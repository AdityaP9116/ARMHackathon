"""Smoke check: do BOTH topologies run, correctly, and faster than native torch?

A fast consolidated sanity test (not a full benchmark) that, across a sweep of
sequence lengths, verifies:

  1. unidirectional  arm_scan.selective_scan       vs  the vendored torch reference
  2. bidirectional   arm_scan.bidirectional_scan   vs  the same reference, both directions

For each (L, topology): correctness (max_abs vs an f64 reference, gated at 1e-4)
and the median speedup over native PyTorch eager. Exits non-zero if any case is
wrong — so it doubles as a CI gate that "both topologies work and beat torch
across L".

No torch.compile here (that is what the full bench/bench_*.py are for, and it is
what OOM'd the runner at L=8192 by unrolling the recurrence into a graph — the
eager reference used here builds no graph, but the default cap stays conservative
regardless). The practical ceiling on L is the TORCH reference (an O(L)
Python-loop scan), not our kernel — which runs in constant memory at any L.

Usage:
    python bench/smoke_topologies.py                         # sweep 512,2048,4096
    python bench/smoke_topologies.py --lengths 512,4096,8192 --reps 3
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "bench"))

import arm_scan  # noqa: E402
from reference import selective_scan_ref  # noqa: E402
from bench_op import make_inputs  # noqa: E402  (shared realistic distribution)

MAX_ABS = 1e-4  # project-wide acceptance gate


def flip_t(x):
    return torch.flip(x, dims=(-1,))


def median_ms(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3


def uni_ref(t, compute_dtype=torch.float32):
    return selective_scan_ref(
        t["u"], t["delta"], t["A"], t["B"], t["C"], D=t["D"], z=t["z"],
        delta_bias=t["delta_bias"], delta_softplus=True,
        compute_dtype=compute_dtype)


def uni_kernel(t):
    return arm_scan.selective_scan(
        t["u"], t["delta"], t["A"], t["B"], t["C"], D=t["D"], z=t["z"],
        delta_bias=t["delta_bias"], delta_softplus=True)


def bidi_ref(t, compute_dtype=torch.float32):
    """Both directions, merged (sum) — what a real bidirectional Mamba does."""
    fwd = uni_ref(t, compute_dtype)
    back = selective_scan_ref(
        flip_t(t["u"]), flip_t(t["delta"]), t["A"], flip_t(t["B"]),
        flip_t(t["C"]), D=t["D"], z=flip_t(t["z"]),
        delta_bias=t["delta_bias"], delta_softplus=True,
        compute_dtype=compute_dtype)
    return fwd + flip_t(back)


def bidi_kernel(t):
    return arm_scan.bidirectional_scan(
        t["u"], t["delta"], t["A"], t["B"], t["C"], D=t["D"], z=t["z"],
        delta_bias=t["delta_bias"], delta_softplus=True, merge="sum")


def check(length, name, kernel_fn, ref_fn, t, warmup, reps):
    """Time one topology at one length; print a table row. Return (ok, speedup)."""
    # Correctness vs a TRUE f64 reference: upcast the inputs AND run the
    # reference in float64. `selective_scan_ref` casts its result back to the
    # input dtype and defaults to compute_dtype=float32, so passing f64 inputs
    # without float64 compute would silently give an f32 "ground truth".
    f64 = {k: (v.double() if torch.is_tensor(v) else v) for k, v in t.items()}
    ref_f64 = ref_fn(f64, compute_dtype=torch.float64)
    got = kernel_fn(t)
    max_abs = (got.double() - ref_f64).abs().max().item()

    # Speed vs native torch EAGER in f32 — the honest baseline.
    kern_ms = median_ms(lambda: kernel_fn(t), warmup, reps)
    eager_ms = median_ms(lambda: ref_fn(t), warmup, reps)
    speedup = eager_ms / kern_ms
    ok = max_abs < MAX_ABS
    print(f"  {length:6d}  {name:15s} {'PASS' if ok else 'FAIL'}  "
          f"{max_abs:.1e}  {kern_ms:9.2f}  {eager_ms:10.2f}  {speedup:6.1f}x")
    return ok, speedup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="512,2048,4096",
                    help="comma-separated L values to sweep")
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--state", type=int, default=16)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    print(f"smoke check — B1 D{args.dim} N{args.state}, sweep L={lengths}  "
          f"reps={args.reps}")
    print(f"kernel: {arm_scan.lib_path()}")
    print("\nNote: the ceiling here is the TORCH reference (an O(L) Python-loop\n"
          "scan we time against), not our kernel — which runs in constant memory\n"
          "at any L. As L grows the eager baseline is what gets slow, which is\n"
          "itself the point: torch is what can't keep up.\n")
    print(f"  {'L':>6}  {'topology':15s} gate  max_abs   kernel(ms)   "
          f"eager(ms)  speedup")

    uni_sp, bidi_sp, all_ok = [], [], True
    with torch.no_grad():
        for length in lengths:
            t = make_inputs(1, args.dim, length, args.state)
            ok_u, sp_u = check(length, "unidirectional", uni_kernel, uni_ref,
                               t, args.warmup, args.reps)
            ok_b, sp_b = check(length, "bidirectional", bidi_kernel, bidi_ref,
                               t, args.warmup, args.reps)
            uni_sp.append(sp_u)
            bidi_sp.append(sp_b)
            all_ok = all_ok and ok_u and ok_b

    print()
    if all_ok:
        print(f"BOTH TOPOLOGIES OK across L={lengths} — vs native torch eager: "
              f"unidirectional {min(uni_sp):.1f}–{max(uni_sp):.1f}x, "
              f"bidirectional {min(bidi_sp):.1f}–{max(bidi_sp):.1f}x.")
    else:
        print("SMOKE CHECK FAILED — a topology is incorrect (see FAIL above).")
        sys.exit(1)


if __name__ == "__main__":
    main()
