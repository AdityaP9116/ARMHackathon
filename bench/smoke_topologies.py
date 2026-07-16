"""Smoke check: do BOTH topologies run, correctly, and faster than native torch?

A fast consolidated sanity test (not a full benchmark) that, for one realistic
shape, verifies:

  1. unidirectional  arm_scan.selective_scan       vs  the vendored torch reference
  2. bidirectional   arm_scan.bidirectional_scan   vs  the same reference, both directions

For each: correctness (max_abs vs an f64 reference, gated at 1e-4) and the median
speedup over native PyTorch eager. Exits non-zero if either is wrong or slower —
so it doubles as a CI gate that "both topologies work and beat torch".

No torch.compile here (that is what the full bench/bench_*.py are for) — this is
meant to run in seconds. Uses the same vendored reference and value distribution
as the real benchmarks, so the numbers are comparable, just single-shape.

Usage:
    python bench/smoke_topologies.py
    python bench/smoke_topologies.py --shape 1,768,512,16 --reps 5
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


def check(name, kernel_fn, ref_fn, t, warmup, reps):
    """Return (ok, max_abs, speedup)."""
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
    status = "PASS" if ok else "FAIL"
    print(f"  {name:16s} {status}  max_abs={max_abs:.2e} (gate {MAX_ABS:g})  "
          f"kernel={kern_ms:7.2f}ms  eager={eager_ms:8.2f}ms  "
          f"=> {speedup:5.1f}x vs torch")
    return ok, speedup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="1,768,512,16", help="B,D,L,N")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    batch, dim, length, state = (int(x) for x in args.shape.split(","))
    print(f"smoke check — shape B{batch} D{dim} L{length} N{state}  "
          f"reps={args.reps}")
    print(f"kernel: {arm_scan.lib_path()}\n")

    with torch.no_grad():
        t = make_inputs(batch, dim, length, state)
        ok_u, sp_u = check("unidirectional", uni_kernel, uni_ref, t,
                           args.warmup, args.reps)
        ok_b, sp_b = check("bidirectional", bidi_kernel, bidi_ref, t,
                           args.warmup, args.reps)

    print()
    if ok_u and ok_b:
        print(f"BOTH TOPOLOGIES OK — unidirectional {sp_u:.1f}x, "
              f"bidirectional {sp_b:.1f}x faster than native torch eager.")
    else:
        print("SMOKE CHECK FAILED — a topology is incorrect (see FAIL above).")
        sys.exit(1)


if __name__ == "__main__":
    main()
