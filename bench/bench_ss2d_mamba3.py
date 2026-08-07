"""2D Mamba-3 cross-scan: our kernel vs PyTorch, at real vision grid sizes.

This is the throughput half of Path C. The accuracy half does not exist and
cannot: no 2D Mamba-3 weights have ever been published, so there is nothing to
measure quality against. Correctness and speed are both weight-independent —
a scan over random weights performs identical arithmetic in an identical memory
pattern — so those we can report, and we report only those.

BASELINES
---------
  ref_eager    the PyTorch recurrence, in fp32. This is what a CPU user would
               have to write, since no 2D Mamba-3 implementation exists at all,
               on any device.
  ref_compile  torch.compile of the same thing. The number that matters. It is
               reported with its COMPILE TIME, because for a sequential scan
               over H*W tokens that cost grows with the grid.

Note the reference here is the f64 oracle run at fp32 (`dtype=torch.float32`).
Timing an f64 baseline against an fp32 kernel would inflate every row.

CORRECTNESS GATES SPEED, per CLAUDE.md: every grid's output is diffed against
the reference before its timing is reported.

Grid sizes are the ones vision models actually use — a 224x224 image at patch
16 is 14x14, at patch 8 is 28x28, and 56x56 is a typical early VMamba stage.

Usage:
    python bench/bench_ss2d_mamba3.py [--quick] [--json out.json]
"""

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tests"))

from arm_scan.ss2d_mamba3 import ss2d_scan_mamba3  # noqa: E402
from reference.mamba3_ref import mamba3_siso_ref  # noqa: E402

# (name, batch, H, W, heads, dv, dqk)
GRIDS = [
    ("14x14_p16", 1, 14, 14, 8, 32, 64),
    ("28x28_p8", 1, 28, 28, 8, 32, 64),
    ("56x56_stage1", 1, 56, 56, 8, 32, 64),
]
QUICK = ["14x14_p16"]


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def make_case(b, hh, ww, h, dv, dqk, seed=0):
    g = torch.Generator().manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=g)

    r = dqk // 4
    dt = torch.nn.functional.softplus(rn(b, h, hh, ww) * 0.5 - 4.5)
    return dict(
        q=rn(b, hh, ww, 1, dqk), k=rn(b, hh, ww, 1, dqk),
        v=rn(b, hh, ww, h, dv), z=rn(b, hh, ww, h, dv),
        adt=-torch.exp(rn(b, h, hh, ww) * 0.5) * dt,
        dt=dt, trap=rn(b, h, hh, ww),
        q_bias=rn(h, dqk), k_bias=rn(h, dqk),
        angles=rn(b, hh, ww, h, r), D=rn(h),
    )


def ref_scan_pair(q, k, v, adt, dt, trap, q_bias, k_bias, angles=None,
                  D=None, z=None):
    """`mamba3_scan_pair`'s signature, backed by the reference at fp32."""
    common = dict(Q_bias=q_bias, K_bias=k_bias, Angles=angles, D=D, Z=z,
                  dtype=torch.float32)
    args = dict(Q=q, K=k, V=v, ADT=adt, DT=dt, Trap=trap)
    return (mamba3_siso_ref(**args, **common, reverse=False),
            mamba3_siso_ref(**args, **common, reverse=True))


def bench(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--max-compile-tokens", type=int, default=256,
                    help="skip torch.compile past this H*W. The graph grows "
                         "with the sequence for a scan, so this bounds a "
                         "compile that would dominate the run")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    names = QUICK if args.quick else [g[0] for g in GRIDS]
    grids = [g for g in GRIDS if g[0] in names]

    meta = {"host": platform.node(), "platform": platform.platform(),
            "machine": platform.machine(), "torch": torch.__version__,
            "threads": torch.get_num_threads(), "git": git_sha()}
    print("2D Mamba-3 cross-scan — 4 directions as two traversal pairs")
    print(f"{meta['machine']}  torch {meta['torch']}  "
          f"threads {meta['threads']}  git {meta['git']}")
    print("\nNo 2D Mamba-3 exists on ANY device, so the baseline is the "
          "PyTorch recurrence.\n")

    hdr = (f"{'grid':>13}  {'tokens':>7}  {'kernel ms':>10}  {'ref ms':>10}  "
           f"{'speedup':>8}  {'compiled ms':>12}  {'vs comp':>8}  "
           f"{'rel err':>9}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for name, b, hh, ww, h, dv, dqk in grids:
        c = make_case(b, hh, ww, h, dv, dqk)
        call = (c["q"], c["k"], c["v"], c["adt"], c["dt"], c["trap"],
                c["q_bias"], c["k_bias"], c["angles"])
        kw = dict(D=c["D"], z=c["z"], merge="sum")

        got = ss2d_scan_mamba3(*call, **kw)
        want = ss2d_scan_mamba3(*call, **kw, scan_pair=ref_scan_pair)
        rel = float((got - want).abs().max()) / max(
            float(want.abs().max()), 1e-30)
        if rel > 1e-4:
            print(f"{name:>13}  CORRECTNESS FAILED: rel {rel:.3e} — "
                  f"not timing this grid")
            rows.append({"grid": name, "correctness": "FAIL", "rel": rel})
            continue

        t_k = bench(lambda: ss2d_scan_mamba3(*call, **kw),
                    args.warmup, args.reps)
        t_r = bench(lambda: ss2d_scan_mamba3(*call, **kw,
                                             scan_pair=ref_scan_pair),
                    args.warmup, args.reps)

        t_c = None
        if not args.no_compile and hh * ww <= args.max_compile_tokens:
            try:
                comp = torch.compile(ref_scan_pair, dynamic=False)
                ss2d_scan_mamba3(*call, **kw, scan_pair=comp)  # compile
                t_c = bench(lambda: ss2d_scan_mamba3(*call, **kw,
                                                     scan_pair=comp),
                            0, args.reps)
            except Exception as exc:  # noqa: BLE001
                print(f"{name:>13}  (torch.compile failed: "
                      f"{type(exc).__name__})")

        print(f"{name:>13}  {hh * ww:>7}  {t_k * 1e3:10.2f}  "
              f"{t_r * 1e3:10.2f}  {t_r / t_k:7.2f}x  "
              f"{(f'{t_c * 1e3:12.2f}' if t_c else f'{chr(45):>12}')}  "
              f"{(f'{t_c / t_k:7.2f}x' if t_c else f'{chr(45):>8}')}  "
              f"{rel:9.2e}")
        rows.append({
            "grid": name, "correctness": "PASS", "tokens": hh * ww,
            "kernel_ms": t_k * 1e3, "ref_eager_ms": t_r * 1e3,
            "ref_compile_ms": t_c * 1e3 if t_c else None,
            "speedup_vs_eager": t_r / t_k,
            "speedup_vs_compile": t_c / t_k if t_c else None, "rel": rel,
        })

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"meta": meta, "rows": rows}, indent=2))
        print(f"\nwrote {args.json}")
    return 0 if all(r.get("correctness") == "PASS" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
