"""Causal vs non-causal 2D Mamba-3, and scan-form vs dense-form.

**This is the measurement Path C exists to produce.** Nobody has published a
causal-vs-non-causal comparison for any Mamba generation, on any hardware, and
the CPU is where the two formulations' asymptotics actually collide.

WHAT IS BEING COMPARED, AND WHY IT IS NOT WHAT THE PLAN EXPECTED
----------------------------------------------------------------
The plan assumed non-causal would mean "a second kernel, two dense GEMMs",
O(L^2), with a thin moat because BLAS is good at GEMMs. The maths says
otherwise: the decay `e^(L_t - L_s)` **factorises**, so the sum over all `s`
splits into a forward scan plus a backward scan minus the double-counted
diagonal. Three formulations therefore exist, and all three are timed here:

  causal        one scan.                       O(L * dv * dqk)
  non-causal    two scans + a diagonal pass.    O(L * dv * dqk), ~2x causal
  dense         the explicit (L, L) mask.       O(L^2 * (dqk + dv)), pure GEMM

The interesting question is where `dense` stops being competitive. It has the
better constant (GEMMs, BLAS, no sequential dependency) and the worse
asymptotics, so there is a crossover, and its location is the result.

CORRECTNESS GATES SPEED: every grid is checked against the dense form before
any timing is reported.

THE ORACLE CAVEAT: no authoritative non-causal implementation exists (VNCT's
code is unreleased), so these numbers describe *our* reading of the operator.
Timing is not affected by that -- the arithmetic is what it is -- but no
accuracy claim is available and none is made.

Usage:
    python bench/bench_mamba3_noncausal.py [--quick] [--json out.json]
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

from arm_scan.mamba3 import mamba3_scan  # noqa: E402
from arm_scan.mamba3_noncausal import (  # noqa: E402
    noncausal_scan, noncausal_scan_dense, ss2d_noncausal_mamba3)
from arm_scan.ss2d_mamba3 import ss2d_scan_mamba3  # noqa: E402

# (name, batch, H, W, heads, dv, dqk) — real vision grids.
GRIDS = [
    ("8x8", 1, 8, 8, 8, 32, 64),
    ("14x14_p16", 1, 14, 14, 8, 32, 64),
    ("28x28_p8", 1, 28, 28, 8, 32, 64),
    ("56x56_stage1", 1, 56, 56, 8, 32, 64),
]
QUICK = ["8x8", "14x14_p16"]


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def make_grid(b, hh, ww, h, dv, dqk, seed=0):
    g = torch.Generator().manual_seed(seed)

    def rn(*s):
        return torch.randn(*s, generator=g)

    dt = torch.nn.functional.softplus(rn(b, h, hh, ww) * 0.5 - 4.5)
    return dict(
        q=rn(b, hh, ww, 1, dqk), k=rn(b, hh, ww, 1, dqk),
        v=rn(b, hh, ww, h, dv), z=rn(b, hh, ww, h, dv),
        adt=-torch.exp(rn(b, h, hh, ww) * 0.5) * dt, dt=dt,
        trap=rn(b, h, hh, ww), q_bias=rn(h, dqk), k_bias=rn(h, dqk),
        angles=rn(b, hh, ww, h, dqk // 4), D=rn(h))


def flatten_rows(c, b, hh, ww, h):
    """The row-major 1D view — what the dense form is timed on."""
    return {kk: (vv.reshape(b, hh * ww, *vv.shape[3:]) if vv.dim() >= 5
                 else vv.reshape(b, h, hh * ww) if vv.dim() == 4 else vv)
            for kk, vv in c.items()}


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
    ap.add_argument("--max-dense-tokens", type=int, default=1024,
                    help="skip the dense form past this H*W. It allocates an "
                         "(L, L) mask per head, so memory alone rules it out "
                         "well before time does -- which is itself the finding")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    names = QUICK if args.quick else [g[0] for g in GRIDS]
    grids = [g for g in GRIDS if g[0] in names]

    meta = {"host": platform.node(), "platform": platform.platform(),
            "machine": platform.machine(), "torch": torch.__version__,
            "threads": torch.get_num_threads(), "git": git_sha()}
    print("Causal vs non-causal 2D Mamba-3 — and scan-form vs dense-form")
    print(f"{meta['machine']}  torch {meta['torch']}  "
          f"threads {meta['threads']}  git {meta['git']}\n")
    print("No authoritative non-causal implementation exists (VNCT unreleased);"
          "\nthese time OUR reading of the operator.\n")

    print("1D columns isolate the cost of dropping causality (1 scan vs 2).")
    print("2D columns are the four-direction cross-scan, which ALREADY runs")
    print("both directions -- so there non-causal only adds a diagonal pass.\n")
    hdr = (f"{'grid':>13}  {'tokens':>7}  {'1D causal':>10}  {'1D nc':>9}  "
           f"{'nc/causal':>10}  {'2D causal':>10}  {'2D nc':>9}  "
           f"{'2D ratio':>9}  {'dense ms':>9}  {'dense/1Dnc':>11}  "
           f"{'rel err':>9}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for name, b, hh, ww, h, dv, dqk in grids:
        c = make_grid(b, hh, ww, h, dv, dqk)
        call = (c["q"], c["k"], c["v"], c["adt"], c["dt"], c["trap"],
                c["q_bias"], c["k_bias"], c["angles"])
        kw = dict(D=c["D"], z=c["z"])
        tokens = hh * ww

        # Correctness first: the 2D non-causal rows plane against the dense
        # form on the same flattened ordering.
        flat = flatten_rows(c, b, hh, ww, h)
        rows_plane, _ = ss2d_noncausal_mamba3(*call, **kw, merge="none")
        want = noncausal_scan_dense(
            flat["q"], flat["k"], flat["v"], flat["adt"], flat["dt"],
            flat["trap"], flat["q_bias"], flat["k_bias"],
            angles=flat["angles"], D=flat["D"], z=flat["z"])
        rel = float((rows_plane.reshape(want.shape) - want).abs().max()) / max(
            float(want.abs().max()), 1e-30)
        if rel > 1e-4:
            print(f"{name:>13}  CORRECTNESS FAILED: rel {rel:.3e}")
            rows.append({"grid": name, "correctness": "FAIL", "rel": rel})
            continue

        # 1D: the honest cost of dropping causality -- one scan vs two.
        f1 = (flat["q"], flat["k"], flat["v"], flat["adt"], flat["dt"],
              flat["trap"], flat["q_bias"], flat["k_bias"])
        fkw = dict(angles=flat["angles"], D=flat["D"], z=flat["z"])
        t_1d_causal = bench(lambda: mamba3_scan(*f1, **fkw),
                            args.warmup, args.reps)
        t_1d_nc = bench(lambda: noncausal_scan(*f1, **fkw),
                        args.warmup, args.reps)

        # 2D: the cross-scan already runs both directions, so this ratio is
        # expected to sit near 1.0 -- it is measuring the diagonal pass, not a
        # second scan.
        t_causal = bench(lambda: ss2d_scan_mamba3(*call, **kw, merge="sum"),
                         args.warmup, args.reps)
        t_nc = bench(lambda: ss2d_noncausal_mamba3(*call, **kw, merge="sum"),
                     args.warmup, args.reps)

        t_dense = None
        if tokens <= args.max_dense_tokens:
            try:
                def run_dense():
                    return noncausal_scan_dense(
                        flat["q"], flat["k"], flat["v"], flat["adt"],
                        flat["dt"], flat["trap"], flat["q_bias"],
                        flat["k_bias"], angles=flat["angles"], D=flat["D"],
                        z=flat["z"])
                t_dense = bench(run_dense, args.warmup, args.reps)
            except RuntimeError as exc:  # OOM on the (L, L) mask
                print(f"{name:>13}  (dense form failed: {str(exc)[:40]})")

        print(f"{name:>13}  {tokens:>7}  {t_1d_causal * 1e3:10.2f}  "
              f"{t_1d_nc * 1e3:9.2f}  {t_1d_nc / t_1d_causal:9.2f}x  "
              f"{t_causal * 1e3:10.2f}  {t_nc * 1e3:9.2f}  "
              f"{t_nc / t_causal:8.2f}x  "
              f"{(f'{t_dense * 1e3:9.2f}' if t_dense else f'{chr(45):>9}')}  "
              f"{(f'{t_dense / t_1d_nc:10.2f}x' if t_dense else f'{chr(45):>11}')}"
              f"  {rel:9.2e}")
        rows.append({
            "grid": name, "correctness": "PASS", "tokens": tokens,
            "d1_causal_ms": t_1d_causal * 1e3, "d1_noncausal_ms": t_1d_nc * 1e3,
            "d1_noncausal_over_causal": t_1d_nc / t_1d_causal,
            "d2_causal_ms": t_causal * 1e3, "d2_noncausal_ms": t_nc * 1e3,
            "d2_noncausal_over_causal": t_nc / t_causal,
            "dense_ms": t_dense * 1e3 if t_dense else None,
            "dense_over_1d_noncausal": t_dense / t_1d_nc if t_dense else None,
            "rel": rel,
        })

    ok_rows = [r for r in rows if r.get("correctness") == "PASS"]
    noisy = [r["grid"] for r in ok_rows if r["d1_causal_ms"] < 1.0]
    if noisy:
        print(f"\nNOTE: {', '.join(noisy)} time under 1 ms per call. Medians "
              f"that short are\ndominated by dispatch overhead on an "
              f"unquiesced box — read those rows as\norder-of-magnitude only, "
              f"not to two significant figures.")
    if ok_rows:
        print("\nThe cost of dropping causality:")
        for r in ok_rows:
            print(f"  {r['grid']:<14} {r['tokens']:>5} tok   "
                  f"1D {r['d1_noncausal_over_causal']:.2f}x   "
                  f"2D {r['d2_noncausal_over_causal']:.2f}x")
        print("\nScan form vs dense form (vs 1D non-causal):")
        for r in ok_rows:
            if r["dense_over_1d_noncausal"] is None:
                print(f"  {r['grid']:<14} {r['tokens']:>5} tok   "
                      f"dense skipped (the (L,L) mask is the limit)")
            else:
                x = r["dense_over_1d_noncausal"]
                who = "dense wins" if x < 1 else "scan wins"
                print(f"  {r['grid']:<14} {r['tokens']:>5} tok   "
                      f"dense/scan {x:.2f}x   -> {who}")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"meta": meta, "rows": rows}, indent=2))
        print(f"\nwrote {args.json}")
    return 0 if all(r.get("correctness") == "PASS" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
