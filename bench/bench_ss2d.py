"""SS2D at the REAL diffusion-workload shapes.

Answers three questions, each with its own row in the JSON:

1. **What did the traversal-pair rewrite buy?** `legacy` stacks the four
   directions into one forward call (Pass A computed 4x, four full-tensor
   `torch.flip` copies); `pair` runs two fused bidirectional calls (Pass A
   computed 2x, no flips). Both go through the same kernel, so the ratio is
   attributable to the rewrite alone.

2. **How much of block time is still NOT scan?** The flip/permute/projection
   overhead split that gates the fully-fused `selective_scan_2d` (>15% rule,
   TOPOLOGY_IMPLEMENTATION_PLAN §3.2).

3. **How does it compare against the baselines that matter?** torch eager and
   `torch.compile` on the reference scan. Per CLAUDE.md, `torch.compile` is the
   baseline judges trust — so it gets measured, and when it cannot compile the
   reference at all (a sequential Python loop over L), that failure is recorded
   as a result rather than quietly skipped.

Shapes from the locked backbone: level-1 384x320 grid @ inner=96 (dim=64 x
expand 1.5), level-2 192x160 @ inner=192 (dim=128); seed-batch 1 and 4. The
torch-reference comparison runs at smaller grids because its Python loop is
linear in L and full-grid reference timing would be minutes per call.

Usage: python bench/bench_ss2d.py [--tag TAG] [--json PATH] [--reps N]
                                  [--no-compile]
"""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from apps.mri_diffusion.backbone.mamba_ss2d import SS2DBlock  # noqa: E402
from arm_scan.ss2d import use_arm_scan  # noqa: E402


def timed_block(module, x, emb, reps, warmup=1, legacy=False):
    """Median (total block seconds, scan-only seconds) over `reps`.

    `module` may be a `torch.compile` wrapper, whose attribute writes would
    land on the wrapper rather than the block; seams are always installed on
    the underlying module so the scan-time accounting stays correct.
    """
    blk = getattr(module, "_orig_mod", module)
    blk.legacy_cross_scan = legacy
    scan_t = []

    def wrap(fn):
        def inner(*a, **k):
            t0 = time.perf_counter()
            r = fn(*a, **k)
            scan_t.append(time.perf_counter() - t0)
            return r
        return inner

    orig_pair, orig_single = blk.scan_pair_fn, blk.scan_fn
    blk.scan_pair_fn, blk.scan_fn = wrap(orig_pair), wrap(orig_single)
    times = []
    try:
        with torch.no_grad():
            for i in range(warmup + reps):
                scan_t.clear()
                t0 = time.perf_counter()
                module(x, emb)
                if i >= warmup:
                    times.append((time.perf_counter() - t0, sum(scan_t)))
    finally:
        blk.scan_pair_fn, blk.scan_fn = orig_pair, orig_single
        blk.legacy_cross_scan = False
    return (statistics.median(t[0] for t in times),
            statistics.median(t[1] for t in times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=platform.node())
    ap.add_argument("--json", default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-compile", action="store_true",
                    help="skip the torch.compile baseline")
    ap.add_argument("--only", default=None,
                    help="substring filter on case name (time-boxed runs)")
    args = ap.parse_args()

    torch.manual_seed(0)
    # name, dim, h, w, batch, compare-against-reference?
    cases = [
        ("L1_384x320_in96_b1", 64, 384, 320, 1, False),
        ("L1_384x320_in96_b4", 64, 384, 320, 4, False),
        ("L2_192x160_in192_b1", 128, 192, 160, 1, False),
        ("L2_192x160_in192_b4", 128, 192, 160, 4, False),
        ("mini_96x80_in96_b1", 64, 96, 80, 1, True),
        ("tiny_32x32_in96_b1", 64, 32, 32, 1, True),
    ]
    out = {"kind": "ss2d", "tag": args.tag, "host": platform.platform(),
           "machine": platform.machine(), "torch": torch.__version__,
           "threads": torch.get_num_threads(), "reps": args.reps,
           "cases": []}
    print(f"host {platform.platform()} / {platform.machine()}, "
          f"torch {torch.__version__}, {torch.get_num_threads()} threads\n")

    if args.only:
        cases = [c for c in cases if args.only in c[0]]
        if not cases:
            raise SystemExit(f"no case matches --only {args.only!r}")

    for name, dim, h, w, b, with_ref in cases:
        blk = SS2DBlock(dim, emb_dim=64, d_state=16).eval()
        x = torch.randn(b, dim, h, w)
        emb = torch.randn(b, 64)
        use_arm_scan(blk)

        pair_tot, pair_scan = timed_block(blk, x, emb, args.reps)
        leg_tot, leg_scan = timed_block(blk, x, emb, args.reps, legacy=True)
        ovh = 100 * (pair_tot - pair_scan) / pair_tot
        row = {"case": name,
               "arm_total_s": pair_tot, "arm_scan_s": pair_scan,
               "legacy_total_s": leg_tot, "legacy_scan_s": leg_scan,
               "pair_speedup_total": leg_tot / pair_tot,
               "pair_speedup_scan": leg_scan / pair_scan,
               "overhead_pct": ovh}
        print(f"{name:22s} pair {pair_tot*1e3:8.1f} ms  "
              f"legacy {leg_tot*1e3:8.1f} ms  "
              f"pair-speedup {leg_tot/pair_tot:4.2f}x  "
              f"(scan {leg_scan/pair_scan:4.2f}x)  overhead {ovh:4.1f}%")

        if with_ref:
            use_arm_scan(blk, enable=False)
            rtot, _ = timed_block(blk, x, emb, max(1, args.reps - 1))
            row["ref_eager_total_s"] = rtot
            print(f"{'':22s} torch eager {rtot*1e3:9.1f} ms  "
                  f"({rtot/pair_tot:.1f}x slower than kernel)")

            if not args.no_compile:
                try:
                    cblk = torch.compile(blk, dynamic=False)
                    ctot, _ = timed_block(cblk, x, emb, 1, warmup=1)
                    row["ref_compile_total_s"] = ctot
                    print(f"{'':22s} torch.compile {ctot*1e3:7.1f} ms  "
                          f"({ctot/pair_tot:.1f}x slower than kernel)")
                except Exception as exc:  # noqa: BLE001 - the failure IS data
                    msg = f"{type(exc).__name__}: {exc}"[:200]
                    row["ref_compile_total_s"] = None
                    row["ref_compile_error"] = msg
                    print(f"{'':22s} torch.compile FAILED — {msg}")
            use_arm_scan(blk)
        out["cases"].append(row)

    real = [c for c in out["cases"] if c["case"].startswith("L")]
    verdict = max(c["overhead_pct"] for c in real)
    out["fused_kernel_justified"] = verdict > 15.0
    out["pair_speedup_geomean"] = (
        statistics.geometric_mean([c["pair_speedup_total"] for c in real]))
    print(f"\ntraversal-pair rewrite: {out['pair_speedup_geomean']:.2f}x "
          f"geomean on the real shapes (block total, same kernel both sides)")
    print(f"P1-7 go/no-go: worst real-shape overhead {verdict:.1f}% "
          f"-> fully fused selective_scan_2d "
          f"{'JUSTIFIED' if verdict > 15 else 'NOT justified'} (15% rule)")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"results written to {args.json}")


if __name__ == "__main__":
    main()
