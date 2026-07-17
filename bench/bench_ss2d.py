"""P0-2 (SS2D_REPOSITIONING_PLAN §5): measure the unfused SS2D path at the
REAL diffusion-workload shapes, and split block time into kernel-scan time
vs everything else (flips/permutes/projections) — the number that decides
whether the fused selective_scan_2d week is justified (>15% overhead rule).

Shapes from the locked backbone: level-1 384x320 grid @ inner=96
(dim=64 x expand 1.5), level-2 192x160 @ inner=192 (dim=128); seed-batch
1 and 4. The torch-reference comparison runs at a 96x80 mini-grid (its
Python loop is linear in L; full-grid ref timing would be minutes/call).

Usage: python bench/bench_ss2d.py [--tag TAG] [--json PATH] [--reps N]
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
from arm_scan.ss2d import scan_fn_arm, use_arm_scan  # noqa: E402


def timed_block(blk, x, emb, reps, warmup=1):
    scan_t = []
    orig = blk.scan_fn

    def wrapped(*a, **k):
        t0 = time.perf_counter()
        r = orig(*a, **k)
        scan_t.append(time.perf_counter() - t0)
        return r

    blk.scan_fn = wrapped
    times = []
    with torch.no_grad():
        for i in range(warmup + reps):
            scan_t.clear()
            t0 = time.perf_counter()
            blk(x, emb)
            if i >= warmup:
                times.append((time.perf_counter() - t0, sum(scan_t)))
    blk.scan_fn = orig
    tot = statistics.median(t[0] for t in times)
    scan = statistics.median(t[1] for t in times)
    return tot, scan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=platform.node())
    ap.add_argument("--json", default=None)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    torch.manual_seed(0)
    cases = [
        ("L1_384x320_in96_b1", 64, 384, 320, 1),
        ("L1_384x320_in96_b4", 64, 384, 320, 4),
        ("L2_192x160_in192_b1", 128, 192, 160, 1),
        ("L2_192x160_in192_b4", 128, 192, 160, 4),
        ("mini_96x80_in96_b1", 64, 96, 80, 1),  # ref-comparable
    ]
    out = {"kind": "ss2d", "tag": args.tag, "host": platform.platform(),
           "machine": platform.machine(), "torch": torch.__version__,
           "reps": args.reps, "cases": []}
    print(f"host {platform.platform()} / {platform.machine()}, "
          f"torch {torch.__version__}\n")

    for name, dim, h, w, b in cases:
        blk = SS2DBlock(dim, emb_dim=64, d_state=16).eval()
        x = torch.randn(b, dim, h, w)
        emb = torch.randn(b, 64)
        use_arm_scan(blk)
        tot, scan = timed_block(blk, x, emb, args.reps)
        ovh = 100 * (tot - scan) / tot
        row = {"case": name, "arm_total_s": tot, "arm_scan_s": scan,
               "overhead_pct": ovh}
        print(f"{name:24s} arm total {tot*1e3:8.1f} ms  "
              f"scan {scan*1e3:8.1f} ms  overhead {ovh:5.1f}%")
        if name.startswith("mini"):
            blk.scan_fn = None
            use_arm_scan(blk, enable=False)
            rtot, _ = timed_block(blk, x, emb, max(1, args.reps - 1))
            row["ref_total_s"] = rtot
            print(f"{'':24s} torch-ref total {rtot*1e3:8.1f} ms  "
                  f"({rtot/tot:.1f}x slower)")
        out["cases"].append(row)

    ovhs = [c["overhead_pct"] for c in out["cases"]
            if not c["case"].startswith("mini")]
    verdict = max(ovhs)
    out["fused_kernel_justified"] = verdict > 15.0
    print(f"\nP1-7 go/no-go: worst real-shape overhead {verdict:.1f}% "
          f"-> fused selective_scan_2d "
          f"{'JUSTIFIED' if verdict > 15 else 'NOT justified'} "
          f"(15% rule)")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"results written to {args.json}")


if __name__ == "__main__":
    main()
