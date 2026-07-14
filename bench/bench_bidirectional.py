"""What do the flip copies actually cost? — the gate on TOPOLOGY_IMPLEMENTATION_PLAN.md §2.2.

The bidirectional scan is correct today (plan §2.1) but pays for six full-tensor
copies per call: it flips u/delta/B/C/z to scan backward, then flips the output
back. A fused `reverse` flag in the kernel (plan §2.2) would delete all six by
walking the sequence backward in place. Whether that is worth building depends
entirely on what the copies cost — which nobody has measured. This measures it.

WHAT IS TIMED

  scan_fwd          one ordinary forward scan. The floor; half the scan work.
  fused_estimate    two forward scans + the merge, with NO flips anywhere.
                    This is the PROXY for a fused `reverse` implementation: the
                    same scan work and the same merge, minus exactly the copy
                    traffic the flag would remove.
  bidirectional     the real thing today: flips + two scans + un-flip + merge.
  flips_only        the six flips alone, no scans — isolates the copy cost so
                    it can be read directly rather than inferred by subtraction.

THE NUMBER THAT DECIDES §2.2

  fusion_headroom = bidirectional / fused_estimate

  ~1.0x  -> the flips are noise. Do NOT build the fused kernel; ship §2.1, and
            say so honestly in the writeup.
  >1.15x -> the flips are real. The fused `reverse` flag pays for itself.

HONESTY ABOUT THE PROXY
`fused_estimate` is an *upper bound* on what fusion can achieve, not a
measurement of the fused path (which does not exist yet). It runs the identical
scan work with zero flip traffic, so it cannot be beaten by a real fused
kernel — a real one also reads the sequence backward, which is less
cache-friendly than the forward stream timed here. Treat the headroom as a
ceiling: if the ceiling is low, fusion is definitively not worth it; if it is
high, the real win will be somewhat less. Reported as such, never as a speedup
the kernel has achieved.

Usage:
    python bench/bench_bidirectional.py            # sweep over L
    python bench/bench_bidirectional.py --quick    # CI-sized
    python bench/bench_bidirectional.py --json out.json
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "bench"))

import arm_scan  # noqa: E402
# Reuse the harness rather than re-deriving it: the realistic value
# distribution (negative A, softplus delta in ~[1e-3, 0.1]) is load-bearing for
# these numbers, and a second copy of it would drift.
from bench_op import bench, env_report, git_sha, make_inputs  # noqa: E402

# Flip traffic is O(B·D·L) while scan work is O(B·D·L·N), so the ratio is
# roughly independent of L in theory — but cache behaviour is not, which is the
# whole reason to sweep it rather than assume.
SUITES = {
    "sweep-len": [(1, 768, l, 16)
                  for l in (128, 512, 1024, 2048, 4096, 8192)],
    "sweep-dim": [(1, d, 1024, 16) for d in (256, 768, 1536, 3072)],
    "basic": [(1, 768, 512, 16), (1, 768, 2048, 16), (1, 1536, 1024, 16)],
}
QUICK_SHAPES = [(1, 768, 128, 16), (1, 768, 512, 16)]

_FLIP_KEYS = ("u", "delta", "B", "C", "z")   # the time-varying tensors


def flip_time(t):
    return torch.flip(t, dims=(-1,))


def scan_fwd(t):
    return arm_scan.selective_scan(
        t["u"], t["delta"], t["A"], t["B"], t["C"], D=t["D"], z=t["z"],
        delta_bias=t["delta_bias"], delta_softplus=True)


def fused_estimate(t):
    """Two forward scans + merge, zero flips — the ceiling a fused `reverse`
    flag could reach. Not a real backward scan: it computes the wrong answer on
    purpose, because we are timing the WORK, not the result. (Correctness of the
    real path is `tests/check_bidirectional.py`'s job, and it is green.)"""
    return scan_fwd(t) + scan_fwd(t)


def bidirectional(t):
    return arm_scan.bidirectional_scan(
        t["u"], t["delta"], t["A"], t["B"], t["C"], D=t["D"], z=t["z"],
        delta_bias=t["delta_bias"], delta_softplus=True, merge="sum")


def flips_only(t):
    """The six copies in isolation: five time-varying inputs in, one output
    back. The list keeps every flip referenced so none can be elided."""
    flipped = [flip_time(t[k]) for k in _FLIP_KEYS]   # 5 input copies
    return flip_time(flipped[0])                       # 1 output copy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=sorted(SUITES), default="sweep-len")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--torch-threads", type=int, default=None)
    ap.add_argument("--tag", type=str, default=platform.node())
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)

    shapes = QUICK_SHAPES if args.quick else SUITES[args.suite]
    reps = args.reps or (5 if args.quick else 10)
    warmup = args.warmup if args.warmup is not None else (1 if args.quick else 3)

    env = env_report()
    env["tag"] = args.tag
    env["git_sha"] = git_sha()
    env["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print("environment:")
    for k, v in env.items():
        print(f"  {k}: {v}")
    print(f"\nflip-overhead study — suite="
          f"{'quick' if args.quick else args.suite} reps={reps} "
          f"warmup={warmup}")
    print("fusion_headroom = bidirectional / fused_estimate  "
          "(a CEILING on what plan §2.2 could win)\n")

    results = {
        "kind": "bidirectional-flip-overhead",
        "env": env,
        "reps": reps,
        "suite": "quick" if args.quick else args.suite,
        "shapes": [],
    }

    with torch.no_grad():
        for batch, dim, length, state in shapes:
            label = f"B{batch}_D{dim}_L{length}_N{state}"
            print(f"=== {label} ===")
            t = make_inputs(batch, dim, length, state)

            # Sanity: the thing we are timing must be the thing that is
            # correct. bidirectional() must equal an explicit flip-scan-flip
            # sum, or the benchmark is measuring the wrong code path.
            manual = scan_fwd(t) + flip_time(scan_fwd(
                {**{k: flip_time(t[k]) for k in _FLIP_KEYS},
                 "A": t["A"], "D": t["D"], "delta_bias": t["delta_bias"]}))
            drift = (bidirectional(t) - manual).abs().max().item()
            if drift > 0.0:
                print(f"  !! bidirectional() does not match flip-scan-flip "
                      f"(max_abs={drift:.3e}) — benchmark is untrustworthy")
                sys.exit(1)

            row = {"shape": [batch, dim, length, state], "timings": {}}
            for name, fn in (
                ("scan_fwd", scan_fwd),
                ("fused_estimate", fused_estimate),
                ("bidirectional", bidirectional),
                ("flips_only", flips_only),
            ):
                r = bench(lambda fn=fn: fn(t), warmup, reps)
                row["timings"][name] = r
                print(f"  {name:16s} {r['median_s']*1e3:9.3f} ms")

            bi = row["timings"]["bidirectional"]["median_s"]
            fu = row["timings"]["fused_estimate"]["median_s"]
            fl = row["timings"]["flips_only"]["median_s"]

            headroom = bi / fu
            flip_share = (bi - fu) / bi * 100.0
            row["fusion_headroom"] = headroom
            row["flip_share_pct"] = flip_share
            row["flips_only_share_pct"] = fl / bi * 100.0

            verdict = ("flips are NOISE — do not fuse" if headroom < 1.05
                       else "marginal" if headroom < 1.15
                       else "flips are REAL — fusion pays")
            print(f"  -> fusion_headroom {headroom:.3f}x  "
                  f"(flips = {flip_share:.1f}% of bidirectional runtime; "
                  f"flips_only measures {row['flips_only_share_pct']:.1f}%)")
            print(f"  -> {verdict}\n")
            results["shapes"].append(row)

    hs = [r["fusion_headroom"] for r in results["shapes"]]
    print(f"fusion_headroom across shapes: min={min(hs):.3f}x "
          f"max={max(hs):.3f}x")
    print("reminder: this is a CEILING (see module docstring) — a real fused "
          "kernel reads backward and will land under it.")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nresults written to {args.json}")


if __name__ == "__main__":
    main()
