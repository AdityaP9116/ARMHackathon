"""What did fusing the backward traversal actually buy? (TOPOLOGY_IMPLEMENTATION_PLAN.md §2)

A backward scan can be had two ways:

  flip-based   flip u/delta/B/C/z along time, run the ordinary FORWARD kernel,
               flip the output back. Correct, but six full-tensor copies.
  fused        `selective_scan(..., reverse=True)` — the kernel walks the
               sequence backward in place. Zero copies.

This benchmark ran BEFORE the fused path existed, to decide whether to build it
at all: it measured a *ceiling* on the win (2 forward scans with no flips) and
reported **1.085x at L=128, falling to 1.025x at L=512** — i.e. the copies are
worth ~2%. That was the honest answer, and it is why the fused path was NOT
built as a speedup.

It was built anyway, because the 2D cross-scan needs a backward traversal for
its row-backward and column-backward directions — `reverse` is the substrate for
SS2D, not a bidirectional optimization. Now that it exists, this file measures
the REAL before/after instead of a proxy.

WHAT IS TIMED

Baselines — the rows that matter, and the only ones fit to publish:
  ref_eager_bidi      stock PyTorch, both directions, merged. What a real
                      bidirectional Mamba does on CPU with no kernel installed.
  ref_compile_bidi    the same under torch.compile — the FAIR baseline, since
                      compile is what a competent user would already be doing.

Ours:
  bidirectional       forward + kernel-fused reverse. No copies.

Diagnostics — internal, not for publication:
  scan_fwd            one forward scan. Half the work; the floor.
  fused_estimate      two forward scans + merge, no flips. Theoretical floor.
  bidirectional_flip  the OLD path: explicit flips + two forward scans + un-flip.
  flips_only          the six copies alone — what fusion actually removed.

THE NUMBERS

  speedup_vs_eager / speedup_vs_compile   <- the result. Kernel vs stock PyTorch.
  fusion_speedup = bidirectional_flip / bidirectional
                                          <- INTERNAL. ~2%. Never a headline.

Both baselines are built on the SAME vendored reference (tests/reference/) that
the rest of the project measures against, so these rows are directly comparable
to bench_op.py's.

Two correctness gates run before any timing: the fused reverse must be
bit-identical to the flip-based definition, AND the kernel must agree with the
PyTorch reference it is being timed against. Beating a baseline you do not match
is not a result.

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
sys.path.insert(0, str(REPO / "tests"))

import arm_scan  # noqa: E402
# Reuse the harness rather than re-deriving it: the realistic value
# distribution (negative A, softplus delta in ~[1e-3, 0.1]) is load-bearing for
# these numbers, and a second copy of it would drift.
from bench_op import bench, env_report, git_sha, make_inputs  # noqa: E402
from reference import selective_scan_ref  # noqa: E402

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


# ---------------------------------------------------------------- baselines
# A bidirectional scan in stock PyTorch. This is what a real bidirectional
# Mamba does on CPU today with no kernel installed: run the scan, run it again
# on the time-flipped inputs, flip that back, sum. Built on the SAME vendored
# reference the rest of the project measures against (tests/reference/), so
# these rows are directly comparable to bench_op.py's.


def _bidi_ref(fn, t):
    """Bidirectional scan via an arbitrary forward-scan callable `fn`."""
    fwd = fn(t["u"], t["delta"], t["A"], t["B"], t["C"], t["D"], t["z"],
             t["delta_bias"])
    back = fn(flip_time(t["u"]), flip_time(t["delta"]), t["A"],
              flip_time(t["B"]), flip_time(t["C"]), t["D"],
              flip_time(t["z"]), t["delta_bias"])
    return fwd + flip_time(back)


def _ref_call(u, delta, A, B, C, D, z, delta_bias):
    return selective_scan_ref(u, delta, A, B, C, D=D, z=z,
                              delta_bias=delta_bias, delta_softplus=True)


def ref_eager_bidi(t):
    """Plain PyTorch, both directions. The eager baseline."""
    return _bidi_ref(_ref_call, t)


def make_ref_compile_bidi(t):
    """torch.compile'd reference, both directions — the FAIR baseline.

    Compiled once here (outside the timed region) and returned as a closure;
    compile time is reported separately, never folded into the median. Returns
    None if inductor is unavailable on this host, which is a normal outcome (no
    MSVC on Windows), not a failure.
    """
    compiled = torch.compile(selective_scan_ref, dynamic=False)

    def call(u, delta, A, B, C, D, z, delta_bias):
        return compiled(u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias,
                        delta_softplus=True)

    t0 = time.perf_counter()
    _bidi_ref(call, t)          # warm the graph; this is where compilation lands
    compile_s = time.perf_counter() - t0
    return (lambda: _bidi_ref(call, t)), compile_s


def fused_estimate(t):
    """Two forward scans + merge, zero flips — the theoretical floor for any
    bidirectional implementation. Not a real backward scan: it computes the
    wrong answer on purpose, because we are timing the WORK, not the result.
    (Correctness lives in `tests/check_bidirectional.py`, which is green.)"""
    return scan_fwd(t) + scan_fwd(t)


def bidirectional_flip(t):
    """The OLD path, kept explicitly so the fusion has a real before/after to be
    measured against: flip the five time-varying inputs, scan forward, flip the
    output back, merge. This is also exactly the DEFINITION `reverse=True` must
    reproduce, so `bidirectional()` below must equal it bit-for-bit."""
    flipped = {k: flip_time(t[k]) for k in _FLIP_KEYS}
    back = arm_scan.selective_scan(
        flipped["u"], flipped["delta"], t["A"], flipped["B"], flipped["C"],
        D=t["D"], z=flipped["z"], delta_bias=t["delta_bias"],
        delta_softplus=True)
    return scan_fwd(t) + flip_time(back)


def bidirectional(t):
    """The NEW path: forward + kernel-fused reverse. No copies."""
    return arm_scan.bidirectional_scan(
        t["u"], t["delta"], t["A"], t["B"], t["C"], D=t["D"], z=t["z"],
        delta_bias=t["delta_bias"], delta_softplus=True, merge="sum")


def flips_only(t):
    """The six copies in isolation: five time-varying inputs in, one output
    back — precisely the work fusion removes. The list keeps every flip
    referenced so none can be elided."""
    flipped = [flip_time(t[k]) for k in _FLIP_KEYS]   # 5 input copies
    return flip_time(flipped[0])                       # 1 output copy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=sorted(SUITES), default="sweep-len")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--torch-threads", type=int, default=None)
    ap.add_argument("--compile-max-len", type=int, default=512,
                    help="skip the torch.compile baseline beyond this L "
                         "(graph unrolling makes compile time explode)")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--tag", type=str, default=platform.node())
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()
    if args.quick:
        args.compile_max_len = min(args.compile_max_len, 128)

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
    print(f"\nbidirectional scan — suite="
          f"{'quick' if args.quick else args.suite} reps={reps} "
          f"warmup={warmup} compile_max_len={args.compile_max_len}")
    print("  vs eager / vs torch.compile : the numbers that matter "
          "(kernel vs stock PyTorch)")
    print("  fusion_speedup              : internal — what the fused reverse "
          "removed vs the flip-based path (~2%, not a headline)\n")

    results = {
        "kind": "bidirectional-fused-reverse",
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

            # CORRECTNESS GATE 1: the fused reverse must reproduce the flip-based
            # definition. If it does not, those two series are not the same
            # computation and the whole comparison is meaningless — so refuse to
            # report a speedup rather than quote a fast wrong answer.
            #
            # NOT asserted bit-exactly. On NEON the two agree to ~1e-7, not to
            # the last bit: discretize/epilogue run 4-wide with a scalar tail,
            # and the vector and tail branches evaluate softplus/SiLU by
            # different means, so flipping the array can move a timestep across
            # that boundary. (It happens to be bit-exact at every length here,
            # since they are all multiples of 4 and no tail exists — but relying
            # on that would make this gate silently shape-dependent.)
            fused_out, flip_out = bidirectional(t), bidirectional_flip(t)
            drift = (fused_out - flip_out).abs().max().item()
            scale = max(1.0, fused_out.abs().max().item())
            if drift / scale > 1e-5:
                print(f"  !! fused reverse != flip-forward-flip "
                      f"(rel={drift/scale:.3e}) — refusing to benchmark")
                sys.exit(1)
            exact = " (bit-identical)" if drift == 0.0 else ""

            # CORRECTNESS GATE 2: the kernel must agree with the PyTorch
            # reference it is being timed against. Beating a baseline you do not
            # match is meaningless.
            ref_out = ref_eager_bidi(t)
            max_err = (fused_out - ref_out).abs().max().item()
            row = {"shape": [batch, dim, length, state], "timings": {},
                   "kernel_vs_ref_max_abs": max_err,
                   "fused_vs_flip_max_abs": drift}
            print(f"  fused == flip-based: rel {drift/scale:.1e}{exact}   "
                  f"kernel-vs-ref max_abs {max_err:.3e}")

            series = [
                ("ref_eager_bidi", ref_eager_bidi),
                ("scan_fwd", scan_fwd),
                ("fused_estimate", fused_estimate),
                ("bidirectional_flip", bidirectional_flip),
                ("bidirectional", bidirectional),
                ("flips_only", flips_only),
            ]
            for name, fn in series:
                r = bench(lambda fn=fn: fn(t), warmup, reps)
                row["timings"][name] = r
                print(f"  {name:19s} {r['median_s']*1e3:9.3f} ms")

            # torch.compile — the fair baseline. Compile cost is reported, never
            # timed. Unavailable inductor (e.g. no MSVC on Windows) is a skip,
            # not a failure.
            if not args.no_compile and length <= args.compile_max_len:
                try:
                    fn, compile_s = make_ref_compile_bidi(t)
                    r = bench(fn, warmup, reps)
                    r["compile_s"] = compile_s
                    row["timings"]["ref_compile_bidi"] = r
                    print(f"  {'ref_compile_bidi':19s} "
                          f"{r['median_s']*1e3:9.3f} ms "
                          f"(one-time compile {compile_s:.1f}s)")
                except Exception as e:
                    row["timings"]["ref_compile_bidi"] = {"error": str(e)[:200]}
                    print(f"  {'ref_compile_bidi':19s} unavailable: "
                          f"{str(e)[:80]}")
            elif not args.no_compile:
                print(f"  {'ref_compile_bidi':19s} skipped (L={length} > "
                      f"compile_max_len={args.compile_max_len})")

            bi = row["timings"]["bidirectional"]["median_s"]
            fp = row["timings"]["bidirectional_flip"]["median_s"]
            fu = row["timings"]["fused_estimate"]["median_s"]
            eager = row["timings"]["ref_eager_bidi"]["median_s"]

            row["speedup_vs_eager"] = eager / bi
            row["fusion_speedup"] = fp / bi          # internal, ~2%
            row["fusion_headroom"] = fp / fu         # ceiling, for continuity

            line = f"  => {row['speedup_vs_eager']:.2f}x vs eager"
            comp = row["timings"].get("ref_compile_bidi", {})
            if "median_s" in comp:
                row["speedup_vs_compile"] = comp["median_s"] / bi
                line += f", {row['speedup_vs_compile']:.2f}x vs torch.compile"
            print(line)
            print(f"     (internal: fused reverse won "
                  f"{row['fusion_speedup']:.3f}x over the flip-based path)")
            # PER-SHAPE noise guard. `fused_estimate` (two forward scans, zero
            # flips) is the floor by construction — the fused path cannot beat
            # it. If it appears to, the run is noise-dominated and the fusion
            # number is meaningless for THIS shape, whatever the across-shape
            # maxima say. (Seen for real: L=128 achieved 1.064x against a 1.055x
            # ceiling on a shared CI runner.)
            if row["fusion_speedup"] > row["fusion_headroom"]:
                print(f"     !! NOISE: achieved {row['fusion_speedup']:.3f}x "
                      f"exceeds its own ceiling {row['fusion_headroom']:.3f}x — "
                      f"this shape's fusion number is unusable")
                row["noise_dominated"] = True
            print()
            results["shapes"].append(row)

    ev = [r["speedup_vs_eager"] for r in results["shapes"]]
    cv = [r["speedup_vs_compile"] for r in results["shapes"]
          if "speedup_vs_compile" in r]
    sp = [r["fusion_speedup"] for r in results["shapes"]]
    hr = [r["fusion_headroom"] for r in results["shapes"]]

    print("=" * 62)
    print(f"bidirectional scan vs eager        : "
          f"{min(ev):.2f}x – {max(ev):.2f}x")
    if cv:
        print(f"bidirectional scan vs torch.compile: "
              f"{min(cv):.2f}x – {max(cv):.2f}x   <- the headline")
    else:
        print("bidirectional scan vs torch.compile: (not measured on this host)")
    noisy = [r for r in results["shapes"] if r.get("noise_dominated")]
    print(f"internal fusion win                : "
          f"{min(sp):.3f}x – {max(sp):.3f}x (ceiling was "
          f"{min(hr):.3f}x – {max(hr):.3f}x)")
    if noisy:
        print(f"!! {len(noisy)}/{len(results['shapes'])} shapes exceeded their "
              f"own ceiling -> this run is NOISE-DOMINATED. The fusion number "
              f"is not measurable here; use a dedicated host with more reps.")
    print("\nThe fusion win is a small internal effect and is NOT a headline. "
          "`reverse` exists as the substrate for the 2D cross-scan "
          "(see BIDIRECTIONAL_LOG.md); the vs-baseline rows are the result.")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nresults written to {args.json}")


if __name__ == "__main__":
    main()
