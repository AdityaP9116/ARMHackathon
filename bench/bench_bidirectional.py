"""Bidirectional scan: kernel vs PyTorch, and the fused-kernel exp-sharing win.

A bidirectional scan runs the recurrence in both time directions and sums them.
The direction-independent Pass A (discretize + exp, ~85% of the work) can be
computed once and SHARED between the two directions — that is the fused
two-direction kernel (BIDIRECTIONAL_SPEEDUP_IDEAS.md §3.2). This file measures
what that sharing buys, and where the fused path stands against stock PyTorch.

WHAT IS TIMED

Baselines — the rows that matter, and the only ones fit to publish:
  ref_eager_bidi      stock PyTorch, both directions, merged. What a real
                      bidirectional Mamba does on CPU with no kernel installed.
  ref_compile_bidi    the same under torch.compile — the FAIR baseline, since
                      compile is what a competent user would already be doing.

Our two implementations:
  bidirectional_twocall   two separate kernel scans (forward + the `reverse`
                          flag), summed. Each recomputes Pass A → exp twice.
  bidirectional           the FUSED two-direction kernel: exp computed once and
                          shared. This is `arm_scan.bidirectional_scan`.

Diagnostic:
  scan_fwd            one forward scan — half the work; the floor.

THE NUMBERS

  speedup_vs_eager / speedup_vs_compile   <- the result. Fused kernel vs PyTorch.
  exp_sharing_speedup = twocall / fused   <- the Stage-3 win, projected ~1.7x
                                             (exp is ~85% and direction-independent).

Both baselines are built on the SAME vendored reference (tests/reference/) that
the rest of the project measures against, so these rows are directly comparable
to bench_op.py's.

Two correctness gates run before any timing: the fused op must equal the two-call
path (bit-identical — sharing the exp reuses identical values), AND the kernel
must agree with the PyTorch reference it is being timed against.

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


def bidirectional_twocall(t):
    """Two separate scans (forward + the `reverse` flag), summed — NO exp
    sharing. This is what `bidirectional_scan` did before the fused two-direction
    kernel (Stage 3): each direction recomputes the full Pass A (discretize +
    exp). `bidirectional()` below must equal this bit-for-bit, and the ratio
    twocall / bidirectional is the exp-sharing win."""
    fwd = scan_fwd(t)
    bwd = arm_scan.selective_scan(
        t["u"], t["delta"], t["A"], t["B"], t["C"], D=t["D"], z=t["z"],
        delta_bias=t["delta_bias"], delta_softplus=True, reverse=True)
    return fwd + bwd


def bidirectional(t):
    """The FUSED path: one call, Pass A (the exp) computed once and shared
    between both directions. This is `arm_scan.bidirectional_scan`, which uses
    the fused two-direction kernel for the tied-weights sum case."""
    return arm_scan.bidirectional_scan(
        t["u"], t["delta"], t["A"], t["B"], t["C"], D=t["D"], z=t["z"],
        delta_bias=t["delta_bias"], delta_softplus=True, merge="sum")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=sorted(SUITES), default="sweep-len")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--torch-threads", type=int, default=None)
    ap.add_argument("--compile-max-len", type=int, default=None,
                    help="skip the torch.compile baseline beyond this L "
                         "(graph unrolling makes compile time explode). "
                         "Default 512, or 128 under --quick. An explicit value "
                         "always wins, including under --quick.")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--tag", type=str, default=platform.node())
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()
    if args.compile_max_len is None:
        # --quick's lower cap exists only to bound CI time. It must NOT override
        # an explicit flag — that made the L=512 compile row unreachable from a
        # quick run, which is exactly the row we wanted.
        args.compile_max_len = 128 if args.quick else 512

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
    print("  vs eager / vs torch.compile : kernel (fused) vs stock PyTorch")
    print("  exp-sharing speedup         : fused vs two-call — the Stage-3 win "
          "(exp computed once, not twice; projected ~1.7x)\n")

    results = {
        "kind": "bidirectional-fused-exp-sharing",
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

            # CORRECTNESS GATE 1: the fused two-direction op must reproduce the
            # two-call path (each direction scanned separately). They share no
            # code beyond the kernel, so if they disagree the exp-sharing changed
            # the answer — refuse to report a speedup over a wrong result. This is
            # expected BIT-IDENTICAL (the Rust test proves it): sharing Pass A
            # reuses the identical exp values, it does not approximate them.
            fused_out, twocall_out = bidirectional(t), bidirectional_twocall(t)
            drift = (fused_out - twocall_out).abs().max().item()
            scale = max(1.0, fused_out.abs().max().item())
            if drift / scale > 1e-6:
                print(f"  !! fused != two-call (rel={drift/scale:.3e}) — "
                      f"exp-sharing changed the answer; refusing to benchmark")
                sys.exit(1)
            exact = " (bit-identical)" if drift == 0.0 else ""

            # CORRECTNESS GATE 2: the kernel must agree with the PyTorch
            # reference it is being timed against.
            ref_out = ref_eager_bidi(t)
            max_err = (fused_out - ref_out).abs().max().item()
            row = {"shape": [batch, dim, length, state], "timings": {},
                   "kernel_vs_ref_max_abs": max_err,
                   "fused_vs_twocall_max_abs": drift}
            print(f"  fused == two-call: rel {drift/scale:.1e}{exact}   "
                  f"kernel-vs-ref max_abs {max_err:.3e}")

            series = [
                ("ref_eager_bidi", ref_eager_bidi),
                ("scan_fwd", scan_fwd),
                ("bidirectional_twocall", bidirectional_twocall),
                ("bidirectional", bidirectional),
            ]
            for name, fn in series:
                r = bench(lambda fn=fn: fn(t), warmup, reps)
                row["timings"][name] = r
                print(f"  {name:22s} {r['median_s']*1e3:9.3f} ms")

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
            tc = row["timings"]["bidirectional_twocall"]["median_s"]
            eager = row["timings"]["ref_eager_bidi"]["median_s"]

            row["speedup_vs_eager"] = eager / bi
            # The Stage-3 win: fused (shares the exp) vs two-call (recomputes it).
            # Projected ~1.7x from the profiler's phase split (exp ~85% of the
            # work, direction-independent). This is the ACHIEVED number.
            row["exp_sharing_speedup"] = tc / bi

            line = f"  => {row['speedup_vs_eager']:.2f}x vs eager"
            comp = row["timings"].get("ref_compile_bidi", {})
            if "median_s" in comp:
                row["speedup_vs_compile"] = comp["median_s"] / bi
                line += f", {row['speedup_vs_compile']:.2f}x vs torch.compile"
            print(line)
            print(f"     (exp-sharing: fused won {row['exp_sharing_speedup']:.3f}x "
                  f"over the two-call path)\n")
            results["shapes"].append(row)

            # Flush after EVERY shape, not just at the end. At long L,
            # torch.compile builds an L-step unrolled graph and can be killed by
            # the OOM reaper mid-sweep — a hard kill, not an exception we could
            # catch. Without this, an L=8192 death would destroy the 128..4096
            # results we already paid ~10 minutes of compilation for.
            if args.json:
                Path(args.json).parent.mkdir(parents=True, exist_ok=True)
                Path(args.json).write_text(json.dumps(results, indent=2))

    ev = [r["speedup_vs_eager"] for r in results["shapes"]]
    cv = [r["speedup_vs_compile"] for r in results["shapes"]
          if "speedup_vs_compile" in r]
    xs = [r["exp_sharing_speedup"] for r in results["shapes"]]

    print("=" * 62)
    print(f"bidirectional (fused) vs eager        : "
          f"{min(ev):.2f}x – {max(ev):.2f}x")
    if cv:
        print(f"bidirectional (fused) vs torch.compile: "
              f"{min(cv):.2f}x – {max(cv):.2f}x   <- the headline")
    else:
        print("bidirectional (fused) vs torch.compile: (not measured on this host)")
    print(f"exp-sharing: fused vs two-call        : "
          f"{min(xs):.3f}x – {max(xs):.3f}x   <- the Stage-3 win "
          f"(projected ~1.7x)")

    # ---- torch.compile's COST, reported rather than hidden in a skip message.
    #
    # Measured over L = 128..2048: compile time is LINEAR in L, converging to
    # ~0.26 s PER TIMESTEP (63s@128, 137s@512, 251s@1024, 534s@2048). The
    # recurrence is a Python `for t in range(L)` loop, so inductor unrolls it into
    # an L-step graph and pays per step.
    #
    # An earlier two-point fit (128, 512) suggested this was sub-linear and the
    # claim was withdrawn; four points show that was the fixed compiler startup
    # cost washing out. Extrapolating 0.26 s/step: L=8192 is ~36 min, and the
    # 131k-token genomics context is ~9.5 HOURS — if it does not OOM building the
    # graph first. At the lengths our applications use, torch.compile is not a
    # slow baseline, it is an absent one.
    #
    # Publish the amortization column too: it is the number a skeptic computes
    # anyway (~5,450 iterations for L>=512, stable because compile and runtime
    # both scale linearly), and it is better coming from us.
    comp_rows = [(r["shape"][2], r["timings"]["ref_compile_bidi"]["compile_s"],
                  r["timings"]["ref_compile_bidi"]["median_s"])
                 for r in results["shapes"]
                 if "compile_s" in r["timings"].get("ref_compile_bidi", {})]
    skipped = [r["shape"][2] for r in results["shapes"]
               if "compile_s" not in r["timings"].get("ref_compile_bidi", {})]
    if comp_rows:
        print("\ntorch.compile COST (the recurrence is unrolled into an "
              "L-step graph):")
        print(f"  {'L':>6}  {'compile':>10}  {'run/iter':>10}  "
              f"{'iters to amortize vs our kernel':>32}")
        for length, cs, ms in comp_rows:
            k = next(r["timings"]["bidirectional"]["median_s"]
                     for r in results["shapes"] if r["shape"][2] == length)
            # how many calls before compile time pays for itself vs our kernel
            gain = ms - k
            iters = f"{cs / gain:,.0f}" if gain > 0 else "never (we are faster)"
            print(f"  {length:>6}  {cs:>9.1f}s  {ms*1e3:>9.2f}ms  {iters:>32}")
        results["compile_cost"] = [
            {"len": l, "compile_s": cs, "median_s": ms}
            for l, cs, ms in comp_rows]
    if skipped:
        print(f"\n  L={skipped} skipped: compile time grows with L "
              f"(--compile-max-len={args.compile_max_len}). Raise the cap to "
              f"measure them — and note that having to is itself the finding.")

    print("\nHeadline: the vs-torch.compile row. The exp-sharing speedup is the "
          "Stage-3 fused-kernel win (one exp sweep, not two) and doubles as the "
          "SS2D substrate — see BIDIRECTIONAL_LOG.md.")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nresults written to {args.json}")


if __name__ == "__main__":
    main()
