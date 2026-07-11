"""Op-level benchmark: the Arm kernel vs PyTorch on the selective scan.

Baselines:
  ref_eager    - the pure-PyTorch reference scan (vendored upstream
                 semantics; the same per-timestep structure HF's
                 slow_forward uses)
  ref_compile  - torch.compile(ref) — the FAIR baseline. Compilation
                 unrolls the sequential recurrence into an L-step graph,
                 which is exactly the structural limitation this kernel
                 exists to beat; compile time explodes with L, so shapes
                 longer than --compile-max-len skip this baseline.
  kernel       - arm_scan.selective_scan (torch custom op -> C ABI ->
                 NEON+rayon on aarch64, scalar+rayon elsewhere)

Methodology: realistic Mamba value distributions (negative A, softplus
delta in ~[1e-3, 0.1]); per-baseline warmup then N timed reps; medians
reported; compile time excluded from timings but reported. A correctness
cross-check (kernel vs ref_eager) is printed for every shape.

Usage:
    python bench/bench_op.py             # full plan shapes
    python bench/bench_op.py --quick     # CI-sized subset
    python bench/bench_op.py --json out.json
"""

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tests"))

import arm_scan  # noqa: E402
from reference import selective_scan_ref  # noqa: E402

FULL_SHAPES = [
    # (batch, dim, len, state) — from INTEGRATION_PLAN.md Phase 6
    (1, 768, 128, 16),
    (1, 768, 512, 16),
    (1, 768, 2048, 16),
    (8, 1536, 1024, 16),
]
QUICK_SHAPES = [
    (1, 768, 128, 16),
    (1, 768, 512, 16),
]


def make_inputs(batch, dim, length, state, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    u = r(batch, dim, length)
    delta = r(batch, dim, length) * 0.5
    delta_bias = torch.empty(dim).uniform_(-6.0, -3.0, generator=g)
    A = -torch.exp(torch.empty(dim, state).uniform_(
        math.log(0.5), math.log(16.0), generator=g))
    B = r(batch, state, length)
    C = r(batch, state, length)
    D = r(dim)
    z = r(batch, dim, length)
    return dict(u=u, delta=delta, A=A, B=B, C=C, D=D, z=z,
                delta_bias=delta_bias)


def ref_call(t):
    return selective_scan_ref(
        t["u"], t["delta"], t["A"], t["B"], t["C"], D=t["D"], z=t["z"],
        delta_bias=t["delta_bias"], delta_softplus=True)


def kernel_call(t):
    return arm_scan.selective_scan(
        t["u"], t["delta"], t["A"], t["B"], t["C"], D=t["D"], z=t["z"],
        delta_bias=t["delta_bias"], delta_softplus=True)


def bench(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "reps": reps,
    }


def env_report():
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "(unknown)",
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "kernel_lib": str(arm_scan.lib_path()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="CI-sized subset (fewer shapes/reps)")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--compile-max-len", type=int, default=512,
                    help="skip the torch.compile baseline beyond this L "
                         "(graph unrolling makes compile time explode)")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    shapes = QUICK_SHAPES if args.quick else FULL_SHAPES
    reps = args.reps or (5 if args.quick else 10)
    warmup = args.warmup if args.warmup is not None else (1 if args.quick else 3)
    if args.quick:
        args.compile_max_len = min(args.compile_max_len, 128)

    env = env_report()
    print("environment:")
    for k, v in env.items():
        print(f"  {k}: {v}")
    print(f"\nreps={reps} warmup={warmup} "
          f"compile_max_len={args.compile_max_len}\n")

    results = {"env": env, "reps": reps, "shapes": []}
    with torch.no_grad():
        for batch, dim, length, state in shapes:
            label = f"B{batch}_D{dim}_L{length}_N{state}"
            print(f"=== {label} ===")
            t = make_inputs(batch, dim, length, state)
            row = {"shape": [batch, dim, length, state], "baselines": {}}

            # correctness cross-check first
            ref_out = ref_call(t)
            kern_out = kernel_call(t)
            max_err = (kern_out - ref_out).abs().max().item()
            scale = ref_out.abs().max().item()
            print(f"  kernel-vs-ref max_abs_err {max_err:.3e} "
                  f"(scale {scale:.1f})")
            row["kernel_vs_ref_max_abs"] = max_err

            r = bench(lambda: ref_call(t), warmup, reps)
            row["baselines"]["ref_eager"] = r
            print(f"  ref_eager    {r['median_s']*1e3:9.2f} ms")

            if not args.no_compile and length <= args.compile_max_len:
                try:
                    compiled = torch.compile(selective_scan_ref, dynamic=False)
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        compiled(t["u"], t["delta"], t["A"], t["B"], t["C"],
                                 D=t["D"], z=t["z"],
                                 delta_bias=t["delta_bias"],
                                 delta_softplus=True)
                    compile_s = time.perf_counter() - t0
                    fn = lambda: compiled(
                        t["u"], t["delta"], t["A"], t["B"], t["C"],
                        D=t["D"], z=t["z"], delta_bias=t["delta_bias"],
                        delta_softplus=True)
                    r = bench(fn, warmup, reps)
                    r["compile_s"] = compile_s
                    row["baselines"]["ref_compile"] = r
                    print(f"  ref_compile  {r['median_s']*1e3:9.2f} ms "
                          f"(one-time compile {compile_s:.1f}s)")
                except Exception as e:  # inductor unavailable on some hosts
                    row["baselines"]["ref_compile"] = {"error": str(e)[:200]}
                    print(f"  ref_compile  unavailable: {str(e)[:120]}")
            elif not args.no_compile:
                print(f"  ref_compile  skipped (L={length} > "
                      f"compile_max_len={args.compile_max_len})")

            r = bench(lambda: kernel_call(t), warmup, reps)
            row["baselines"]["kernel"] = r
            print(f"  kernel       {r['median_s']*1e3:9.2f} ms")

            k = row["baselines"]["kernel"]["median_s"]
            e = row["baselines"]["ref_eager"]["median_s"]
            line = f"  speedup: {e/k:.2f}x vs eager"
            c = row["baselines"].get("ref_compile", {})
            if "median_s" in c:
                line += f", {c['median_s']/k:.2f}x vs torch.compile"
            print(line + "\n")
            results["shapes"].append(row)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"results written to {args.json}")


if __name__ == "__main__":
    main()
