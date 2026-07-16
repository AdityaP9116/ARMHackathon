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

# (batch, dim, len, state) — presets from BASELINE_TEST_PLAN.md §4.2
SUITES = {
    # the plan's headline shapes (INTEGRATION_PLAN.md Phase 6)
    "basic": [
        (1, 768, 128, 16),
        (1, 768, 512, 16),
        (1, 768, 2048, 16),
        (8, 1536, 1024, 16),
    ],
    # O(L) curve: the kernel-vs-eager gap as a function of sequence length
    "sweep-len": [(1, 768, l, 16)
                  for l in (64, 128, 256, 512, 1024, 2048, 4096, 8192)],
    # threading saturation as channel count grows
    "sweep-dim": [(1, d, 512, 16) for d in (256, 768, 1536, 3072)],
    # batch scaling at the mamba-130m-like shape
    "sweep-batch": [(b, 1536, 1024, 16) for b in (1, 4, 8)],
    # single shape used by the RAYON_NUM_THREADS scaling loop in
    # bench/run_baseline.sh (one process per thread count)
    "scaling-point": [(1, 1536, 512, 16)],
}
QUICK_SHAPES = [
    (1, 768, 128, 16),
    (1, 768, 512, 16),
]


def parse_shapes(spec):
    """--shapes 'B,D,L,N;B,D,L,N' -> list of tuples."""
    out = []
    for part in spec.split(";"):
        vals = tuple(int(v) for v in part.split(","))
        if len(vals) != 4:
            raise ValueError(f"bad shape '{part}': need B,D,L,N")
        out.append(vals)
    return out


def git_sha():
    import subprocess
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


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
    ap.add_argument("--suite", choices=sorted(SUITES), default="basic",
                    help="shape preset (see BASELINE_TEST_PLAN.md)")
    ap.add_argument("--shapes", type=str, default=None,
                    help="explicit shapes 'B,D,L,N;...' (overrides --suite)")
    ap.add_argument("--quick", action="store_true",
                    help="CI-sized subset (fewer shapes/reps)")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--compile-max-len", type=int, default=None,
                    help="skip the torch.compile baseline beyond this L "
                         "(graph unrolling makes compile time explode). "
                         "Default 512, or 128 under --quick. An explicit value "
                         "always wins, including under --quick.")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--torch-threads", type=int, default=None,
                    help="pin torch intra-op threads (fairness control)")
    ap.add_argument("--tag", type=str, default=platform.node(),
                    help="host label embedded in the JSON (e.g. ampere-a1)")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)

    if args.shapes:
        shapes = parse_shapes(args.shapes)
    elif args.quick:
        shapes = QUICK_SHAPES
    else:
        shapes = SUITES[args.suite]
    reps = args.reps or (5 if args.quick else 10)
    warmup = args.warmup if args.warmup is not None else (1 if args.quick else 3)
    if args.compile_max_len is None:
        # --quick's lower cap bounds CI time only; an explicit flag must win.
        args.compile_max_len = 128 if args.quick else 512

    env = env_report()
    env["tag"] = args.tag
    env["git_sha"] = git_sha()
    env["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print("environment:")
    for k, v in env.items():
        print(f"  {k}: {v}")
    print(f"\nsuite={args.suite if not args.shapes else 'custom'} "
          f"reps={reps} warmup={warmup} "
          f"compile_max_len={args.compile_max_len}\n")

    results = {
        "kind": "op",
        "env": env,
        "reps": reps,
        "suite": args.suite if not args.shapes else "custom",
        "args": {k: v for k, v in vars(args).items() if k != "json"},
        "shapes": [],
    }
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
