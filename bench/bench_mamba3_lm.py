"""The real 187M Mamba-3 on CPU: our kernel vs the only CPU alternative.

WHAT THE BASELINE IS, AND WHY IT IS NOT "UPSTREAM"
--------------------------------------------------
There is no upstream CPU baseline to beat. `mamba_ssm`'s Mamba-3 kernels are
Triton / TileLang / CuTe, the package does not install without `nvcc`, and
`mamba_ssm.modules.mamba3` raises on import for a CPU-only user. So the honest
framing of this benchmark is not "N x faster than the standard path" — it is
**"the standard path does not exist, and here is what the alternative costs."**

The alternative a CPU user actually has is to write the recurrence in PyTorch,
which is exactly `tests/reference/mamba3_ref.py`. Two baselines, per CLAUDE.md:

  ref_eager    that reference, in fp32 — NOT the f64 oracle it defaults to.
               Timing an f64 baseline against an fp32 kernel would inflate
               every speedup on this page by roughly the f64/f32 ratio, which
               would be measuring our own thumb on the scale.
  ref_compile  torch.compile of the same thing. This is the number an Arm
               engineer will look at, and it is reported with its COMPILE TIME,
               because for a sequential scan that cost is the story: the loop
               is L iterations of data-dependent work, so the graph grows with
               sequence length. Compilation is skipped past
               `--max-compile-len` and reported as skipped rather than
               silently omitted.

CORRECTNESS GATES SPEED. Every configuration's output is diffed against the
reference before any timing is reported.

Methodology per bench/README.md: fixed thread count, warmup then medians,
host/git-tagged JSON. Quiesce the machine first.

Usage:
    python bench/bench_mamba3_lm.py [--quick] [--json out.json]
"""

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "apps"))
sys.path.insert(0, str(REPO / "tests"))

import mamba3_lm.block as block_mod  # noqa: E402
from mamba3_lm import load_model  # noqa: E402
from reference.mamba3_ref import mamba3_mimo_ref, mamba3_siso_ref  # noqa: E402

LENGTHS = [128, 256, 512, 1024]
QUICK = [128, 256]


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _ref_scan(q, k, v, adt, dt, trap, q_bias, k_bias, angles=None, D=None,
              z=None, reverse=False, cos=None, sin=None):
    """`arm_scan.mamba3_scan`'s signature, backed by the PyTorch reference.

    fp32, not the reference's f64 default — see the module docstring.
    """
    if reverse:
        raise NotImplementedError("bench does not exercise the reverse pass")
    return mamba3_siso_ref(
        Q=q, K=k, V=v, ADT=adt, DT=dt, Trap=trap, Q_bias=q_bias,
        K_bias=k_bias, Angles=angles, D=D, Z=z,
        dtype=torch.float32).to(v.dtype)


def _ref_mimo_scan(q, k, v, adt, dt, trap, q_bias, k_bias, psi=None,
                   zeta=None, phi=None, angles=None, D=None, z=None,
                   reverse=False, cos=None, sin=None):
    """`arm_scan.mamba3_mimo_scan`'s signature, backed by the reference."""
    if reverse:
        raise NotImplementedError("bench does not exercise the reverse pass")
    return mamba3_mimo_ref(
        Q=q, K=k, V=v, ADT=adt, DT=dt, Trap=trap, Q_bias=q_bias, K_bias=k_bias,
        MIMO_V=psi, MIMO_Z=zeta, MIMO_Out=phi, Angles=angles, D=D, Z=z,
        dtype=torch.float32).to(v.dtype)


@contextmanager
def scan_backend(fn, mimo=False):
    """Swap the scan the model's mixers call, then put it back.

    The model is built once and reused across backends so that weight loading
    and allocator state cannot differ between the rows being compared. `mimo`
    selects WHICH entry point to swap: the two families call different
    functions, so patching the wrong one silently benchmarks the kernel against
    itself and reports a 1.0x speedup that looks like a real measurement.
    """
    name = "mamba3_mimo_scan" if mimo else "mamba3_scan"
    original = getattr(block_mod.arm_scan, name)
    setattr(block_mod.arm_scan, name, fn)
    try:
        yield
    finally:
        setattr(block_mod.arm_scan, name, original)


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
    ap.add_argument("--model", default=None,
                    help="checkpoint id or path (default: the 187M SISO)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-compile-len", type=int, default=256,
                    help="skip torch.compile past this length. The compiled "
                         "graph grows with L for a sequential scan, so this "
                         "bounds a compile that would otherwise dominate the "
                         "whole run")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    lengths = QUICK if args.quick else LENGTHS

    kw = {"model_id_or_path": args.model} if args.model else {}
    print("loading checkpoint ...")
    model = load_model(**kw)
    n_params = sum(p.numel() for p in model.parameters())
    # Which family did we actually load? Ask the model, do not infer from the
    # name -- the reference and the swapped entry point must both match it.
    is_mimo = bool(getattr(model.backbone.layers[0].mixer, "is_mimo", False))
    ref_fn = _ref_mimo_scan if is_mimo else _ref_scan

    meta = {
        "host": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "git": git_sha(),
        "params": n_params,
        "model": args.model or "state-spaces/mamba3-siso-187m",
        "variant": "mimo" if is_mimo else "siso",
        "mimo_rank": getattr(model.backbone.layers[0].mixer, "mimo_rank", 1),
    }
    print(f"\n{meta['model']}  ({n_params / 1e6:.1f}M params, "
          f"{meta['variant'].upper()}"
          f"{f", rank {meta['mimo_rank']}" if is_mimo else ''})")
    print(f"{meta['machine']}  torch {meta['torch']}  "
          f"threads {meta['threads']}  git {meta['git']}")
    print("\nbaseline = the PyTorch recurrence (fp32). Upstream has NO CPU "
          "path at all.\n")

    hdr = (f"{'L':>6}  {'kernel ms':>10}  {'ref ms':>10}  {'speedup':>8}  "
           f"{'compiled ms':>12}  {'vs comp':>8}  {'compile s':>10}  "
           f"{'max|d|':>9}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for L in lengths:
        torch.manual_seed(0)
        ids = torch.randint(0, 1000, (1, L))

        with torch.no_grad():
            got = model(ids)
            with scan_backend(ref_fn, mimo=is_mimo):
                want = model(ids)
        delta = float((got - want).abs().max())
        # Gate before timing. The two paths differ only in the scan, both in
        # fp32, so this is a tight bound rather than a bf16-scale one.
        scale = max(float(want.abs().max()), 1e-30)
        if delta / scale > 1e-3:
            print(f"{L:>6}  CORRECTNESS FAILED: max|delta| {delta:.3e} "
                  f"(rel {delta / scale:.3e}) — not timing this shape")
            rows.append({"len": L, "correctness": "FAIL", "max_abs": delta})
            continue

        with torch.no_grad():
            t_kernel = bench(lambda: model(ids), args.warmup, args.reps)
            with scan_backend(ref_fn, mimo=is_mimo):
                t_ref = bench(lambda: model(ids), args.warmup, args.reps)

        t_comp, compile_s = None, None
        if not args.no_compile and L <= args.max_compile_len:
            try:
                compiled = torch.compile(ref_fn, dynamic=False)
                t0 = time.perf_counter()
                with torch.no_grad(), scan_backend(compiled, mimo=is_mimo):
                    model(ids)                       # triggers compilation
                compile_s = time.perf_counter() - t0
                with torch.no_grad(), scan_backend(compiled, mimo=is_mimo):
                    t_comp = bench(lambda: model(ids), 0, args.reps)
            except Exception as exc:  # noqa: BLE001
                print(f"{L:>6}  (torch.compile failed: "
                      f"{type(exc).__name__}: {str(exc)[:40]})")

        print(f"{L:>6}  {t_kernel * 1e3:10.2f}  {t_ref * 1e3:10.2f}  "
              f"{t_ref / t_kernel:7.2f}x  "
              f"{(f'{t_comp * 1e3:12.2f}' if t_comp else f'{"skipped":>12}')}  "
              f"{(f'{t_comp / t_kernel:7.2f}x' if t_comp else f'{"-":>8}')}  "
              f"{(f'{compile_s:10.1f}' if compile_s else f'{"-":>10}')}  "
              f"{delta:9.2e}")

        rows.append({
            "len": L, "correctness": "PASS", "max_abs": delta,
            "kernel_ms": t_kernel * 1e3, "ref_eager_ms": t_ref * 1e3,
            "ref_compile_ms": t_comp * 1e3 if t_comp else None,
            "compile_s": compile_s,
            "speedup_vs_eager": t_ref / t_kernel,
            "speedup_vs_compile": t_comp / t_kernel if t_comp else None,
            "tokens_per_s": L / t_kernel,
        })

    print("\ntokens/s (prefill, our kernel):")
    for r in rows:
        if r.get("correctness") == "PASS":
            print(f"  L={r['len']:<6} {r['tokens_per_s']:10.1f} tok/s")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"meta": meta, "rows": rows}, indent=2))
        print(f"\nwrote {args.json}")

    return 0 if all(r.get("correctness") == "PASS" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
