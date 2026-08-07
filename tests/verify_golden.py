"""Independently verify the golden vectors.

Recomputes every golden case with a second implementation that shares no
code with the generator: pure numpy float64, explicit Python loops, no
einsum, no torch. If a transcription error existed in the vendored torch
reference (wrong index order, wrong discretization, wrong gate), the two
implementations would disagree and this script fails loudly.

Also checks:
  - stored `last_state_f64` against the naive recurrence's final state,
  - the f32/f64 gap (`out_f32` vs `out_f64`) stays below the kernel
    acceptance tolerance from INTEGRATION_PLAN.md (max_abs < 1e-4),
  - regenerating every core case reproduces the stored inputs bit-for-bit
    (determinism of the draws in golden_inputs.py).

Numpy-only on purpose — including the determinism check, which is why this
whole file runs in requirements.txt's torch-free tier and in CI's `test` job.

Usage: python tests/verify_golden.py
"""

import json
import sys
from pathlib import Path

import numpy as np

GOLDEN_DIR = Path(__file__).parent / "golden"

# agreement required between the two independent float64 implementations
F64_RTOL, F64_ATOL = 1e-8, 1e-10
# acceptance tolerance for any f32 kernel (from INTEGRATION_PLAN.md)
KERNEL_MAX_ABS = 1e-4


def softplus(x):
    # matches torch.nn.functional.softplus(beta=1, threshold=20)
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def silu(x):
    return x / (1.0 + np.exp(-x))


def naive_scan_f64(u, delta, A, B, C, D_skip=None, z=None, delta_bias=None,
                   delta_softplus=False):
    """Loop-based float64 selective scan. Deliberately naive."""
    u = u.astype(np.float64)
    delta = delta.astype(np.float64)
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    C = C.astype(np.float64)

    batch, dim, L = u.shape
    N = A.shape[1]
    grouped = B.ndim == 4
    group_size = dim // B.shape[1] if grouped else None

    if delta_bias is not None:
        delta = delta + delta_bias.astype(np.float64)[None, :, None]
    if delta_softplus:
        delta = softplus(delta)

    out = np.empty((batch, dim, L), dtype=np.float64)
    last_state = np.empty((batch, dim, N), dtype=np.float64)
    for b in range(batch):
        for d in range(dim):
            g = d // group_size if grouped else None
            h = np.zeros(N, dtype=np.float64)
            for t in range(L):
                dt = delta[b, d, t]
                b_vec = B[b, g, :, t] if grouped else B[b, :, t]
                c_vec = C[b, g, :, t] if grouped else C[b, :, t]
                h = np.exp(dt * A[d, :]) * h + (dt * u[b, d, t]) * b_vec
                acc = 0.0
                for n in range(N):          # no np.dot: independent op order
                    acc += c_vec[n] * h[n]
                out[b, d, t] = acc
            last_state[b, d, :] = h

    if D_skip is not None:
        out = out + u * D_skip.astype(np.float64)[None, :, None]
    if z is not None:
        out = out * silu(z.astype(np.float64))
    return out, last_state


def check_case(path):
    data = np.load(path)
    meta = json.loads(bytes(data["meta_json"]).decode())
    name = meta["name"]

    out, last_state = naive_scan_f64(
        data["u"], data["delta"], data["A"], data["B"], data["C"],
        D_skip=data["D_skip"] if "D_skip" in data else None,
        z=data["z"] if "z" in data else None,
        delta_bias=data["delta_bias"] if "delta_bias" in data else None,
        delta_softplus=meta["delta_softplus"])

    errs = []
    def close(a, b, what):
        diff = np.abs(a - b)
        denom = np.maximum(np.abs(b), F64_ATOL / F64_RTOL)
        rel = (diff / denom).max()
        if not np.allclose(a, b, rtol=F64_RTOL, atol=F64_ATOL):
            errs.append(f"{what}: max_abs={diff.max():.3e} max_rel={rel:.3e}")
        return diff.max()

    f64_err = close(out, data["out_f64"], "out_f64 vs naive")
    close(last_state, data["last_state_f64"], "last_state_f64 vs naive")

    f32_gap = np.abs(data["out_f32"].astype(np.float64) - data["out_f64"]).max()
    if f32_gap >= KERNEL_MAX_ABS:
        errs.append(f"f32 floor {f32_gap:.3e} >= kernel tolerance {KERNEL_MAX_ABS}")
    if abs(f32_gap - meta["f32_max_abs_err"]) > 1e-12:
        errs.append("manifest f32_max_abs_err does not match stored arrays")

    status = "FAIL" if errs else "ok"
    print(f"  {name:24s} f64 agreement={f64_err:.3e}  "
          f"f32 floor={f32_gap:.3e}  {status}")
    for e in errs:
        print(f"      !! {e}")
    return not errs


def check_determinism(manifest):
    """Regenerating every core case must reproduce stored inputs bit-for-bit.

    This runs unconditionally, on every platform, in CI's torch-free `test`
    job — which is the point. It used to import the draws from `gen_golden`,
    which pulls in torch, so it had to degrade to a skip rather than take the
    whole run down with it (an earlier variant did take it down: `make test`
    failed on all three platforms from Jul 17 onward and, because every other
    job declares `needs: test`, silently blocked the rest of the pipeline).

    A skip is not a gate, though, and this one hid a real failure: torch does
    not keep its RNG stream stable across releases, so the goldens drawn under
    torch 2.11 stopped redrawing bit-for-bit under 2.13 — failing for anyone
    with the full requirements installed while CI stayed green. The draws now
    live in `golden_inputs`, which is numpy-only and pinned to an algorithm
    spelled out in that file, so there is nothing left to skip for.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from golden_inputs import CORE_CASES, INPUT_DRAW_SPEC, draw_inputs

    # only the cases this file draws — hf_mixer_layer0 is captured from a real
    # HF forward pass by check_hf_slow_path.py and has no draw spec
    spec = {m["name"]: m.get("input_draw") for m in manifest}
    stale = sorted(name for name, *_ in CORE_CASES
                   if name in spec and spec[name] != INPUT_DRAW_SPEC)
    if stale:
        print(f"  determinism FAIL: {len(stale)} case(s) drawn by a different "
              f"input spec than {INPUT_DRAW_SPEC}: {', '.join(stale)}")
        print("  (the draws changed; regenerate with python tests/gen_golden.py)")
        return False

    ok, checked = True, 0
    keys = ("u", "delta", "A", "B", "C", "D_skip", "z", "delta_bias")
    for name, B, D, L, N, kw in CORE_CASES:
        path = GOLDEN_DIR / f"{name}.npz"
        if not path.exists():          # --large cases are never committed
            continue
        data = np.load(path)
        checked += 1
        drawn = draw_inputs(name, B, D, L, N, **kw)
        for key, arr in zip(keys, drawn):
            stored = data[key] if key in data else None
            if arr is None or stored is None:
                if (arr is None) != (stored is None):
                    print(f"  determinism FAIL: {name}/{key} present in one "
                          f"of stored/regenerated but not the other")
                    ok = False
                continue
            if not np.array_equal(stored, arr):
                print(f"  determinism FAIL: {name}/{key} differs on regeneration")
                ok = False
    if ok:
        print(f"  determinism ok: {checked} cases redrawn bit-identical "
              f"({INPUT_DRAW_SPEC}, no torch)")
    return ok


def main():
    with open(GOLDEN_DIR / "manifest.json") as f:
        manifest = json.load(f)
    files = sorted(GOLDEN_DIR.glob("*.npz"))
    names = {m["name"] for m in manifest}
    listed = {p.stem for p in files}
    if not names <= listed:
        print(f"missing files for manifest entries: {names - listed}")
        sys.exit(1)

    print(f"verifying {len(files)} golden cases against independent "
          f"numpy implementation:")
    ok = all([check_case(p) for p in files])
    ok = check_determinism(manifest) and ok
    if not ok:
        print("\nVERIFICATION FAILED")
        sys.exit(1)
    print("\nall golden vectors verified")


if __name__ == "__main__":
    main()
