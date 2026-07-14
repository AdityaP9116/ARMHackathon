"""Python-level golden check through the C ABI.

Runs every tests/golden/*.npz case through the actual cdylib via
ctypes — the tier that catches ABI, layout, and ownership bugs that Rust
unit tests structurally cannot see (they never cross the boundary).

numpy-only (no torch), so CI can run it on every platform in seconds.

Usage:
    cargo build --release -p arm-scan-ffi   # (in kernel/)
    python tests/check_ffi.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
# ARM_SCAN_INSTALLED=1 tests a pip-installed arm-scan wheel instead of the
# in-repo package (the wheel job's mode)
if not os.environ.get("ARM_SCAN_INSTALLED"):
    sys.path.insert(0, str(REPO / "python"))

import arm_scan  # noqa: E402

GOLDEN_DIR = REPO / "tests" / "golden"
MAX_ABS = 1e-4      # kernel acceptance tolerance (INTEGRATION_PLAN.md)
FLOOR_FACTOR = 50.0  # same criterion as the Rust golden test


def run_case(meta):
    data = np.load(GOLDEN_DIR / f"{meta['name']}.npz")
    get = lambda k: data[k] if k in data else None

    out, last = arm_scan.selective_scan_numpy(
        data["u"], data["delta"], data["A"], data["B"], data["C"],
        D=get("D_skip"), z=get("z"), delta_bias=get("delta_bias"),
        delta_softplus=meta["delta_softplus"], return_last_state=True,
    )

    out_err = np.abs(out.astype(np.float64) - data["out_f64"]).max()
    last_f64 = data["last_state_f64"]
    last_scale = max(1.0, np.abs(last_f64).max())
    last_rel = np.abs(last.astype(np.float64) - last_f64).max() / last_scale
    return out_err, last_rel


def check_streaming():
    """h0 through the Python->C path: a scan split in two, with the first
    half's last_state resumed as initial_state, matches the one-shot scan."""
    rng = np.random.default_rng(0)
    B, D, L, N, split = 1, 4, 12, 16, 5
    f = lambda *s: rng.standard_normal(s).astype(np.float32)
    u, delta, z = f(B, D, L), f(B, D, L), f(B, D, L)
    A = (-np.abs(f(D, N)) - 0.5).astype(np.float32)
    Bm, Cm, Dv = f(B, N, L), f(B, N, L), f(D)

    kw = dict(D=Dv, delta_softplus=True)
    out_full, _ = arm_scan.selective_scan_numpy(
        u, delta, A, Bm, Cm, z=z, return_last_state=True, **kw)
    out1, state = arm_scan.selective_scan_numpy(
        u[..., :split], delta[..., :split], A, Bm[..., :split],
        Cm[..., :split], z=z[..., :split], return_last_state=True, **kw)
    out2 = arm_scan.selective_scan_numpy(
        u[..., split:], delta[..., split:], A, Bm[..., split:],
        Cm[..., split:], z=z[..., split:], initial_state=state, **kw)
    return float(max(np.abs(out_full[..., :split] - out1).max(),
                     np.abs(out_full[..., split:] - out2).max()))


def main():
    print(f"kernel library: {arm_scan.lib_path()}")
    manifest = json.loads((GOLDEN_DIR / "manifest.json").read_text())
    assert manifest, "empty manifest"

    failures = []
    for meta in manifest:
        out_err, last_rel = run_case(meta)
        floor_bound = max(FLOOR_FACTOR * meta["f32_max_abs_err"], 1e-6)
        ok = out_err < MAX_ABS and out_err < floor_bound and last_rel < 1e-4
        print(f"  {meta['name']:24s} out_max_abs={out_err:.3e} "
              f"(floor {meta['f32_max_abs_err']:.3e})  "
              f"last_rel={last_rel:.3e}  {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(meta["name"])

    serr = check_streaming()
    ok = serr < 1e-5
    print(f"  {'streaming (h0 resume)':24s} max_abs={serr:.3e}  "
          f"{'ok' if ok else 'FAIL'}")
    if not ok:
        failures.append("streaming-h0")

    # error paths must reject, not crash or write garbage
    try:
        arm_scan.selective_scan_numpy(
            np.zeros((1, 2, 3), np.float32), np.zeros((1, 2, 3), np.float32),
            np.zeros((5, 4), np.float32),  # wrong dim
            np.zeros((1, 4, 3), np.float32), np.zeros((1, 4, 3), np.float32))
        failures.append("shape-mismatch-not-rejected")
    except ValueError:
        print("  shape validation            rejects bad A dim  ok")

    if failures:
        print(f"\nFFI GOLDEN CHECK FAILED: {failures}")
        sys.exit(1)
    print(f"\nall {len(manifest)} golden cases pass through the C ABI")


if __name__ == "__main__":
    main()
