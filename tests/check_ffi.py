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
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
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
