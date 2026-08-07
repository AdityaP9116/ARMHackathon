"""Generate golden test vectors for the Arm selective-scan kernel.

For each case this script:
  1. draws float32 inputs with realistic Mamba value distributions
     (A negative and O(1..16), softplus(delta) in ~[1e-3, 0.1], randn
     activations),
  2. computes the output twice with the vendored upstream reference:
       - at float64 (inputs upcast) -> `out_f64`, the correctness ground truth
       - at float32 (exactly upstream semantics) -> `out_f32`, which
         establishes the tolerance floor any correct f32 kernel should meet,
  3. saves everything to tests/golden/<name>.npz plus a manifest.json.

Determinism: the inputs come from `golden_inputs.draw_inputs`, a torch-free
draw seeded per case from the case name, so cases are independent and the
whole set is reproducible bit-for-bit on any torch and any numpy — see that
module for why the draws no longer go through `torch.Generator`.
`verify_golden.py` re-runs the same draws and compares against the committed
inputs, and being torch-free it does so in CI's torch-free `test` job.

Usage:
    python tests/gen_golden.py            # core (committed) cases, small
    python tests/gen_golden.py --large    # also emit large benchmark-shaped
                                          # cases (NOT committed; .gitignored)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from golden_inputs import (CORE_CASES, INPUT_DRAW_SPEC, LARGE_CASES,
                           case_seed, draw_inputs)
from reference import selective_scan_ref

GOLDEN_DIR = Path(__file__).parent / "golden"


def generate_case(name, B, D, L, N, **kw):
    softplus = kw.get("softplus", True)
    drawn = draw_inputs(name, B, D, L, N, **kw)
    # the draws are numpy (torch-free by design); the reference is torch
    u, delta, A, Bmat, Cmat, D_skip, z, delta_bias = [
        None if a is None else torch.from_numpy(a) for a in drawn]

    # float64 ground truth: upcast the *same f32 values* so the comparison
    # with any f32 kernel is apples-to-apples.
    f64 = lambda t: None if t is None else t.double()
    out_f64, last_state_f64 = selective_scan_ref(
        f64(u), f64(delta), f64(A), f64(Bmat), f64(Cmat), f64(D_skip), f64(z),
        f64(delta_bias), delta_softplus=softplus, return_last_state=True,
        compute_dtype=torch.float64)

    # float32 run, exactly upstream semantics -> tolerance floor
    out_f32, last_state_f32 = selective_scan_ref(
        u, delta, A, Bmat, Cmat, D_skip, z, delta_bias,
        delta_softplus=softplus, return_last_state=True,
        compute_dtype=torch.float32)

    arrays = {
        "u": u.numpy(), "delta": delta.numpy(), "A": A.numpy(),
        "B": Bmat.numpy(), "C": Cmat.numpy(),
        "out_f64": out_f64.numpy(), "last_state_f64": last_state_f64.numpy(),
        "out_f32": out_f32.numpy(), "last_state_f32": last_state_f32.numpy(),
    }
    if D_skip is not None:
        arrays["D_skip"] = D_skip.numpy()
    if z is not None:
        arrays["z"] = z.numpy()
    if delta_bias is not None:
        arrays["delta_bias"] = delta_bias.numpy()

    meta = {
        "name": name, "batch": B, "dim": D, "len": L, "state": N,
        "groups": kw.get("groups"), "delta_softplus": softplus,
        "has_z": z is not None, "has_D": D_skip is not None,
        "has_delta_bias": delta_bias is not None,
        # `seed` and `input_draw` pin the INPUTS (torch-free, reproducible
        # anywhere); `torch_version` records only what computed the outputs.
        "seed": case_seed(name), "input_draw": INPUT_DRAW_SPEC,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        # observed f32-vs-f64 gap, the floor a correct f32 kernel should hit
        "f32_max_abs_err": float((out_f32.double() - out_f64).abs().max()),
    }
    arrays["meta_json"] = np.frombuffer(
        json.dumps(meta).encode(), dtype=np.uint8).copy()

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GOLDEN_DIR / f"{name}.npz", **arrays)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--large", action="store_true",
                    help="also generate large benchmark-shaped cases")
    args = ap.parse_args()

    cases = CORE_CASES + (LARGE_CASES if args.large else [])
    manifest = []
    for name, B, D, L, N, kw in cases:
        meta = generate_case(name, B, D, L, N, **kw)
        manifest.append(meta)
        print(f"  {name:24s} (B={B} D={D} L={L} N={N})  "
              f"f32 floor={meta['f32_max_abs_err']:.3e}")

    # merge with any entries other scripts added (e.g. hf_mixer_layer0 from
    # check_hf_slow_path.py) instead of clobbering them
    manifest_path = GOLDEN_DIR / "manifest.json"
    ours = {m["name"] for m in manifest}
    if manifest_path.exists():
        manifest += [m for m in json.loads(manifest_path.read_text())
                     if m["name"] not in ours]
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(cases)} cases generated -> {GOLDEN_DIR} "
          f"({len(manifest)} total in manifest)")


if __name__ == "__main__":
    main()
