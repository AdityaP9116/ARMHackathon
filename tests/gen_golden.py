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

Determinism: every case uses its own torch.Generator seeded from the case
name, so cases are independent and the whole set is reproducible bit-for-bit
on the same torch version.

Usage:
    python tests/gen_golden.py            # core (committed) cases, small
    python tests/gen_golden.py --large    # also emit large benchmark-shaped
                                          # cases (NOT committed; .gitignored)
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from reference import selective_scan_ref

GOLDEN_DIR = Path(__file__).parent / "golden"


def case_seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")


def draw_inputs(name, B, D, L, N, *, groups=None, with_z=True, with_D=True,
                with_bias=True, softplus=True, delta_style="normal"):
    g = torch.Generator().manual_seed(case_seed(name))

    def randn(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32)

    def uniform(lo, hi, *shape):
        return torch.empty(*shape, dtype=torch.float32).uniform_(lo, hi, generator=g)

    u = randn(B, D, L)
    # A: negative, magnitudes spanning the trained-Mamba range (init is
    # -[1..N] per channel; training spreads it out).
    A = -torch.exp(uniform(math.log(0.5), math.log(16.0), D, N))

    if softplus:
        # raw (pre-softplus) delta; bias chosen so softplus(delta+bias)
        # lands in the realistic ~[1e-3, 0.1] region.
        delta = randn(B, D, L) * 0.5
        delta_bias = uniform(-6.0, -3.0, D) if with_bias else None
        if not with_bias:
            delta = delta - 4.5
        if delta_style == "extreme":
            # stress exp underflow: softplus(delta) up to ~10 -> exp(delta*A)
            # down to exp(-160) == 0.0 in f32
            delta = uniform(-8.0, 10.0, B, D, L)
    else:
        # delta used directly as the (positive) timestep
        delta = uniform(1e-3, 0.1, B, D, L)
        delta_bias = None

    bc_batch_shape = (B, groups, N, L) if groups else (B, N, L)
    Bmat = randn(*bc_batch_shape)
    Cmat = randn(*bc_batch_shape)
    D_skip = randn(D) if with_D else None
    z = randn(B, D, L) if with_z else None
    return u, delta, A, Bmat, Cmat, D_skip, z, delta_bias


def generate_case(name, B, D, L, N, **kw):
    softplus = kw.get("softplus", True)
    u, delta, A, Bmat, Cmat, D_skip, z, delta_bias = draw_inputs(
        name, B, D, L, N, **kw)

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
        "seed": case_seed(name), "torch_version": torch.__version__,
        # observed f32-vs-f64 gap, the floor a correct f32 kernel should hit
        "f32_max_abs_err": float((out_f32.double() - out_f64).abs().max()),
    }
    arrays["meta_json"] = np.frombuffer(
        json.dumps(meta).encode(), dtype=np.uint8).copy()

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GOLDEN_DIR / f"{name}.npz", **arrays)
    return meta


CORE_CASES = [
    # (name, B, D, L, N, kwargs) — full Mamba config (z, D, bias, softplus)
    # unless overridden
    ("tiny",                1, 4,    8,    16, {}),
    ("small",               2, 8,    32,   16, {}),
    ("medium",              2, 64,   128,  16, {}),
    ("channels",            1, 256,  64,   16, {}),
    ("long_seq",            1, 16,   1024, 16, {}),
    ("edge_L1",             2, 8,    1,    16, {}),
    ("edge_D1",             1, 1,    32,   16, {}),
    ("state8",              2, 8,    32,   8,  {}),
    ("state13_neon_tail",   2, 8,    32,   13, {}),
    ("no_z",                2, 8,    32,   16, {"with_z": False}),
    ("no_D",                2, 8,    32,   16, {"with_D": False}),
    ("no_bias",             2, 8,    32,   16, {"with_bias": False}),
    ("no_softplus",         2, 8,    32,   16, {"softplus": False}),
    ("extreme_delta",       2, 8,    64,   16, {"delta_style": "extreme"}),
    ("grouped_BC",          2, 8,    32,   16, {"groups": 2}),
]

LARGE_CASES = [
    # benchmark-shaped; regenerate on demand, never committed
    ("large_mamba130m", 1, 1536, 512,  16, {}),
    ("large_batch",     4, 768,  1024, 16, {}),
]


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
