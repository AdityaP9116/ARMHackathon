"""Generate 2D cross-scan (SS2D) golden vectors.

TOPOLOGY_IMPLEMENTATION_PLAN.md §3.3. Same discipline as the 1D goldens
(`gen_golden.py`): ground truth from the vendored upstream reference at
float64, plus the same computation at float32 to record the tolerance FLOOR a
correct f32 kernel should land near — not merely under.

WHAT IS STORED, AND WHY PER-DIRECTION
-------------------------------------
Each case stores the four direction planes SEPARATELY, before any merge:

    row_fwd, row_bwd, col_fwd, col_bwd     each (b, d, h, w)

so a kernel bug and a merge-strategy bug cannot be confused for one another
(§3.3 is explicit about this). The merge itself is one addition in Python and
is checked by the app-level parity test, not here.

Ground truth is built the way §3.1 DEFINES the cross-scan: take the grid, form
the row-major and column-major token orderings, and for each run the reference
scan forward and on the flipped sequence (flipping the output back). That is
also exactly what `arm_scan.ss2d.ss2d_scan(..., merge="none")` must reproduce.

Grid coverage mirrors the 1D edge philosophy: square, non-square, and H/W not
multiples of 4 (the transpose/tail paths), plus a d_state that is not a
multiple of 4 to exercise the NEON general path the way `state13_neon_tail`
does in 1D.

Determinism: the inputs come from `golden_inputs.draw_inputs_2d`, a torch-free
draw seeded per case from the case name — the same module and the same reason
as the 1D set. Torch does not keep its RNG stream stable across releases, so
the draws this script used to make through `torch.Generator` could not be
reproduced under a different torch. `verify_golden_2d.py` now redraws every
case with numpy alone and compares bit-for-bit.

Usage: python tests/gen_golden_2d.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from golden_inputs import (CORE_CASES_2D, INPUT_DRAW_SPEC,  # noqa: E402
                           case_seed, draw_inputs_2d)
from reference import selective_scan_ref  # noqa: E402

# Own subdirectory, deliberately: the 1D verifier globs `golden/*.npz` and the
# two case schemas are different (four direction planes here vs one output
# there). Keeping them apart means neither script can pick up the other's files.
GOLDEN_DIR = Path(__file__).parent / "golden" / "2d"


def _views(t):
    """Grid -> (row-major seq, col-major seq). Mirrors arm_scan.ss2d."""
    b, c, h, w = t.shape
    return t.reshape(b, c, h * w), t.transpose(2, 3).reshape(b, c, w * h)


def cross_scan_reference(inp, dtype):
    """The four direction planes, computed with the vendored 1D reference.

    This is the DEFINITION the kernel is held to: backward directions are
    flip -> forward scan -> flip back.
    """
    cast = {k: (v.to(dtype) if torch.is_tensor(v) else v)
            for k, v in inp.items()}
    b, d, h, w = cast["u"].shape

    u_r, u_c = _views(cast["u"])
    dl_r, dl_c = _views(cast["delta"])
    B_r, B_c = _views(cast["B"])
    C_r, C_c = _views(cast["C"])

    def scan(u, delta, Bs, Cs):
        # compute_dtype is explicit: upcasting the inputs alone would still
        # let the reference accumulate in f32 (gen_golden.py does the same).
        return selective_scan_ref(
            u, delta, cast["A"], Bs, Cs, D=cast["D"], z=None,
            delta_bias=cast["delta_bias"], delta_softplus=True,
            return_last_state=False, compute_dtype=dtype)

    def pair(u, delta, Bs, Cs):
        fwd = scan(u, delta, Bs, Cs)
        bwd = scan(u.flip(-1), delta.flip(-1), Bs.flip(-1),
                   Cs.flip(-1)).flip(-1)
        return fwd, bwd

    row_f, row_b = pair(u_r, dl_r, B_r, C_r)
    col_f, col_b = pair(u_c, dl_c, B_c, C_c)

    to_grid_r = lambda t: t.reshape(b, d, h, w)                 # noqa: E731
    to_grid_c = lambda t: t.reshape(b, d, w, h).transpose(2, 3)  # noqa: E731
    return (to_grid_r(row_f), to_grid_r(row_b),
            to_grid_c(col_f), to_grid_c(col_b))


def main():
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    print(f"torch {torch.__version__}, numpy {np.__version__} "
          f"(inputs: {INPUT_DRAW_SPEC})\n")

    for name, b, d, h, w, n in CORE_CASES_2D:
        # the draws are numpy (torch-free by design); the reference is torch
        inp = {k: torch.from_numpy(v)
               for k, v in draw_inputs_2d(name, b, d, h, w, n).items()}
        planes64 = cross_scan_reference(inp, torch.float64)
        planes32 = cross_scan_reference(inp, torch.float32)

        # The floor: what upstream's own f32 evaluation costs against f64.
        floor = max(float((p32.to(torch.float64) - p64).abs().max())
                    for p32, p64 in zip(planes32, planes64))

        keys = ("row_fwd", "row_bwd", "col_fwd", "col_bwd")
        arrays = {k: v.numpy().astype(np.float32)
                  for k, v in inp.items()}
        arrays.update({f"out_{k}": p.numpy().astype(np.float64)
                       for k, p in zip(keys, planes64)})
        np.savez_compressed(GOLDEN_DIR / f"{name}.npz", **arrays)

        manifest.append(dict(
            name=name, batch=b, dim=d, height=h, width=w, state=n,
            groups=None, delta_softplus=True, has_D=True,
            has_delta_bias=True,
            # `seed` and `input_draw` pin the INPUTS (torch-free, reproducible
            # anywhere); `torch_version` records only what computed the planes.
            seed=case_seed(name), input_draw=INPUT_DRAW_SPEC,
            torch_version=torch.__version__, numpy_version=np.__version__,
            f32_max_abs_err=floor))
        print(f"{name:16s} b{b} d{d} {h}x{w} n{n}  "
              f"L={h*w:5d}  f32 floor {floor:.3e}")

    path = GOLDEN_DIR / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {len(manifest)} cases + {path.name}")


if __name__ == "__main__":
    main()
