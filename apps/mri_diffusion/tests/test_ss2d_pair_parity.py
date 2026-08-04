"""K2 gate: the traversal-PAIR cross-scan == the legacy four-direction stack.

`SS2DBlock._cross_scan` (shipping) runs SS2D as two traversal-order pairs so
the kernel shares Pass A between each pair's two directions.
`SS2DBlock._cross_scan_legacy` (oracle) runs the older formulation: four
directions stacked into one forward call, with four `torch.flip` copies. The
two are algebraically identical, and this file is what says so.

Checked on BOTH backends, because they fail differently:

  reference path  — pure torch on both sides. Any mismatch here is a REFACTOR
                    bug (view/reshape/transpose order, D applied the wrong
                    number of times). Tolerance is tight.
  kernel path     — arm_scan on both sides. Any mismatch here is a KERNEL bug
                    in the fused bidirectional traversal. Tolerance is the
                    standing fp32 one: NEON `reverse` agrees with
                    flip-forward-flip to ~1e-7, not bit-exactly, because the
                    4-timestep vector body and the scalar tail evaluate
                    softplus/SiLU by different means and flipping moves
                    timesteps across that boundary (documented on
                    `ScanInput::reverse` in kernel/arm-scan-core/src/lib.rs).

Grids deliberately include a non-square case and one whose H and W are not
multiples of 4, to exercise the same tail paths `state13_neon_tail` covers in
1D. Run under RAYON_NUM_THREADS in {1,2,8} to also cover rayon bit-identity.

Usage: python -u apps/mri_diffusion/tests/test_ss2d_pair_parity.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from apps.mri_diffusion.backbone.mamba_ss2d import SS2DBlock  # noqa: E402

# (h, w): square, non-square, and non-multiple-of-4 in both axes.
GRIDS = [(8, 8), (6, 10), (7, 5), (12, 4)]
REF_TOL = 2e-5   # torch-vs-torch: refactor bug detector
KERNEL_TOL = 1e-4  # standing golden gate (CLAUDE.md)


def _block(dim=8, d_state=16, seed=0):
    torch.manual_seed(seed)
    blk = SS2DBlock(dim, emb_dim=16, d_state=d_state).eval()
    # out_proj is zero-init (identity-at-init residual), which would make the
    # block's OUTPUT identical no matter what the scan did. Compare the scan
    # section directly instead, and give D/A real values.
    with torch.no_grad():
        blk.D.uniform_(0.5, 1.5)
    return blk


def _compare(blk, s, label, tol):
    with torch.no_grad():
        new = blk._cross_scan(s)
        old = blk._cross_scan_legacy(s)
    err = (new - old).abs().max().item()
    scale = old.abs().max().item()
    ok = err < tol
    print(f"    {label:28s} max_abs {err:.3e}  (scale {scale:.3e})  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok, err


def main():
    torch.manual_seed(0)
    all_ok = True

    print("reference path (torch pair vs torch 4-stack) — refactor gate")
    for h, w in GRIDS:
        blk = _block()
        s = torch.randn(2, blk.inner, h, w)
        ok, _ = _compare(blk, s, f"grid {h}x{w}", REF_TOL)
        all_ok &= ok

    print("\nkernel path (arm_scan pair vs arm_scan 4-stack) — kernel gate")
    try:
        sys.path.insert(0, str(ROOT / "python"))
        from arm_scan._ffi import load
        from arm_scan.ss2d import use_arm_scan
        load()
    except Exception as exc:  # noqa: BLE001 - report, don't mask
        print(f"    SKIPPED — arm_scan cdylib not loadable: {exc}")
        print("    (build it: cargo build --release -p arm-scan-ffi)")
    else:
        for h, w in GRIDS:
            blk = _block()
            n = use_arm_scan(blk)
            assert n == 1, f"expected 1 block switched, got {n}"
            s = torch.randn(2, blk.inner, h, w)
            ok, _ = _compare(blk, s, f"grid {h}x{w}", KERNEL_TOL)
            all_ok &= ok

        print("\ncross-backend (kernel pair vs torch pair) — numerics gate")
        for h, w in GRIDS:
            blk = _block()
            s = torch.randn(2, blk.inner, h, w)
            with torch.no_grad():
                ref = blk._cross_scan(s)
                use_arm_scan(blk)
                got = blk._cross_scan(s)
            err = (got - ref).abs().max().item()
            ok = err < KERNEL_TOL
            print(f"    {f'grid {h}x{w}':28s} max_abs {err:.3e}  "
                  f"{'PASS' if ok else 'FAIL'}")
            all_ok &= ok

    print("\nRESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
