"""arm_scan — Arm-optimized Mamba selective scan for the PyTorch ecosystem.

Quick start (PyTorch):
    import arm_scan
    arm_scan.patch()          # HF transformers Mamba now uses the kernel
    ...                        # run any Mamba model on CPU as usual
    arm_scan.stats()          # confirm the kernel actually ran

Direct op:  arm_scan.selective_scan(u, delta, A, B, C, D=..., z=...)
Bidirectional: arm_scan.bidirectional_scan(...)  (both time directions, merged)
2D cross-scan: arm_scan.ss2d_scan(u, delta, A, B, C, ...)  ((b,d,h,w) grid,
    VMamba-style 4 directions as two pairs on the fused bidirectional kernel)
Mamba-3:  arm_scan.mamba3_scan(q, k, v, adt, dt, trap, q_bias, k_bias,
    angles=...)  -- SISO; mamba3_scan_pair(...) gives both traversal
    directions, which is what the bidirectional and 2D topologies build on
2D Mamba-3: arm_scan.ss2d_scan_mamba3(...)  (grid-shaped inputs, four
    directions as two pairs -- no CPU implementation of this exists elsewhere)
NumPy-only: arm_scan.selective_scan_numpy(...)  (no torch required)
"""

from ._ffi import lib_path
from .numpy_api import selective_scan_numpy

__all__ = [
    "selective_scan_numpy",
    "lib_path",
    "selective_scan",
    "bidirectional_scan",
    "ss2d_scan",
    "use_arm_scan",
    "mamba3_scan",
    "mamba3_scan_pair",
    "ss2d_scan_mamba3",
    "angles_to_cos_sin",
    "patch",
    "unpatch",
    "stats",
]


def __getattr__(name):
    # torch-dependent pieces load lazily so numpy-only users (and CI's
    # torch-free golden check) never import torch. importlib avoids the
    # `from . import x` -> package-getattr -> recursion trap.
    import importlib

    if name == "selective_scan":
        return importlib.import_module(".op", __name__).selective_scan
    if name == "bidirectional_scan":
        return importlib.import_module(
            ".bidirectional", __name__).bidirectional_scan
    if name in ("ss2d_scan", "use_arm_scan"):
        return getattr(importlib.import_module(".ss2d", __name__), name)
    if name in ("mamba3_scan", "mamba3_scan_pair", "angles_to_cos_sin"):
        return getattr(importlib.import_module(".mamba3", __name__), name)
    if name == "ss2d_scan_mamba3":
        return importlib.import_module(
            ".ss2d_mamba3", __name__).ss2d_scan_mamba3
    if name in ("patch", "unpatch", "stats"):
        return getattr(importlib.import_module(".patch", __name__), name)
    raise AttributeError(f"module 'arm_scan' has no attribute '{name}'")
