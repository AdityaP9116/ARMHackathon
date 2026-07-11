"""arm_scan — Arm-optimized Mamba selective scan for the PyTorch ecosystem.

Quick start (PyTorch):
    import arm_scan
    arm_scan.patch()          # HF transformers Mamba now uses the kernel
    ...                        # run any Mamba model on CPU as usual
    arm_scan.stats()          # confirm the kernel actually ran

Direct op:  arm_scan.selective_scan(u, delta, A, B, C, D=..., z=...)
NumPy-only: arm_scan.selective_scan_numpy(...)  (no torch required)
"""

from ._ffi import lib_path
from .numpy_api import selective_scan_numpy

__all__ = [
    "selective_scan_numpy",
    "lib_path",
    "selective_scan",
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
    if name in ("patch", "unpatch", "stats"):
        return getattr(importlib.import_module(".patch", __name__), name)
    raise AttributeError(f"module 'arm_scan' has no attribute '{name}'")
