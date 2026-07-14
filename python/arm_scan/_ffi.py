"""ctypes bindings to the arm-scan-ffi cdylib.

Library search order:
  1. ARM_SCAN_LIB environment variable (full path to the library)
  2. next to this package (wheel layout, Phase 5)
  3. the repo's cargo output dirs (kernel/target/{release,debug})
"""

import ctypes
import os
import sys
from pathlib import Path

ABI_VERSION = 3   # 3: added the `reverse` parameter

_LIB_NAMES = {
    "win32": ["arm_scan_ffi.dll"],
    "darwin": ["libarm_scan_ffi.dylib"],
}.get(sys.platform, ["libarm_scan_ffi.so"])

ERROR_NAMES = {
    0: "ok",
    1: "null pointer",
    2: "invalid dims/shapes",
    3: "backend unavailable on this platform",
    4: "bad backend/threading enum",
    5: "panic inside kernel",
}

BACKENDS = {"auto": 0, "scalar": 1, "neon": 2}
THREADING = {"auto": 0, "sequential": 1, "seq": 1, "rayon": 2}


class ArmScanDims(ctypes.Structure):
    _fields_ = [
        ("batch", ctypes.c_size_t),
        ("dim", ctypes.c_size_t),
        ("len", ctypes.c_size_t),
        ("state", ctypes.c_size_t),
        ("groups", ctypes.c_size_t),
    ]


def _candidate_paths():
    env = os.environ.get("ARM_SCAN_LIB")
    if env:
        yield Path(env)
    here = Path(__file__).resolve().parent
    for name in _LIB_NAMES:
        yield here / name
    repo_root = here.parents[1]  # python/arm_scan -> repo root
    for profile in ("release", "debug"):
        for name in _LIB_NAMES:
            yield repo_root / "kernel" / "target" / profile / name


_lib = None
_lib_path = None


def load():
    """Load (once) and return the ctypes library handle."""
    global _lib, _lib_path
    if _lib is not None:
        return _lib
    tried = []
    for path in _candidate_paths():
        tried.append(str(path))
        if path.is_file():
            lib = ctypes.CDLL(str(path))
            version = lib.arm_scan_abi_version()
            if version != ABI_VERSION:
                raise RuntimeError(
                    f"{path} has ABI version {version}, this package needs "
                    f"{ABI_VERSION}; rebuild with `cargo build --release "
                    f"-p arm-scan-ffi`"
                )
            lib.arm_scan_selective_scan_f32.restype = ctypes.c_int
            lib.arm_scan_selective_scan_f32.argtypes = [
                ctypes.POINTER(ArmScanDims),
                *([ctypes.c_void_p] * 8),  # u delta a b c d_skip z delta_bias
                ctypes.c_int,  # delta_softplus
                ctypes.c_int,  # reverse
                ctypes.c_int,  # backend
                ctypes.c_int,  # threading
                ctypes.c_void_p,  # out
                ctypes.c_void_p,  # last_state
            ]
            _lib, _lib_path = lib, path
            return lib
    raise OSError(
        "arm_scan_ffi library not found; build it with `cargo build "
        "--release -p arm-scan-ffi` (searched: " + ", ".join(tried) + ")"
    )


def lib_path():
    load()
    return _lib_path


def scan_raw(dims, ptr_u, ptr_delta, ptr_a, ptr_b, ptr_c, ptr_d_skip, ptr_z,
             ptr_delta_bias, delta_softplus, backend, threading, ptr_out,
             ptr_last, *, reverse=False):
    """Thin call-through. Pointers are integer addresses; 0 means null.

    `reverse` is keyword-only on purpose: every caller here passes positionally,
    so slotting it into the middle (where it sits in the C signature) would
    silently shift `backend` into it. Position in Python need not match C.
    """
    lib = load()
    code = lib.arm_scan_selective_scan_f32(
        ctypes.byref(dims), ptr_u, ptr_delta, ptr_a, ptr_b, ptr_c,
        ptr_d_skip or None, ptr_z or None, ptr_delta_bias or None,
        int(bool(delta_softplus)), int(bool(reverse)),
        BACKENDS[backend], THREADING[threading],
        ptr_out, ptr_last or None,
    )
    if code != 0:
        raise RuntimeError(
            f"arm_scan kernel error {code}: "
            f"{ERROR_NAMES.get(code, 'unknown')}"
        )
