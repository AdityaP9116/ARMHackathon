"""NumPy front-end for the kernel — torch-free, used by tests/check_ffi.py
and anyone who wants the scan without the PyTorch dependency."""

import numpy as np

from . import _ffi


def _prep(x, name, shape=None):
    arr = np.ascontiguousarray(x, dtype=np.float32)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {arr.shape}")
    return arr


def selective_scan_numpy(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                         delta_softplus=False, return_last_state=False,
                         backend="auto", threading="auto", initial_state=None):
    """Selective scan on numpy arrays (see kernel docs for semantics).

    u, delta: (batch, dim, len); A: (dim, state);
    B, C: (batch, state, len) or (batch, groups, state, len);
    D, delta_bias: (dim,); z: (batch, dim, len).
    Returns out (batch, dim, len) [, last_state (batch, dim, state)].
    """
    u = _prep(u, "u")
    if u.ndim != 3:
        raise ValueError(f"u must be (batch, dim, len), got {u.shape}")
    batch, dim, length = u.shape
    delta = _prep(delta, "delta", (batch, dim, length))
    A = _prep(A, "A")
    if A.ndim != 2 or A.shape[0] != dim:
        raise ValueError(f"A must be ({dim}, state), got {A.shape}")
    state = A.shape[1]

    B = _prep(B, "B")
    C = _prep(C, "C")
    if B.ndim == 3:
        B = B.reshape(batch, 1, state, length)
    if C.ndim == 3:
        C = C.reshape(batch, 1, state, length)
    groups = B.shape[1]
    for name, m in (("B", B), ("C", C)):
        if m.shape != (batch, groups, state, length):
            raise ValueError(
                f"{name}: expected ({batch}, {groups}, {state}, {length}), "
                f"got {m.shape}")

    D = None if D is None else _prep(D, "D", (dim,))
    z = None if z is None else _prep(z, "z", (batch, dim, length))
    delta_bias = (None if delta_bias is None
                  else _prep(delta_bias, "delta_bias", (dim,)))
    h0 = (None if initial_state is None
          else _prep(initial_state, "initial_state", (batch, dim, state)))

    out = np.empty((batch, dim, length), dtype=np.float32)
    last = np.empty((batch, dim, state), dtype=np.float32)

    dims = _ffi.ArmScanDims(batch, dim, length, state, groups)
    ptr = lambda a: 0 if a is None else a.ctypes.data
    _ffi.scan_raw(
        dims, ptr(u), ptr(delta), ptr(A), ptr(B), ptr(C), ptr(D), ptr(z),
        ptr(delta_bias), delta_softplus, backend, threading,
        out.ctypes.data, last.ctypes.data, ptr_h0=ptr(h0),
    )
    return (out, last) if return_last_state else out
