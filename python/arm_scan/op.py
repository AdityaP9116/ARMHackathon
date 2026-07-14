"""PyTorch front-end: a `torch.library` custom op wrapping the kernel.

Registered as `arm_scan::selective_scan` with a fake (meta) kernel so it
composes with `torch.compile` instead of graph-breaking — important because
the project's fair baseline IS torch.compile.

Inference-only: no autograd formula is registered. CPU float32 tensors.
"""

from typing import Optional, Tuple

import torch

from . import _ffi

_CALLS = {"n": 0}


def _c(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().float()


@torch.library.custom_op("arm_scan::selective_scan", mutates_args=())
def _selective_scan_op(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d_skip: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    delta_bias: Optional[torch.Tensor],
    h0: Optional[torch.Tensor],
    delta_softplus: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, dim, length = u.shape
    state = a.shape[1]
    groups = b.shape[1] if b.dim() == 4 else 1

    out = torch.empty_like(u)
    last_state = u.new_empty((batch, dim, state))
    dims = _ffi.ArmScanDims(batch, dim, length, state, groups)
    ptr = lambda t: 0 if t is None else t.data_ptr()
    _ffi.scan_raw(
        dims, u.data_ptr(), delta.data_ptr(), a.data_ptr(), b.data_ptr(),
        c.data_ptr(), ptr(d_skip), ptr(z), ptr(delta_bias),
        delta_softplus, "auto", "auto", out.data_ptr(),
        last_state.data_ptr(), ptr_h0=ptr(h0),
    )
    _CALLS["n"] += 1
    return out, last_state


@_selective_scan_op.register_fake
def _(u, delta, a, b, c, d_skip, z, delta_bias, h0, delta_softplus):
    return torch.empty_like(u), u.new_empty(
        (u.shape[0], u.shape[1], a.shape[1]))


def selective_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                   delta_softplus=False, return_last_state=False,
                   initial_state=None):
    """Selective scan on CPU float32 torch tensors.

    u, delta, z: (batch, dim, len); A: (dim, state);
    B, C: (batch, state, len) or (batch, groups, state, len);
    D, delta_bias: (dim,).
    initial_state: optional (batch, dim, state) SSM state to resume from
    (defaults to zeros); pair with return_last_state to stream/decode.
    Returns out (batch, dim, len) [, last_state (batch, dim, state)].

    Tensors are made contiguous/f32 here, so callers can pass transposed
    views directly.
    """
    batch, dim, length = u.shape
    state = A.shape[1]
    if B.dim() == 3:
        B = B.reshape(batch, 1, state, length)
    if C.dim() == 3:
        C = C.reshape(batch, 1, state, length)
    out, last_state = _selective_scan_op(
        _c(u), _c(delta), _c(A), _c(B), _c(C),
        None if D is None else _c(D),
        None if z is None else _c(z),
        None if delta_bias is None else _c(delta_bias),
        None if initial_state is None else _c(initial_state),
        delta_softplus,
    )
    return (out, last_state) if return_last_state else out


def kernel_calls() -> int:
    """How many times the native kernel has been invoked (engagement
    check for tests and benchmarks)."""
    return _CALLS["n"]
