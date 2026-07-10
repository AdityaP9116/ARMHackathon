"""Vendored reference selective scan from state-spaces/mamba (mamba-ssm).

Source: mamba_ssm/ops/selective_scan_interface.py :: selective_scan_ref
        https://github.com/state-spaces/mamba (Apache-2.0)
        Copyright (c) 2023, Tri Dao, Albert Gu.

Vendored verbatim except for three documented deviations:
  1. einops removed:
       rearrange(D, "d -> d 1")            -> D.unsqueeze(-1)
       repeat(x, "B G N L -> B (G H) N L") -> x.repeat_interleave(H, dim=1)
     (repeat_interleave on dim=1 produces exactly the (G H) grouped-channel
      ordering: output channel d = g*H + h.)
  2. `compute_dtype` parameter added. Upstream hard-casts inputs with
     .float() (i.e. float32). With compute_dtype=torch.float32 (the default)
     this function is semantically identical to upstream; passing
     torch.float64 lets us compute high-precision golden outputs.
  3. Complex A is not supported (raises NotImplementedError). Standard Mamba
     uses real A; the complex path is dead code for every model we target.
  4. Time-invariant (2-dim) B/C are not supported (raises
     NotImplementedError). Every Mamba call site passes input-dependent
     (variable) B and C; the S4-style time-invariant path is dead code here.

This function is the correctness ground truth for the Rust kernel. Do not
"improve" it — fidelity to upstream is the whole point.
"""

import torch
import torch.nn.functional as F


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                       delta_softplus=False, return_last_state=False,
                       compute_dtype=torch.float32):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L) or r(B G N L)
    C: r(B N L) or r(B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D)

    out: r(B D L)
    last_state (optional): r(B D dstate)
    """
    if A.is_complex():
        raise NotImplementedError("complex A not supported in vendored reference")
    dtype_in = u.dtype
    u = u.to(compute_dtype)
    delta = delta.to(compute_dtype)
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].to(compute_dtype)
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if not (is_variable_B and is_variable_C):
        raise NotImplementedError(
            "non-variable (time-invariant) B/C not supported: every Mamba "
            "selective-scan call site uses input-dependent B and C")
    A = A.to(compute_dtype)
    B = B.to(compute_dtype)
    C = C.to(compute_dtype)
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if B.dim() == 3:
        deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
    else:
        B = B.repeat_interleave(dim // B.shape[1], dim=1)
        deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = C.repeat_interleave(dim // C.shape[1], dim=1)
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if C.dim() == 3:
            y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
        else:
            y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        ys.append(y)
    y = torch.stack(ys, dim=2)  # (batch dim L)
    out = y if D is None else y + u * D.to(compute_dtype).unsqueeze(-1)
    if z is not None:
        out = out * F.silu(z.to(compute_dtype))
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)
