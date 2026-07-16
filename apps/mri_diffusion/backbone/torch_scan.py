"""Pure-PyTorch selective scan — the backbone's REFERENCE path.

Same semantics as the vendored upstream reference
(tests/reference/selective_scan_ref.py) restricted to what the SS2D block
uses: variable 3-dim B/C, delta_softplus fused, no z-gating here (the block
gates outside the scan). Differentiable, so Phase-B training works.

Phase C swaps this call for arm_scan (ss2d unfused path) behind the same
signature — that seam is the whole point of this file.
"""

import torch
import torch.nn.functional as F


def selective_scan_torch(u, delta, A, B, C, D=None, delta_bias=None,
                         delta_softplus=True):
    """u,delta:(b,d,l)  A:(d,n)  B,C:(b,n,l)  D,delta_bias:(d,) -> (b,d,l)"""
    if delta_bias is not None:
        delta = delta + delta_bias[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    x = A.new_zeros((u.shape[0], u.shape[1], A.shape[1]))
    ys = []
    for t in range(u.shape[2]):
        x = deltaA[:, :, t] * x + deltaB_u[:, :, t]
        ys.append(torch.einsum("bdn,bn->bd", x, C[:, :, t]))
    y = torch.stack(ys, dim=2)
    return y if D is None else y + u * D[None, :, None]
