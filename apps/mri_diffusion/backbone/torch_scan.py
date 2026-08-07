"""Pure-PyTorch selective scan — the backbone's REFERENCE path.

Same semantics as the vendored upstream reference
(tests/reference/selective_scan_ref.py) restricted to what the SS2D block
uses: variable 3-dim B/C, delta_softplus fused, no z-gating here (the block
gates outside the scan). Differentiable, so Phase-B training works.

Two seams live here, and the kernel has a drop-in for each:

  `selective_scan_torch`      <-> `arm_scan.ss2d.scan_fn_arm`
  `selective_scan_pair_torch` <-> `arm_scan.ss2d.scan_pair_arm`

The pair form is the one `SS2DBlock.forward` uses: it returns both time
directions, because SS2D's four directions are two traversal-order pairs and
the kernel can emit a pair from a single shared Pass A. Here in the reference
the backward direction is written as flip-forward-flip, which is the
*specification* the kernel's `reverse` traversal is defined against (enforced
bit-for-bit on the scalar backend by `reverse_matches_flip_forward_flip` in
the Rust property tests, and in numpy by tests/check_bidirectional_math.py).
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

    # `unbind` the time axis ONCE rather than indexing `deltaA[:, :, t]` inside
    # the loop. Semantically identical, but it decides whether training is
    # feasible at all: each in-loop `select` registers its own backward, which
    # scatters a (b, d, l, n) zero-filled gradient buffer per timestep — L of
    # them, so backward cost grows as O(L * b*d*l*n) and dwarfs forward. On a
    # 32x32 grid that measured 147s backward against 6.3s forward (23x); one
    # `unbind` (whose backward is a single stack) brings it back in line.
    dA_t, dBu_t = deltaA.unbind(2), deltaB_u.unbind(2)
    C_t = C.unbind(-1)
    x = A.new_zeros((u.shape[0], u.shape[1], A.shape[1]))
    ys = []
    for t in range(u.shape[2]):
        x = dA_t[t] * x + dBu_t[t]
        ys.append(torch.einsum("bdn,bn->bd", x, C_t[t]))
    y = torch.stack(ys, dim=2)
    return y if D is None else y + u * D[None, :, None]


def selective_scan_pair_torch(u, delta, A, B, C, D=None, delta_bias=None,
                              delta_softplus=True):
    """Both time directions of the same sequence -> `(fwd, bwd)`.

    Signature-compatible with `arm_scan.ss2d.scan_pair_arm`. Both outputs are
    indexed at `t` in forward-time order: reversing changes the recurrence's
    traversal order, never the layout, so `bwd` is flipped back before return.

    `D` is applied inside BOTH directions (each is a full scan). Summing the
    pair therefore carries `2*D*u` — matching the four-separate-scans
    formulation this replaces. See the note in `arm_scan.ss2d.ss2d_scan`.

    The two directions are STACKED on the batch axis and scanned in one call.
    That matters: this reference is a Python loop over the time axis, so two
    separate calls would run the loop twice and make training (which must use
    this path — the kernel registers no autograd) 2x slower than the
    formulation it replaced. Batching keeps the loop count identical; only the
    batch widens. The kernel gets its win a different way, by sharing Pass A,
    which a torch reference cannot express.
    """
    b = u.shape[0]

    def both(t):
        return torch.cat((t, t.flip(-1)), dim=0)

    out = selective_scan_torch(
        both(u), both(delta), A, both(B), both(C), D=D,
        delta_bias=delta_bias, delta_softplus=delta_softplus)
    return out[:b], out[b:].flip(-1)
