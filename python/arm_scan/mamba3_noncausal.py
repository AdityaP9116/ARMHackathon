"""Non-causal Mamba-3 aggregation — 1D and 2D, two ways.

**Non-causal** means every position aggregates over *every* other position
rather than only its own past. That is what a vision model wants: a pixel has
no "past". VNCT lifts Mamba-3's trapezoidal dynamics this way; its code is
unreleased, so everything here is our reading of the operator, and the caveat
at the bottom of this docstring is not decoration.

THE RESULT THAT SHAPED THIS MODULE
-----------------------------------
The plan predicted non-causal would need "a second kernel — two dense GEMMs",
O(L^2), with a thin moat because GEMMs are what BLAS is good at. **That is not
what the maths says.** Unrolling the recurrence gives

    y_t = sum_s M[t,s] * (q_t . k_s) * v_s + D*v_t,     M[t,s] = e^(L_t - L_s) * scale_s

and because the decay `e^(L_t - L_s)` **factorises** into `e^(L_t) * e^(-L_s)`,
the sum over `s < t` is exactly a forward scan and the sum over `s > t` is
exactly a backward one. So

    sum_{all s}  =  sum_{s<=t}  +  sum_{s>=t}  -  sum_{s=t}

    non-causal   =    forward    +   backward   -   diagonal

Both directions already exist (`mamba3_scan_pair`). **Non-causal costs 2x a
causal scan, not O(L^2), and needs no new kernel at all.** `noncausal_scan`
below is that route; `noncausal_scan_dense` is the O(L^2) mask form, kept as an
independent oracle and as the other half of the comparison.

The diagonal correction needs `q_t . k_t`, which looks like it needs the
rotated q/k the kernel computes internally — it does not. **A dot product is
invariant under rotating both operands by the same angle**, and RoPE rotates
`q_t` and `k_t` by exactly the same `theta_t`. So the correction is computable
from the unrotated tensors, which is why this composes over the public op
instead of needing kernel surgery.

THE ORACLE CAVEAT, STATED NOT BURIED
-------------------------------------
For 1D we captured ground truth from the official GPU kernels. **For non-causal
2D nothing authoritative exists** — VNCT's repository is unreleased (404,
checked twice). The gates here prove the two routes agree with each other and
that the causal path is reproduced exactly when the mask is restricted; they do
**not** prove this is what VNCT's authors intended. No accuracy claim is
available for this operator and none is made.
"""

import torch

from .ss2d import _MERGES, grid_to_views
from .ss2d_mamba3 import grid_to_views_time_major, views_to_grid_time_major


def _diagonal_term(q, k, v, dt, trap, q_bias, k_bias, D=None, z=None):
    """The `s == t` contribution: `(D + gamma_t * (q_t . k_t)) * v_t`, gated.

    Both `forward` and `backward` include this, so the non-causal sum must
    subtract one copy.

    Computed from UNROTATED q/k on purpose: `q_t . k_t` is invariant under a
    shared rotation, and RoPE gives `q_t` and `k_t` the same angle. Rotating
    here would be wasted work and would need the angle pre-pass duplicated.

    q, k    : (b, l, 1, dqk) or (b, l, dqk)
    v, z    : (b, l, h, dv)
    dt, trap: (b, h, l)
    """
    if q.dim() == 4:
        q = q[:, :, 0, :]
    if k.dim() == 4:
        k = k[:, :, 0, :]
    b, length, dqk = q.shape
    h, dv = v.shape[2], v.shape[3]

    # (b, l, dqk) + (h, dqk) -> (b, l, h, dqk), then contract the head dim.
    qb = q.unsqueeze(2) + q_bias.view(1, 1, h, dqk)
    kb = k.unsqueeze(2) + k_bias.view(1, 1, h, dqk)
    qk = (qb * kb).sum(-1)                                   # (b, l, h)

    gamma = dt * torch.sigmoid(trap)                         # (b, h, l)
    coef = gamma.permute(0, 2, 1) * qk                       # (b, l, h)
    if D is not None:
        coef = coef + D.view(1, 1, h)
    out = coef.unsqueeze(-1) * v
    if z is not None:
        out = out * (z * torch.sigmoid(z))
    return out


def noncausal_scan(q, k, v, adt, dt, trap, q_bias, k_bias, angles=None,
                   D=None, z=None, cos=None, sin=None):
    """Non-causal 1D aggregation, via two scans. `(b, l, h, dv)`.

    O(L) per position — the same cost class as the causal scan, twice over.
    """
    from .mamba3 import mamba3_scan_pair
    fwd, bwd = mamba3_scan_pair(q, k, v, adt, dt, trap, q_bias, k_bias,
                                angles=angles, D=D, z=z, cos=cos, sin=sin)
    return fwd + bwd - _diagonal_term(q, k, v, dt, trap, q_bias, k_bias, D, z)


def noncausal_scan_dense(q, k, v, adt, dt, trap, q_bias, k_bias, angles=None,
                         D=None, z=None, cos=None, sin=None, causal=False):
    """Non-causal 1D aggregation, via the explicit `(L, L)` mask.

    `causal=True` drops the upper half of the mask, giving the **dual form of
    the causal scan**. That is not a curiosity: it is an O(L^2) GEMM algorithm
    sharing no code with the kernel, so making it reproduce `mamba3_scan`
    independently validates the mask derivation — and therefore the non-causal
    mask, which is the same construction run twice. `check_mamba3_noncausal.py`
    uses it for exactly that.

    `y = (M o (Q K^T)) V + D*V`, gated. O(L^2) in time and memory, but every
    step is a GEMM. This exists for two reasons: it is an independent algorithm
    to check `noncausal_scan` against, and it is the other half of the
    scan-vs-dense comparison that `bench/bench_mamba3_noncausal.py` publishes.

    Pure PyTorch by design. A hand-written Rust GEMM would be competing with
    BLAS on BLAS's home ground, which the plan already predicted would be a
    thin moat — and the measurement below shows the dense form loses on
    asymptotics long before kernel quality matters.
    """
    from .mamba3 import angles_to_cos_sin

    if q.dim() == 3:
        q = q.unsqueeze(2)
    if k.dim() == 3:
        k = k.unsqueeze(2)
    b, length, _, dqk = q.shape
    h, dv = v.shape[2], v.shape[3]
    half = dqk // 2
    if cos is None:
        if angles is None:
            raise ValueError("pass either `angles` or both `cos` and `sin`")
        cos, sin = angles_to_cos_sin(angles, dt, half)

    # Bias, then rotate — interleaved pairs, matching the SISO kernel.
    qb = q[:, :, 0, :].unsqueeze(2) + q_bias.view(1, 1, h, dqk)
    kb = k[:, :, 0, :].unsqueeze(2) + k_bias.view(1, 1, h, dqk)

    def rope(t):
        v0, v1 = t[..., 0::2], t[..., 1::2]
        out = torch.empty_like(t)
        out[..., 0::2] = v0 * cos - v1 * sin
        out[..., 1::2] = v0 * sin + v1 * cos
        return out

    qr, kr = rope(qb), rope(kb)

    lam = torch.sigmoid(trap)
    gamma = dt * lam

    def half_mask(reverse):
        """Strictly-triangular mask for one traversal, in original order."""
        a, d_, l_ = (adt, dt, lam)
        if reverse:
            a, d_, l_ = (torch.flip(x, dims=[-1]) for x in (a, d_, l_))
        sh = torch.zeros_like(d_)
        sh[..., :-1] = d_[..., 1:] * (1.0 - l_[..., 1:])
        sc = d_ * l_ + sh
        cum = torch.cumsum(a, dim=-1)
        m = torch.exp(cum.unsqueeze(-1) - cum.unsqueeze(-2)) * sc.unsqueeze(-2)
        m = m * torch.tril(torch.ones(length, length, dtype=m.dtype,
                                      device=m.device), -1)
        return torch.flip(m, dims=[-1, -2]) if reverse else m

    M = half_mask(False) + torch.diag_embed(gamma)
    if not causal:
        M = M + half_mask(True)
    qk = torch.einsum("bthd,bshd->bhts", qr, kr)
    out = torch.einsum("bhts,bhts,bshp->bthp", M, qk, v)
    if D is not None:
        out = out + D.view(1, 1, h, 1) * v
    if z is not None:
        out = out * (z * torch.sigmoid(z))
    return out


def ss2d_noncausal_mamba3(q, k, v, adt, dt, trap, q_bias, k_bias, angles,
                          D=None, z=None, merge="sum"):
    """4-direction **non-causal** 2D aggregation over a token grid.

    Same layouts as `arm_scan.ss2d_scan_mamba3`. Each of the two token
    orderings (row-major, column-major) is aggregated non-causally, so the
    result is

        (row_fwd + row_bwd - diag) + (col_fwd + col_bwd - diag)

    which is the existing four-direction sum **minus two diagonals** — the
    causal cross-scan already computes every term needed. That is the practical
    consequence of the factorisation in the module docstring: going non-causal
    in 2D costs one extra elementwise pass, not a new kernel.

    merge "none" returns the two per-ordering planes `(rows, cols)`, each
    `(b, H, W, h, dv)`, rather than four causal directions.
    """
    if merge not in _MERGES:
        raise ValueError(f"merge must be one of {_MERGES}, got {merge!r}")
    from .mamba3 import mamba3_scan_pair

    b, hh, ww = q.shape[:3]
    fwd, bwd = mamba3_scan_pair(
        grid_to_views_time_major(q), grid_to_views_time_major(k),
        grid_to_views_time_major(v),
        grid_to_views(adt), grid_to_views(dt), grid_to_views(trap),
        q_bias, k_bias, angles=grid_to_views_time_major(angles), D=D,
        z=None if z is None else grid_to_views_time_major(z),
    )
    diag = _diagonal_term(
        grid_to_views_time_major(q), grid_to_views_time_major(k),
        grid_to_views_time_major(v), grid_to_views(dt), grid_to_views(trap),
        q_bias, k_bias, D, None if z is None else grid_to_views_time_major(z))

    nc = fwd + bwd - diag
    rows, cols = views_to_grid_time_major(nc[:b], nc[b:], hh, ww)
    if merge == "none":
        return rows, cols
    out = rows + cols
    return out if merge == "sum" else out * 0.5
