"""SS2D for Mamba-3 — the 4-direction 2D cross-scan on the Mamba-3 primitive.

The same topology as `ss2d.py`, over a different recurrence. Four directions are
two *traversal-order pairs* over two *token orderings*:

    rows:  row-major  (h, w) -> t        forward + backward
    cols:  col-major  (w, h) -> t        forward + backward

and `arm_scan.mamba3_scan_pair` already supplies both directions of a pair. So
this module is layout only — **no new kernel code** — which is exactly what
`ss2d.py`'s helpers were designed to allow.

WHY A SEPARATE MODULE RATHER THAN WIDENING `ss2d_scan`'s SEAM
-------------------------------------------------------------
`ss2d_scan`'s `scan_pair` seam speaks Mamba-1's parameter list — five
`(b, d, l)` tensors plus scalars. Mamba-3 hands the scan eleven tensors across
**two different layout families** (below), so a genuinely shared seam would be a
union type that every caller has to disambiguate. The repo's standing rule
applies (CLAUDE.md): the two kernels stay separate and share threading,
packaging and tests, not entry points. The *layout helpers* are shared, and
those are the part with the actual logic.

TWO LAYOUT FAMILIES, WHICH IS THE ONLY REAL COMPLICATION HERE
--------------------------------------------------------------
Mamba-3's 1D tensors are not uniformly laid out, so their grid forms are not
either, and the two need different view builders:

    time-major   q, k     (b, H, W, g, dqk)      -> (2b, H*W, g, dqk)
                 v, z     (b, H, W, h, dv)       -> (2b, H*W, h, dv)
                 angles   (b, H, W, h, r)        -> (2b, H*W, h, r)
    head-major   adt, dt, trap  (b, h, H, W)     -> (2b, h, H*W)

`q_bias`, `k_bias` and `D` are parameters, not time-varying, and pass through.

THE ANGLE PRE-PASS MUST RUN ON THE VIEWS, NOT THE GRID
-------------------------------------------------------
`theta = cumsum(tanh(angle) * PI * dt)` accumulates **along the traversal
order**, and row-major and column-major are different orders — so the two
orderings have genuinely different `theta`. Building the views first and letting
`mamba3_scan_pair` run the pre-pass on them is what keeps that right: the two
orderings sit on the batch axis, and `cumsum` treats batch rows independently.
Running the pre-pass on the grid and then reordering would give both orderings
the row-major angles, which is wrong and would not raise.

Within a pair, sharing `cos`/`sin` between forward and backward IS correct:
`reverse=True` walks the recurrence backward while each token keeps its own
position's `theta`. `tests/check_mamba3_op.py` pins that equivalence.

Inference-only, like `ss2d.py`: the kernel registers no autograd.
"""

import torch

from .ss2d import _MERGES, grid_to_views


def grid_to_views_time_major(t):
    """`(b, H, W, *rest)` -> `(2b, H*W, *rest)`: row-major stacked over col-major.

    The time-major counterpart of `ss2d.grid_to_views`, which handles the
    `(b, c, H, W)` (head-major) family. Both put the two orderings on the batch
    axis so one kernel call covers both.
    """
    if t.dim() < 3:
        raise ValueError(f"expected (b, H, W, ...), got {tuple(t.shape)}")
    b, h, w = t.shape[:3]
    rest = t.shape[3:]
    rows = t.reshape(b, h * w, *rest)
    cols = t.transpose(1, 2).reshape(b, w * h, *rest)
    return torch.cat((rows, cols), dim=0)


def views_to_grid_time_major(row_seq, col_seq, h, w):
    """Inverse of `grid_to_views_time_major`'s split halves."""
    b = row_seq.shape[0]
    rest = row_seq.shape[2:]
    rows = row_seq.reshape(b, h, w, *rest)
    cols = col_seq.reshape(b, w, h, *rest).transpose(1, 2)
    return rows, cols


def scan_pair_mamba3(q, k, v, adt, dt, trap, q_bias, k_bias, angles=None,
                     D=None, z=None):
    """The default two-direction primitive: `arm_scan.mamba3_scan_pair`.

    Split out so `ss2d_scan_mamba3`'s `scan_pair` argument has a named default
    and so a reference implementation can be substituted with the identical
    dataflow — which is what the parity gate exercises.
    """
    from .mamba3 import mamba3_scan_pair
    return mamba3_scan_pair(q, k, v, adt, dt, trap, q_bias, k_bias,
                            angles=angles, D=D, z=z)


def _check_grid(name, t, b, h, w, dim, head_major=False):
    if t.dim() != dim:
        raise ValueError(
            f"{name} must be {dim}-D "
            f"{'(b, heads, H, W)' if head_major else '(b, H, W, ...)'}, "
            f"got {tuple(t.shape)}")
    got = (t.shape[0], t.shape[2], t.shape[3]) if head_major else tuple(
        t.shape[:3])
    want = (b, h, w)
    if got != want:
        raise ValueError(
            f"{name} grid axes {got} do not match (b, H, W) = {want}; "
            f"full shape {tuple(t.shape)}")


def ss2d_scan_mamba3(q, k, v, adt, dt, trap, q_bias, k_bias, angles,
                     D=None, z=None, merge="sum", scan_pair=None):
    """4-direction 2D cross-scan over a token grid, on the Mamba-3 recurrence.

    Layouts — everything time-varying is grid-shaped:
        q, k     : (b, H, W, g, dqk)    g = 1 (SISO)
        v, z     : (b, H, W, h, dv)
        adt, dt  : (b, h, H, W)         adt = A*dt (<= 0); dt post-softplus
        trap     : (b, h, H, W)         PRE-sigmoid
        angles   : (b, H, W, h, r)      raw; the pre-pass runs on the views
        q_bias,
        k_bias   : (h, dqk)             parameters
        D        : (h,)

    merge:
        "sum"  row_fwd + row_bwd + col_fwd + col_bwd   (VMamba's combine)
        "mean" the same, divided by 4
        "none" the four direction planes unmerged, each `(b, H, W, h, dv)`, in
               order (row_fwd, row_bwd, col_fwd, col_bwd) — for a learned
               combine, and the form the goldens are checked against, so a
               kernel bug stays isolated from a merge-strategy bug.

    `scan_pair` is the two-direction primitive; defaults to the kernel. Passing
    a reference implementation runs the identical dataflow without the kernel.

    NOTE ON `D` AND `z`: both are applied inside every direction, so a "sum"
    merge carries four copies of the skip and gates each direction separately.
    That mirrors `ss2d_scan`'s Mamba-1 behaviour and is what a four-separate-
    scans formulation does. Pass `D=None` and add the skip once yourself if you
    want it counted once.
    """
    if merge not in _MERGES:
        raise ValueError(f"merge must be one of {_MERGES}, got {merge!r}")
    if q.dim() != 5:
        raise ValueError(
            f"q must be (b, H, W, groups, dqk), got {tuple(q.shape)}")

    b, h, w = q.shape[:3]
    _check_grid("q", q, b, h, w, 5)
    _check_grid("k", k, b, h, w, 5)
    _check_grid("v", v, b, h, w, 5)
    _check_grid("angles", angles, b, h, w, 5)
    for name, t in (("adt", adt), ("dt", dt), ("trap", trap)):
        _check_grid(name, t, b, h, w, 4, head_major=True)
    if z is not None:
        _check_grid("z", z, b, h, w, 5)
    if scan_pair is None:
        scan_pair = scan_pair_mamba3

    # Views first, THEN the scan — so the angle pre-pass accumulates along each
    # traversal order rather than along row-major twice. See the module
    # docstring; this ordering is the one correctness trap in the file.
    fwd, bwd = scan_pair(
        grid_to_views_time_major(q), grid_to_views_time_major(k),
        grid_to_views_time_major(v),
        grid_to_views(adt), grid_to_views(dt), grid_to_views(trap),
        q_bias, k_bias,
        angles=grid_to_views_time_major(angles), D=D,
        z=None if z is None else grid_to_views_time_major(z),
    )

    row_f, col_f = views_to_grid_time_major(fwd[:b], fwd[b:], h, w)
    row_b, col_b = views_to_grid_time_major(bwd[:b], bwd[b:], h, w)

    if merge == "none":
        return row_f, row_b, col_f, col_b
    out = row_f + row_b + col_f + col_b
    return out if merge == "sum" else out * 0.25
