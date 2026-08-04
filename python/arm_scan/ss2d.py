"""SS2D — the VMamba-style 4-direction 2D cross-scan, on CPU.

TOPOLOGY_IMPLEMENTATION_PLAN.md §3.1 (the `(B, D, H, W)` contract) executed on
top of the **fused bidirectional kernel** rather than four forward scans.

WHY TWO KERNEL CALLS AND NOT FOUR
---------------------------------
The four cross-scan directions are two *traversal-order pairs* over two
*distinct token orderings*:

    rows:  row-major  (h, w) -> t        forward + backward
    cols:  col-major  (w, h) -> t        forward + backward

Within a pair, both directions consume **the same** (u, delta, B, C) tensors and
differ only in the order the recurrence walks them. That is exactly the
precondition for `selective_scan_bidirectional`, which computes the
direction-independent Pass A — discretize + `exp`, ~85% of kernel time — **once**
and emits both directions from it. Measured at **1.58-1.75x (geomean ~1.67x)**
across four CI runs; see BIDIRECTIONAL_LOG.md Step 7.

So SS2D costs two Pass A evaluations, not four. The earlier path stacked
`[rows, rows.flip, cols, cols.flip]` into one 4B forward call, which recomputed
Pass A four times and paid for four `torch.flip` copies of a
`(1, 96, 122880)` tensor per block per denoiser call.

PROJECTIONS ARE COMPUTED ON THE GRID, NOT PER DIRECTION
-------------------------------------------------------
`delta`/`B`/`C` come from token-wise projections of the feature map, so
projecting the grid and then reordering is identical to reordering and then
projecting. This module therefore takes the time-varying tensors in **grid
form** `(b, ., h, w)` and builds the traversal views itself — one projection
pass in the caller instead of four.

WHAT THIS IS NOT
----------------
Not a fused `selective_scan_2d` (a single Rust entry point owning all four
directions with an in-kernel tile transpose). That remains the identified next
lever — see TOPOLOGY_IMPLEMENTATION_PLAN.md §3.2 and the measured
flip/permute overhead in `bench/results/ss2d_*.json`. This module is the
already-verified-kernel path that captures the larger share of that overhead
today.

Inference-only: the kernel registers no autograd. Train through the reference
seam (`torch_scan.selective_scan_pair_torch`) and swap `scan_pair_arm` in for
inference — `use_arm_scan(module)` does that for every SS2D block in a model.
"""

import torch

_MERGES = ("sum", "mean", "none")


def grid_to_views(t):
    """`(b, c, h, w)` -> `(2b, c, h*w)`: row-major stacked over column-major.

    The two orderings go on the BATCH axis so a single kernel call covers both,
    doubling the rayon row count for free. Column-major is the transposed grid
    flattened, i.e. the scan axis walks down columns.
    """
    b, c, h, w = t.shape
    rows = t.reshape(b, c, h * w)
    cols = t.transpose(2, 3).reshape(b, c, w * h)
    return torch.cat((rows, cols), dim=0)


def views_to_grid(row_seq, col_seq, h, w):
    """Inverse of the `grid_to_views` split halves, back to `(b, c, h, w)`."""
    b, c, _ = row_seq.shape
    rows = row_seq.reshape(b, c, h, w)
    cols = col_seq.reshape(b, c, w, h).transpose(2, 3)
    return rows, cols


def scan_pair_arm(u, delta, A, B, C, D=None, delta_bias=None,
                  delta_softplus=True):
    """`(fwd, bwd)` from ONE fused kernel call that shares Pass A.

    Signature matches `arm_scan.selective_scan` (1D, `(b, d, l)` tensors); the
    only difference is that two outputs come back instead of one. This is the
    seam `ss2d_scan` calls and the one `use_arm_scan` swaps.
    """
    from .bidirectional import bidirectional_scan
    return bidirectional_scan(
        u, delta, A, B, C, D=D, delta_bias=delta_bias,
        delta_softplus=delta_softplus, merge="none",
    )


def scan_fn_arm(u, delta, A, B, C, D=None, delta_bias=None,
                delta_softplus=True):
    """Single-direction 1D scan through the kernel.

    Retained because `SS2DBlock._forward_legacy` — the four-separate-directions
    formulation that `forward` is gated against — is defined in terms of it.
    New code should use `ss2d_scan`.
    """
    from .op import selective_scan
    return selective_scan(u, delta, A, B, C, D=D, delta_bias=delta_bias,
                          delta_softplus=delta_softplus)


def ss2d_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True,
              merge="sum", scan_pair=None):
    """4-direction 2D cross-scan over a token grid.

    Layouts — everything time-varying is grid-shaped:
        u, delta : (b, d, h, w)
        A        : (d, n)
        B, C     : (b, n, h, w)  or grouped (b, groups, n, h, w)
        D, delta_bias : (d,)

    merge:
        "sum"  row_fwd + row_bwd + col_fwd + col_bwd   (VMamba's combine)
        "mean" the same, divided by 4
        "none" the four direction planes unmerged, each `(b, d, h, w)`, in
               order (row_fwd, row_bwd, col_fwd, col_bwd) — for a learned
               combine, and the form the 2D goldens are checked against
               (per-direction, before merge, so kernel bugs stay isolated from
               merge-strategy bugs).

    `scan_pair` is the two-direction scan primitive; defaults to the fused
    kernel. Passing a reference implementation runs the identical dataflow
    without the kernel — that substitution is what the parity gate exercises.

    NOTE ON `D`: the skip is applied inside every direction, so a "sum" merge
    carries `4*D*u`. That is what the four-separate-scans formulation does and
    what trained SS2D weights expect. Pass `D=None` and add the skip yourself
    if you want it counted once.
    """
    if merge not in _MERGES:
        raise ValueError(f"merge must be one of {_MERGES}, got {merge!r}")
    if u.dim() != 4:
        raise ValueError(f"u must be (b, d, h, w), got {tuple(u.shape)}")
    if delta.shape != u.shape:
        raise ValueError(
            f"delta {tuple(delta.shape)} must match u {tuple(u.shape)}")
    if scan_pair is None:
        scan_pair = scan_pair_arm

    b, _, h, w = u.shape
    for name, t in (("B", B), ("C", C)):
        if t.shape[0] != b or tuple(t.shape[-2:]) != (h, w):
            raise ValueError(
                f"{name} must be (b={b}, ..., {h}, {w}), got {tuple(t.shape)}")

    # Grouped B/C arrive as (b, g, n, h, w): fold the group axis into channels
    # for the view build, then restore it — the 1D kernel's own contract
    # already accepts (batch, groups, state, len).
    def _views(t):
        if t.dim() == 5:
            bb, g, n, hh, ww = t.shape
            v = grid_to_views(t.reshape(bb, g * n, hh, ww))
            return v.reshape(2 * bb, g, n, hh * ww)
        return grid_to_views(t)

    fwd, bwd = scan_pair(
        grid_to_views(u), grid_to_views(delta), A, _views(B), _views(C),
        D=D, delta_bias=delta_bias, delta_softplus=delta_softplus,
    )

    row_f, col_f = views_to_grid(fwd[:b], fwd[b:], h, w)
    row_b, col_b = views_to_grid(bwd[:b], bwd[b:], h, w)

    if merge == "none":
        return row_f, row_b, col_f, col_b
    out = row_f + row_b + col_f + col_b
    return out if merge == "sum" else out * 0.25


def use_arm_scan(module, enable=True):
    """Swap the scan implementation on every SS2D block in `module`.

    Blocks expose two seams; both switch together so the legacy
    single-direction path (kept as the parity oracle) and the pair path stay on
    the same backend:

      `scan_pair_fn` -> `scan_pair_arm`           (what `forward` uses)
      `scan_fn`      -> `scan_fn_arm`             (what `_forward_legacy` uses)

    Returns the number of blocks switched.
    """
    n = 0
    for m in module.modules():
        touched = False
        if hasattr(m, "scan_pair_fn"):
            if enable:
                m.scan_pair_fn = scan_pair_arm
            else:
                from apps.mri_diffusion.backbone.torch_scan import \
                    selective_scan_pair_torch
                m.scan_pair_fn = selective_scan_pair_torch
            touched = True
        if hasattr(m, "scan_fn"):
            if enable:
                m.scan_fn = scan_fn_arm
            else:
                from apps.mri_diffusion.backbone.torch_scan import \
                    selective_scan_torch
                m.scan_fn = selective_scan_torch
            touched = True
        n += int(touched)
    return n
