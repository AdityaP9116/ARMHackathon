"""Bidirectional selective scan — the correctness path (TOPOLOGY_IMPLEMENTATION_PLAN.md §2.1).

Runs the recurrence over the sequence in both time directions and merges the
two outputs. Built entirely on the existing 1D op: no new Rust, no new FFI.
The fused version (a `reverse` flag in the kernel, §2.2 of the plan) replaces
only this module's internals — `_scan_reverse` below is the single seam.

WHICH BIDIRECTIONAL MODELS THIS IS FOR
--------------------------------------
Two patterns exist in the wild and they are not interchangeable:

  "outer" (Caduceus's BiMambaWrapper, Vim): the WHOLE mixer — causal conv,
      x_proj, dt_proj — is re-run on `x.flip(time)`. A causal conv over
      flipped input is not the flip of the conv over input, so the two
      directions' scan inputs are genuinely different tensors. Such a model
      does not need this module: it already calls `selective_scan` twice, and
      both calls are ordinary FORWARD scans. A kernel `reverse` flag buys it
      nothing.

  "inner" (VMamba/SS2D-style cross-scan, and bidirectional variants that flip
      after the projections): the SAME projected tensors are traversed in both
      time directions. Flipping commutes with the time-pointwise projections,
      so the backward direction's (u, delta, B, C) are exactly the flips of the
      forward direction's. THIS is the pattern this module implements, and the
      one a fused `reverse` flag actually accelerates — it is also the 1D case
      of the SS2D cross-scan.

Check which one your checkpoint is before wiring this in. Getting it wrong
produces plausible-looking output that is quietly wrong.

MERGE
-----
`merge="sum"` matches the common case. Note that for any LINEAR merge (sum,
mean), gating inside each direction with the (correspondingly flipped) `z` is
algebraically identical to gating once after the merge:

    fwd: y_f[t]·silu(z[t]);  bwd (after un-flipping): y_b[t]·silu(z[t])
    sum: (y_f[t] + y_b[t])·silu(z[t])

so we let the kernel apply the gate in both passes and do not special-case it.
For anything non-linear (a learned gated combine), pass `merge="none"` and do
the combination yourself on the two returned tensors — the primitive should not
guess at a model-specific merge.

GOTCHA: with `merge="sum"` and a shared `D`, the skip connection is applied in
BOTH directions, so the merged output carries 2·D·u, not D·u. That is what real
bidirectional Mambas do (each direction's mixer applies its own D, then the
outputs are summed), but it surprises people — so it is pinned by an assertion
in `tests/check_bidirectional_math.py`. If your model wants D counted once,
pass `D=None` here and add the skip yourself after merging.
"""

from typing import Optional

import torch

from .op import selective_scan

_MERGES = ("sum", "mean", "concat", "none")


def _flip_time(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Reverse the time axis, which is last for every time-varying tensor in
    the layout contract (u/delta/z: (B,D,L); B/C: (B,[G,]N,L))."""
    return None if t is None else torch.flip(t, dims=(-1,))


def _scan_reverse(u, delta, A, B, C, D, z, delta_bias, delta_softplus,
                  return_last_state):
    """Scan the sequence backward in time.

    THE SEAM: today this flips the time-varying inputs, runs the forward
    kernel, and flips the output back — correct, but it pays for four
    full-tensor copies in and one out. Once the kernel grows the `reverse`
    flag (plan §2.2), the body becomes a single
    `selective_scan(..., reverse=True)` with no flips, and every caller of this
    module gets the win for free.

    `last_state` here is the state after consuming t=0 (the last step a
    backward scan sees) — i.e. the state at the START of the sequence. It is
    not a resumable cache the way the forward scan's last_state is;
    bidirectional models are non-causal and generally ignore it.
    """
    out = selective_scan(
        _flip_time(u), _flip_time(delta), A, _flip_time(B), _flip_time(C),
        D=D, z=_flip_time(z), delta_bias=delta_bias,
        delta_softplus=delta_softplus, return_last_state=return_last_state,
    )
    if return_last_state:
        out, last = out
        return _flip_time(out), last
    return _flip_time(out)


def bidirectional_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                       delta_softplus=False, merge="sum",
                       reverse_params=None, return_last_state=False):
    """Scan `u` forward and backward in time and merge the two outputs.

    Tensor layouts are exactly `arm_scan.selective_scan`'s:
      u, delta, z: (batch, dim, len);  A: (dim, state);
      B, C: (batch, state, len) or (batch, groups, state, len);
      D, delta_bias: (dim,).

    merge:
      "sum"    out_fwd + out_bwd                      (the common case)
      "mean"   (out_fwd + out_bwd) / 2
      "concat" cat along the channel axis -> (batch, 2*dim, len)
      "none"   return (out_fwd, out_bwd) unmerged, for a learned/gated combine

    reverse_params: for models whose backward direction has its OWN weights
      (Vim's `bimamba_type="v2"` has a separate A_b, D_b, dt_proj_b), pass a
      dict overriding any of "A", "D", "delta", "delta_bias", "B", "C" for the
      backward pass. Time-varying overrides (delta/B/C) are used as given and
      are NOT flipped — supply them already in forward-time order, exactly as
      you would for the forward pass. Default (None) is the weight-tied case:
      the backward pass reuses A/D/delta_bias and the flipped forward tensors.

    Returns out (batch, dim, len) — or (batch, 2*dim, len) for "concat", or a
    2-tuple for "none". With return_last_state=True, each output is paired with
    its last_state; see `_scan_reverse` on what the backward one means.
    """
    if merge not in _MERGES:
        raise ValueError(f"merge must be one of {_MERGES}, got {merge!r}")

    p = reverse_params or {}
    rev = _scan_reverse(
        u,
        p.get("delta", delta),
        p.get("A", A),
        p.get("B", B),
        p.get("C", C),
        p.get("D", D),
        z,
        p.get("delta_bias", delta_bias),
        delta_softplus,
        return_last_state,
    )
    fwd = selective_scan(
        u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias,
        delta_softplus=delta_softplus, return_last_state=return_last_state,
    )

    if return_last_state:
        (out_f, last_f), (out_b, last_b) = fwd, rev
    else:
        out_f, out_b = fwd, rev

    if merge == "sum":
        out = out_f + out_b
    elif merge == "mean":
        out = (out_f + out_b) * 0.5
    elif merge == "concat":
        out = torch.cat((out_f, out_b), dim=1)
    else:  # "none"
        out = (out_f, out_b)

    if not return_last_state:
        return out
    return (out, (last_f, last_b)) if merge != "none" else (
        (out_f, last_f), (out_b, last_b))
