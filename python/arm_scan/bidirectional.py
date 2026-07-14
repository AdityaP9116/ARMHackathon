"""Bidirectional selective scan (TOPOLOGY_IMPLEMENTATION_PLAN.md §2).

Runs the recurrence over the sequence in both time directions and merges the
two outputs. The backward direction is **fused in the kernel** — it walks the
sequence backward in place via `selective_scan(..., reverse=True)` rather than
materializing flipped copies of u/delta/B/C/z and un-flipping the result.

Honest framing of what that fusion is worth: it removes six full-tensor copies,
which `bench/bench_bidirectional.py` measured at **~2%** of runtime (falling as
the sequence lengthens). It is not a speedup story. It was built because the 2D
cross-scan needs a backward traversal anyway — its column-backward and
row-backward directions are this same primitive — so `reverse` is the substrate
for SS2D, not a bidirectional optimization. See BIDIRECTIONAL_LOG.md.

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
mean), gating inside each direction with `z` is algebraically identical to
gating once after the merge:

    fwd: y_f[t]·silu(z[t]);  bwd: y_b[t]·silu(z[t])
    sum: (y_f[t] + y_b[t])·silu(z[t])

so we let the kernel apply the gate in both passes and do not special-case it.
(The reverse scan reads `z` at index t like everything else — `reverse` changes
traversal order, never layout — so no flipping is involved on either side.)
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

import torch

from .op import selective_scan

_MERGES = ("sum", "mean", "concat", "none")


def _scan_reverse(u, delta, A, B, C, D, z, delta_bias, delta_softplus,
                  return_last_state):
    """Scan the sequence backward in time — fused, no copies.

    THE SEAM (now closed). This used to flip the five time-varying inputs, run
    the forward kernel, and flip the output back — correct, but six full-tensor
    copies per call. The kernel now walks the sequence backward in place, so the
    whole thing is one call and the copies are gone. Nothing above this function
    changed.

    Kept as a named function rather than inlined because it is the documented
    seam, and because the flip-based definition it replaces is still the
    specification: `reverse=True` is *defined* as flip-forward-flip, enforced
    bit-for-bit by `reverse_matches_flip_forward_flip` in the Rust property
    tests and by `tests/check_bidirectional_math.py` in numpy.

    `last_state` here is the state after consuming t=0 (the last step a
    backward scan sees) — i.e. the state at the START of the sequence. It is
    not a resumable cache the way the forward scan's last_state is;
    bidirectional models are non-causal and generally ignore it.
    """
    return selective_scan(
        u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias,
        delta_softplus=delta_softplus, return_last_state=return_last_state,
        reverse=True,
    )


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
      backward pass. Supply time-varying overrides (delta/B/C) in ordinary
      forward-time order, exactly as you would for the forward pass — the kernel
      reverses the traversal, not the layout, so nothing is ever pre-flipped.
      Default (None) is the weight-tied case: the backward pass reuses
      A/D/delta_bias and the same forward tensors.

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
