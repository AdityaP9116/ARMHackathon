"""PyTorch front-end for the Mamba-3 SISO scan.

Registered as `arm_scan::mamba3_scan` with a fake (meta) kernel so it composes
with `torch.compile` instead of graph-breaking — which matters because
`torch.compile` IS this project's fair baseline, so an op that forces a graph
break would quietly flatter every number measured against it.

Inference-only: no autograd formula. CPU float32.

Two contracts worth reading before calling this
-----------------------------------------------
**`trap` is pre-sigmoid.** The kernel applies the sigmoid, matching upstream,
so a caller cannot accidentally apply it twice.

**Angles come in raw; this module runs the pre-pass.** Upstream splits the
rotation across two kernels — `angle_dt_fwd` accumulates
`theta = cumsum(tanh(angle) * PI * dt)`, then `mamba3_siso_fwd` consumes the
result — and we mirror that split, so our goldens stay directly comparable to
the captured ground truth. `angles_to_cos_sin` is that pre-pass.

The `tanh(.) * PI` squashing is the part that appears in no published
description of Mamba-3. It was recovered by diffing the official kernel's
internal buffers; omitting it leaves ~7% relative error, which is close enough
to read as a rounding problem and send you hunting the wrong thing.
"""

import math
from typing import Optional, Tuple

import torch

from . import _ffi
from .op import kernel_calls as _kernel_calls  # noqa: F401  (re-export)
from .op import _CALLS


def _c(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().float()


def angles_to_cos_sin(angles: torch.Tensor, dt: torch.Tensor,
                      half: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """The RoPE angle pre-pass: `theta = cumsum(tanh(angle) * PI * dt)`.

    angles : (b, l, h, r)   raw, straight off the model's projection
    dt     : (b, h, l)      post-softplus
    half   : dqk // 2       the number of rotation PAIRS the kernel expects
    returns  (cos, sin), each (b, l, h, half)

    `r` is normally SMALLER than `half`: at the published `rope_fraction=0.5`
    only the first half of the head dimension rotates, so `r == dqk // 4`. The
    remaining pairs get angle 0 — the identity rotation — which is why the tail
    is padded with cos=1, sin=0 rather than with zeros in both.

    Getting that padding wrong does not raise: the kernel is handed a raw
    pointer and told how many elements to read, so a short `cos` is an
    out-of-bounds read that surfaces as NaN much later. `mamba3_scan` therefore
    checks the shapes before the call.

    Kept in Python rather than fused into the kernel for the same reason
    upstream keeps it in a separate kernel: it is pointwise and cheap next to
    the scan, and fusing it would put a new NEON transcendental on the critical
    path for no measured gain. Revisit only if a profile says otherwise.
    """
    if angles.dim() != 4:
        raise ValueError(
            f"angles must be (b, l, h, r), got {tuple(angles.shape)}")
    r = angles.shape[-1]
    if r > half:
        raise ValueError(
            f"angles has {r} rotation pairs but the head dim allows {half}")
    # Compute at the caller's precision, floored at fp32. This used to force
    # `.float()` unconditionally, which is right for the kernel path (the FFI
    # boundary downcasts anyway) but wrong for anyone doing f64 analysis with
    # this helper: it silently capped an f64 pipeline at fp32, and the
    # non-causal dense form then could not be compared against the reference at
    # better than ~1e-8. fp32 callers are unaffected — `promote_types` returns
    # float32 and the casts below are no-ops.
    dtype = torch.promote_types(angles.dtype, torch.float32)
    # dt is (b, h, l) -> (b, l, h, 1) to broadcast over the rotation pairs.
    dt_blh = dt.permute(0, 2, 1).unsqueeze(-1).to(dtype)
    theta = torch.cumsum(
        torch.tanh(angles.to(dtype)) * math.pi * dt_blh, dim=1)
    cos, sin = torch.cos(theta), torch.sin(theta)
    if r < half:
        pad = (*theta.shape[:-1], half - r)
        cos = torch.cat([cos, torch.ones(pad, dtype=cos.dtype)], dim=-1)
        sin = torch.cat([sin, torch.zeros(pad, dtype=sin.dtype)], dim=-1)
    return cos, sin


@torch.library.custom_op("arm_scan::mamba3_scan", mutates_args=())
def _mamba3_scan_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    adt: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    d_skip: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    reverse: bool,
    psi: Optional[torch.Tensor] = None,
    zeta: Optional[torch.Tensor] = None,
    phi: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # SISO passes q as (b, l, 1, dqk) -- axis 2 is the B/C GROUP axis, always 1.
    # MIMO passes (b, l, rank, dqk) -- axis 2 is the RANK. Same position, and
    # the kernel reads it the same way, but they mean different things, so the
    # distinction is carried by whether the projections are present.
    b, length, rank, dqk = q.shape
    h, dv = v.shape[2], v.shape[3]

    dims = _ffi.ArmMamba3Dims(b, h, dv, dqk, length, rank)
    # The kernel writes head-major (b, h, l, dv); the model's world is
    # time-major. Permuting here keeps the strided-mutable-slice problem inside
    # the kernel, where it is a safety question, instead of leaking into the
    # public contract. See arm_scan_core::mamba3 for the full reasoning.
    out_hm = torch.empty((b, h, length, dv), dtype=torch.float32)

    def ptr(t):
        return 0 if t is None else t.data_ptr()

    _ffi.mamba3_raw(
        dims,
        [q.data_ptr(), k.data_ptr(), v.data_ptr(), adt.data_ptr(),
         dt.data_ptr(), trap.data_ptr(), q_bias.data_ptr(), k_bias.data_ptr(),
         cos.data_ptr(), sin.data_ptr(), ptr(d_skip), ptr(z),
         ptr(psi), ptr(zeta), ptr(phi)],
        1 if reverse else 0,
        _ffi.BACKENDS["auto"],
        _ffi.THREADING["auto"],
        out_hm.data_ptr(),
        0,
        0,
    )
    _CALLS["n"] += 1
    return out_hm.permute(0, 2, 1, 3).contiguous()


@_mamba3_scan_op.register_fake
def _(q, k, v, adt, dt, trap, q_bias, k_bias, cos, sin, d_skip, z, reverse,
      psi=None, zeta=None, phi=None):
    b, length = q.shape[0], q.shape[1]
    h, dv = v.shape[2], v.shape[3]
    return q.new_empty((b, length, h, dv))


def _check_shapes(q, k, v, adt, dt, trap, q_bias, k_bias, cos, sin, D, z):
    """Validate every tensor against the layout contract before the FFI call.

    This is not belt-and-braces — it is the only real check there is. The C
    entry point receives raw pointers and is *told* how many elements to read
    (`from_raw_parts(ptr, expected)`), so the Rust-side length validation is
    vacuous for FFI callers: a short buffer is an out-of-bounds read, not a
    rejection. It surfaces as NaN somewhere downstream, which is exactly how a
    missing RoPE pad was found here.

    So the Python layer owns shape correctness, and says so loudly.
    """
    b, length, groups, dqk = q.shape
    if groups != 1:
        raise ValueError(
            f"SISO expects one B/C group, got {groups}. Multi-group and MIMO "
            "are out of scope — see MAMBA3_IMPLEMENTATION_PLAN.md")
    if dqk % 2:
        raise ValueError(f"dqk must be even (RoPE lane pairs), got {dqk}")
    if v.dim() != 4:
        raise ValueError(f"v must be (b, l, h, dv), got {tuple(v.shape)}")
    h, dv = v.shape[2], v.shape[3]
    half = dqk // 2
    want = {
        "q": (b, length, 1, dqk), "k": (b, length, 1, dqk),
        "v": (b, length, h, dv),
        "adt": (b, h, length), "dt": (b, h, length), "trap": (b, h, length),
        "q_bias": (h, dqk), "k_bias": (h, dqk),
        "cos": (b, length, h, half), "sin": (b, length, h, half),
    }
    got = dict(q=q, k=k, v=v, adt=adt, dt=dt, trap=trap, q_bias=q_bias,
               k_bias=k_bias, cos=cos, sin=sin)
    if D is not None:
        want["D"], got["D"] = (h,), D
    if z is not None:
        want["z"], got["z"] = (b, length, h, dv), z
    for name, shape in want.items():
        actual = tuple(got[name].shape)
        if actual != shape:
            raise ValueError(
                f"{name} must be {shape}, got {actual}. Passing a short buffer "
                "to the kernel is an out-of-bounds read, not an error, so this "
                "is checked here.")


def mamba3_scan(q, k, v, adt, dt, trap, q_bias, k_bias, angles=None,
                D=None, z=None, reverse=False, cos=None, sin=None):
    """Mamba-3 SISO selective scan.

    q, k    : (b, l, 1, dqk)   or (b, l, dqk) — the SISO group axis is optional
    v, z    : (b, l, h, dv)
    adt, dt : (b, h, l)        adt = A*dt (<=0); dt post-softplus
    trap    : (b, h, l)        PRE-sigmoid
    angles  : (b, l, h, r)     raw; the pre-pass runs here
    q_bias, k_bias : (h, dqk)
    D       : (h,)
    returns : (b, l, h, dv)

    Pass `cos`/`sin` directly instead of `angles` when they are already
    computed — the traversal-pair path does this so the pre-pass runs once for
    both directions rather than twice.
    """
    if q.dim() == 3:
        q = q.unsqueeze(2)
    if k.dim() == 3:
        k = k.unsqueeze(2)
    if (cos is None) != (sin is None):
        raise ValueError("pass cos and sin together, or neither")
    half = q.shape[-1] // 2
    if cos is None:
        if angles is None:
            raise ValueError("pass either `angles` or both `cos` and `sin`")
        cos, sin = angles_to_cos_sin(angles, dt, half)
    _check_shapes(q, k, v, adt, dt, trap, q_bias, k_bias, cos, sin, D, z)
    return _mamba3_scan_op(
        _c(q), _c(k), _c(v), _c(adt), _c(dt), _c(trap), _c(q_bias), _c(k_bias),
        _c(cos), _c(sin),
        None if D is None else _c(D), None if z is None else _c(z),
        bool(reverse),
    )


def mamba3_scan_pair(q, k, v, adt, dt, trap, q_bias, k_bias, angles=None,
                     D=None, z=None, cos=None, sin=None):
    """Both traversal directions from one set of inputs -> `(fwd, bwd)`.

    This is the seam the bidirectional and 2D cross-scan topologies plug into:
    both are traversal orders over the same primitive, so neither needs any
    Mamba-3-specific orchestration of its own.

    Two forward calls today rather than one fused pass. Mamba-1 fuses its pair
    because Pass A (discretize + exp) is ~85% of its runtime and is
    direction-independent, so computing it once is most of the win. **Mamba-3's
    arithmetic is not distributed that way** — Pass B's state update dominates
    at ~3*dv*dqk per timestep against O(dqk) of shared per-step work, so the
    same trick would buy a few percent. Fuse only if a profile disagrees; do
    not assume the Mamba-1 result transfers.
    """
    if cos is None:
        if angles is None:
            raise ValueError("pass either `angles` or both `cos` and `sin`")
        cos, sin = angles_to_cos_sin(angles, dt, q.shape[-1] // 2)
    common = dict(q_bias=q_bias, k_bias=k_bias, D=D, z=z, cos=cos, sin=sin)
    fwd = mamba3_scan(q, k, v, adt, dt, trap, reverse=False, **common)
    bwd = mamba3_scan(q, k, v, adt, dt, trap, reverse=True, **common)
    return fwd, bwd


def mamba3_mimo_scan(q, k, v, adt, dt, trap, q_bias, k_bias, psi, zeta, phi,
                     angles=None, D=None, z=None, reverse=False,
                     cos=None, sin=None):
    """Mamba-3 **MIMO** (rank-r) selective scan.

    q, k    : (b, l, r, 1, dqk)  or (b, l, r, dqk) — the group axis is optional
    v, z    : (b, l, h, dv)
    adt, dt : (b, h, l)          adt = A*dt (<=0); dt post-softplus
    trap    : (b, h, l)          PRE-sigmoid
    q_bias,
    k_bias  : (h, r, dqk)        per-head AND per-rank
    psi,
    zeta,
    phi     : (h, r, dv)         input / gate / output projections
    angles  : (b, l, h, a)       raw; the pre-pass runs here
    D       : (h,)
    returns : (b, l, h, dv)

    **Not interchangeable with `mamba3_scan` at r=1.** The two families rotate
    different lane pairs — SISO interleaved `(2i, 2i+1)`, MIMO split-halves
    `(i, i + dqk/2)` — so a rank-1 MIMO call is *not* a SISO call. They agree
    only when the rotation is the identity; `check_rank1_collapse` measures
    exactly that (7.7e-16 with angles zeroed, 3.8e-01 without).

    The angle pre-pass IS shared: upstream calls the same `angle_dt_fwd` for
    both, so `angles_to_cos_sin` applies unchanged.
    """
    if q.dim() == 5:
        if q.shape[3] != 1:
            raise ValueError(
                f"MIMO expects one B/C group, got {q.shape[3]}. Multi-group is "
                "out of scope — see MAMBA3_IMPLEMENTATION_PLAN.md")
        q = q[:, :, :, 0, :]
    if k.dim() == 5:
        k = k[:, :, :, 0, :]
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError(
            f"q/k must be (b, l, r, dqk) after squeezing the group axis; got "
            f"{tuple(q.shape)} and {tuple(k.shape)}")

    rank, dqk = q.shape[2], q.shape[3]
    h, dv = v.shape[2], v.shape[3]
    if (cos is None) != (sin is None):
        raise ValueError("pass cos and sin together, or neither")
    if cos is None:
        if angles is None:
            raise ValueError("pass either `angles` or both `cos` and `sin`")
        cos, sin = angles_to_cos_sin(angles, dt, dqk // 2)

    # The same argument as `_check_shapes`: the C entry point is told how many
    # elements to read, so a short buffer is an out-of-bounds read rather than
    # an error. Shape correctness is owned here.
    for name, t, want in (
        ("q", q, (q.shape[0], q.shape[1], rank, dqk)),
        ("k", k, (q.shape[0], q.shape[1], rank, dqk)),
        ("q_bias", q_bias, (h, rank, dqk)),
        ("k_bias", k_bias, (h, rank, dqk)),
        ("psi", psi, (h, rank, dv)),
        ("zeta", zeta, (h, rank, dv)),
        ("phi", phi, (h, rank, dv)),
    ):
        if tuple(t.shape) != want:
            raise ValueError(
                f"{name} must be {want}, got {tuple(t.shape)}")
    if z is not None and tuple(z.shape) != tuple(v.shape):
        raise ValueError(
            f"z must match v {tuple(v.shape)}, got {tuple(z.shape)}")

    return _mamba3_scan_op(
        _c(q), _c(k), _c(v), _c(adt), _c(dt), _c(trap), _c(q_bias), _c(k_bias),
        _c(cos), _c(sin),
        None if D is None else _c(D), None if z is None else _c(z),
        bool(reverse), _c(psi), _c(zeta), _c(phi),
    )
