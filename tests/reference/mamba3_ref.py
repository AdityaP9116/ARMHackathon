"""Mamba-3 SISO selective scan — the CPU reference implementation.

WHAT THIS IS
------------
The oracle for every Mamba-3 kernel we write, playing exactly the role
`selective_scan_ref.py` plays for Mamba-1 and `scalar.rs` plays inside the
crate: a slow, obvious, plain-PyTorch transcription that everything faster is
diffed against.

It exists because **upstream has no CPU path**. `mamba_ssm/modules/mamba3.py`
imports Triton/TileLang/CuTe kernels and asserts if they are missing, so there
was nothing to check a CPU implementation against. Stage 0 captured
(inputs -> outputs) pairs from the official GPU kernel; this file is the
formulation that has to reproduce them.

WHICH RECURRENCE — the question this file settles
-------------------------------------------------
The paper and the community re-implementation
(`rishikksh20/mamba3-pytorch`) disagree by max_abs 1.06 on O(1) states, and no
gate remapping reconciles them (`tools/check_mamba3_recurrence.py`). The
difference is structural: the paper carries the state decay on the PREVIOUS
input term, the community version does not.

Reading the official kernel settles it in favour of the paper. In
`mamba3_siso_fwd.py` phase 1:

    trap          = sigmoid(Trap_t)
    gamma_t       = dt_t * trap_t                        # weight of Bx_t at t
    shifted_gamma = dt_{t+1} * (1 - trap_{t+1})          # weight of Bx_t at t+1
    scale_t       = gamma_t + shifted_gamma_t

and the intra-chunk matrix is *strictly* lower triangular with `scale` on the
off-diagonal and `gamma` on the diagonal. That is only consistent with

    h_t = a_t*h_{t-1} + a_t*dt_t*(1-lam_t)*Bx_{t-1} + dt_t*lam_t*Bx_t

because the shifted term must share the `a_{t+1}` factor with `h_t` for the two
contributions of `Bx_t` to collapse into a single `scale_t` carried by the
decay. That is the paper's form, decay on the previous term included.

THE FORM IMPLEMENTED HERE
-------------------------
The official kernel is chunked (SSD/dual form: cumulative decays, Gram
products, per-chunk state carries). We deliberately implement the equivalent
**sequential** recurrence instead — it is far easier to read and to be
confident in, and being a slow independent formulation is precisely what makes
it a useful oracle. Per (batch, head), with S of shape (headdim_v, headdim_qk):

    S_t = a_t * S_{t-1} + scale_t * (v_t k_t^T)
    y_u = a_u * (q_u @ S_{u-1}^T) + (D + gamma_u * (q_u . k_u)) * v_u
    y_u = y_u * silu(z_u)

where q/k are bias-added then RoPE-rotated. Note `q_u . k_u` on the diagonal is
rotation-invariant (both are rotated by the same angle), which is why the
kernel can compute it pre-rotation and we can compute it post-rotation.

PRECISION
---------
Runs in f64 by default. The kernel is mixed precision and emits **bf16**, so
the honest comparison is `bf16(reference)` against the golden to ~1 ULP — NOT
`< 1e-4`, which bf16's ~0.4% relative epsilon can never satisfy. See
`verify_golden_mamba3.py`.

STATUS — structure CONFIRMED against the kernel's own buffers; one open gap
--------------------------------------------------------------------------
Verified by running the kernel with `store_states_adt_outv=True`, which makes
it dump its internal buffers, and diffing term by term (far more decisive than
comparing final outputs — see the `_rope` note for how output-only comparison
actively misled us):

* `Gamma_store`  vs `dt*sigmoid(Trap)`         -> **3.4e-8  EXACT**
* `Scale_store`  vs `gamma + shifted`          -> **3.4e-8  EXACT**
* `Q_store`      vs `rope(Q + Q_bias, Angles)` -> 3.5e-3 (bf16 level)
* `K_store`      vs `rope(K + K_bias)*scale`   -> 2.3e-3 (bf16 level)
* `QK_store`     vs `(q.k)*gamma`              -> 2.8e-4
* `DA_CS`        -> differs by exactly 1/log2(e); the kernel stores decay in
  log2 units and uses exp2. Mathematically identical, not an error.

**`Scale_store` matching exactly is the decisive result of Stage 1.** It is
direct evidence from the kernel's own memory that the trapezoid is
`scale = dt*lam + dt_next*(1 - lam_next)`, which is only consistent with the
PAPER's recurrence (decay carried on the previous term). The community
formulation is ruled out — not by argument, by measurement.

**GATE PASSES.** Worst case across all 10 goldens is **4.47 bf16 ULP** measured
at tensor scale (bound: 8). The residual is bf16 output quantisation, not
error: golden _04 is L=1 with no accumulation whatsoever and still sits at
0.45 ULP, which is the floor for comparing anything against a bf16-stored
result.

The last bug, and it was invisible from the scan kernel alone: the rotation
angle is accumulated by a **separate pre-pass kernel**, `angle_dt_fwd`, which
`mamba3_siso_combined` runs before `mamba3_siso_fwd`. So the scan kernel's
`Q_store` legitimately matches RAW angles (it receives them already
accumulated), while the captured `Angles` — recorded at the outer boundary —
are pre-accumulation. The pre-pass computes

    theta = cumsum(tanh(angle) * PI * dt)   (mod 2*pi)

and the `tanh(.) * PI` squashing is the part no amount of end-to-end sweeping
would have found: omitting it leaves ~7e-2 relative error, close enough to look
like a rounding problem and send you hunting the wrong thing. Applying it took
the suite from 1.6e-1 to bf16 level.

Method note worth keeping: diffing the kernel's INTERNAL buffers
(`store_states_adt_outv=True` exposes Gamma/Scale/Q/K/QK/DA_CS) settled in one
shot what twenty end-to-end variant sweeps could not, and twice caught cases
where two errors were partially cancelling and flattering a wrong hypothesis.
Reach for that first next time.
"""

import torch


def _rope(vec, angles):
    """Rotate `vec` (..., d) by `angles` (..., d//2), INTERLEAVED convention.

    Pairs adjacent dimensions (2i, 2i+1) — NOT the split-halves (i, i+d/2)
    convention. From the kernel:

        k0, k1 = tl.split(tl.reshape(k, [CHUNK, HEADDIM_QK // 2, 2]))
        ...
        k = tl.reshape(tl.join(ko0, ko1), [CHUNK, HEADDIM_QK])

    reshaping to (..., d/2, 2) and splitting the trailing axis takes even and
    odd indices, and the join/reshape writes them back interleaved.

    This distinction is invisible on the diagonal — `q_t . k_t` is invariant
    under any rotation applied to both — so an L=1 golden passes with either
    convention. It only shows up off-diagonal, where q_u and k_t carry
    different angles. Getting it wrong cost a full verify cycle; the L=1 case
    passing while every multi-step case failed is what localised it.

    Angles shorter than d//2 are zero-padded: that is how `rope_fraction < 1`
    is expressed, and a zero angle is the identity, so the tail passes through.
    """
    d = vec.shape[-1]
    half = d // 2
    ang = torch.zeros(*vec.shape[:-1], half, dtype=vec.dtype, device=vec.device)
    n = min(angles.shape[-1], half)
    ang[..., :n] = angles[..., :n].to(vec.dtype)
    cos, sin = torch.cos(ang), torch.sin(ang)
    v0, v1 = vec[..., 0::2], vec[..., 1::2]
    out = torch.empty_like(vec)
    out[..., 0::2] = v0 * cos - v1 * sin
    out[..., 1::2] = v0 * sin + v1 * cos
    return out


def mamba3_siso_ref(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles,
                    D=None, Z=None, dtype=torch.float64):
    """Sequential Mamba-3 SISO scan. Tensor names match the official kernel.

    Q, K      (b, l, ngroups, dqk)   ngroups=1 for SISO; shared across heads
    V, Z      (b, l, h, dv)
    ADT,DT,   (b, h, l)              ADT = A*dt (<=0), DT = dt (post-softplus),
    Trap                             Trap is PRE-sigmoid (the kernel applies it)
    Q_bias,   (h, dqk)               per-head, added BEFORE the rotation
    K_bias
    Angles    (b, l, h, r)           r <= dqk//2; zero-padded to dqk//2
    D         (h,)                   skip connection
    returns   (b, l, h, dv)
    """
    Q, K, V = Q.to(dtype), K.to(dtype), V.to(dtype)
    ADT, DT, Trap = ADT.to(dtype), DT.to(dtype), Trap.to(dtype)
    Q_bias, K_bias, Angles = (Q_bias.to(dtype), K_bias.to(dtype),
                              Angles.to(dtype))
    b, l, _, dv = V.shape
    h, dqk = Q_bias.shape

    lam = torch.sigmoid(Trap)                      # (b, h, l)
    gamma = DT * lam
    # shifted_gamma_t = dt_{t+1} * (1 - lam_{t+1}); the last step has no
    # successor, so it contributes nothing (the kernel masks it to 0.0).
    shifted = torch.zeros_like(gamma)
    shifted[..., :-1] = DT[..., 1:] * (1.0 - lam[..., 1:])
    scale = gamma + shifted
    a = torch.exp(ADT)                             # (b, h, l), ADT <= 0

    # Bias is per-head while Q/K are shared across heads (ngroups=1), so
    # broadcast the group axis out to heads before adding.
    q = Q[:, :, 0, :].unsqueeze(2) + Q_bias.view(1, 1, h, dqk)   # (b,l,h,dqk)
    k = K[:, :, 0, :].unsqueeze(2) + K_bias.view(1, 1, h, dqk)
    # The rotation angle is accumulated by a SEPARATE pre-pass kernel
    # (`angle_dt_fwd`) that `mamba3_siso_combined` runs before the scan; the
    # scan kernel itself then consumes the result raw. From angle_dt.py:
    #
    #     angle_vals = tanh(angle) * PI
    #     vals       = angle_vals * dt
    #     out        = cumsum(vals) + carry           (mod 2*pi)
    #
    # The `tanh(.) * PI` squashing is the part that is impossible to guess: it
    # bounds each step's rotation to +/-pi. Omitting it leaves you at ~7e-2
    # relative error, close enough to look like a rounding problem and send you
    # hunting the wrong thing.
    #
    # (mod 2*pi is omitted here — cos/sin are 2*pi-periodic, so it is a
    # numerical-range convenience in the kernel, not part of the maths.)
    theta = torch.cumsum(
        torch.tanh(Angles) * torch.pi * DT.permute(0, 2, 1).unsqueeze(-1),
        dim=1)
    q = _rope(q, theta)
    k = _rope(k, theta)

    out = torch.zeros(b, l, h, dv, dtype=dtype)
    S = torch.zeros(b, h, dv, dqk, dtype=dtype)
    for t in range(l):
        a_t = a[:, :, t].view(b, h, 1, 1)
        q_t, k_t, v_t = q[:, t], k[:, t], V[:, t]                # (b,h,*)
        # y from the running state, carrying this step's decay
        y = a_t.squeeze(-1) * torch.einsum("bhd,bhvd->bhv", q_t, S)
        # diagonal term: D-skip plus this step's own contribution
        qk = (q_t * k_t).sum(-1)                                 # (b,h)
        coef = qk * gamma[:, :, t]
        if D is not None:
            coef = coef + D.to(dtype).view(1, h)
        out[:, t] = y + coef.unsqueeze(-1) * v_t
        # state update
        S = a_t * S + (scale[:, :, t].view(b, h, 1, 1)
                       * v_t.unsqueeze(-1) * k_t.unsqueeze(-2))
    if Z is not None:
        z = Z.to(dtype)
        out = out * (z * torch.sigmoid(z))                       # silu
    return out
