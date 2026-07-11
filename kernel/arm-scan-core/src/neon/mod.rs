//! NEON (aarch64) implementation of the selective scan.
//!
//! Register-level design (see INTEGRATION_PLAN.md Phase 2):
//!  - The per-channel state `h` (N floats) lives in NEON registers across
//!    the entire sequence loop — four `float32x4_t` for the N=16 fast path
//!    that every stock Mamba model uses. Zero state memory traffic.
//!  - The channel's A row is loop-invariant and hoisted into registers.
//!  - Discretization, recurrence, and the C dot-product are fused into one
//!    in-register loop; the only intermediate that ever touches memory is
//!    the transposed B/C scratch below.
//!  - `exp` is the hand-written 4-lane polynomial in [`exp`].
//!
//! Layout: b/c arrive as (batch, groups, state, len) — at a fixed timestep
//! the N state values are strided by `len`, which NEON cannot load
//! directly. We transpose each (batch, group) plane once into (len, N4)
//! scratch (N4 = N rounded up to a multiple of 4, zero-padded). The
//! transpose is O(N*len) and is amortized over the dim/groups channels
//! (1536 of them for mamba-130m) that then stream it contiguously.
//! Zero padding makes the general-N vector loop exact with no masked tail:
//! padded lanes compute h = exp(dt*0)*0 + dt_u*0 = 0 and contribute
//! c*h = 0 to the output.

mod exp;

use core::arch::aarch64::*;

use crate::{Float, ScanDims, ScanInput};

/// Entry point from the dispatcher. Plain safe fn: NEON is architecturally
/// mandatory on aarch64, so no runtime feature probe is needed.
pub(crate) fn scan(
    dims: &ScanDims,
    input: &ScanInput<'_, f32>,
    out: &mut [f32],
    last_state: Option<&mut [f32]>,
) {
    // SAFETY: neon is always available on aarch64 targets.
    unsafe { scan_inner(dims, input, out, last_state) }
}

#[target_feature(enable = "neon")]
unsafe fn scan_inner(
    dims: &ScanDims,
    input: &ScanInput<'_, f32>,
    out: &mut [f32],
    mut last_state: Option<&mut [f32]>,
) {
    let ScanDims {
        batch,
        dim,
        len,
        state,
        groups,
    } = *dims;
    let group_size = dim / groups;
    let n4 = state.div_ceil(4) * 4;

    // (len, n4) transposed scratch for the current (batch, group) plane.
    // Zero-initialized once; padding columns (n >= state) are never written
    // again, so they stay zero across reuse.
    let mut bt = vec![0.0_f32; len * n4];
    let mut ct = vec![0.0_f32; len * n4];
    let mut a_pad = vec![0.0_f32; n4];
    let mut h_buf = vec![0.0_f32; n4];

    for bi in 0..batch {
        for g in 0..groups {
            let plane = (bi * groups + g) * state * len;
            for n in 0..state {
                let src = &input.b[plane + n * len..plane + (n + 1) * len];
                let src_c = &input.c[plane + n * len..plane + (n + 1) * len];
                for t in 0..len {
                    bt[t * n4 + n] = src[t];
                    ct[t * n4 + n] = src_c[t];
                }
            }

            for d in g * group_size..(g + 1) * group_size {
                let row = (bi * dim + d) * len;
                let ch = Channel {
                    u: &input.u[row..row + len],
                    delta: &input.delta[row..row + len],
                    z: input.z.map(|z| &z[row..row + len]),
                    bias: input.delta_bias.map_or(0.0, |v| v[d]),
                    d_skip: input.d_skip.map(|v| v[d]),
                    softplus: input.delta_softplus,
                };
                let out_row = &mut out[row..row + len];
                let last = last_state
                    .as_deref_mut()
                    .map(|ls| &mut ls[(bi * dim + d) * state..(bi * dim + d + 1) * state]);

                let a_row = &input.a[d * state..(d + 1) * state];
                if state == 16 {
                    channel_n16(a_row, &bt, &ct, &ch, out_row, last);
                } else {
                    a_pad[..state].copy_from_slice(a_row);
                    channel_general(&a_pad, &bt, &ct, &ch, out_row, last, &mut h_buf);
                }
            }
        }
    }
}

/// Per-channel scalars/rows shared by both code paths.
struct Channel<'a> {
    u: &'a [f32],
    delta: &'a [f32],
    z: Option<&'a [f32]>,
    bias: f32,
    d_skip: Option<f32>,
    softplus: bool,
}

impl Channel<'_> {
    /// Discretization + output epilogue that stays scalar (1 op per
    /// timestep vs N vector lanes; not worth vectorizing).
    #[inline(always)]
    fn dt_at(&self, t: usize) -> f32 {
        let mut dt = self.delta[t] + self.bias;
        if self.softplus {
            dt = dt.softplus();
        }
        dt
    }

    #[inline(always)]
    fn epilogue(&self, t: usize, mut y: f32) -> f32 {
        if let Some(ds) = self.d_skip {
            y += ds * self.u[t];
        }
        if let Some(z) = self.z {
            y *= z[t].silu();
        }
        y
    }
}

/// N=16 fast path: the whole SSM state lives in four q-registers.
#[target_feature(enable = "neon")]
unsafe fn channel_n16(
    a_row: &[f32],
    bt: &[f32],
    ct: &[f32],
    ch: &Channel<'_>,
    out_row: &mut [f32],
    last_state: Option<&mut [f32]>,
) {
    let a = a_row.as_ptr();
    let a0 = vld1q_f32(a);
    let a1 = vld1q_f32(a.add(4));
    let a2 = vld1q_f32(a.add(8));
    let a3 = vld1q_f32(a.add(12));

    let mut h0 = vdupq_n_f32(0.0);
    let mut h1 = vdupq_n_f32(0.0);
    let mut h2 = vdupq_n_f32(0.0);
    let mut h3 = vdupq_n_f32(0.0);

    for (t, out_slot) in out_row.iter_mut().enumerate() {
        let dt = ch.dt_at(t);
        let vdt = vdupq_n_f32(dt);
        let vdtu = vdupq_n_f32(dt * ch.u[t]);
        let b = bt.as_ptr().add(t * 16);
        let c = ct.as_ptr().add(t * 16);

        let e0 = exp::vexpq_f32(vmulq_f32(vdt, a0));
        let e1 = exp::vexpq_f32(vmulq_f32(vdt, a1));
        let e2 = exp::vexpq_f32(vmulq_f32(vdt, a2));
        let e3 = exp::vexpq_f32(vmulq_f32(vdt, a3));

        // h = e ⊙ h + (dt*u) * b
        h0 = vfmaq_f32(vmulq_f32(vld1q_f32(b), vdtu), e0, h0);
        h1 = vfmaq_f32(vmulq_f32(vld1q_f32(b.add(4)), vdtu), e1, h1);
        h2 = vfmaq_f32(vmulq_f32(vld1q_f32(b.add(8)), vdtu), e2, h2);
        h3 = vfmaq_f32(vmulq_f32(vld1q_f32(b.add(12)), vdtu), e3, h3);

        // y = c · h
        let mut acc = vmulq_f32(vld1q_f32(c), h0);
        acc = vfmaq_f32(acc, vld1q_f32(c.add(4)), h1);
        acc = vfmaq_f32(acc, vld1q_f32(c.add(8)), h2);
        acc = vfmaq_f32(acc, vld1q_f32(c.add(12)), h3);

        *out_slot = ch.epilogue(t, vaddvq_f32(acc));
    }

    if let Some(ls) = last_state {
        let p = ls.as_mut_ptr();
        vst1q_f32(p, h0);
        vst1q_f32(p.add(4), h1);
        vst1q_f32(p.add(8), h2);
        vst1q_f32(p.add(12), h3);
    }
}

/// General-N path: state in a zero-padded buffer, processed 4 lanes at a
/// time with no tail special-casing (see module docs). Correct for any N;
/// the N=16 specialization above is the one that gets the register file.
#[target_feature(enable = "neon")]
unsafe fn channel_general(
    a_pad: &[f32],
    bt: &[f32],
    ct: &[f32],
    ch: &Channel<'_>,
    out_row: &mut [f32],
    last_state: Option<&mut [f32]>,
    h_buf: &mut [f32],
) {
    let n4 = a_pad.len();
    h_buf.fill(0.0);

    for (t, out_slot) in out_row.iter_mut().enumerate() {
        let dt = ch.dt_at(t);
        let vdt = vdupq_n_f32(dt);
        let vdtu = vdupq_n_f32(dt * ch.u[t]);
        let b = bt.as_ptr().add(t * n4);
        let c = ct.as_ptr().add(t * n4);

        let mut acc = vdupq_n_f32(0.0);
        for i in (0..n4).step_by(4) {
            let a_v = vld1q_f32(a_pad.as_ptr().add(i));
            let h_v = vld1q_f32(h_buf.as_ptr().add(i));
            let e = exp::vexpq_f32(vmulq_f32(vdt, a_v));
            let h_new = vfmaq_f32(vmulq_f32(vld1q_f32(b.add(i)), vdtu), e, h_v);
            vst1q_f32(h_buf.as_mut_ptr().add(i), h_new);
            acc = vfmaq_f32(acc, vld1q_f32(c.add(i)), h_new);
        }
        *out_slot = ch.epilogue(t, vaddvq_f32(acc));
    }

    if let Some(ls) = last_state {
        ls.copy_from_slice(&h_buf[..ls.len()]);
    }
}
