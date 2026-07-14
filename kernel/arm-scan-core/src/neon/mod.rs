//! NEON (aarch64) implementation of the selective scan.
//!
//! Phase 2 held the SSM state in registers; Phase 3 restructures each
//! channel into a chunked two-pass pipeline and threads across channels:
//!
//!  Pass A (per chunk of `CHUNK` timesteps, vectorized ACROSS TIME):
//!    A1: dt = softplus(delta + bias) and dt·u, 4 timesteps per iteration
//!        using the NEON softplus — no scalar libm on the hot path.
//!    A2: ābar = exp(dt·A) and b̄ = (dt·u)·B for the whole chunk. The exps
//!        have no cross-timestep dependency here, so they issue at full
//!        pipeline throughput instead of being interleaved with the
//!        recurrence's serial FMA chain.
//!  Pass B (vectorized ACROSS STATE): the recurrence h = ābar⊙h + b̄ and
//!    the C·h dot product — nothing but loads and FMAs, with h resident in
//!    four q-registers for the N=16 fast path.
//!  Epilogue (vectorized across time): out = (y + D·u) · silu(z) over the
//!    whole row, 4 timesteps at a time.
//!
//! Channel iteration goes through `parallel::for_each_channel` (rayon),
//! with all chunk buffers in per-thread scratch. The transposed B/C planes
//! (see below) are built once up front and shared read-only by all workers.
//!
//! Layout: b/c arrive as (batch, groups, state, len) — at a fixed timestep
//! the N state values are strided by `len`, which NEON cannot load
//! directly. We transpose each (batch, group) plane once into (len, N4)
//! scratch (N4 = N rounded up to a multiple of 4, zero-padded). The
//! transpose is O(N*len) per plane and is amortized over the dim/groups
//! channels (1536 of them for mamba-130m) that then stream it
//! contiguously. Zero padding makes the general-N vector loop exact with
//! no masked tail: padded lanes compute h = exp(dt*0)*0 + dt_u*0 = 0 and
//! contribute c*h = 0 to the output.

mod exp;
mod math;
#[cfg(feature = "profiling")]
pub mod profile;

use core::arch::aarch64::*;

use crate::{Float, ScanDims, ScanInput, Threading};

/// Timesteps per chunk. 128 keeps the per-thread scratch
/// (dt + dtu + ābar + b̄ at N=16: ~17 KB) inside L1.
const CHUNK: usize = 128;

/// Entry point from the dispatcher. NEON is architecturally mandatory on
/// aarch64, so no runtime feature probe is needed.
pub(crate) fn scan(
    dims: &ScanDims,
    input: &ScanInput<'_, f32>,
    out: &mut [f32],
    last_state: Option<&mut [f32]>,
    threading: Threading,
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

    // Transpose every (batch, group) B/C plane to (len, n4), zero-padded.
    let planes = batch * groups;
    let mut bt = vec![0.0_f32; planes * len * n4];
    let mut ct = vec![0.0_f32; planes * len * n4];
    for p in 0..planes {
        let src_base = p * state * len;
        let dst_base = p * len * n4;
        for n in 0..state {
            let src_b = &input.b[src_base + n * len..src_base + (n + 1) * len];
            let src_c = &input.c[src_base + n * len..src_base + (n + 1) * len];
            for t in 0..len {
                bt[dst_base + t * n4 + n] = src_b[t];
                ct[dst_base + t * n4 + n] = src_c[t];
            }
        }
    }
    let (bt, ct) = (&bt[..], &ct[..]);

    crate::parallel::for_each_channel(
        len,
        state,
        out,
        last_state,
        threading,
        || Scratch::new(n4),
        |scratch, ch_idx, out_row, last| {
            let (bi, d) = (ch_idx / dim, ch_idx % dim);
            let plane = (bi * groups + d / group_size) * len * n4;
            let row = ch_idx * len;
            let ch = Channel {
                u: &input.u[row..row + len],
                delta: &input.delta[row..row + len],
                z: input.z.map(|z| &z[row..row + len]),
                bias: input.delta_bias.map_or(0.0, |v| v[d]),
                d_skip: input.d_skip.map(|v| v[d]),
                softplus: input.delta_softplus,
            };
            let a_row = &input.a[d * state..(d + 1) * state];
            let bt_plane = &bt[plane..plane + len * n4];
            let ct_plane = &ct[plane..plane + len * n4];

            // SAFETY: NEON is always available on aarch64.
            unsafe {
                if state == 16 {
                    channel_n16(a_row, bt_plane, ct_plane, &ch, out_row, last, scratch);
                } else {
                    scratch.a_pad[..state].copy_from_slice(a_row);
                    channel_general(bt_plane, ct_plane, &ch, out_row, last, scratch);
                }
                epilogue_row(&ch, out_row);
            }
        },
    );
}

/// Per-thread working buffers (allocated once per rayon worker).
struct Scratch {
    dt: Vec<f32>,    // CHUNK
    dtu: Vec<f32>,   // CHUNK
    abar: Vec<f32>,  // CHUNK * n4
    bbar: Vec<f32>,  // CHUNK * n4
    a_pad: Vec<f32>, // n4, zero-padded A row (general path)
    h_buf: Vec<f32>, // n4 (general path)
}

impl Scratch {
    fn new(n4: usize) -> Self {
        Scratch {
            dt: vec![0.0; CHUNK],
            dtu: vec![0.0; CHUNK],
            abar: vec![0.0; CHUNK * n4],
            bbar: vec![0.0; CHUNK * n4],
            a_pad: vec![0.0; n4],
            h_buf: vec![0.0; n4],
        }
    }
}

/// Per-channel inputs shared by both code paths.
struct Channel<'a> {
    u: &'a [f32],
    delta: &'a [f32],
    z: Option<&'a [f32]>,
    bias: f32,
    d_skip: Option<f32>,
    softplus: bool,
}

/// Pass A1: dt = softplus(delta + bias), dtu = dt*u for `tlen` timesteps
/// starting at `start`, vectorized across time with a scalar tail.
#[target_feature(enable = "neon")]
unsafe fn discretize_chunk(ch: &Channel<'_>, start: usize, tlen: usize, scratch: &mut Scratch) {
    let delta = ch.delta[start..start + tlen].as_ptr();
    let u = ch.u[start..start + tlen].as_ptr();
    let vbias = vdupq_n_f32(ch.bias);

    let mut t = 0;
    while t + 4 <= tlen {
        let mut v = vaddq_f32(vld1q_f32(delta.add(t)), vbias);
        if ch.softplus {
            v = math::vsoftplusq_f32(v);
        }
        vst1q_f32(scratch.dt.as_mut_ptr().add(t), v);
        vst1q_f32(
            scratch.dtu.as_mut_ptr().add(t),
            vmulq_f32(v, vld1q_f32(u.add(t))),
        );
        t += 4;
    }
    while t < tlen {
        let mut dt = *delta.add(t) + ch.bias;
        if ch.softplus {
            dt = dt.softplus();
        }
        scratch.dt[t] = dt;
        scratch.dtu[t] = dt * *u.add(t);
        t += 1;
    }
}

/// Epilogue: out = (y + D·u) * silu(z), vectorized across the whole row.
#[target_feature(enable = "neon")]
unsafe fn epilogue_row(ch: &Channel<'_>, out_row: &mut [f32]) {
    if ch.d_skip.is_none() && ch.z.is_none() {
        return;
    }
    let len = out_row.len();
    let out = out_row.as_mut_ptr();
    let u = ch.u.as_ptr();
    let ds = ch.d_skip.unwrap_or(0.0);
    let vds = vdupq_n_f32(ds);

    let mut t = 0;
    while t + 4 <= len {
        let mut y = vld1q_f32(out.add(t));
        if ch.d_skip.is_some() {
            y = vfmaq_f32(y, vds, vld1q_f32(u.add(t)));
        }
        if let Some(z) = ch.z {
            y = vmulq_f32(y, math::vsiluq_f32(vld1q_f32(z.as_ptr().add(t))));
        }
        vst1q_f32(out.add(t), y);
        t += 4;
    }
    while t < len {
        let mut y = *out.add(t);
        if ch.d_skip.is_some() {
            y += ds * *u.add(t);
        }
        if let Some(z) = ch.z {
            y *= z[t].silu();
        }
        *out.add(t) = y;
        t += 1;
    }
}

/// N=16 fast path: chunked two-pass with the state in four q-registers.
#[target_feature(enable = "neon")]
unsafe fn channel_n16(
    a_row: &[f32],
    bt: &[f32],
    ct: &[f32],
    ch: &Channel<'_>,
    out_row: &mut [f32],
    last_state: Option<&mut [f32]>,
    scratch: &mut Scratch,
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

    let len = out_row.len();
    let mut start = 0;
    while start < len {
        let tlen = CHUNK.min(len - start);

        // Pass A1: discretization across time
        discretize_chunk(ch, start, tlen, scratch);

        // Pass A2: batch all exps + input projections for the chunk
        let abar = scratch.abar.as_mut_ptr();
        let bbar = scratch.bbar.as_mut_ptr();
        for t in 0..tlen {
            let vdt = vdupq_n_f32(scratch.dt[t]);
            let vdtu = vdupq_n_f32(scratch.dtu[t]);
            let b = bt.as_ptr().add((start + t) * 16);
            let o = t * 16;
            // dt*A is always <= 0 (A < 0, dt >= 0); the decay factor tolerates
            // the degree-3 exp (contraction + golden margin) -> nonpos_fast.
            vst1q_f32(abar.add(o), exp::vexpq_f32_nonpos_fast(vmulq_f32(vdt, a0)));
            vst1q_f32(abar.add(o + 4), exp::vexpq_f32_nonpos_fast(vmulq_f32(vdt, a1)));
            vst1q_f32(abar.add(o + 8), exp::vexpq_f32_nonpos_fast(vmulq_f32(vdt, a2)));
            vst1q_f32(abar.add(o + 12), exp::vexpq_f32_nonpos_fast(vmulq_f32(vdt, a3)));
            vst1q_f32(bbar.add(o), vmulq_f32(vld1q_f32(b), vdtu));
            vst1q_f32(bbar.add(o + 4), vmulq_f32(vld1q_f32(b.add(4)), vdtu));
            vst1q_f32(bbar.add(o + 8), vmulq_f32(vld1q_f32(b.add(8)), vdtu));
            vst1q_f32(bbar.add(o + 12), vmulq_f32(vld1q_f32(b.add(12)), vdtu));
        }

        // Pass B: pure-FMA recurrence + output dot product
        let abar = scratch.abar.as_ptr();
        let bbar = scratch.bbar.as_ptr();
        for t in 0..tlen {
            let o = t * 16;
            let c = ct.as_ptr().add((start + t) * 16);

            h0 = vfmaq_f32(vld1q_f32(bbar.add(o)), vld1q_f32(abar.add(o)), h0);
            h1 = vfmaq_f32(vld1q_f32(bbar.add(o + 4)), vld1q_f32(abar.add(o + 4)), h1);
            h2 = vfmaq_f32(vld1q_f32(bbar.add(o + 8)), vld1q_f32(abar.add(o + 8)), h2);
            h3 = vfmaq_f32(vld1q_f32(bbar.add(o + 12)), vld1q_f32(abar.add(o + 12)), h3);

            let mut acc = vmulq_f32(vld1q_f32(c), h0);
            acc = vfmaq_f32(acc, vld1q_f32(c.add(4)), h1);
            acc = vfmaq_f32(acc, vld1q_f32(c.add(8)), h2);
            acc = vfmaq_f32(acc, vld1q_f32(c.add(12)), h3);
            *out_row.get_unchecked_mut(start + t) = vaddvq_f32(acc);
        }

        start += tlen;
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
/// Uses the chunked discretization but keeps exp inline in the recurrence.
#[target_feature(enable = "neon")]
unsafe fn channel_general(
    bt: &[f32],
    ct: &[f32],
    ch: &Channel<'_>,
    out_row: &mut [f32],
    last_state: Option<&mut [f32]>,
    scratch: &mut Scratch,
) {
    let n4 = scratch.a_pad.len();
    scratch.h_buf.fill(0.0);

    let len = out_row.len();
    let mut start = 0;
    while start < len {
        let tlen = CHUNK.min(len - start);
        discretize_chunk(ch, start, tlen, scratch);

        for t in 0..tlen {
            let vdt = vdupq_n_f32(scratch.dt[t]);
            let vdtu = vdupq_n_f32(scratch.dtu[t]);
            let b = bt.as_ptr().add((start + t) * n4);
            let c = ct.as_ptr().add((start + t) * n4);

            let mut acc = vdupq_n_f32(0.0);
            for i in (0..n4).step_by(4) {
                let a_v = vld1q_f32(scratch.a_pad.as_ptr().add(i));
                let h_v = vld1q_f32(scratch.h_buf.as_ptr().add(i));
                // dt*A <= 0 on real lanes; zero-padded A lanes give exp(0)=1.
                let e = exp::vexpq_f32_nonpos_fast(vmulq_f32(vdt, a_v));
                let h_new = vfmaq_f32(vmulq_f32(vld1q_f32(b.add(i)), vdtu), e, h_v);
                vst1q_f32(scratch.h_buf.as_mut_ptr().add(i), h_new);
                acc = vfmaq_f32(acc, vld1q_f32(c.add(i)), h_new);
            }
            *out_row.get_unchecked_mut(start + t) = vaddvq_f32(acc);
        }
        start += tlen;
    }

    if let Some(ls) = last_state {
        ls.copy_from_slice(&scratch.h_buf[..ls.len()]);
    }
}
