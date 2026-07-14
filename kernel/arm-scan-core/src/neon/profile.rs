//! Phase-level timing harness for the NEON selective scan.
//!
//! This is a DIAGNOSTIC copy of the production `scan`/`channel_n16` path in
//! [`super`], instrumented with per-phase `Instant` timers and run
//! single-threaded so the accumulation is trivial and the phase *ratios* are
//! exact. It answers the one question that reorders the optimization backlog:
//! how is kernel time split across transpose / discretize / exp / projection /
//! recurrence / epilogue.
//!
//! Read the RELATIVE split, not the absolute nanoseconds: totals here are
//! ~single-core and include a small `Instant::now()` overhead per chunk-phase
//! (amortized over `CHUNK` timesteps, so it barely shifts the ratios). Because
//! the transpose runs once serially before the (normally parallel) channel
//! loop, its share here is the *one-core* share — in a C-core run its
//! wall-clock fraction grows by ~C (Amdahl), which is exactly why a large
//! transpose share flags §4.1/§4.2 in `IMPROVEMENT_IDEAS.md`.
//!
//! Kept deliberately close to `mod.rs`; if the production kernel's phase
//! structure changes, update this in lockstep. Behind the `profiling` feature
//! so it never touches a shipping build.

use core::arch::aarch64::*;
use std::time::Instant;

use super::{exp, Channel, Scratch, CHUNK};
use crate::{ScanDims, ScanInput};

/// Accumulated nanoseconds per phase over a whole (sequential) scan.
#[derive(Clone, Copy, Debug, Default)]
pub struct PhaseTimings {
    /// Build the transposed, zero-padded (len, N) B/C planes (serial prologue).
    pub transpose_ns: u128,
    /// Pass A1: dt = softplus(delta + bias), dt·u.
    pub discretize_ns: u128,
    /// Pass A2, exp part: ābar = exp(dt·A) — the hypothesized hot spot.
    pub exp_ns: u128,
    /// Pass A2, projection part: b̄ = (dt·u)·B.
    pub proj_ns: u128,
    /// Pass B: recurrence h = ābar⊙h + b̄ and the C·h dot product.
    pub recurrence_ns: u128,
    /// Epilogue: out = (y + D·u)·silu(z) over the row.
    pub epilogue_ns: u128,
    /// Total channels processed (batch·dim).
    pub channels: usize,
    pub len: usize,
    pub state: usize,
}

impl PhaseTimings {
    /// Sum of all phase timers (the profiler's own wall clock, minus setup).
    pub fn total_ns(&self) -> u128 {
        self.transpose_ns
            + self.discretize_ns
            + self.exp_ns
            + self.proj_ns
            + self.recurrence_ns
            + self.epilogue_ns
    }
}

/// One channel of the N=16 path, split so each phase is timed independently.
/// Mirrors [`super::channel_n16`]; the A2 loop is split into an exp loop and a
/// projection loop purely so the two can be attributed separately.
///
/// # Safety
/// Requires NEON (architecturally guaranteed on aarch64). `a_row` has 16
/// elements; `bt`/`ct` are (len, 16) planes; `out_row` has `len` elements.
#[target_feature(enable = "neon")]
unsafe fn channel_profiled(
    a_row: &[f32],
    bt: &[f32],
    ct: &[f32],
    ch: &Channel<'_>,
    out_row: &mut [f32],
    scratch: &mut Scratch,
    t: &mut PhaseTimings,
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

        // Pass A1: discretization across time.
        let c0 = Instant::now();
        super::discretize_chunk(ch, start, tlen, scratch);
        t.discretize_ns += c0.elapsed().as_nanos();

        // Pass A2 (exp): ābar = exp(dt·A) for the chunk.
        let abar = scratch.abar.as_mut_ptr();
        let c0 = Instant::now();
        for i in 0..tlen {
            let vdt = vdupq_n_f32(scratch.dt[i]);
            let o = i * 16;
            vst1q_f32(abar.add(o), exp::vexpq_f32_nonpos_fast(vmulq_f32(vdt, a0)));
            vst1q_f32(abar.add(o + 4), exp::vexpq_f32_nonpos_fast(vmulq_f32(vdt, a1)));
            vst1q_f32(abar.add(o + 8), exp::vexpq_f32_nonpos_fast(vmulq_f32(vdt, a2)));
            vst1q_f32(abar.add(o + 12), exp::vexpq_f32_nonpos_fast(vmulq_f32(vdt, a3)));
        }
        t.exp_ns += c0.elapsed().as_nanos();

        // Pass A2 (projection): b̄ = (dt·u)·B for the chunk.
        let bbar = scratch.bbar.as_mut_ptr();
        let c0 = Instant::now();
        for i in 0..tlen {
            let vdtu = vdupq_n_f32(scratch.dtu[i]);
            let b = bt.as_ptr().add((start + i) * 16);
            let o = i * 16;
            vst1q_f32(bbar.add(o), vmulq_f32(vld1q_f32(b), vdtu));
            vst1q_f32(bbar.add(o + 4), vmulq_f32(vld1q_f32(b.add(4)), vdtu));
            vst1q_f32(bbar.add(o + 8), vmulq_f32(vld1q_f32(b.add(8)), vdtu));
            vst1q_f32(bbar.add(o + 12), vmulq_f32(vld1q_f32(b.add(12)), vdtu));
        }
        t.proj_ns += c0.elapsed().as_nanos();

        // Pass B: pure-FMA recurrence + output dot product.
        let abar = scratch.abar.as_ptr();
        let bbar = scratch.bbar.as_ptr();
        let c0 = Instant::now();
        for i in 0..tlen {
            let o = i * 16;
            let c = ct.as_ptr().add((start + i) * 16);
            h0 = vfmaq_f32(vld1q_f32(bbar.add(o)), vld1q_f32(abar.add(o)), h0);
            h1 = vfmaq_f32(vld1q_f32(bbar.add(o + 4)), vld1q_f32(abar.add(o + 4)), h1);
            h2 = vfmaq_f32(vld1q_f32(bbar.add(o + 8)), vld1q_f32(abar.add(o + 8)), h2);
            h3 = vfmaq_f32(vld1q_f32(bbar.add(o + 12)), vld1q_f32(abar.add(o + 12)), h3);

            let mut acc = vmulq_f32(vld1q_f32(c), h0);
            acc = vfmaq_f32(acc, vld1q_f32(c.add(4)), h1);
            acc = vfmaq_f32(acc, vld1q_f32(c.add(8)), h2);
            acc = vfmaq_f32(acc, vld1q_f32(c.add(12)), h3);
            *out_row.get_unchecked_mut(start + i) = vaddvq_f32(acc);
        }
        t.recurrence_ns += c0.elapsed().as_nanos();

        start += tlen;
    }

    let c0 = Instant::now();
    super::epilogue_row(ch, out_row);
    t.epilogue_ns += c0.elapsed().as_nanos();
}

/// Run the NEON scan single-threaded with per-phase timing. Supports the
/// N=16 fast path only (the production shape for every mamba checkpoint).
///
/// Panics if `dims.state != 16`.
pub fn scan_profiled(dims: &ScanDims, input: &ScanInput<'_, f32>, out: &mut [f32]) -> PhaseTimings {
    assert_eq!(
        dims.state, 16,
        "scan_profiled supports the N=16 fast path only"
    );
    let ScanDims {
        batch,
        dim,
        len,
        state,
        groups,
    } = *dims;
    let group_size = dim / groups;
    let n4 = 16;

    let mut timings = PhaseTimings {
        channels: batch * dim,
        len,
        state,
        ..Default::default()
    };

    // Transpose every (batch, group) B/C plane to (len, 16), zero-padded.
    let planes = batch * groups;
    let mut bt = vec![0.0_f32; planes * len * n4];
    let mut ct = vec![0.0_f32; planes * len * n4];
    let c0 = Instant::now();
    for p in 0..planes {
        let src_base = p * state * len;
        let dst_base = p * len * n4;
        for n in 0..state {
            let src_b = &input.b[src_base + n * len..src_base + (n + 1) * len];
            let src_c = &input.c[src_base + n * len..src_base + (n + 1) * len];
            for tt in 0..len {
                bt[dst_base + tt * n4 + n] = src_b[tt];
                ct[dst_base + tt * n4 + n] = src_c[tt];
            }
        }
    }
    timings.transpose_ns = c0.elapsed().as_nanos();
    let (bt, ct) = (&bt[..], &ct[..]);

    let mut scratch = Scratch::new(n4);
    for ch_idx in 0..batch * dim {
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
        let out_row = &mut out[row..row + len];
        // SAFETY: NEON is always available on aarch64; shapes validated above.
        unsafe {
            channel_profiled(
                a_row,
                bt_plane,
                ct_plane,
                &ch,
                out_row,
                &mut scratch,
                &mut timings,
            );
        }
    }
    timings
}
