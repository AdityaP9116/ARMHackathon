//! Portable scalar Mamba-3 SISO scan.
//!
//! Three jobs, exactly as `scalar.rs` does for Mamba-1: the in-crate correctness
//! oracle that the NEON path is diffed against, the non-Arm fallback, and what
//! keeps x86 CI meaningful. Clarity over speed — **do not optimise this file**.
//!
//! A direct transcription of `tests/reference/mamba3_ref.py`, which reproduces
//! the official Triton kernel to 4.47 bf16 ULP on the captured ground truth.

use super::{Mamba3Dims, Mamba3Input};
use crate::{parallel, Float, Threading};

/// Per-head scratch: the `(dv, dqk)` state matrix, the rotated q/k for the
/// current step, and the 2-tap carry. Allocated once per rayon worker.
struct Scratch<T> {
    s: Vec<T>,  // dv * dqk, row-major
    q: Vec<T>,  // dqk — bias-added and rotated
    k: Vec<T>,  // dqk
    bx: Vec<T>, // dqk — k_t * scale_t from the previous step in scan order
}

impl<T: Float> Scratch<T> {
    fn new(dv: usize, dqk: usize) -> Self {
        Scratch {
            s: vec![T::ZERO; dv * dqk],
            q: vec![T::ZERO; dqk],
            k: vec![T::ZERO; dqk],
            bx: vec![T::ZERO; dqk],
        }
    }
}

/// Rotate `dst` in place from `src + bias` using precomputed cos/sin.
///
/// **Interleaved convention** — pairs `(2i, 2i+1)`, not split-halves. This is
/// invisible on the diagonal (`q.k` is unchanged when both are rotated by the
/// same angle), so an L=1 test passes under either convention; only
/// off-diagonal terms expose it. Established by diffing the official kernel's
/// own `Q_store` buffer.
#[inline]
fn rope<T: Float>(src: &[T], bias: &[T], cos: &[T], sin: &[T], dst: &mut [T]) {
    for i in 0..src.len() / 2 {
        let (a, b) = (src[2 * i] + bias[2 * i], src[2 * i + 1] + bias[2 * i + 1]);
        let (c, s) = (cos[i], sin[i]);
        dst[2 * i] = a * c - b * s;
        dst[2 * i + 1] = a * s + b * c;
    }
}

pub(crate) fn scan<T: Float>(
    dims: &Mamba3Dims,
    input: &Mamba3Input<'_, T>,
    out: &mut [T],
    last_state: Option<&mut [T]>,
    last_bx: Option<&mut [T]>,
    threading: Threading,
) {
    let Mamba3Dims {
        batch: _,
        heads,
        dv,
        dqk,
        len,
        // SISO by construction: validation rejects rank > 1 without the MIMO
        // projections, and dispatch routes MIMO to `mimo::scan` before here.
        rank: _,
    } = *dims;
    let half = dqk / 2;

    parallel::for_each_head(
        len,
        dv,
        dqk,
        out,
        last_state,
        last_bx,
        threading,
        || Scratch::<T>::new(dv, dqk),
        |scratch, bh_idx, out_row, last_s, last_b| {
            let (bi, h) = (bh_idx / heads, bh_idx % heads);
            let q_bias = &input.q_bias[h * dqk..(h + 1) * dqk];
            let k_bias = &input.k_bias[h * dqk..(h + 1) * dqk];
            let d_skip = input.d_skip.map_or(T::ZERO, |d| d[h]);
            // (batch, heads, len) — contiguous in t for this (bi, h)
            let gate_base = (bi * heads + h) * len;

            scratch.s.iter_mut().for_each(|x| *x = T::ZERO);
            scratch.bx.iter_mut().for_each(|x| *x = T::ZERO);

            for i in 0..len {
                // `reverse` picks which timestep this step of the recurrence
                // consumes; the output still lands at index `t`.
                let t = if input.reverse { len - 1 - i } else { i };

                // --- gates -------------------------------------------------
                // The trapezoid's second term reads the NEXT timestep in scan
                // order: t+1 going forward, t-1 going backward. This is the one
                // place `reverse` is more than an index flip, and getting it
                // wrong is the highest-risk error in this kernel.
                let g = gate_base + t;
                let lam = input.trap[g].sigmoid();
                let gamma = input.dt[g] * lam;
                let nxt = if input.reverse {
                    (i + 1 < len).then(|| gate_base + t - 1)
                } else {
                    (t + 1 < len).then(|| g + 1)
                };
                // The final step of the traversal has no successor, so it
                // contributes nothing — matching the kernel, which masks the
                // out-of-range load to 0.
                let shifted = nxt.map_or(T::ZERO, |n| {
                    input.dt[n] * (T::ONE - input.trap[n].sigmoid())
                });
                let scale = gamma + shifted;
                let alpha = input.adt[g].exp();

                // --- rotate q/k --------------------------------------------
                // q/k are (batch, len, 1, dqk); cos/sin are (b, l, h, dqk/2).
                let qk_off = (bi * len + t) * dqk;
                let cs_off = ((bi * len + t) * heads + h) * half;
                let cos = &input.cos[cs_off..cs_off + half];
                let sin = &input.sin[cs_off..cs_off + half];
                rope(
                    &input.q[qk_off..qk_off + dqk],
                    q_bias,
                    cos,
                    sin,
                    &mut scratch.q,
                );
                rope(
                    &input.k[qk_off..qk_off + dqk],
                    k_bias,
                    cos,
                    sin,
                    &mut scratch.k,
                );

                // v/z are (batch, len, heads, dv) — strided per head.
                let v_off = ((bi * len + t) * heads + h) * dv;
                let v = &input.v[v_off..v_off + dv];

                // --- output from the running state -------------------------
                // y = alpha * (q . S^T), then the diagonal term.
                let qk_dot = scratch
                    .q
                    .iter()
                    .zip(scratch.k.iter())
                    .fold(T::ZERO, |acc, (&a, &b)| acc + a * b);
                let diag = d_skip + gamma * qk_dot;

                let o = t * dv;
                let dst = &mut out_row[o..o + dv];
                for ((slot, row), &vr) in dst
                    .iter_mut()
                    .zip(scratch.s.chunks_exact(dqk))
                    .zip(v.iter())
                {
                    let acc = scratch
                        .q
                        .iter()
                        .zip(row.iter())
                        .fold(T::ZERO, |a, (&qj, &sj)| a + qj * sj);
                    *slot = alpha * acc + diag * vr;
                }

                // --- state update: S = alpha*S + scale * (v (x) k) ----------
                for (row, &vr) in scratch.s.chunks_exact_mut(dqk).zip(v.iter()) {
                    let sv = scale * vr;
                    for (sj, &kj) in row.iter_mut().zip(scratch.k.iter()) {
                        *sj = alpha * *sj + sv * kj;
                    }
                }
                // Carry the weighted k for the resumable contract; the 2-tap's
                // cross-chunk continuation needs it alongside S.
                for (bx, &kj) in scratch.bx.iter_mut().zip(scratch.k.iter()) {
                    *bx = scale * kj;
                }

                // --- gate --------------------------------------------------
                if let Some(z) = input.z {
                    let zs = &z[v_off..v_off + dv];
                    for (slot, &zi) in out_row[o..o + dv].iter_mut().zip(zs.iter()) {
                        *slot = *slot * zi.silu();
                    }
                }
            }

            if let Some(ls) = last_s {
                ls.copy_from_slice(&scratch.s);
            }
            if let Some(lb) = last_b {
                lb.copy_from_slice(&scratch.bx);
            }
        },
    );
}
