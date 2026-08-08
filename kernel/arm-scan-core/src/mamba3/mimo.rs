//! Portable scalar Mamba-3 **MIMO** scan.
//!
//! A direct transcription of `mamba3_mimo_ref` in `tests/reference/mamba3_ref.py`,
//! which reproduces the official `mamba3_mimo_combined` TileLang kernel to
//! **2.40 bf16 ULP** across the ground truth in `tests/golden/mamba3_mimo/`
//! (captured from the real `state-spaces/mamba3-mimo-187m` checkpoint).
//!
//! Clarity over speed, exactly like `scalar.rs`: this is the in-crate oracle
//! and the portable fallback. **Do not optimise this file.**
//!
//! # What MIMO changes, and what it does not
//!
//! The discretization is **identical** to SISO — same `gamma`, same forward-
//! looking `scale`, same `alpha`, and upstream even calls the same angle
//! pre-pass. Everything below is what actually differs:
//!
//! ```text
//!   x_r[p]     = psi[h][r][p] * v[p]                 elementwise, not a matmul
//!   y_r[p]     = alpha * sum_n q_r[n] * S[p][n]      every rank reads ONE state
//!   diag_r[p]  = gamma * sum_{r2} (q_r . k_r2) * x_r2[p]  +  D * x_r[p]
//!   S[p][n]    = alpha * S[p][n] + scale * sum_r x_r[p] * k_r[n]
//!   out[p]     = sum_r phi[h][r][p] * silu(z[p] * zeta[h][r][p]) * (y_r + diag_r)
//! ```
//!
//! Three details that are easy to get subtly wrong, all confirmed against the
//! kernel source rather than inferred:
//!
//! * the diagonal is a **rank-by-rank contraction** — output rank `r` collects
//!   a contribution from *every* input rank `r2`, not just its own;
//! * `D` multiplies the **projected** `x_r`, and is **not** scaled by `gamma`;
//! * the gate is applied **per rank, before** `phi` reduces the rank axis.
//!
//! # The RoPE convention is NOT the same as SISO's
//!
//! SISO's Triton kernel rotates interleaved lanes `(2i, 2i+1)`. MIMO's TileLang
//! kernel rotates **split halves** `(i, i + dqk/2)`, and only for
//! `i < dqk/4` — the remaining lanes pass through untouched. That is how
//! `rope_fraction = 0.5` is expressed here, where the SISO path instead
//! zero-pads its angles.
//!
//! Two kernels of one model family disagreeing on rotation convention is
//! surprising enough to be worth stating twice. It is read from
//! `mamba3_mimo_fwd.py` and confirmed by measurement: a rank-1 MIMO scan
//! matches a SISO scan to 7.7e-16 when the angles are zeroed, and diverges to
//! 3.8e-01 when they are not. See `check_rank1_collapse`.

use super::{Mamba3Dims, Mamba3Input};
use crate::{parallel, Float, Threading};

/// Per-head scratch. `q`/`k`/`x` carry every rank for the current step.
struct Scratch<T> {
    s: Vec<T>,   // dv * dqk, row-major — SHARED across ranks
    q: Vec<T>,   // rank * dqk, bias-added and rotated
    k: Vec<T>,   // rank * dqk
    x: Vec<T>,   // rank * dv — psi_r * v
    acc: Vec<T>, // rank * dv — per-rank result before the phi reduction
    qk: Vec<T>,  // rank * rank — the rank-by-rank Gram matrix
}

impl<T: Float> Scratch<T> {
    fn new(dv: usize, dqk: usize, rank: usize) -> Self {
        Scratch {
            s: vec![T::ZERO; dv * dqk],
            q: vec![T::ZERO; rank * dqk],
            k: vec![T::ZERO; rank * dqk],
            x: vec![T::ZERO; rank * dv],
            acc: vec![T::ZERO; rank * dv],
            qk: vec![T::ZERO; rank * rank],
        }
    }
}

/// Rotate `dst` in place from `src + bias`, **split-halves** convention.
///
/// Pairs lane `i` with lane `i + d/2` for `i < n_pairs`, where `n_pairs` is
/// however many angles were supplied (`dqk/4` at the published
/// `rope_fraction = 0.5`). Lanes outside those pairs are copied through
/// unrotated — this path expresses partial rotation by *rotating fewer lanes*,
/// where `mamba3/scalar.rs` expresses it by zero-padding the angle array.
///
/// See the module docs: this is deliberately not the same convention as SISO's.
#[inline]
fn rope_split_halves<T: Float>(src: &[T], bias: &[T], cos: &[T], sin: &[T], dst: &mut [T]) {
    let d = src.len();
    let half = d / 2;
    let pairs = cos.len().min(half);
    for i in 0..d {
        dst[i] = src[i] + bias[i];
    }
    for i in 0..pairs {
        let (a, b) = (dst[i], dst[half + i]);
        let (c, s) = (cos[i], sin[i]);
        dst[i] = a * c - b * s;
        dst[half + i] = a * s + b * c;
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
        rank,
    } = *dims;
    let half = dqk / 2;
    let mimo = input
        .mimo
        .as_ref()
        .expect("mimo::scan requires the MIMO projections; dispatch guarantees it");

    parallel::for_each_head(
        len,
        dv,
        dqk,
        out,
        last_state,
        last_bx,
        threading,
        || Scratch::<T>::new(dv, dqk, rank),
        |scratch, bh_idx, out_row, last_s, last_b| {
            let (bi, h) = (bh_idx / heads, bh_idx % heads);
            // Biases are (heads, rank, dqk); projections are (heads, rank, dv).
            let bias_base = h * rank * dqk;
            let proj_base = h * rank * dv;
            let d_skip = input.d_skip.map_or(T::ZERO, |d| d[h]);
            let gate_base = (bi * heads + h) * len;

            scratch.s.iter_mut().for_each(|x| *x = T::ZERO);

            for i in 0..len {
                let t = if input.reverse { len - 1 - i } else { i };

                // --- gates: identical to SISO ------------------------------
                let g = gate_base + t;
                let lam = input.trap[g].sigmoid();
                let gamma = input.dt[g] * lam;
                let nxt = if input.reverse {
                    (i + 1 < len).then(|| gate_base + t - 1)
                } else {
                    (t + 1 < len).then(|| g + 1)
                };
                let shifted = nxt.map_or(T::ZERO, |n| {
                    input.dt[n] * (T::ONE - input.trap[n].sigmoid())
                });
                let scale = gamma + shifted;
                let alpha = input.adt[g].exp();

                // cos/sin are per (b, l, h) and SHARED across ranks.
                let cs_off = ((bi * len + t) * heads + h) * half;
                let cos = &input.cos[cs_off..cs_off + half];
                let sin = &input.sin[cs_off..cs_off + half];

                // q/k are (batch, len, rank, dqk).
                let qk_off = (bi * len + t) * rank * dqk;
                for r in 0..rank {
                    let src = qk_off + r * dqk;
                    let b0 = bias_base + r * dqk;
                    rope_split_halves(
                        &input.q[src..src + dqk],
                        &input.q_bias[b0..b0 + dqk],
                        cos,
                        sin,
                        &mut scratch.q[r * dqk..(r + 1) * dqk],
                    );
                    rope_split_halves(
                        &input.k[src..src + dqk],
                        &input.k_bias[b0..b0 + dqk],
                        cos,
                        sin,
                        &mut scratch.k[r * dqk..(r + 1) * dqk],
                    );
                }

                // v/z are (batch, len, heads, dv) — strided per head.
                let v_off = ((bi * len + t) * heads + h) * dv;
                let v = &input.v[v_off..v_off + dv];

                // --- project v down to r streams, elementwise --------------
                for r in 0..rank {
                    let p0 = proj_base + r * dv;
                    for (xr, (&vr, &psi)) in scratch.x[r * dv..(r + 1) * dv]
                        .iter_mut()
                        .zip(v.iter().zip(mimo.psi[p0..p0 + dv].iter()))
                    {
                        *xr = psi * vr;
                    }
                }

                // --- rank-by-rank Gram matrix ------------------------------
                for a in 0..rank {
                    for b in 0..rank {
                        scratch.qk[a * rank + b] = scratch.q[a * dqk..(a + 1) * dqk]
                            .iter()
                            .zip(scratch.k[b * dqk..(b + 1) * dqk].iter())
                            .fold(T::ZERO, |s, (&qi, &ki)| s + qi * ki);
                    }
                }

                // --- per-rank output ---------------------------------------
                for r in 0..rank {
                    let qr = &scratch.q[r * dqk..(r + 1) * dqk];
                    let dst = &mut scratch.acc[r * dv..(r + 1) * dv];
                    for (p, (slot, row)) in
                        dst.iter_mut().zip(scratch.s.chunks_exact(dqk)).enumerate()
                    {
                        // Every rank reads the SAME state — the arithmetic
                        // intensity argument for MIMO in one line.
                        let acc = qr
                            .iter()
                            .zip(row.iter())
                            .fold(T::ZERO, |a, (&qj, &sj)| a + qj * sj);
                        // Diagonal: this rank collects from every input rank.
                        let mut d = T::ZERO;
                        for b in 0..rank {
                            d = d + scratch.qk[r * rank + b] * scratch.x[b * dv + p];
                        }
                        // D multiplies the PROJECTED x, and is not scaled by
                        // gamma — unlike SISO, where it rides the same
                        // coefficient as the qk term.
                        *slot = alpha * acc + gamma * d + d_skip * scratch.x[r * dv + p];
                    }
                }

                // --- state update: one state, r rank-1 terms ---------------
                for (p, row) in scratch.s.chunks_exact_mut(dqk).enumerate() {
                    for (n, sj) in row.iter_mut().enumerate() {
                        let mut upd = T::ZERO;
                        for r in 0..rank {
                            upd = upd + scratch.x[r * dv + p] * scratch.k[r * dqk + n];
                        }
                        *sj = alpha * *sj + scale * upd;
                    }
                }
                // No `bx` carry here on purpose. SISO stores `scale * k_t`,
                // one vector per step; a rank-r step has `r` of them and no
                // `(dqk,)` vector summarises them. Validation rejects the
                // carry for MIMO rather than letting this write rank 0's slice
                // and look meaningful. See `validate`.

                // --- gate per rank, then reduce the rank axis --------------
                let o = t * dv;
                let dst = &mut out_row[o..o + dv];
                for slot in dst.iter_mut() {
                    *slot = T::ZERO;
                }
                for r in 0..rank {
                    let p0 = proj_base + r * dv;
                    let phi = &mimo.phi[p0..p0 + dv];
                    let zeta = &mimo.zeta[p0..p0 + dv];
                    let ar = &scratch.acc[r * dv..(r + 1) * dv];
                    match input.z {
                        Some(z) => {
                            let zs = &z[v_off..v_off + dv];
                            for (p, slot) in dst.iter_mut().enumerate() {
                                *slot = *slot + phi[p] * (zs[p] * zeta[p]).silu() * ar[p];
                            }
                        }
                        None => {
                            for (p, slot) in dst.iter_mut().enumerate() {
                                *slot = *slot + phi[p] * ar[p];
                            }
                        }
                    }
                }
            }

            if let Some(ls) = last_s {
                ls.copy_from_slice(&scratch.s);
            }
            // `last_b` is always None here: validation rejects the carry for
            // MIMO, so there is deliberately nothing to write.
            debug_assert!(last_b.is_none());
        },
    );
}
