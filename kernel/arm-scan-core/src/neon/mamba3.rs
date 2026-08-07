//! NEON (aarch64) Mamba-3 SISO scan.
//!
//! **This file mirrors `mamba3/tiled.rs` loop for loop.** That file is the
//! portable twin, validated on x86 against the naive oracle *and* against the
//! captured official-kernel ground truth; this one substitutes vector ops for
//! the inner arithmetic and changes nothing structural. If the two disagree,
//! the bug is in the intrinsics here, not in the algorithm — which is the whole
//! point of having written the blocking portably first.
//!
//! # Where the time goes, and therefore what is vectorised
//!
//! Per timestep the work is:
//!
//! | Phase | Cost | Vectorised |
//! |---|---|---|
//! | Pass B (state update + output projection) | `~3 * dv * dqk` | **yes — the target** |
//! | RoPE on q and k | `2 * dqk` | yes |
//! | `q . k` for the diagonal term | `dqk` | yes |
//! | epilogue (D-skip, silu gate) | `dv` | yes |
//! | gates (sigmoid, exp, trapezoid blend) | `O(1)` | **no — deliberately** |
//!
//! Pass B dominates by a factor of `dv` (64x at the 187M shape), so the gates
//! stay scalar: they are a handful of transcendentals per timestep against tens
//! of thousands of FMAs, and vectorising them across chunks would mean
//! materialising per-chunk gate buffers for no measurable return. That is a
//! measurement-backed choice, not an omission — revisit only if a profile says
//! otherwise.
//!
//! # The ordering constraint
//!
//! `y` is read from `S` **before** the update, in the same pass that writes the
//! new value. See `mamba3/tiled.rs` for why reading it after is wrong by exactly
//! the trapezoid's shifted term.

use core::arch::aarch64::*;

use super::math::vsiluq_f32;
use crate::mamba3::{Mamba3Dims, Mamba3Input, TILE};
use crate::{parallel, Threading};

struct Scratch {
    s: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    y: Vec<f32>,
    bx: Vec<f32>,
}

impl Scratch {
    fn new(dv: usize, dqk: usize) -> Self {
        Scratch {
            s: vec![0.0; dv * dqk],
            q: vec![0.0; dqk],
            k: vec![0.0; dqk],
            y: vec![0.0; dv],
            bx: vec![0.0; dqk],
        }
    }
}

/// Bias-add then rotate, interleaved `(2i, 2i+1)` pairs.
///
/// `vld2q_f32` de-interleaves in one instruction: `.0` collects the even lanes
/// and `.1` the odd, which is exactly the pair layout RoPE wants, and
/// `vst2q_f32` re-interleaves on the way out. No shuffles needed.
///
/// # Safety
/// All four inputs must have at least `half*2` (`cos`/`sin`: `half`) readable
/// elements, and `dst` that many writable. Callers pass exact per-head slices.
#[inline]
unsafe fn rope_neon(src: &[f32], bias: &[f32], cos: &[f32], sin: &[f32], dst: &mut [f32]) {
    let half = cos.len();
    let mut i = 0;
    while i + 4 <= half {
        // SAFETY: i+4 <= half, and src/bias/dst hold 2*half elements.
        let a2 = vld2q_f32(src.as_ptr().add(2 * i));
        let b2 = vld2q_f32(bias.as_ptr().add(2 * i));
        let a = vaddq_f32(a2.0, b2.0);
        let b = vaddq_f32(a2.1, b2.1);
        let c = vld1q_f32(cos.as_ptr().add(i));
        let s = vld1q_f32(sin.as_ptr().add(i));
        let o0 = vfmsq_f32(vmulq_f32(a, c), b, s); // a*c - b*s
        let o1 = vfmaq_f32(vmulq_f32(a, s), b, c); // a*s + b*c
        vst2q_f32(dst.as_mut_ptr().add(2 * i), float32x4x2_t(o0, o1));
        i += 4;
    }
    // Scalar tail for a `dqk/2` that is not a multiple of 4.
    while i < half {
        let a = src[2 * i] + bias[2 * i];
        let b = src[2 * i + 1] + bias[2 * i + 1];
        let (c, s) = (cos[i], sin[i]);
        dst[2 * i] = a * c - b * s;
        dst[2 * i + 1] = a * s + b * c;
        i += 1;
    }
}

/// `sum(a[i] * b[i])`.
///
/// # Safety
/// `a` and `b` must be the same length.
#[inline]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 4 <= n {
        // SAFETY: i+4 <= n for both slices.
        acc = vfmaq_f32(
            acc,
            vld1q_f32(a.as_ptr().add(i)),
            vld1q_f32(b.as_ptr().add(i)),
        );
        i += 4;
    }
    let mut s = vaddvq_f32(acc);
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

pub(crate) fn scan(
    dims: &Mamba3Dims,
    input: &Mamba3Input<'_, f32>,
    out: &mut [f32],
    last_state: Option<&mut [f32]>,
    last_bx: Option<&mut [f32]>,
    threading: Threading,
) {
    let Mamba3Dims {
        heads,
        dv,
        dqk,
        len,
        ..
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
        || Scratch::new(dv, dqk),
        |sc, bh_idx, out_row, last_s, last_b| {
            let (bi, h) = (bh_idx / heads, bh_idx % heads);
            let q_bias = &input.q_bias[h * dqk..(h + 1) * dqk];
            let k_bias = &input.k_bias[h * dqk..(h + 1) * dqk];
            let d_skip = input.d_skip.map_or(0.0, |d| d[h]);
            let gate_base = (bi * heads + h) * len;

            sc.s.iter_mut().for_each(|x| *x = 0.0);
            sc.bx.iter_mut().for_each(|x| *x = 0.0);

            for i in 0..len {
                let t = if input.reverse { len - 1 - i } else { i };

                // --- gates (scalar by design; O(1) against Pass B's dv*dqk) ---
                // The trapezoid's second term reads the next step in SCAN order:
                // t+1 forward, t-1 backward.
                let g = gate_base + t;
                let lam = 1.0 / (1.0 + (-input.trap[g]).exp());
                let gamma = input.dt[g] * lam;
                let nxt = if input.reverse {
                    (i + 1 < len).then(|| g - 1)
                } else {
                    (t + 1 < len).then(|| g + 1)
                };
                let shifted = nxt.map_or(0.0, |n| {
                    let l2 = 1.0 / (1.0 + (-input.trap[n]).exp());
                    input.dt[n] * (1.0 - l2)
                });
                let scale = gamma + shifted;
                let alpha = input.adt[g].exp();

                let qk_off = (bi * len + t) * dqk;
                let cs_off = ((bi * len + t) * heads + h) * half;
                let cos = &input.cos[cs_off..cs_off + half];
                let sin = &input.sin[cs_off..cs_off + half];
                // SAFETY: slices are exact per-head/per-timestep views whose
                // lengths validation has already pinned to dqk / half.
                unsafe {
                    rope_neon(&input.q[qk_off..qk_off + dqk], q_bias, cos, sin, &mut sc.q);
                    rope_neon(&input.k[qk_off..qk_off + dqk], k_bias, cos, sin, &mut sc.k);
                }

                let v_off = ((bi * len + t) * heads + h) * dv;
                let v = &input.v[v_off..v_off + dv];

                // SAFETY: sc.q and sc.k are both dqk long.
                let qk_dot = unsafe { dot_neon(&sc.q, &sc.k) };
                let diag = d_skip + gamma * qk_dot;

                sc.y.iter_mut().for_each(|x| *x = 0.0);

                // --- THE BLOCKED CORE, vectorised ---------------------------
                // One pass over S: accumulate the OLD value into y, write the
                // new one back. Mirrors mamba3/tiled.rs exactly.
                for c0 in (0..dqk).step_by(TILE) {
                    let c1 = (c0 + TILE).min(dqk);
                    for (r, (row, &vr)) in sc.s.chunks_exact_mut(dqk).zip(v.iter()).enumerate() {
                        let sv = scale * vr;
                        // SAFETY: c0..c1 is within row (len dqk); the vector
                        // body steps 4 at a time and stops 4 short of c1, with
                        // a scalar tail.
                        let acc = unsafe {
                            let mut acc = vdupq_n_f32(0.0);
                            let mut j = c0;
                            while j + 4 <= c1 {
                                let old = vld1q_f32(row.as_ptr().add(j));
                                let qj = vld1q_f32(sc.q.as_ptr().add(j));
                                let kj = vld1q_f32(sc.k.as_ptr().add(j));
                                acc = vfmaq_f32(acc, qj, old);
                                // new = alpha*old + sv*k
                                let new = vfmaq_n_f32(vmulq_n_f32(old, alpha), kj, sv);
                                vst1q_f32(row.as_mut_ptr().add(j), new);
                                j += 4;
                            }
                            let mut s = vaddvq_f32(acc);
                            while j < c1 {
                                let old = row[j];
                                s += sc.q[j] * old;
                                row[j] = alpha * old + sv * sc.k[j];
                                j += 1;
                            }
                            s
                        };
                        sc.y[r] += acc;
                    }
                }

                // --- epilogue: out = (alpha*y + diag*v) * silu(z) -----------
                let o = t * dv;
                let dst = &mut out_row[o..o + dv];
                let zs = input.z.map(|z| &z[v_off..v_off + dv]);
                // SAFETY: dst, sc.y, v and zs are all dv long; the vector body
                // stops 4 short and a scalar tail finishes.
                unsafe {
                    let va = vdupq_n_f32(alpha);
                    let vd = vdupq_n_f32(diag);
                    let mut r = 0;
                    while r + 4 <= dv {
                        let yv = vld1q_f32(sc.y.as_ptr().add(r));
                        let vv = vld1q_f32(v.as_ptr().add(r));
                        let mut o4 = vfmaq_f32(vmulq_f32(va, yv), vd, vv);
                        if let Some(z) = zs {
                            o4 = vmulq_f32(o4, vsiluq_f32(vld1q_f32(z.as_ptr().add(r))));
                        }
                        vst1q_f32(dst.as_mut_ptr().add(r), o4);
                        r += 4;
                    }
                    while r < dv {
                        let mut val = alpha * sc.y[r] + diag * v[r];
                        if let Some(z) = zs {
                            let zr = z[r];
                            val *= zr / (1.0 + (-zr).exp());
                        }
                        dst[r] = val;
                        r += 1;
                    }
                }

                for (bx, &kj) in sc.bx.iter_mut().zip(sc.k.iter()) {
                    *bx = scale * kj;
                }
            }

            if let Some(ls) = last_s {
                ls.copy_from_slice(&sc.s);
            }
            if let Some(lb) = last_b {
                lb.copy_from_slice(&sc.bx);
            }
        },
    );
}
