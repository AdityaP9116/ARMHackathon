//! Cache-blocked Mamba-3 scan — portable, and the structural template for NEON.
//!
//! # Why this file exists
//!
//! Mamba-1's state is 16 f32 (64 bytes) and lives in four NEON registers, which
//! is the whole reason its inner loop is fast. Mamba-3's state is a **matrix per
//! head**, `dv x dqk`: 8 KB at the sweep shapes and **32 KB** at the 187M shape,
//! against 64 KB of L1d per core on the Neoverse N2 we measured. It cannot be
//! register-resident, so the recurrence has to be blocked.
//!
//! That restructuring — not the intrinsics — is where the bugs are. So the
//! blocking lives here, portably, where it can be tested on any machine
//! (including x86 CI and a dev box with no Arm hardware). `neon/mamba3.rs`
//! mirrors this loop nest exactly and substitutes vector ops for the inner
//! arithmetic; if the two ever disagree, the difference is in the intrinsics,
//! not in the algorithm.
//!
//! # The ordering constraint that is easy to get wrong
//!
//! `y` must be read from the state **before** that timestep's update:
//!
//! ```text
//!   y_t = alpha * (q . S_old^T) + (D + gamma * (q.k)) * v
//!   S   = alpha * S_old + scale * (v (x) k)
//! ```
//!
//! A naive fusion that updates `S` first and then reads it computes
//! `q . S_new^T = alpha*(q . S_old^T) + scale*(q.k)*v` — wrong by
//! `(scale - gamma)*(q.k)*v`, i.e. by exactly the trapezoid's *shifted* term.
//! It would still run, still produce plausible numbers, and be wrong in a way
//! that grows with sequence length.
//!
//! The fix costs nothing: read the old value, accumulate it into `y`, then write
//! the new value back — one pass over `S`, each element loaded and stored once.
//! That is what the loop below does, and it is why `y` is scaled by `alpha`
//! *after* the accumulation rather than inside it.

use super::{Mamba3Dims, Mamba3Input};
use crate::{parallel, Float, Threading};

/// Columns of the state matrix processed per tile.
///
/// Sized so one tile-row of `S` plus the corresponding `q`/`k` slices stay
/// comfortably in L1 while leaving room for the streaming inputs. 32 f32 is
/// 8 NEON q-registers' worth, which is the natural unit for the vector port.
///
/// **This is a tunable to be swept, not a derived constant** — see
/// `MAMBA3_KERNEL_WORKPLAN.md` §M4. Sweep {16, 32, 64} at the production shapes
/// before treating it as settled.
pub(crate) const TILE: usize = 32;

struct Scratch<T> {
    s: Vec<T>,
    q: Vec<T>,
    k: Vec<T>,
    y: Vec<T>,
    bx: Vec<T>,
}

impl<T: Float> Scratch<T> {
    fn new(dv: usize, dqk: usize) -> Self {
        Scratch {
            s: vec![T::ZERO; dv * dqk],
            q: vec![T::ZERO; dqk],
            k: vec![T::ZERO; dqk],
            y: vec![T::ZERO; dv],
            bx: vec![T::ZERO; dqk],
        }
    }
}

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
        || Scratch::<T>::new(dv, dqk),
        |sc, bh_idx, out_row, last_s, last_b| {
            let (bi, h) = (bh_idx / heads, bh_idx % heads);
            let q_bias = &input.q_bias[h * dqk..(h + 1) * dqk];
            let k_bias = &input.k_bias[h * dqk..(h + 1) * dqk];
            let d_skip = input.d_skip.map_or(T::ZERO, |d| d[h]);
            let gate_base = (bi * heads + h) * len;

            sc.s.iter_mut().for_each(|x| *x = T::ZERO);
            sc.bx.iter_mut().for_each(|x| *x = T::ZERO);

            for i in 0..len {
                let t = if input.reverse { len - 1 - i } else { i };

                // --- gates. The trapezoid's second term reads the next step in
                // SCAN order: t+1 forward, t-1 backward.
                let g = gate_base + t;
                let lam = input.trap[g].sigmoid();
                let gamma = input.dt[g] * lam;
                let nxt = if input.reverse {
                    (i + 1 < len).then(|| g - 1)
                } else {
                    (t + 1 < len).then(|| g + 1)
                };
                let shifted = nxt.map_or(T::ZERO, |n| {
                    input.dt[n] * (T::ONE - input.trap[n].sigmoid())
                });
                let scale = gamma + shifted;
                let alpha = input.adt[g].exp();

                // --- rotate q/k
                let qk_off = (bi * len + t) * dqk;
                let cs_off = ((bi * len + t) * heads + h) * half;
                let cos = &input.cos[cs_off..cs_off + half];
                let sin = &input.sin[cs_off..cs_off + half];
                rope(&input.q[qk_off..qk_off + dqk], q_bias, cos, sin, &mut sc.q);
                rope(&input.k[qk_off..qk_off + dqk], k_bias, cos, sin, &mut sc.k);

                let v_off = ((bi * len + t) * heads + h) * dv;
                let v = &input.v[v_off..v_off + dv];

                let qk_dot =
                    sc.q.iter()
                        .zip(sc.k.iter())
                        .fold(T::ZERO, |a, (&x, &y)| a + x * y);
                let diag = d_skip + gamma * qk_dot;

                sc.y.iter_mut().for_each(|x| *x = T::ZERO);

                // --- THE BLOCKED CORE ---------------------------------------
                // One pass over S: read the old value into y, write the new one
                // back. Tiling is over the dqk (column) axis so each tile-row
                // slice is contiguous.
                for c0 in (0..dqk).step_by(TILE) {
                    let c1 = (c0 + TILE).min(dqk);
                    let qt = &sc.q[c0..c1];
                    let kt = &sc.k[c0..c1];
                    for (r, (row, &vr)) in sc.s.chunks_exact_mut(dqk).zip(v.iter()).enumerate() {
                        let sv = scale * vr;
                        let mut acc = T::ZERO;
                        for ((sj, &qj), &kj) in row[c0..c1].iter_mut().zip(qt.iter()).zip(kt.iter())
                        {
                            let old = *sj;
                            acc = acc + qj * old;
                            *sj = alpha * old + sv * kj;
                        }
                        sc.y[r] = sc.y[r] + acc;
                    }
                }

                // `alpha` applies to the whole q.S_old product, so it is folded
                // in once here rather than inside the tile loop.
                let o = t * dv;
                for ((slot, &yr), &vr) in
                    out_row[o..o + dv].iter_mut().zip(sc.y.iter()).zip(v.iter())
                {
                    *slot = alpha * yr + diag * vr;
                }

                for (bx, &kj) in sc.bx.iter_mut().zip(sc.k.iter()) {
                    *bx = scale * kj;
                }

                if let Some(z) = input.z {
                    let zs = &z[v_off..v_off + dv];
                    for (slot, &zi) in out_row[o..o + dv].iter_mut().zip(zs.iter()) {
                        *slot = *slot * zi.silu();
                    }
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
