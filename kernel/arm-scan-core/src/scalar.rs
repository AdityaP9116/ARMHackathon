//! Scalar reference implementation of the selective scan.
//!
//! Direct transcription of the recurrence — clarity over speed. This is the
//! correctness baseline every optimized variant (NEON, chunked, threaded)
//! must reproduce; it is also the portable fallback on non-Arm hosts.
//! Channel iteration goes through [`crate::parallel::for_each_channel`], so
//! even the fallback scales across cores.

use crate::{Float, ScanDims, ScanInput, Threading};

pub(crate) fn scan<T: Float>(
    dims: &ScanDims,
    input: &ScanInput<'_, T>,
    h0: Option<&[T]>,
    out: &mut [T],
    last_state: Option<&mut [T]>,
    threading: Threading,
) {
    let ScanDims {
        batch: _,
        dim,
        len,
        state,
        groups,
    } = *dims;
    let group_size = dim / groups;

    crate::parallel::for_each_channel(
        len,
        state,
        out,
        last_state,
        threading,
        || vec![T::ZERO; state],
        |h, ch_idx, out_row, last| {
            let (bi, d) = (ch_idx / dim, ch_idx % dim);
            let a_row = &input.a[d * state..(d + 1) * state];
            let bias = input.delta_bias.map_or(T::ZERO, |v| v[d]);
            let d_skip = input.d_skip.map(|v| v[d]);
            // base offset of b/c row (bi, g, n=0, t=0); n advances by `len`
            let bc_base = (bi * groups + d / group_size) * state * len;
            let row = ch_idx * len;
            let u_row = &input.u[row..row + len];
            let delta_row = &input.delta[row..row + len];
            let z_row = input.z.map(|z| &z[row..row + len]);

            // `h0` seeds the state; `reverse` picks the traversal direction.
            // The two are orthogonal: a backward scan resumed from a prior
            // state is perfectly coherent (and is what SS2D will want).
            match h0 {
                Some(h0) => h.copy_from_slice(&h0[ch_idx * state..(ch_idx + 1) * state]),
                None => h.fill(T::ZERO),
            }
            for i in 0..len {
                // The ONLY thing `reverse` changes: which timestep this step of
                // the recurrence consumes. Output still lands at index `t`, and
                // the pointwise D-skip / z-gate still read index `t`, so the
                // layout is untouched — see `ScanInput::reverse`.
                let t = if input.reverse { len - 1 - i } else { i };

                let mut dt = delta_row[t] + bias;
                if input.delta_softplus {
                    dt = dt.softplus();
                }
                let dt_u = dt * u_row[t];

                let mut y = T::ZERO;
                for (n, h_n) in h.iter_mut().enumerate() {
                    let bc_idx = bc_base + n * len + t;
                    let new = (dt * a_row[n]).exp() * *h_n + dt_u * input.b[bc_idx];
                    *h_n = new;
                    y = y + input.c[bc_idx] * new;
                }

                if let Some(ds) = d_skip {
                    y = y + ds * u_row[t];
                }
                if let Some(z) = z_row {
                    y = y * z[t].silu();
                }
                out_row[t] = y;
            }

            if let Some(ls) = last {
                ls.copy_from_slice(h);
            }
        },
    );
}

/// Per-channel scratch for the fused bidirectional scan: the shared,
/// direction-independent Pass-A products (materialized for the whole row so
/// both directions can read them) plus one recurrence state.
struct BidirScratch<T> {
    abar: Vec<T>, // len * state — exp(dt*A), computed once
    bbar: Vec<T>, // len * state — dt*u*B, computed once
    h: Vec<T>,    // state — the recurrence state, reused per direction
}

/// Fused bidirectional scan (scalar). Computes Pass A — discretize, exp, input
/// projection — **once** per channel, then runs the recurrence in both time
/// directions over those shared products. Because Pass A is pointwise in time
/// (direction-independent) and the exp is ~85% of the work, sharing it is the
/// point: this is one exp sweep + two cheap FMA sweeps, versus two full scans.
///
/// Output is **bit-identical** to two standalone scans — `selective_scan` with
/// `reverse: false` into `out_fwd`, and `reverse: true` into `out_bwd` — because
/// the shared products hold exactly the same values the standalone paths compute
/// inline, consumed in the same order. Enforced by
/// `fused_bidirectional_matches_two_scans` in `tests/property.rs`.
///
/// No `h0`: bidirectional models are non-causal and seed both directions from
/// zero. `input.reverse` is ignored (both directions are produced regardless).
pub(crate) fn scan_bidirectional<T: Float>(
    dims: &ScanDims,
    input: &ScanInput<'_, T>,
    out_fwd: &mut [T],
    out_bwd: &mut [T],
    last_fwd: Option<&mut [T]>,
    last_bwd: Option<&mut [T]>,
    threading: Threading,
) {
    let ScanDims {
        batch: _,
        dim,
        len,
        state,
        groups,
    } = *dims;
    let group_size = dim / groups;

    crate::parallel::for_each_channel_bidir(
        len,
        state,
        out_fwd,
        out_bwd,
        last_fwd,
        last_bwd,
        threading,
        || BidirScratch {
            abar: vec![T::ZERO; len * state],
            bbar: vec![T::ZERO; len * state],
            h: vec![T::ZERO; state],
        },
        |scratch, ch_idx, out_fwd_row, out_bwd_row, last_f, last_b| {
            let (bi, d) = (ch_idx / dim, ch_idx % dim);
            let a_row = &input.a[d * state..(d + 1) * state];
            let bias = input.delta_bias.map_or(T::ZERO, |v| v[d]);
            let d_skip = input.d_skip.map(|v| v[d]);
            let bc_base = (bi * groups + d / group_size) * state * len;
            let row = ch_idx * len;
            let u_row = &input.u[row..row + len];
            let delta_row = &input.delta[row..row + len];
            let z_row = input.z.map(|z| &z[row..row + len]);

            // Pass A (shared): discretize + exp + input projection, ONCE.
            // Identical values to what each standalone direction computes inline.
            for t in 0..len {
                let mut dt = delta_row[t] + bias;
                if input.delta_softplus {
                    dt = dt.softplus();
                }
                let dt_u = dt * u_row[t];
                for (n, &a_n) in a_row.iter().enumerate() {
                    let bc_idx = bc_base + n * len + t;
                    scratch.abar[t * state + n] = (dt * a_n).exp();
                    scratch.bbar[t * state + n] = dt_u * input.b[bc_idx];
                }
            }

            // Pass B, both directions. `t` is the timestep consumed at step `i`;
            // output still lands at index `t` (layout never flips) — matching
            // scalar::scan's reverse handling exactly.
            let BidirScratch { abar, bbar, h } = &mut *scratch;
            for (reverse, out_row, last) in
                [(false, out_fwd_row, last_f), (true, out_bwd_row, last_b)]
            {
                h.fill(T::ZERO);
                for i in 0..len {
                    let t = if reverse { len - 1 - i } else { i };
                    let mut y = T::ZERO;
                    for (n, h_n) in h.iter_mut().enumerate() {
                        let new = abar[t * state + n] * *h_n + bbar[t * state + n];
                        *h_n = new;
                        y = y + input.c[bc_base + n * len + t] * new;
                    }
                    if let Some(ds) = d_skip {
                        y = y + ds * u_row[t];
                    }
                    if let Some(z) = z_row {
                        y = y * z[t].silu();
                    }
                    out_row[t] = y;
                }
                if let Some(ls) = last {
                    ls.copy_from_slice(h);
                }
            }
        },
    );
}

#[cfg(test)]
mod tests {
    use crate::{Float, ScanDims, ScanInput};

    /// Single-timestep case checked against the closed-form recurrence,
    /// worked in f64 independently of any golden file:
    ///   h = exp(dt*a) * 0 + dt*u*b ;  y = c*h + d_skip*u ;  out = y*silu(z)
    #[test]
    fn hand_computed_single_step() {
        let dims = ScanDims {
            batch: 1,
            dim: 1,
            len: 1,
            state: 2,
            groups: 1,
        };
        let (u, dt_raw, a, b, c, ds, z, bias) = (
            0.7_f64,
            -1.2_f64,
            [-1.5_f64, -4.0],
            [0.3_f64, -0.8],
            [1.1_f64, 0.4],
            0.9_f64,
            0.25_f64,
            0.5_f64,
        );
        let input = ScanInput {
            u: &[u],
            delta: &[dt_raw],
            a: &a,
            b: &b,
            c: &c,
            d_skip: Some(&[ds]),
            z: Some(&[z]),
            delta_bias: Some(&[bias]),
            delta_softplus: true,
            reverse: false,
        };
        let mut out = [0.0_f64];
        let mut last = [0.0_f64; 2];
        selective_scan_for_test(&dims, &input, &mut out, &mut last);

        let dt = ((dt_raw + bias).exp()).ln_1p();
        let h: Vec<f64> = (0..2).map(|n| dt * u * b[n]).collect();
        let y = c[0] * h[0] + c[1] * h[1] + ds * u;
        let expected = y * (z / (1.0 + (-z).exp()));
        assert!(
            (out[0] - expected).abs() < 1e-15,
            "{} vs {expected}",
            out[0]
        );
        assert!((last[0] - h[0]).abs() < 1e-15);
        assert!((last[1] - h[1]).abs() < 1e-15);
    }

    /// Two timesteps: the state must carry over with the exp(dt*a) decay.
    #[test]
    fn hand_computed_state_carryover() {
        let dims = ScanDims {
            batch: 1,
            dim: 1,
            len: 2,
            state: 1,
            groups: 1,
        };
        let (u, dt, a, b, c) = (
            [1.0_f64, 2.0],
            [0.1_f64, 0.2],
            [-2.0_f64],
            [1.0_f64, 3.0],
            [1.0_f64, 1.0],
        );
        let input = ScanInput {
            u: &u,
            delta: &dt,
            a: &a,
            b: &b,
            c: &c,
            d_skip: None,
            z: None,
            delta_bias: None,
            delta_softplus: false,
            reverse: false,
        };
        let mut out = [0.0_f64; 2];
        selective_scan_for_test(&dims, &input, &mut out, &mut []);

        let h1 = 0.1 * 1.0 * 1.0; // dt*u*b
        let h2 = (0.2_f64 * -2.0).exp() * h1 + 0.2 * 2.0 * 3.0;
        assert!((out[0] - h1).abs() < 1e-15);
        assert!((out[1] - h2).abs() < 1e-15);
    }

    fn selective_scan_for_test<T: Float>(
        dims: &ScanDims,
        input: &ScanInput<'_, T>,
        out: &mut [T],
        last: &mut [T],
    ) {
        let last_opt = if last.is_empty() { None } else { Some(last) };
        crate::selective_scan(dims, input, out, last_opt).unwrap();
    }
}
