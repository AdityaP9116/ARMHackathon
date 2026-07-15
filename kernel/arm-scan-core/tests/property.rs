//! Property tests: the f32 kernel is fuzzed against the same kernel run in
//! f64 across random shapes, flag combinations, and value ranges (including
//! the exp-underflow regime). Catches indexing bugs, flag mishandling, and
//! numeric blowups that fixed golden shapes might miss.

use proptest::prelude::*;

use arm_scan_core::{
    selective_scan, selective_scan_bidirectional, selective_scan_with_backend,
    selective_scan_with_options, selective_scan_with_state, Backend, ScanDims, ScanInput,
    ScanOptions, Threading,
};

#[derive(Debug, Clone)]
struct Case {
    dims: ScanDims,
    u: Vec<f32>,
    delta: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
    d_skip: Option<Vec<f32>>,
    z: Option<Vec<f32>>,
    delta_bias: Option<Vec<f32>>,
    delta_softplus: bool,
    reverse: bool,
}

fn vecf(n: usize, lo: f32, hi: f32) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(lo..hi, n)
}

fn case_strategy() -> impl Strategy<Value = Case> {
    (
        1usize..=2,
        1usize..=8,
        1usize..=32,
        1usize..=8,
        prop::bool::ANY,
    )
        .prop_flat_map(|(batch, dim, len, state, grouped)| {
            // groups must divide dim; use 2 groups when possible
            let groups = if grouped && dim % 2 == 0 { 2 } else { 1 };
            let bdl = batch * dim * len;
            let bgnl = batch * groups * state * len;
            (
                Just(ScanDims {
                    batch,
                    dim,
                    len,
                    state,
                    groups,
                }),
                vecf(bdl, -3.0, 3.0),                   // u
                vecf(bdl, -8.0, 8.0),                   // delta (raw)
                vecf(dim * state, -16.0, -0.01),        // a (negative, Mamba-like)
                vecf(bgnl, -3.0, 3.0),                  // b
                vecf(bgnl, -3.0, 3.0),                  // c
                prop::option::of(vecf(dim, -2.0, 2.0)), // d_skip
                prop::option::of(vecf(bdl, -4.0, 4.0)), // z
                prop::option::of(vecf(dim, -6.0, 1.0)), // delta_bias
                // paired: proptest's tuple Strategy impl stops at 10 elements,
                // and this tuple is already at 10.
                (prop::bool::ANY, prop::bool::ANY), // (delta_softplus, reverse)
            )
        })
        .prop_map(
            |(dims, u, delta, a, b, c, d_skip, z, delta_bias, (sp, reverse))| {
                let mut case = Case {
                    dims,
                    u,
                    delta,
                    a,
                    b,
                    c,
                    d_skip,
                    z,
                    delta_bias,
                    delta_softplus: sp,
                    reverse,
                };
                if !case.delta_softplus {
                    // raw delta is the timestep: must be positive like a
                    // real post-softplus value
                    for v in &mut case.delta {
                        *v = v.abs() * 0.01 + 1e-3;
                    }
                    case.delta_bias = None;
                }
                case
            },
        )
}

fn widen(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

/// Reverse the time axis of a contiguous tensor whose LAST dim is `len`
/// (u/delta/z: (B,D,L); b/c: (B,G,N,L); out: (B,D,L) — all are rows of `len`).
fn flip_time(v: &[f32], len: usize) -> Vec<f32> {
    v.chunks_exact(len)
        .flat_map(|row| row.iter().rev().copied())
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// `reverse: true` must equal flipping the time axis of every time-varying
    /// input, scanning FORWARD, and flipping the output back — the definition
    /// the fused traversal exists to implement without the copies.
    ///
    /// TWO STRENGTHS, and the difference is the interesting part:
    ///
    /// * **Scalar: BIT-for-bit.** The scalar path runs one uniform code path per
    ///   timestep, so both routes apply identical arithmetic to identical values
    ///   in identical order. Any difference at all is an indexing bug, not
    ///   rounding — which is exactly what we want to pin down. Note the two
    ///   routes also land on DIFFERENT chunk boundaries (forward-on-flipped
    ///   splits the flipped axis; reverse splits the original), so passing
    ///   bit-exactly additionally proves chunking never leaks into the math.
    ///
    /// * **NEON: tight tolerance, NOT bit-exact — and it cannot be.** Two of the
    ///   NEON passes process 4 timesteps at a time with a **scalar tail**:
    ///   `discretize_chunk` (softplus) and `epilogue_row` (SiLU). The vector and
    ///   tail branches use different implementations of the same function — the
    ///   NEON polynomial vs libm — which agree to ~1e-7 but not bit-for-bit.
    ///   Which branch a timestep takes depends on its ARRAY POSITION, and
    ///   flipping the array moves timesteps across that boundary. So at
    ///   `len = 31`, timestep 29 is in the scalar tail when scanned in place and
    ///   in the vector body when scanned flipped — same value, ~1 ulp apart.
    ///   This is a property of the existing forward kernel, not of `reverse`;
    ///   demanding bit-equality here would be asserting something false.
    ///
    /// Mirrors `tests/check_bidirectional_math.py`, which proves the same
    /// identity in numpy against an independently-written backward recurrence.
    #[test]
    fn reverse_matches_flip_forward_flip(case in case_strategy()) {
        let len = case.dims.len;
        let n_out = case.dims.batch * case.dims.dim * len;
        let n_last = case.dims.batch * case.dims.dim * case.dims.state;

        // flipped copies: A / d_skip / delta_bias have no time axis.
        let uf = flip_time(&case.u, len);
        let deltaf = flip_time(&case.delta, len);
        let bf = flip_time(&case.b, len);
        let cf = flip_time(&case.c, len);
        let zf = case.z.as_deref().map(|z| flip_time(z, len));

        for backend in [Backend::Scalar, Backend::Auto] {
            // (a) the fused backward traversal
            let mut out_rev = vec![0.0_f32; n_out];
            let mut last_rev = vec![0.0_f32; n_last];
            selective_scan_with_backend(
                &case.dims,
                &ScanInput {
                    u: &case.u, delta: &case.delta, a: &case.a, b: &case.b,
                    c: &case.c,
                    d_skip: case.d_skip.as_deref(),
                    z: case.z.as_deref(),
                    delta_bias: case.delta_bias.as_deref(),
                    delta_softplus: case.delta_softplus,
                    reverse: true,
                },
                &mut out_rev,
                Some(&mut last_rev),
                backend,
            ).unwrap();

            // (b) flip the inputs, scan forward, flip the output back
            let mut out_fwd = vec![0.0_f32; n_out];
            let mut last_fwd = vec![0.0_f32; n_last];
            selective_scan_with_backend(
                &case.dims,
                &ScanInput {
                    u: &uf, delta: &deltaf, a: &case.a, b: &bf, c: &cf,
                    d_skip: case.d_skip.as_deref(),
                    z: zf.as_deref(),
                    delta_bias: case.delta_bias.as_deref(),
                    delta_softplus: case.delta_softplus,
                    reverse: false,
                },
                &mut out_fwd,
                Some(&mut last_fwd),
                backend,
            ).unwrap();
            let out_fwd = flip_time(&out_fwd, len);

            if backend == Backend::Scalar {
                prop_assert!(
                    out_rev.iter().zip(&out_fwd)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "scalar: reverse != flip-forward-flip (must be bit-exact), \
                     dims={:?} softplus={}",
                    case.dims, case.delta_softplus
                );
                prop_assert!(
                    last_rev.iter().zip(&last_fwd)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "scalar: last_state differs (must be bit-exact), dims={:?}",
                    case.dims
                );
            } else {
                // Scale-relative, matching `auto_backend_matches_scalar`'s bar:
                // the gap is one SIMD-vs-libm transcendental, ~1e-7.
                let scale = out_rev.iter().fold(1.0_f32, |m, v| m.max(v.abs()));
                for (i, (r, f)) in out_rev.iter().zip(&out_fwd).enumerate() {
                    let rel = (r - f).abs() / scale;
                    prop_assert!(
                        rel < 1e-5,
                        "neon: out[{i}] reverse={r} flip={f} rel={rel:.3e} \
                         dims={:?}", case.dims
                    );
                }
                let ls = last_rev.iter().fold(1.0_f32, |m, v| m.max(v.abs()));
                for (i, (r, f)) in last_rev.iter().zip(&last_fwd).enumerate() {
                    let rel = (r - f).abs() / ls;
                    prop_assert!(
                        rel < 1e-5,
                        "neon: last_state[{i}] reverse={r} flip={f} \
                         rel={rel:.3e} dims={:?}", case.dims
                    );
                }
            }
        }
    }

    /// The fused bidirectional scan must be **bit-identical** to running the
    /// scan twice — once forward, once reversed. That is the definition it
    /// shares Pass A under: the shared exp/discretize/projection products hold
    /// exactly the values each standalone direction computes inline, consumed in
    /// the same order, so the result is not merely close but bit-for-bit equal.
    ///
    /// Checked on the SAME backend on both sides, so any difference is a fusion
    /// bug, not a SIMD-vs-libm gap. Runs on `Scalar` AND `Auto` — on aarch64 the
    /// latter exercises the NEON fused path against NEON two-scans; on x86 it is
    /// the scalar path again (harmless). Both threadings, so the new two-output
    /// parallel driver is covered (small proptest shapes keep Auto sequential).
    #[test]
    fn fused_bidirectional_matches_two_scans(case in case_strategy()) {
        let n_out = case.dims.batch * case.dims.dim * case.dims.len;
        let n_last = case.dims.batch * case.dims.dim * case.dims.state;
        let mk = |reverse| ScanInput {
            u: &case.u, delta: &case.delta, a: &case.a, b: &case.b, c: &case.c,
            d_skip: case.d_skip.as_deref(),
            z: case.z.as_deref(),
            delta_bias: case.delta_bias.as_deref(),
            delta_softplus: case.delta_softplus,
            reverse,
        };
        let bits_eq = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());

        for backend in [Backend::Scalar, Backend::Auto] {
            // reference: two standalone scans on this backend
            let mut ref_fwd = vec![0.0_f32; n_out];
            let mut ref_fwd_last = vec![0.0_f32; n_last];
            selective_scan_with_backend(
                &case.dims, &mk(false), &mut ref_fwd, Some(&mut ref_fwd_last), backend,
            ).unwrap();
            let mut ref_bwd = vec![0.0_f32; n_out];
            let mut ref_bwd_last = vec![0.0_f32; n_last];
            selective_scan_with_backend(
                &case.dims, &mk(true), &mut ref_bwd, Some(&mut ref_bwd_last), backend,
            ).unwrap();

            for threading in [Threading::Sequential, Threading::Rayon] {
                let mut f_fwd = vec![0.0_f32; n_out];
                let mut f_bwd = vec![0.0_f32; n_out];
                let mut f_fwd_last = vec![0.0_f32; n_last];
                let mut f_bwd_last = vec![0.0_f32; n_last];
                selective_scan_bidirectional(
                    &case.dims, &mk(false), &mut f_fwd, &mut f_bwd,
                    Some(&mut f_fwd_last), Some(&mut f_bwd_last),
                    ScanOptions { backend, threading },
                ).unwrap();

                prop_assert!(bits_eq(&f_fwd, &ref_fwd), "fwd differs ({backend:?}/{threading:?}) dims={:?}", case.dims);
                prop_assert!(bits_eq(&f_bwd, &ref_bwd), "bwd differs ({backend:?}/{threading:?}) dims={:?}", case.dims);
                prop_assert!(bits_eq(&f_fwd_last, &ref_fwd_last), "fwd last_state differs ({backend:?}/{threading:?}) dims={:?}", case.dims);
                prop_assert!(bits_eq(&f_bwd_last, &ref_bwd_last), "bwd last_state differs ({backend:?}/{threading:?}) dims={:?}", case.dims);
            }
        }
    }

    /// Guard against a vacuous pass: if `reverse` were silently ignored, the
    /// equivalence test above would still hold (both sides would be forward
    /// scans of... no — but the FFI/plumbing could still drop the flag). A
    /// reversed scan must actually differ from a forward one on the same input.
    #[test]
    fn reverse_actually_reverses(case in case_strategy()) {
        // L=1 is symmetric under reversal, and a constant sequence can be too.
        prop_assume!(case.dims.len >= 4);

        let n_out = case.dims.batch * case.dims.dim * case.dims.len;
        let mk = |reverse| ScanInput {
            u: &case.u, delta: &case.delta, a: &case.a, b: &case.b, c: &case.c,
            d_skip: case.d_skip.as_deref(),
            z: case.z.as_deref(),
            delta_bias: case.delta_bias.as_deref(),
            delta_softplus: case.delta_softplus,
            reverse,
        };

        let mut out_f = vec![0.0_f32; n_out];
        selective_scan(&case.dims, &mk(false), &mut out_f, None).unwrap();
        let mut out_r = vec![0.0_f32; n_out];
        selective_scan(&case.dims, &mk(true), &mut out_r, None).unwrap();

        let scale = out_f.iter().fold(1e-6_f32, |m, v| m.max(v.abs()));
        let max_rel = out_f.iter().zip(&out_r)
            .map(|(a, b)| (a - b).abs() / scale)
            .fold(0.0_f32, f32::max);
        prop_assert!(
            max_rel > 1e-4,
            "reverse produced ~the forward answer (max_rel={max_rel:.3e}) — \
             is the flag being dropped? dims={:?}", case.dims
        );
    }

    #[test]
    fn f32_matches_f64(case in case_strategy()) {
        let n_out = case.dims.batch * case.dims.dim * case.dims.len;
        let n_last = case.dims.batch * case.dims.dim * case.dims.state;

        let mut out32 = vec![0.0_f32; n_out];
        let mut last32 = vec![0.0_f32; n_last];
        selective_scan(
            &case.dims,
            &ScanInput {
                u: &case.u, delta: &case.delta, a: &case.a, b: &case.b,
                c: &case.c,
                d_skip: case.d_skip.as_deref(),
                z: case.z.as_deref(),
                delta_bias: case.delta_bias.as_deref(),
                delta_softplus: case.delta_softplus,
                reverse: case.reverse,
            },
            &mut out32,
            Some(&mut last32),
        ).unwrap();

        let (u, delta, a, b, c) = (
            widen(&case.u), widen(&case.delta), widen(&case.a),
            widen(&case.b), widen(&case.c),
        );
        let d_skip = case.d_skip.as_deref().map(widen);
        let z = case.z.as_deref().map(widen);
        let delta_bias = case.delta_bias.as_deref().map(widen);
        let mut out64 = vec![0.0_f64; n_out];
        selective_scan(
            &case.dims,
            &ScanInput {
                u: &u, delta: &delta, a: &a, b: &b, c: &c,
                d_skip: d_skip.as_deref(),
                z: z.as_deref(),
                delta_bias: delta_bias.as_deref(),
                delta_softplus: case.delta_softplus,
                reverse: case.reverse,
            },
            &mut out64,
            None,
        ).unwrap();

        for (i, (k, r)) in out32.iter().zip(out64.iter()).enumerate() {
            let err = (*k as f64 - r).abs();
            // f32 rounding over <=32 sequential steps with bounded values
            prop_assert!(
                err < 1e-3,
                "idx {i}: f32={k} f64={r} err={err:.3e} dims={:?}",
                case.dims
            );
            prop_assert!(k.is_finite(), "non-finite output at {i}: {k}");
        }
    }

    /// Dispatched backend (NEON on aarch64) vs scalar, on the same f32
    /// inputs. Differences come only from the polynomial exp and 4-lane
    /// summation order, so the scale-relative gap must stay tiny.
    #[test]
    fn auto_backend_matches_scalar(case in case_strategy()) {
        let n_out = case.dims.batch * case.dims.dim * case.dims.len;
        let n_last = case.dims.batch * case.dims.dim * case.dims.state;
        let input = ScanInput {
            u: &case.u, delta: &case.delta, a: &case.a, b: &case.b,
            c: &case.c,
            d_skip: case.d_skip.as_deref(),
            z: case.z.as_deref(),
            delta_bias: case.delta_bias.as_deref(),
            delta_softplus: case.delta_softplus,
            reverse: case.reverse,
        };

        let mut out_scalar = vec![0.0_f32; n_out];
        let mut last_scalar = vec![0.0_f32; n_last];
        selective_scan_with_backend(
            &case.dims, &input, &mut out_scalar, Some(&mut last_scalar),
            Backend::Scalar,
        ).unwrap();

        let mut out_auto = vec![0.0_f32; n_out];
        let mut last_auto = vec![0.0_f32; n_last];
        selective_scan_with_backend(
            &case.dims, &input, &mut out_auto, Some(&mut last_auto),
            Backend::Auto,
        ).unwrap();

        let scale = out_scalar.iter().fold(1.0_f32, |m, v| m.max(v.abs()));
        for (i, (s, a)) in out_scalar.iter().zip(out_auto.iter()).enumerate() {
            let rel = (s - a).abs() / scale;
            prop_assert!(
                rel < 1e-5,
                "out idx {i}: scalar={s} auto={a} rel={rel:.3e} dims={:?}",
                case.dims
            );
        }
        let ls_scale = last_scalar.iter().fold(1.0_f32, |m, v| m.max(v.abs()));
        for (i, (s, a)) in last_scalar.iter().zip(last_auto.iter()).enumerate() {
            let rel = (s - a).abs() / ls_scale;
            prop_assert!(
                rel < 1e-5,
                "last_state idx {i}: scalar={s} auto={a} rel={rel:.3e}",
            );
        }
    }

    /// Channels are independent, so forced-rayon output must be
    /// BIT-IDENTICAL to sequential — any divergence means rows are not
    /// disjoint or scheduling leaked into the math. Checked for both
    /// backends via Auto (NEON on aarch64, scalar elsewhere) and Scalar.
    #[test]
    fn parallel_is_bit_identical(case in case_strategy()) {
        let n_out = case.dims.batch * case.dims.dim * case.dims.len;
        let n_last = case.dims.batch * case.dims.dim * case.dims.state;
        let input = ScanInput {
            u: &case.u, delta: &case.delta, a: &case.a, b: &case.b,
            c: &case.c,
            d_skip: case.d_skip.as_deref(),
            z: case.z.as_deref(),
            delta_bias: case.delta_bias.as_deref(),
            delta_softplus: case.delta_softplus,
            reverse: case.reverse,
        };

        for backend in [Backend::Auto, Backend::Scalar] {
            let mut out_seq = vec![0.0_f32; n_out];
            let mut last_seq = vec![0.0_f32; n_last];
            selective_scan_with_options(
                &case.dims, &input, &mut out_seq, Some(&mut last_seq),
                ScanOptions { backend, threading: Threading::Sequential },
            ).unwrap();

            let mut out_par = vec![0.0_f32; n_out];
            let mut last_par = vec![0.0_f32; n_last];
            selective_scan_with_options(
                &case.dims, &input, &mut out_par, Some(&mut last_par),
                ScanOptions { backend, threading: Threading::Rayon },
            ).unwrap();

            prop_assert!(
                out_seq.iter().zip(&out_par).all(|(a, b)| a.to_bits() == b.to_bits()),
                "out differs between sequential and rayon ({backend:?}) dims={:?}",
                case.dims
            );
            prop_assert!(
                last_seq.iter().zip(&last_par).all(|(a, b)| a.to_bits() == b.to_bits()),
                "last_state differs between sequential and rayon ({backend:?})"
            );
        }
    }
}

/// Streaming contract: scanning a prefix, then resuming the rest from the
/// prefix's `last_state` fed back as `h0`, must reproduce the one-shot scan of
/// the whole sequence. This is what autoregressive decode relies on. Checked
/// for both backends, at N=16 (the fast path) and N=8 (the general path), with
/// a mid-sequence split that is not a chunk boundary.
#[test]
fn streaming_matches_oneshot() {
    fn fill(v: &mut [f32], mut seed: u32, lo: f32, hi: f32) {
        for x in v.iter_mut() {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *x = lo + (hi - lo) * (seed as f32 / u32::MAX as f32);
        }
    }
    // Extract timesteps [t0, t1) from a (rows, len) row-major tensor.
    fn slice_l(src: &[f32], rows: usize, len: usize, t0: usize, t1: usize) -> Vec<f32> {
        let w = t1 - t0;
        let mut out = vec![0.0_f32; rows * w];
        for r in 0..rows {
            out[r * w..(r + 1) * w].copy_from_slice(&src[r * len + t0..r * len + t1]);
        }
        out
    }

    fn check(state: usize) {
        let (batch, dim, len, split) = (1usize, 4usize, 16usize, 7usize);
        let bdl = batch * dim * len;
        let bnl = batch * state * len;

        let mut u = vec![0.0_f32; bdl];
        let mut delta = vec![0.0_f32; bdl];
        let mut a = vec![0.0_f32; dim * state];
        let mut b = vec![0.0_f32; bnl];
        let mut c = vec![0.0_f32; bnl];
        let mut z = vec![0.0_f32; bdl];
        let mut d_skip = vec![0.0_f32; dim];
        let mut bias = vec![0.0_f32; dim];
        fill(&mut u, 1, -3.0, 3.0);
        fill(&mut delta, 2, -2.0, 2.0);
        fill(&mut a, 3, -16.0, -0.5);
        fill(&mut b, 4, -3.0, 3.0);
        fill(&mut c, 5, -3.0, 3.0);
        fill(&mut z, 6, -3.0, 3.0);
        fill(&mut d_skip, 7, -1.0, 1.0);
        fill(&mut bias, 8, -6.0, -3.0);

        let dims = ScanDims {
            batch,
            dim,
            len,
            state,
            groups: 1,
        };
        let full = ScanInput {
            u: &u,
            delta: &delta,
            a: &a,
            b: &b,
            c: &c,
            d_skip: Some(&d_skip),
            z: Some(&z),
            delta_bias: Some(&bias),
            delta_softplus: true,
            reverse: false,
        };

        for backend in [Backend::Scalar, Backend::Auto] {
            let opts = ScanOptions {
                backend,
                threading: Threading::Sequential,
            };

            let mut out_full = vec![0.0_f32; bdl];
            selective_scan_with_state(&dims, &full, &mut out_full, None, None, opts).unwrap();

            // Part 1: timesteps [0, split), capturing the intermediate state.
            let (u1, d1, b1, c1, z1) = (
                slice_l(&u, dim, len, 0, split),
                slice_l(&delta, dim, len, 0, split),
                slice_l(&b, state, len, 0, split),
                slice_l(&c, state, len, 0, split),
                slice_l(&z, dim, len, 0, split),
            );
            let dims1 = ScanDims { len: split, ..dims };
            let in1 = ScanInput {
                u: &u1,
                delta: &d1,
                a: &a,
                b: &b1,
                c: &c1,
                d_skip: Some(&d_skip),
                z: Some(&z1),
                delta_bias: Some(&bias),
                delta_softplus: true,
                reverse: false,
            };
            let mut out1 = vec![0.0_f32; dim * split];
            let mut mid = vec![0.0_f32; dim * state];
            selective_scan_with_state(&dims1, &in1, &mut out1, Some(&mut mid), None, opts).unwrap();

            // Part 2: timesteps [split, len), resuming from `mid` as h0.
            let rem = len - split;
            let (u2, d2, b2, c2, z2) = (
                slice_l(&u, dim, len, split, len),
                slice_l(&delta, dim, len, split, len),
                slice_l(&b, state, len, split, len),
                slice_l(&c, state, len, split, len),
                slice_l(&z, dim, len, split, len),
            );
            let dims2 = ScanDims { len: rem, ..dims };
            let in2 = ScanInput {
                u: &u2,
                delta: &d2,
                a: &a,
                b: &b2,
                c: &c2,
                d_skip: Some(&d_skip),
                z: Some(&z2),
                delta_bias: Some(&bias),
                delta_softplus: true,
                reverse: false,
            };
            let mut out2 = vec![0.0_f32; dim * rem];
            selective_scan_with_state(&dims2, &in2, &mut out2, None, Some(&mid), opts).unwrap();

            for dd in 0..dim {
                for tt in 0..split {
                    let (f, s) = (out_full[dd * len + tt], out1[dd * split + tt]);
                    assert!(
                        (f - s).abs() < 1e-6,
                        "part1 {backend:?} N={state} d={dd} t={tt}: full={f} stream={s}"
                    );
                }
                for tt in 0..rem {
                    let (f, s) = (out_full[dd * len + split + tt], out2[dd * rem + tt]);
                    assert!(
                        (f - s).abs() < 1e-6,
                        "part2 {backend:?} N={state} d={dd} t={tt}: full={f} stream={s}"
                    );
                }
            }
        }
    }

    check(16);
    check(8);
}

/// Explicitly requesting NEON must work on aarch64 and error elsewhere.
#[test]
fn neon_backend_availability() {
    use arm_scan_core::ScanError;
    let dims = ScanDims {
        batch: 1,
        dim: 1,
        len: 1,
        state: 1,
        groups: 1,
    };
    let one = [0.5_f32];
    let input = ScanInput {
        u: &one,
        delta: &one,
        a: &[-1.0],
        b: &one,
        c: &one,
        d_skip: None,
        z: None,
        delta_bias: None,
        delta_softplus: false,
        reverse: false,
    };
    let mut out = [0.0_f32];
    let res = selective_scan_with_backend(&dims, &input, &mut out, None, Backend::Neon);
    if cfg!(target_arch = "aarch64") {
        assert!(res.is_ok(), "NEON must be available on aarch64: {res:?}");
    } else {
        assert_eq!(res, Err(ScanError::BackendUnavailable(Backend::Neon)));
    }
}

/// Shape validation must reject wrong slice lengths rather than index OOB.
#[test]
fn validation_rejects_bad_shapes() {
    let dims = ScanDims {
        batch: 1,
        dim: 2,
        len: 3,
        state: 2,
        groups: 1,
    };
    let ok = vec![0.0_f32; 6];
    let a = vec![0.0_f32; 4];
    let bc = vec![0.0_f32; 6];
    let mut out = vec![0.0_f32; 6];

    // u too short
    let bad_u = vec![0.0_f32; 5];
    let input = ScanInput {
        u: &bad_u,
        delta: &ok,
        a: &a,
        b: &bc,
        c: &bc,
        d_skip: None,
        z: None,
        delta_bias: None,
        delta_softplus: false,
        reverse: false,
    };
    assert!(selective_scan(&dims, &input, &mut out, None).is_err());

    // groups don't divide dim
    let dims_bad = ScanDims {
        batch: 1,
        dim: 3,
        len: 1,
        state: 1,
        groups: 2,
    };
    let three = vec![0.0_f32; 3];
    let two = vec![0.0_f32; 2];
    let mut out3 = vec![0.0_f32; 3];
    let input = ScanInput {
        u: &three,
        delta: &three,
        a: &three,
        b: &two,
        c: &two,
        d_skip: None,
        z: None,
        delta_bias: None,
        delta_softplus: false,
        reverse: false,
    };
    assert!(selective_scan(&dims_bad, &input, &mut out3, None).is_err());
}
