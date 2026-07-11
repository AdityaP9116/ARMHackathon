//! Property tests: the f32 kernel is fuzzed against the same kernel run in
//! f64 across random shapes, flag combinations, and value ranges (including
//! the exp-underflow regime). Catches indexing bugs, flag mishandling, and
//! numeric blowups that fixed golden shapes might miss.

use proptest::prelude::*;

use arm_scan_core::{selective_scan, selective_scan_with_backend, Backend, ScanDims, ScanInput};

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
                prop::bool::ANY,                        // delta_softplus
            )
        })
        .prop_map(|(dims, u, delta, a, b, c, d_skip, z, delta_bias, sp)| {
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
        })
}

fn widen(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

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
    };
    assert!(selective_scan(&dims_bad, &input, &mut out3, None).is_err());
}
