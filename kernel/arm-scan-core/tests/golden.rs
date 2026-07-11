//! Golden-vector tests: the f32 kernel must reproduce the float64 PyTorch
//! reference (tests/golden/*.npz, see tests/README.md at the repo root) to
//! within the acceptance tolerance, on every case in the manifest.

use std::fs::File;
use std::path::PathBuf;

use ndarray::{ArrayD, IxDyn, OwnedRepr};
use ndarray_npy::NpzReader;
use serde::Deserialize;

use arm_scan_core::{selective_scan_with_backend, Backend, ScanDims, ScanInput};

/// Kernel acceptance tolerance from INTEGRATION_PLAN.md.
const MAX_ABS: f64 = 1e-4;
/// A correct f32 kernel should land near the recorded torch-f32 floor, not
/// orders of magnitude above it (different summation order costs a small
/// constant factor, not 100x).
const FLOOR_FACTOR: f64 = 50.0;

#[derive(Deserialize)]
struct CaseMeta {
    name: String,
    batch: usize,
    dim: usize,
    len: usize,
    state: usize,
    groups: Option<usize>,
    delta_softplus: bool,
    f32_max_abs_err: f64,
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden")
}

fn read_arr<T: ndarray_npy::ReadableElement>(
    npz: &mut NpzReader<File>,
    name: &str,
) -> Option<ArrayD<T>> {
    npz.by_name::<OwnedRepr<T>, IxDyn>(&format!("{name}.npy"))
        .or_else(|_| npz.by_name::<OwnedRepr<T>, IxDyn>(name))
        .ok()
}

fn to_vec_f32(a: ArrayD<f32>) -> Vec<f32> {
    a.as_standard_layout().iter().copied().collect()
}

struct CaseResult {
    out: Vec<f32>,
    out_err: f64,
    last_rel_err: f64,
}

fn run_case(meta: &CaseMeta, backend: Backend) -> CaseResult {
    let path = golden_dir().join(format!("{}.npz", meta.name));
    let mut npz = NpzReader::new(
        File::open(&path).unwrap_or_else(|e| panic!("cannot open {}: {e}", path.display())),
    )
    .expect("bad npz");

    let u = to_vec_f32(read_arr::<f32>(&mut npz, "u").expect("u"));
    let delta = to_vec_f32(read_arr::<f32>(&mut npz, "delta").expect("delta"));
    let a = to_vec_f32(read_arr::<f32>(&mut npz, "A").expect("A"));
    let b = to_vec_f32(read_arr::<f32>(&mut npz, "B").expect("B"));
    let c = to_vec_f32(read_arr::<f32>(&mut npz, "C").expect("C"));
    let d_skip = read_arr::<f32>(&mut npz, "D_skip").map(to_vec_f32);
    let z = read_arr::<f32>(&mut npz, "z").map(to_vec_f32);
    let delta_bias = read_arr::<f32>(&mut npz, "delta_bias").map(to_vec_f32);
    let out_f64 = read_arr::<f64>(&mut npz, "out_f64").expect("out_f64");
    let last_f64 = read_arr::<f64>(&mut npz, "last_state_f64").expect("last_state_f64");

    let dims = ScanDims {
        batch: meta.batch,
        dim: meta.dim,
        len: meta.len,
        state: meta.state,
        groups: meta.groups.unwrap_or(1),
    };
    let input = ScanInput {
        u: &u,
        delta: &delta,
        a: &a,
        b: &b,
        c: &c,
        d_skip: d_skip.as_deref(),
        z: z.as_deref(),
        delta_bias: delta_bias.as_deref(),
        delta_softplus: meta.delta_softplus,
    };
    let mut out = vec![0.0_f32; dims.batch * dims.dim * dims.len];
    let mut last = vec![0.0_f32; dims.batch * dims.dim * dims.state];
    selective_scan_with_backend(&dims, &input, &mut out, Some(&mut last), backend)
        .expect("scan failed");

    let out_err = out
        .iter()
        .zip(out_f64.iter())
        .map(|(k, r)| (*k as f64 - r).abs())
        .fold(0.0_f64, f64::max);
    // last_state error is judged relative to the state's own magnitude
    let last_scale = last_f64.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
    let last_err = last
        .iter()
        .zip(last_f64.iter())
        .map(|(k, r)| (*k as f64 - r).abs())
        .fold(0.0_f64, f64::max)
        / last_scale;
    CaseResult {
        out,
        out_err,
        last_rel_err: last_err,
    }
}

#[test]
fn all_golden_cases() {
    let manifest_path = golden_dir().join("manifest.json");
    let manifest: Vec<CaseMeta> =
        serde_json::from_reader(File::open(&manifest_path).unwrap_or_else(|e| {
            panic!(
                "cannot open {} (run `python tests/gen_golden.py` first): {e}",
                manifest_path.display()
            )
        }))
        .expect("bad manifest");
    assert!(!manifest.is_empty(), "empty manifest");

    let auto_is_neon = cfg!(target_arch = "aarch64");
    println!(
        "backend Auto resolves to {} on this host",
        if auto_is_neon { "NEON" } else { "scalar" }
    );

    let mut failures = Vec::new();
    for meta in &manifest {
        let scalar = run_case(meta, Backend::Scalar);
        let auto = run_case(meta, Backend::Auto);

        // parity between the dispatched backend and the scalar baseline
        // (on aarch64 this is the NEON-vs-scalar cross-check on real data)
        let out_scale = auto.out.iter().fold(1.0_f32, |m, v| m.max(v.abs())) as f64;
        let parity = scalar
            .out
            .iter()
            .zip(auto.out.iter())
            .map(|(s, a)| (*s as f64 - *a as f64).abs())
            .fold(0.0_f64, f64::max)
            / out_scale;

        let floor_bound = (FLOOR_FACTOR * meta.f32_max_abs_err).max(1e-6);
        let ok = |r: &CaseResult| {
            r.out_err < MAX_ABS && r.out_err < floor_bound && r.last_rel_err < 1e-4
        };
        let pass = ok(&scalar) && ok(&auto) && parity < 1e-5;
        println!(
            "  {:24} scalar={:.3e} auto={:.3e} (floor {:.3e})  parity={:.3e}  last_rel={:.3e}  {}",
            meta.name,
            scalar.out_err,
            auto.out_err,
            meta.f32_max_abs_err,
            parity,
            auto.last_rel_err,
            if pass { "ok" } else { "FAIL" }
        );
        if !pass {
            failures.push(meta.name.clone());
        }
    }
    assert!(failures.is_empty(), "golden failures: {failures:?}");
}
