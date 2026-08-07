//! Mamba-3 golden test: the Rust kernel must reproduce ground truth captured
//! from the **official** `mamba3_siso_combined` Triton kernel
//! (`tests/golden/mamba3/`, see `MAMBA3_IMPLEMENTATION_PLAN.md` §3).
//!
//! # Tolerance — read before changing it
//!
//! The official kernel emits **bf16**, whose relative epsilon is ~0.4% (8
//! mantissa bits). An f64 reference cannot agree with a bf16-stored result more
//! closely than bf16 can express, so the plan's original "< 1e-4" gate is
//! unsatisfiable and was corrected. We compare in units of **one bf16 ULP at
//! the tensor's scale**, matching `tests/verify_golden_mamba3.py`.
//!
//! The bound is set from evidence, not chosen to pass: golden `_04` is L=1, so
//! it accumulates nothing, and it still sits at ~0.45 ULP purely from output
//! quantisation. A structurally wrong kernel is off by thousands — every failing
//! iteration during development was.
//!
//! Note the goldens store **post-cast** inputs: the official kernel downcasts
//! `Q/K/V/Trap/Angles/Z` to bf16 on entry, so these are the values it actually
//! consumed, not the ones it was handed.

use std::fs::File;
use std::path::PathBuf;

use ndarray::{ArrayD, IxDyn, OwnedRepr};
use ndarray_npy::NpzReader;

use arm_scan_core::{
    mamba3_scan_with_options, Backend, Mamba3Dims, Mamba3Input, ScanOptions, Threading,
};

/// bf16 keeps 8 mantissa bits.
const BF16_EPS: f64 = 1.0 / 256.0;
/// Worst-case bf16 ULPs (at tensor scale) the kernel may differ by.
const MAX_ULPS: f64 = 8.0;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/mamba3")
}

fn read(npz: &mut NpzReader<File>, name: &str) -> Option<ArrayD<f32>> {
    npz.by_name::<OwnedRepr<f32>, IxDyn>(&format!("{name}.npy"))
        .or_else(|_| npz.by_name::<OwnedRepr<f32>, IxDyn>(name))
        .ok()
}

fn vec_of(a: &ArrayD<f32>) -> Vec<f32> {
    a.as_standard_layout().iter().copied().collect()
}

/// Reproduce the Python pre-pass: `theta = cumsum(tanh(angle) * PI * dt)`, then
/// cos/sin. Upstream runs this as a separate kernel (`angle_dt_fwd`) before the
/// scan, and we mirror that split — so the test has to do it too.
fn angles_to_cos_sin(
    angles: &ArrayD<f32>,
    dt: &[f32],
    b: usize,
    l: usize,
    h: usize,
    half: usize,
) -> (Vec<f32>, Vec<f32>) {
    let a = vec_of(angles);
    let r = angles.shape()[angles.ndim() - 1];
    let mut cos = vec![0.0f32; b * l * h * half];
    let mut sin = vec![0.0f32; b * l * h * half];
    for bi in 0..b {
        for hi in 0..h {
            let mut acc = vec![0.0f64; half];
            for t in 0..l {
                let dt_v = dt[(bi * h + hi) * l + t] as f64;
                for j in 0..half {
                    if j < r {
                        let ang = a[((bi * l + t) * h + hi) * r + j] as f64;
                        acc[j] += ang.tanh() * std::f64::consts::PI * dt_v;
                    }
                    let o = ((bi * l + t) * h + hi) * half + j;
                    cos[o] = acc[j].cos() as f32;
                    sin[o] = acc[j].sin() as f32;
                }
            }
        }
    }
    (cos, sin)
}

struct CaseResult {
    /// Worst deviation from the golden, in bf16 ULPs at tensor scale.
    ulps: f64,
    scale: f64,
    len: usize,
    /// Raw kernel output, head-major. Kept so the threading test can compare
    /// the actual numbers rather than a summary statistic — a bit-identity
    /// claim checked through `max()` would pass for two different outputs that
    /// happened to share a worst case.
    out: Vec<f32>,
}

fn run_case(path: &PathBuf, backend: Backend, threading: Threading) -> CaseResult {
    let mut npz = NpzReader::new(File::open(path).expect("open npz")).expect("npz");
    let q = read(&mut npz, "kw_Q").expect("Q");
    let k = read(&mut npz, "kw_K").expect("K");
    let v = read(&mut npz, "kw_V").expect("V");
    let adt = read(&mut npz, "kw_ADT").expect("ADT");
    let dt = read(&mut npz, "kw_DT").expect("DT");
    let trap = read(&mut npz, "kw_Trap").expect("Trap");
    let qb = read(&mut npz, "kw_Q_bias").expect("Q_bias");
    let kb = read(&mut npz, "kw_K_bias").expect("K_bias");
    let angles = read(&mut npz, "kw_Angles").expect("Angles");
    let d = read(&mut npz, "kw_D");
    let z = read(&mut npz, "kw_Z");
    let gold = read(&mut npz, "out").expect("out");

    // V is (batch, len, heads, dv); Q is (batch, len, groups, dqk).
    let (b, l, h, dv) = (v.shape()[0], v.shape()[1], v.shape()[2], v.shape()[3]);
    let dqk = q.shape()[3];
    let half = dqk / 2;

    let dt_v = vec_of(&dt);
    let (cos, sin) = angles_to_cos_sin(&angles, &dt_v, b, l, h, half);
    let (qv, kv, vv) = (vec_of(&q), vec_of(&k), vec_of(&v));
    let (adtv, trapv) = (vec_of(&adt), vec_of(&trap));
    let (qbv, kbv) = (vec_of(&qb), vec_of(&kb));
    let dv_skip = d.as_ref().map(vec_of);
    let zv = z.as_ref().map(vec_of);

    let dims = Mamba3Dims {
        batch: b,
        heads: h,
        dv,
        dqk,
        len: l,
    };
    let input = Mamba3Input {
        q: &qv,
        k: &kv,
        v: &vv,
        adt: &adtv,
        dt: &dt_v,
        trap: &trapv,
        q_bias: &qbv,
        k_bias: &kbv,
        cos: &cos,
        sin: &sin,
        d_skip: dv_skip.as_deref(),
        z: zv.as_deref(),
        reverse: false,
    };
    // Kernel output is head-major (batch, heads, len, dv); the goldens are
    // time-major (batch, len, heads, dv). See `parallel::for_each_head`.
    let mut out = vec![0.0f32; b * h * l * dv];
    mamba3_scan_with_options(
        &dims,
        &input,
        &mut out,
        None,
        None,
        ScanOptions { backend, threading },
    )
    .expect("mamba3 scan");

    let g = vec_of(&gold);
    let scale = g.iter().fold(0.0f64, |m, &x| m.max((x as f64).abs()));
    let ulp = scale.max(1e-30) * BF16_EPS;
    let mut worst = 0.0f64;
    for bi in 0..b {
        for t in 0..l {
            for hi in 0..h {
                for r in 0..dv {
                    let got = out[((bi * h + hi) * l + t) * dv + r] as f64;
                    let want = g[((bi * l + t) * h + hi) * dv + r] as f64;
                    // Compare like with like: the golden is bf16-quantised, so
                    // round ours the same way before differencing.
                    let got_bf16 = bf16_round(got);
                    worst = worst.max((got_bf16 - want).abs());
                }
            }
        }
    }
    CaseResult {
        ulps: worst / ulp,
        scale,
        len: l,
        out,
    }
}

/// Round an f64 to the nearest bf16 value (round-to-nearest-even on the top 16
/// bits of the f32 representation), returned as f64.
fn bf16_round(x: f64) -> f64 {
    let f = x as f32;
    let bits = f.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = (bits + 0x7fff + lsb) & 0xffff_0000;
    f32::from_bits(rounded) as f64
}

fn cases() -> Vec<PathBuf> {
    let dir = golden_dir();
    let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().is_some_and(|x| x == "npz")
                && p.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("mamba3_siso"))
        })
        .collect();
    v.sort();
    v
}

#[test]
fn mamba3_golden_cases() {
    let cases = cases();
    assert!(
        cases.len() >= 6,
        "expected the Stage-0 gate's >=6 goldens, found {}",
        cases.len()
    );
    let mut worst = 0.0f64;
    // Both the naive oracle and the cache-blocked path (the NEON kernel's
    // structural twin) must clear the bound on the REAL captured ground truth,
    // not just on synthetic property-test data.
    for (backend, label) in [(Backend::Scalar, "naive"), (Backend::Auto, "tiled")] {
        for path in &cases {
            let r = run_case(path, backend, Threading::Sequential);
            let name = format!("{}[{label}]", path.file_stem().unwrap().to_string_lossy());
            println!(
                "  {name:<28} L={:<5} scale={:8.3} {:8.2} bf16 ULP",
                r.len, r.scale, r.ulps
            );
            assert!(
                r.ulps <= MAX_ULPS,
                "{name}: {:.2} bf16 ULP exceeds {MAX_ULPS}. Do NOT raise the \
             bound to pass — a structurally wrong kernel is off by thousands.",
                r.ulps
            );
            // Guard against a vacuous pass: an all-zero or constant output would
            // sail through a tolerance check on some cases.
            assert!(
                r.out.iter().any(|&x| x != 0.0) && r.out.iter().all(|x| x.is_finite()),
                "{name}: output is all zeros or non-finite"
            );
            worst = worst.max(r.ulps);
        }
    }
    println!(
        "  worst across {} goldens x 2 backends: {worst:.2} bf16 ULP",
        cases.len()
    );
}

/// Threaded output must be **bit-identical** to sequential, element for element.
/// Holds by construction — `(batch, head)` pairs are disjoint with no
/// cross-thread reduction — but a guarantee is only worth what its test asserts.
#[test]
fn mamba3_parallel_is_bit_identical() {
    for path in cases().iter() {
        let seq = run_case(path, Backend::Scalar, Threading::Sequential);
        let par = run_case(path, Backend::Scalar, Threading::Rayon);
        assert_eq!(seq.out.len(), par.out.len());
        for (i, (a, b)) in seq.out.iter().zip(par.out.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{}: element {i} differs between sequential ({a}) and threaded ({b})",
                path.display()
            );
        }
    }
}
