//! Mamba-3 property tests — the invariants that must hold regardless of shape.
//!
//! The headline one is `reverse`. For Mamba-1 that flag is a pure index flip and
//! is bit-identical to flip-forward-flip on the scalar path. **Mamba-3 makes it
//! more than an index flip**: the trapezoid's second term reads the *next*
//! timestep in scan order, so a forward scan pairs `t` with `t+1` while a
//! backward scan must pair `t` with `t-1`. Get that backwards and the kernel
//! still runs, still produces plausible numbers, and is wrong — and it would be
//! wrong in *both* the bidirectional and the 2D cross-scan topologies, since
//! both are built on backward traversal.
//!
//! So `reverse` is defined here as an equivalence and checked as one.

use arm_scan_core::{
    mamba3_scan_with_options, Backend, Mamba3Dims, Mamba3Input, ScanOptions, Threading,
};

/// Deterministic, dependency-free pseudo-randomness (SplitMix64 -> f32 in
/// [-1, 1)). Fixed seeds keep failures reproducible.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32 / 8_388_608.0) - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
    /// Non-positive, as `A*dt` always is — the same precondition the fast NEON
    /// `exp` relies on.
    fn vec_nonpos(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| -(self.next_f32().abs()) - 0.01).collect()
    }
    /// Strictly positive, as post-softplus `dt` always is.
    fn vec_pos(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32().abs() + 0.01).collect()
    }
}

struct Case {
    dims: Mamba3Dims,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    adt: Vec<f32>,
    dt: Vec<f32>,
    trap: Vec<f32>,
    q_bias: Vec<f32>,
    k_bias: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    d: Vec<f32>,
    z: Vec<f32>,
}

fn make(seed: u64, batch: usize, heads: usize, dv: usize, dqk: usize, len: usize) -> Case {
    let mut r = Rng(seed);
    let half = dqk / 2;
    Case {
        dims: Mamba3Dims {
            batch,
            heads,
            dv,
            dqk,
            len,
            rank: 1,
        },
        q: r.vec(batch * len * dqk),
        k: r.vec(batch * len * dqk),
        v: r.vec(batch * len * heads * dv),
        adt: r.vec_nonpos(batch * heads * len),
        dt: r.vec_pos(batch * heads * len),
        trap: r.vec(batch * heads * len),
        q_bias: r.vec(heads * dqk),
        k_bias: r.vec(heads * dqk),
        cos: r.vec(batch * len * heads * half),
        sin: r.vec(batch * len * heads * half),
        d: r.vec(heads),
        z: r.vec(batch * len * heads * dv),
    }
}

impl Case {
    fn input(&self, reverse: bool) -> Mamba3Input<'_, f32> {
        Mamba3Input {
            q: &self.q,
            k: &self.k,
            v: &self.v,
            adt: &self.adt,
            dt: &self.dt,
            trap: &self.trap,
            q_bias: &self.q_bias,
            k_bias: &self.k_bias,
            cos: &self.cos,
            sin: &self.sin,
            d_skip: Some(&self.d),
            z: Some(&self.z),
            reverse,
            mimo: None,
        }
    }

    fn run(&self, reverse: bool, threading: Threading) -> Vec<f32> {
        self.run_with(Backend::Scalar, reverse, threading)
    }

    fn run_with(&self, backend: Backend, reverse: bool, threading: Threading) -> Vec<f32> {
        let d = self.dims;
        let mut out = vec![0.0f32; d.batch * d.heads * d.len * d.dv];
        mamba3_scan_with_options(
            &d,
            &self.input(reverse),
            &mut out,
            None,
            None,
            ScanOptions { backend, threading },
        )
        .expect("scan");
        out
    }

    /// A copy with every time-varying tensor reversed along the time axis.
    /// `q_bias`/`k_bias`/`d` are per-head parameters and stay put.
    fn time_flipped(&self) -> Case {
        let d = self.dims;
        let (b, h, dv, dqk, l) = (d.batch, d.heads, d.dv, d.dqk, d.len);
        let half = dqk / 2;
        // (b, l, X) layouts: flip the l axis.
        let flip_bl = |src: &[f32], inner: usize| -> Vec<f32> {
            let mut out = vec![0.0; src.len()];
            for bi in 0..b {
                for t in 0..l {
                    let s = (bi * l + t) * inner;
                    let dst = (bi * l + (l - 1 - t)) * inner;
                    out[dst..dst + inner].copy_from_slice(&src[s..s + inner]);
                }
            }
            out
        };
        // (b, h, l) layout: flip the trailing axis.
        let flip_bhl = |src: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0; src.len()];
            for i in 0..b * h {
                for t in 0..l {
                    out[i * l + (l - 1 - t)] = src[i * l + t];
                }
            }
            out
        };
        Case {
            dims: d,
            q: flip_bl(&self.q, dqk),
            k: flip_bl(&self.k, dqk),
            v: flip_bl(&self.v, h * dv),
            adt: flip_bhl(&self.adt),
            dt: flip_bhl(&self.dt),
            trap: flip_bhl(&self.trap),
            q_bias: self.q_bias.clone(),
            k_bias: self.k_bias.clone(),
            cos: flip_bl(&self.cos, h * half),
            sin: flip_bl(&self.sin, h * half),
            d: self.d.clone(),
            z: flip_bl(&self.z, h * dv),
        }
    }
}

const SHAPES: &[(usize, usize, usize, usize, usize)] = &[
    // batch, heads, dv, dqk, len
    (1, 2, 4, 8, 1),   // L=1 edge: no successor at all
    (1, 2, 4, 8, 2),   // the smallest case where the 2-tap direction matters
    (1, 3, 8, 16, 7),  // odd length
    (2, 4, 8, 16, 16), // batch > 1
    (1, 2, 16, 32, 33),
];

/// `reverse` is DEFINED as: flip time, scan forward, flip the output back.
/// This is the test that catches a 2-tap pointing the wrong way.
#[test]
fn mamba3_reverse_matches_flip_forward_flip() {
    for (i, &(b, h, dv, dqk, l)) in SHAPES.iter().enumerate() {
        let c = make(0xC0FFEE + i as u64, b, h, dv, dqk, l);
        let rev = c.run(true, Threading::Sequential);
        let fwd_flipped = c.time_flipped().run(false, Threading::Sequential);

        // Output is head-major (b, h, l, dv); un-flip the time axis.
        let mut worst = 0.0f32;
        for bh in 0..b * h {
            for t in 0..l {
                for r in 0..dv {
                    let a = rev[(bh * l + t) * dv + r];
                    let e = fwd_flipped[(bh * l + (l - 1 - t)) * dv + r];
                    worst = worst.max((a - e).abs());
                }
            }
        }
        assert!(
            worst < 1e-5,
            "shape {b}x{h}x{dv}x{dqk}x{l}: reverse deviates from \
             flip-forward-flip by {worst:e}. The trapezoid's second term must \
             read t-1 going backward, not t+1."
        );
    }
}

/// A reverse scan must actually depend on direction — otherwise the test above
/// could pass against a `reverse` flag that does nothing at all.
#[test]
fn mamba3_reverse_actually_differs() {
    let c = make(7, 1, 2, 8, 16, 12);
    let f = c.run(false, Threading::Sequential);
    let r = c.run(true, Threading::Sequential);
    let diff = f
        .iter()
        .zip(r.iter())
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        diff > 1e-3,
        "forward and reverse produced near-identical output ({diff:e}) — the \
         reverse flag is not being honoured"
    );
}

/// Threaded output must be bit-identical to sequential, in both directions.
#[test]
fn mamba3_parallel_bit_identical_both_directions() {
    for &(b, h, dv, dqk, l) in SHAPES {
        let c = make(99, b, h, dv, dqk, l);
        for reverse in [false, true] {
            let seq = c.run(reverse, Threading::Sequential);
            let par = c.run(reverse, Threading::Rayon);
            for (i, (a, b2)) in seq.iter().zip(par.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b2.to_bits(),
                    "reverse={reverse} element {i}: {a} vs {b2}"
                );
            }
        }
    }
}

/// Shape validation must reject rather than silently truncate — the "wrong but
/// green" failure mode this repo keeps finding.
#[test]
fn mamba3_validation_rejects_bad_shapes() {
    let c = make(1, 1, 2, 4, 8, 4);
    let d = c.dims;
    let mut out = vec![0.0f32; d.batch * d.heads * d.len * d.dv];

    // Odd dqk has no meaning: RoPE rotates lane pairs.
    let bad = Mamba3Dims { dqk: 7, ..d };
    assert!(mamba3_scan_with_options(
        &bad,
        &c.input(false),
        &mut out,
        None,
        None,
        ScanOptions::default()
    )
    .is_err());

    // Zero dims.
    let zero = Mamba3Dims { heads: 0, ..d };
    assert!(mamba3_scan_with_options(
        &zero,
        &c.input(false),
        &mut out,
        None,
        None,
        ScanOptions::default()
    )
    .is_err());

    // Wrong output length.
    let mut short = vec![0.0f32; 3];
    assert!(mamba3_scan_with_options(
        &d,
        &c.input(false),
        &mut short,
        None,
        None,
        ScanOptions::default()
    )
    .is_err());
}

/// The blocked kernel (`Backend::Auto`, `mamba3/tiled.rs`) must agree with the
/// naive oracle (`Backend::Scalar`, `mamba3/scalar.rs`).
///
/// Not bit-identical, and it should not be: tiling splits each `q . S_row` dot
/// product across tiles, so the summation order differs and f32 rounding
/// follows. What must hold is agreement to the level that reassociation alone
/// explains — a genuine ordering bug (reading `S` after its update rather than
/// before) is off by the trapezoid's shifted term and grows with sequence
/// length, which is orders of magnitude larger.
#[test]
fn mamba3_tiled_matches_naive() {
    for &(b, h, dv, dqk, l) in SHAPES {
        let c = make(0xB10C, b, h, dv, dqk, l);
        for reverse in [false, true] {
            let naive = c.run_with(Backend::Scalar, reverse, Threading::Sequential);
            let tiled = c.run_with(Backend::Auto, reverse, Threading::Sequential);
            let scale = naive.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-6);
            let worst = naive
                .iter()
                .zip(tiled.iter())
                .fold(0.0f32, |m, (a, t)| m.max((a - t).abs()));
            assert!(
                worst / scale < 1e-5,
                "shape {b}x{h}x{dv}x{dqk}x{l} reverse={reverse}: tiled deviates \
                 from naive by {:.3e} relative — too large for reassociation. \
                 Check that y is read from S BEFORE the update.",
                worst / scale
            );
        }
    }
}

/// The tiling must not depend on `dqk` being a multiple of `TILE` — the tail
/// tile is where an off-by-one lives.
#[test]
fn mamba3_tiled_handles_ragged_dqk() {
    // TILE is 32; 48 gives one full tile plus a 16-wide tail, 34 a 2-wide tail.
    for &dqk in &[34usize, 48, 66] {
        let c = make(5, 1, 2, 8, dqk, 9);
        let naive = c.run_with(Backend::Scalar, false, Threading::Sequential);
        let tiled = c.run_with(Backend::Auto, false, Threading::Sequential);
        let scale = naive.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-6);
        let worst = naive
            .iter()
            .zip(tiled.iter())
            .fold(0.0f32, |m, (a, t)| m.max((a - t).abs()));
        assert!(
            worst / scale < 1e-5,
            "dqk={dqk}: ragged tail tile deviates by {:.3e}",
            worst / scale
        );
    }
}

/// Explicit NEON-vs-naive parity, aarch64 only.
///
/// `mamba3_tiled_matches_naive` already covers this incidentally, because
/// `Backend::Auto` resolves to NEON here — but a gate that only works by
/// accident of dispatch is a gate that stops working when dispatch changes.
/// This one names the backend.
#[cfg(target_arch = "aarch64")]
#[test]
fn mamba3_neon_matches_naive() {
    for &(b, h, dv, dqk, l) in SHAPES {
        let c = make(0x1234, b, h, dv, dqk, l);
        for reverse in [false, true] {
            let naive = c.run_with(Backend::Scalar, reverse, Threading::Sequential);
            let neon = c.run_with(Backend::Neon, reverse, Threading::Sequential);
            let scale = naive.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-6);
            let worst = naive
                .iter()
                .zip(neon.iter())
                .fold(0.0f32, |m, (a, n)| m.max((a - n).abs()));
            assert!(
                worst / scale < 1e-5,
                "shape {b}x{h}x{dv}x{dqk}x{l} reverse={reverse}: NEON deviates \
                 from the scalar oracle by {:.3e} relative",
                worst / scale
            );
        }
    }
}

/// NEON must be bit-identical across thread counts too — vectorisation must not
/// have introduced any cross-head sharing.
#[cfg(target_arch = "aarch64")]
#[test]
fn mamba3_neon_parallel_bit_identical() {
    for &(b, h, dv, dqk, l) in SHAPES {
        let c = make(0x5678, b, h, dv, dqk, l);
        let seq = c.run_with(Backend::Neon, false, Threading::Sequential);
        let par = c.run_with(Backend::Neon, false, Threading::Rayon);
        for (i, (a, b2)) in seq.iter().zip(par.iter()).enumerate() {
            assert_eq!(a.to_bits(), b2.to_bits(), "element {i}: {a} vs {b2}");
        }
    }
}
