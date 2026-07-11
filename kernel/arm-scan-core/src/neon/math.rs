//! Vectorized softplus and SiLU for the discretization/gate passes.
//!
//! Phase 2 profiling showed the recurrence loop is not the only cost: each
//! timestep also paid one *scalar* libm softplus (discretization) and one
//! scalar libm silu (gate). These helpers move both onto 4-lane NEON so the
//! chunked kernel can precompute them across time.
//!
//! `vlogq_f32` is a Cephes-style natural log (mantissa/exponent split +
//! degree-9 polynomial). It requires positive, normal, finite input — which
//! `vsoftplusq_f32`, its only caller, guarantees by construction (argument
//! is 1 + t with t in (0, 1]).

use core::arch::aarch64::*;

use super::exp::vexpq_f32;

// Cephes logf polynomial coefficients
const LP0: f32 = 7.037_683_6e-2;
const LP1: f32 = -1.151_461e-1;
const LP2: f32 = 1.167_699_9e-1;
const LP3: f32 = -1.242_014_1e-1;
const LP4: f32 = 1.424_932_3e-1;
const LP5: f32 = -1.666_805_7e-1;
const LP6: f32 = 2.000_071_5e-1;
const LP7: f32 = -2.499_999_4e-1;
const LP8: f32 = 3.333_333e-1;
const SQRTHF: f32 = 0.707_106_77;
const LN2_HI: f32 = 0.693_359_4;
const LN2_LO: f32 = -2.121_944_4e-4;

/// Four-lane ln(x) for positive normal finite x.
///
/// # Safety
/// Requires NEON. Caller must ensure every lane is a positive, normal,
/// finite float (no 0 / negative / subnormal / inf / NaN handling).
#[target_feature(enable = "neon")]
pub(super) unsafe fn vlogq_f32(x: float32x4_t) -> float32x4_t {
    // split into exponent and mantissa m in [0.5, 1)
    let ix = vreinterpretq_s32_f32(x);
    let e_raw = vsubq_s32(vshrq_n_s32(ix, 23), vdupq_n_s32(126));
    let m = vreinterpretq_f32_s32(vorrq_s32(
        vandq_s32(ix, vdupq_n_s32(0x007F_FFFF)),
        vdupq_n_s32(0x3F00_0000),
    ));

    // if m < sqrt(0.5): e -= 1, m = 2m; keeps m in [sqrt(0.5), sqrt(2))
    let small = vcltq_f32(m, vdupq_n_f32(SQRTHF));
    let e = vsubq_s32(
        e_raw,
        vandq_s32(vreinterpretq_s32_u32(small), vdupq_n_s32(1)),
    );
    let m = vbslq_f32(small, vaddq_f32(m, m), m);
    let xm = vsubq_f32(m, vdupq_n_f32(1.0));
    let ef = vcvtq_f32_s32(e);

    // polynomial: y = xm^3 * P(xm)
    let mut p = vdupq_n_f32(LP0);
    p = vfmaq_f32(vdupq_n_f32(LP1), p, xm);
    p = vfmaq_f32(vdupq_n_f32(LP2), p, xm);
    p = vfmaq_f32(vdupq_n_f32(LP3), p, xm);
    p = vfmaq_f32(vdupq_n_f32(LP4), p, xm);
    p = vfmaq_f32(vdupq_n_f32(LP5), p, xm);
    p = vfmaq_f32(vdupq_n_f32(LP6), p, xm);
    p = vfmaq_f32(vdupq_n_f32(LP7), p, xm);
    p = vfmaq_f32(vdupq_n_f32(LP8), p, xm);
    let z = vmulq_f32(xm, xm);
    let mut y = vmulq_f32(vmulq_f32(p, z), xm);

    // assemble: log = xm - z/2 + y + e*ln2 (hi/lo split)
    y = vfmaq_f32(y, ef, vdupq_n_f32(LN2_LO));
    y = vfmaq_f32(y, z, vdupq_n_f32(-0.5));
    let r = vaddq_f32(xm, y);
    vfmaq_f32(r, ef, vdupq_n_f32(LN2_HI))
}

/// Four-lane softplus(x) = ln(1 + e^x), computed stably as
/// max(x, 0) + log1p(e^{-|x|}). Matches torch's threshold-20 behavior
/// automatically: for x > ~17, e^{-x} vanishes below f32 eps and the
/// result is exactly x.
///
/// # Safety
/// Requires NEON.
#[target_feature(enable = "neon")]
pub(super) unsafe fn vsoftplusq_f32(x: float32x4_t) -> float32x4_t {
    let t = vexpq_f32(vnegq_f32(vabsq_f32(x)));
    // t in (0, 1] -> 1 + t in (1, 2]: safe domain for vlogq
    let log1p_t = vlogq_f32(vaddq_f32(vdupq_n_f32(1.0), t));
    vaddq_f32(vmaxq_f32(x, vdupq_n_f32(0.0)), log1p_t)
}

/// Four-lane silu(x) = x * sigmoid(x) = x / (1 + e^{-x}).
///
/// # Safety
/// Requires NEON.
#[target_feature(enable = "neon")]
pub(super) unsafe fn vsiluq_f32(x: float32x4_t) -> float32x4_t {
    let denom = vaddq_f32(vdupq_n_f32(1.0), vexpq_f32(vnegq_f32(x)));
    vdivq_f32(x, denom)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply4(f: unsafe fn(float32x4_t) -> float32x4_t, xs: [f32; 4]) -> [f32; 4] {
        unsafe {
            let v = vld1q_f32(xs.as_ptr());
            let r = f(v);
            let mut out = [0.0_f32; 4];
            vst1q_f32(out.as_mut_ptr(), r);
            out
        }
    }

    /// vlogq over its softplus-use domain [1, 2] plus a wide positive range.
    #[test]
    fn log_sweep() {
        let mut worst = 0.0_f64;
        // dense over [1, 2] (the actual call-site domain)
        for i in 0..1_000_000_usize {
            let x = 1.0_f32 + i as f32 / 1_000_000.0;
            let got = apply4(vlogq_f32, [x; 4])[0] as f64;
            let expect = (x as f64).ln();
            let err = if expect.abs() < 1e-3 {
                (got - expect).abs() // near ln(1)=0 use absolute
            } else {
                ((got - expect) / expect).abs()
            };
            worst = worst.max(err);
            assert!(err < 1e-6, "ln({x}) got {got} expect {expect}");
        }
        // coarse over positive normals
        for i in 0..100_000_usize {
            let x = f32::exp2(-120.0 + 240.0 * i as f32 / 100_000.0);
            let got = apply4(vlogq_f32, [x; 4])[0] as f64;
            let expect = (x as f64).ln();
            let rel = ((got - expect) / expect).abs();
            worst = worst.max(rel);
            assert!(rel < 1e-6, "ln({x:e}) got {got} expect {expect}");
        }
        println!("vlogq_f32 worst error: {worst:.3e}");
    }

    /// softplus against f64 reference over the delta ranges the scan sees.
    #[test]
    fn softplus_sweep() {
        let mut worst = 0.0_f64;
        for i in 0..2_000_000_usize {
            let x = -30.0_f32 + 60.0 * i as f32 / 2_000_000.0;
            let got = apply4(vsoftplusq_f32, [x; 4])[0] as f64;
            let expect = (x as f64).exp().ln_1p();
            // absolute near zero output, relative elsewhere
            let err = if expect < 1e-3 {
                (got - expect).abs()
            } else {
                ((got - expect) / expect).abs()
            };
            worst = worst.max(err);
            assert!(err < 2e-6, "softplus({x}) got {got} expect {expect}");
        }
        // threshold behavior: large x passes through exactly
        assert_eq!(apply4(vsoftplusq_f32, [25.0; 4])[0], 25.0);
        assert_eq!(apply4(vsoftplusq_f32, [88.0; 4])[0], 88.0);
        println!("vsoftplusq_f32 worst error: {worst:.3e}");
    }

    /// silu against f64 reference; extremes must stay finite.
    #[test]
    fn silu_sweep() {
        let mut worst = 0.0_f64;
        for i in 0..2_000_000_usize {
            let x = -40.0_f32 + 80.0 * i as f32 / 2_000_000.0;
            let got = apply4(vsiluq_f32, [x; 4])[0] as f64;
            let xf = x as f64;
            let expect = xf / (1.0 + (-xf).exp());
            let err = if expect.abs() < 1e-3 {
                (got - expect).abs()
            } else {
                ((got - expect) / expect).abs()
            };
            worst = worst.max(err);
            assert!(err < 2e-6, "silu({x}) got {got} expect {expect}");
        }
        let extreme = apply4(vsiluq_f32, [-200.0, 200.0, -88.0, 88.0]);
        assert!(extreme.iter().all(|v| v.is_finite()), "{extreme:?}");
        println!("vsiluq_f32 worst error: {worst:.3e}");
    }
}
