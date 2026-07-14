//! Fast NEON `exp` for four f32 lanes.
//!
//! Cephes-style: range-reduce x = k·ln2 + r with a hi/lo split of ln2, a
//! degree-6 polynomial for e^r on r ∈ [-ln2/2, ln2/2], then scale by 2^k via
//! exponent-field integer arithmetic. ~2 ulp over the full range.
//!
//! This is the single hottest operation in the whole kernel (16 exps per
//! channel per timestep): one call replaces four libm `expf` calls with
//! ~12 vector instructions.
//!
//! Deviation from a pure polynomial: inputs below the normal-f32 range
//! (x < -87.34) return exactly 0.0 (a `vbslq` select). True f32 exp yields
//! subnormals down to x ~ -103.3 before hitting 0; flushing that whole
//! region to zero costs at most ~1.2e-38 of absolute error, which is
//! irrelevant against the kernel's 1e-4 tolerance and matches where the
//! deeply-negative `delta * A` scan arguments end up anyway.

use core::arch::aarch64::*;

const LOG2E: f32 = std::f32::consts::LOG2_E;
// ln2 split so that k*LN2_HI is exact for |k| < 2^15
const LN2_HI: f32 = 0.693_359_4;
const LN2_LO: f32 = -2.121_944_4e-4;
// Cephes expf polynomial coefficients (p5 = 1/2 handles the r^2/2! term)
const P0: f32 = 1.987_569_1e-4;
const P1: f32 = 1.398_199_9e-3;
const P2: f32 = 8.333_452e-3;
const P3: f32 = 4.166_579_6e-2;
const P4: f32 = 1.666_666_6e-1;
const P5: f32 = 5e-1;
/// Above this, 2^k would overflow the exponent field (k = 128).
const MAX_X: f32 = 88.02;
/// Smallest x whose exp is still a *normal* f32 (k >= -126). Below this we
/// return exactly 0.0 rather than modeling the subnormal range
/// [e^-103.28, e^-87.34): the 2^k exponent trick wraps for k < -126, and a
/// worst-case absolute error of ~1.2e-38 versus true f32 exp is 34 orders
/// of magnitude beneath the kernel's 1e-4 acceptance tolerance.
const MIN_X: f32 = -87.336_54;

/// Four-lane e^x.
///
/// # Safety
/// Requires NEON (architecturally guaranteed on aarch64).
#[target_feature(enable = "neon")]
pub(super) unsafe fn vexpq_f32(x: float32x4_t) -> float32x4_t {
    let clamped = vmaxq_f32(vminq_f32(x, vdupq_n_f32(MAX_X)), vdupq_n_f32(MIN_X));

    // k = round-to-nearest(x / ln2); kf is exactly integral
    let k = vcvtnq_s32_f32(vmulq_n_f32(clamped, LOG2E));
    let kf = vcvtq_f32_s32(k);

    // r = x - k*ln2, in two steps for precision
    let r = vfmaq_f32(clamped, kf, vdupq_n_f32(-LN2_HI));
    let r = vfmaq_f32(r, kf, vdupq_n_f32(-LN2_LO));

    // Horner: p = ((((P0*r + P1)*r + P2)*r + P3)*r + P4)*r + P5
    let mut p = vdupq_n_f32(P0);
    p = vfmaq_f32(vdupq_n_f32(P1), p, r);
    p = vfmaq_f32(vdupq_n_f32(P2), p, r);
    p = vfmaq_f32(vdupq_n_f32(P3), p, r);
    p = vfmaq_f32(vdupq_n_f32(P4), p, r);
    p = vfmaq_f32(vdupq_n_f32(P5), p, r);

    // e^r = 1 + r + r^2 * p
    let e_r = vfmaq_f32(vaddq_f32(r, vdupq_n_f32(1.0)), p, vmulq_f32(r, r));

    // scale by 2^k through the exponent field. The clamp above keeps
    // k in [-126, 127], so the biased exponent never wraps.
    let two_k = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(k, vdupq_n_s32(127)), 23));
    let result = vmulq_f32(e_r, two_k);

    // exact zero where f32 exp underflows (keeps parity with torch)
    let underflow = vcltq_f32(x, vdupq_n_f32(MIN_X));
    vbslq_f32(underflow, vdupq_n_f32(0.0), result)
}

/// Four-lane e^x specialized for **non-positive** input (`x <= 0` on every
/// lane), which is exactly the scan's Pass-A2 argument `dt * A` (A < 0, the
/// post-softplus dt >= 0, and the zero-padded A lanes give exactly 0). Three
/// things the general [`vexpq_f32`] does for positive / edge inputs are
/// dropped, because this domain never needs them:
///
///  1. **No upper (overflow) clamp** — `x <= 0` means `2^k <= 1`, never inf.
///  2. **No exact-zero underflow select** — deep-underflow lanes return
///     `~2^-126 ≈ 1.2e-38` instead of exactly 0. That is 34 orders of
///     magnitude below the kernel's 1e-4 tolerance, and the recurrence
///     `h = abar·h + b̄` with `abar < 1` is a contraction, so it cannot
///     amplify the difference. The low clamp to `MIN_X` stays, purely to keep
///     the `2^k` exponent field from wrapping.
///  3. **One lower polynomial degree** — the dropped `P0·r^5` term of the
///     mantissa polynomial is `< ~1.3e-7` over `|r| <= ln2/2`, so worst-case
///     accuracy stays `~6e-7` over `[-104, 0]`, still comfortably tighter than
///     the golden gate. (`vexpq_f32` is untouched and keeps its ~2 ulp.)
///
/// # Safety
/// Requires NEON. Every lane must be `<= 0`; positive input can overflow to
/// `inf` because the upper clamp is absent (the scan never passes positive).
#[target_feature(enable = "neon")]
pub(super) unsafe fn vexpq_f32_nonpos(x: float32x4_t) -> float32x4_t {
    // Low clamp only (no upper clamp): keeps k >= -126 so 2^k never wraps.
    let clamped = vmaxq_f32(x, vdupq_n_f32(MIN_X));

    // k = round-to-nearest(x / ln2) <= 0; kf is exactly integral.
    let k = vcvtnq_s32_f32(vmulq_n_f32(clamped, LOG2E));
    let kf = vcvtq_f32_s32(k);

    // r = x - k*ln2, hi/lo split for precision.
    let r = vfmaq_f32(clamped, kf, vdupq_n_f32(-LN2_HI));
    let r = vfmaq_f32(r, kf, vdupq_n_f32(-LN2_LO));

    // Horner from P1 (P0 dropped): p = (((P1*r + P2)*r + P3)*r + P4)*r + P5.
    let mut p = vdupq_n_f32(P1);
    p = vfmaq_f32(vdupq_n_f32(P2), p, r);
    p = vfmaq_f32(vdupq_n_f32(P3), p, r);
    p = vfmaq_f32(vdupq_n_f32(P4), p, r);
    p = vfmaq_f32(vdupq_n_f32(P5), p, r);

    // e^r = 1 + r + r^2 * p, then scale by 2^k through the exponent field.
    let e_r = vfmaq_f32(vaddq_f32(r, vdupq_n_f32(1.0)), p, vmulq_f32(r, r));
    let two_k = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(k, vdupq_n_s32(127)), 23));
    vmulq_f32(e_r, two_k)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exp4(xs: [f32; 4]) -> [f32; 4] {
        unsafe {
            let v = vld1q_f32(xs.as_ptr());
            let r = vexpq_f32(v);
            let mut out = [0.0_f32; 4];
            vst1q_f32(out.as_mut_ptr(), r);
            out
        }
    }

    /// Dense sweep over the scan's actual argument domain (delta*A is
    /// non-positive; we still sweep the positive side for robustness):
    /// 4M points across [-104, 88], relative error must stay within ~4 ulp.
    #[test]
    fn dense_sweep_accuracy() {
        let (lo, hi, n) = (-104.0_f64, 88.0_f64, 4_000_000_usize);
        let step = (hi - lo) / n as f64;
        let mut worst_rel = 0.0_f64;
        let mut i = 0;
        while i < n {
            let xs = [
                (lo + step * i as f64) as f32,
                (lo + step * (i + 1) as f64) as f32,
                (lo + step * (i + 2) as f64) as f32,
                (lo + step * (i + 3) as f64) as f32,
            ];
            let got = exp4(xs);
            for lane in 0..4 {
                let expect = (xs[lane] as f64).exp();
                let g = got[lane] as f64;
                if expect < f32::MIN_POSITIVE as f64 {
                    // underflow region: allow 0 or a tiny subnormal
                    assert!(
                        g.abs() <= f32::MIN_POSITIVE as f64,
                        "x={} got {g}, expected underflow-scale",
                        xs[lane]
                    );
                } else {
                    let rel = ((g - expect) / expect).abs();
                    worst_rel = worst_rel.max(rel);
                    assert!(
                        rel < 5e-7,
                        "x={} got {g} expect {expect} rel {rel:.3e}",
                        xs[lane]
                    );
                }
            }
            i += 4;
        }
        println!("vexpq_f32 worst relative error over sweep: {worst_rel:.3e}");
    }

    fn exp4_nonpos(xs: [f32; 4]) -> [f32; 4] {
        unsafe {
            let v = vld1q_f32(xs.as_ptr());
            let r = vexpq_f32_nonpos(v);
            let mut out = [0.0_f32; 4];
            vst1q_f32(out.as_mut_ptr(), r);
            out
        }
    }

    /// vexpq_f32_nonpos over its declared domain [-104, 0]: the same 4M-point
    /// dense sweep as vexpq_f32, but the accuracy budget reflects the dropped
    /// polynomial term (worst ~6e-7; asserted < 1.5e-6 with margin). Deep
    /// underflow returns a ~1e-38-scale value rather than exact 0 by design
    /// (see the function docs), so that region only checks the result is
    /// negligibly small, not exactly zero.
    #[test]
    fn nonpos_dense_sweep_accuracy() {
        let (lo, hi, n) = (-104.0_f64, 0.0_f64, 4_000_000_usize);
        let step = (hi - lo) / n as f64;
        let mut worst_rel = 0.0_f64;
        let mut i = 0;
        while i < n {
            let xs = [
                (lo + step * i as f64) as f32,
                (lo + step * (i + 1) as f64) as f32,
                (lo + step * (i + 2) as f64) as f32,
                (lo + step * (i + 3) as f64) as f32,
            ];
            let got = exp4_nonpos(xs);
            for lane in 0..4 {
                let expect = (xs[lane] as f64).exp();
                let g = got[lane] as f64;
                if expect < 1e-30 {
                    // deep underflow: only require a negligibly small result
                    assert!(
                        g <= 1e-30,
                        "x={} got {g}, expected underflow-scale",
                        xs[lane]
                    );
                } else {
                    let rel = ((g - expect) / expect).abs();
                    worst_rel = worst_rel.max(rel);
                    assert!(
                        rel < 1.5e-6,
                        "x={} got {g} expect {expect} rel {rel:.3e}",
                        xs[lane]
                    );
                }
            }
            i += 4;
        }
        println!("vexpq_f32_nonpos worst relative error over sweep: {worst_rel:.3e}");
    }

    #[test]
    fn nonpos_special_values() {
        let out = exp4_nonpos([0.0, -1.0, -10.0, -87.0]);
        assert_eq!(out[0], 1.0, "exp(0) must be exactly 1");
        assert!((out[1] - (-1.0_f32).exp()).abs() < 2e-6);
        assert!((out[2] as f64 - (-10.0_f64).exp()).abs() / (-10.0_f64).exp() < 1.5e-6);
        assert!((out[3] as f64 - (-87.0_f64).exp()).abs() / (-87.0_f64).exp() < 1.5e-6);
        // deep underflow stays finite and negligibly small (not necessarily 0)
        let out = exp4_nonpos([-160.0, -1000.0, -104.0, -300.0]);
        assert!(
            out.iter().all(|v| v.is_finite() && *v >= 0.0 && *v < 1e-30),
            "{out:?}"
        );
    }

    #[test]
    fn special_values() {
        let out = exp4([0.0, 1.0, -1.0, -87.0]);
        assert_eq!(out[0], 1.0, "exp(0) must be exactly 1");
        assert!((out[1] - core::f32::consts::E).abs() < 3e-7);
        assert!((out[2] - (-1.0_f32).exp()).abs() < 3e-8);
        assert!((out[3] as f64 - (-87.0_f64).exp()).abs() / (-87.0_f64).exp() < 5e-7);

        // deep underflow -> exact zero, like f32::exp
        let out = exp4([-160.0, -1000.0, f32::MIN, -103.9]);
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0]);

        // just above the overflow clamp stays finite (saturates near f32::MAX
        // scale rather than producing inf) — scan never feeds this, but the
        // function must not explode
        let out = exp4([100.0, 200.0, 88.0, 50.0]);
        assert!(out[2].is_finite() && out[3].is_finite());
        assert!(out[0].is_finite() && out[1].is_finite());
    }
}
