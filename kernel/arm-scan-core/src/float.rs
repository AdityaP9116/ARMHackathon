//! Minimal float abstraction so the scalar kernel is generic over
//! f32 (production) and f64 (used by tests as an in-Rust precision
//! reference). Deliberately tiny — not a general numerics trait.

use core::ops::{Add, Div, Mul, Neg, Sub};

pub trait Float:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + core::fmt::Debug
    + Send
    + Sync
    + 'static
{
    const ZERO: Self;
    const ONE: Self;
    /// softplus falls back to identity above this (matches torch's
    /// `F.softplus(..., threshold=20)`)
    const SOFTPLUS_THRESHOLD: Self;

    fn exp(self) -> Self;
    fn ln_1p(self) -> Self;

    /// softplus(x) = ln(1 + e^x), linear above the threshold like torch
    #[inline]
    fn softplus(self) -> Self {
        if self > Self::SOFTPLUS_THRESHOLD {
            self
        } else {
            self.exp().ln_1p()
        }
    }

    /// silu(x) = x * sigmoid(x)
    #[inline]
    fn silu(self) -> Self {
        self / (Self::ONE + (-self).exp())
    }

    /// sigmoid(x) = 1 / (1 + e^-x) — Mamba-3's trapezoid gate.
    ///
    /// Written as the branch below rather than the direct formula because
    /// `exp(-x)` overflows to +inf for x <~ -88 in f32, and inf/inf is NaN.
    /// Reflecting to the other branch keeps every argument non-positive, which
    /// is the same precondition `vexpq_f32_nonpos` exploits on the NEON side.
    #[inline]
    fn sigmoid(self) -> Self {
        if self >= Self::ZERO {
            Self::ONE / (Self::ONE + (-self).exp())
        } else {
            let e = self.exp();
            e / (Self::ONE + e)
        }
    }

    /// tanh(x) — used only for the Mamba-3 RoPE angle squashing
    /// (`theta = cumsum(tanh(angle) * PI * dt)`).
    ///
    /// `tanh(x) = 1 - 2/(e^{2x} + 1)`, evaluated on the non-positive branch and
    /// reflected for the same overflow reason as `sigmoid`. Saturates cleanly:
    /// at |x| >= ~20 the result is +/-1 to well within f32.
    #[inline]
    fn tanh(self) -> Self {
        let two = Self::ONE + Self::ONE;
        if self >= Self::ZERO {
            let e = (-two * self).exp();
            (Self::ONE - e) / (Self::ONE + e)
        } else {
            let e = (two * self).exp();
            -((Self::ONE - e) / (Self::ONE + e))
        }
    }
}

macro_rules! impl_float {
    ($t:ty) => {
        impl Float for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const SOFTPLUS_THRESHOLD: Self = 20.0;

            #[inline]
            fn exp(self) -> Self {
                self.exp()
            }
            #[inline]
            fn ln_1p(self) -> Self {
                self.ln_1p()
            }
        }
    };
}

impl_float!(f32);
impl_float!(f64);
