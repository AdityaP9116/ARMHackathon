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
