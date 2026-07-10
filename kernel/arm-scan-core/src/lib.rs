//! Arm-optimized Mamba selective-scan kernel — core library.
//!
//! Semantics match `selective_scan_ref` from `state-spaces/mamba`
//! (vendored at `tests/reference/selective_scan_ref.py` in this repo), which
//! in turn matches Hugging Face transformers' `MambaMixer.slow_forward`
//! bit-for-bit. Golden vectors generated from that reference live in
//! `tests/golden/` and are enforced by `tests/golden.rs`.
//!
//! Layout contract (all row-major, fully contiguous):
//!   u, delta, z, out : (batch, dim, len)
//!   a                : (dim, state)
//!   b, c             : (batch, groups, state, len)   — groups == 1 for the
//!                      standard Mamba (B, N, L) case, which is the same
//!                      memory layout
//!   d_skip, delta_bias : (dim,)
//!   last_state       : (batch, dim, state)

mod float;
mod scalar;

pub use float::Float;

/// Problem dimensions. `groups` must divide `dim`; channel `d` reads
/// b/c group `d / (dim / groups)` (matches `repeat_interleave` upstream).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScanDims {
    pub batch: usize,
    pub dim: usize,
    pub len: usize,
    pub state: usize,
    pub groups: usize,
}

/// Borrowed input tensors. See module docs for layouts.
pub struct ScanInput<'a, T> {
    pub u: &'a [T],
    pub delta: &'a [T],
    pub a: &'a [T],
    pub b: &'a [T],
    pub c: &'a [T],
    pub d_skip: Option<&'a [T]>,
    pub z: Option<&'a [T]>,
    pub delta_bias: Option<&'a [T]>,
    /// Apply softplus to (delta + delta_bias) inside the kernel (fused
    /// discretization). HF's slow path pre-applies softplus, so the patch
    /// layer passes `false`; mamba-ssm call sites pass `true`.
    pub delta_softplus: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanError {
    ZeroDim(&'static str),
    GroupsDontDivideDim {
        dim: usize,
        groups: usize,
    },
    BadLen {
        tensor: &'static str,
        expected: usize,
        got: usize,
    },
}

impl core::fmt::Display for ScanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ScanError::ZeroDim(name) => write!(f, "dimension `{name}` is zero"),
            ScanError::GroupsDontDivideDim { dim, groups } => {
                write!(f, "groups ({groups}) does not divide dim ({dim})")
            }
            ScanError::BadLen {
                tensor,
                expected,
                got,
            } => {
                write!(
                    f,
                    "tensor `{tensor}` has {got} elements, expected {expected}"
                )
            }
        }
    }
}

impl std::error::Error for ScanError {}

fn validate<T>(
    dims: &ScanDims,
    input: &ScanInput<'_, T>,
    out: &[T],
    last_state: Option<&[T]>,
) -> Result<(), ScanError> {
    let ScanDims {
        batch,
        dim,
        len,
        state,
        groups,
    } = *dims;
    for (n, v) in [
        ("batch", batch),
        ("dim", dim),
        ("len", len),
        ("state", state),
        ("groups", groups),
    ] {
        if v == 0 {
            return Err(ScanError::ZeroDim(n));
        }
    }
    if dim % groups != 0 {
        return Err(ScanError::GroupsDontDivideDim { dim, groups });
    }
    let bdl = batch * dim * len;
    let bgnl = batch * groups * state * len;
    let checks: [(&'static str, usize, usize); 5] = [
        ("u", input.u.len(), bdl),
        ("delta", input.delta.len(), bdl),
        ("a", input.a.len(), dim * state),
        ("b", input.b.len(), bgnl),
        ("c", input.c.len(), bgnl),
    ];
    for (tensor, got, expected) in checks {
        if got != expected {
            return Err(ScanError::BadLen {
                tensor,
                expected,
                got,
            });
        }
    }
    let optional: [(&'static str, Option<usize>, usize); 5] = [
        ("d_skip", input.d_skip.map(<[T]>::len), dim),
        ("z", input.z.map(<[T]>::len), bdl),
        ("delta_bias", input.delta_bias.map(<[T]>::len), dim),
        ("out", Some(out.len()), bdl),
        (
            "last_state",
            last_state.map(<[T]>::len),
            batch * dim * state,
        ),
    ];
    for (tensor, got, expected) in optional {
        if let Some(got) = got {
            if got != expected {
                return Err(ScanError::BadLen {
                    tensor,
                    expected,
                    got,
                });
            }
        }
    }
    Ok(())
}

/// Run the selective scan. Writes the gated output into `out` and, when
/// requested, the final SSM state into `last_state`.
///
/// This is the scalar implementation; NEON/chunked/threaded variants slot
/// in behind this same signature in later phases.
pub fn selective_scan<T: Float>(
    dims: &ScanDims,
    input: &ScanInput<'_, T>,
    out: &mut [T],
    last_state: Option<&mut [T]>,
) -> Result<(), ScanError> {
    validate(dims, input, out, last_state.as_deref())?;
    scalar::scan(dims, input, out, last_state);
    Ok(())
}
