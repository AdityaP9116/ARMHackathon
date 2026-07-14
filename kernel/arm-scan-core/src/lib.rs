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
#[cfg(target_arch = "aarch64")]
mod neon;
mod parallel;
mod scalar;

pub use float::Float;

/// Phase-level profiling of the NEON fast path. Diagnostic tool, not part of
/// the stable API — only present on aarch64 with the `profiling` feature.
/// See `neon/profile.rs` and `PROFILING.md`.
#[cfg(all(target_arch = "aarch64", feature = "profiling"))]
pub use neon::profile::{scan_profiled, PhaseTimings};

/// Which implementation to run. `Auto` picks the fastest correct backend
/// for the platform and element type; the explicit variants exist for A/B
/// parity testing and benchmark ladders.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// NEON for f32 on aarch64, scalar otherwise.
    Auto,
    /// Portable scalar implementation (any platform, f32 or f64).
    Scalar,
    /// NEON implementation. Errors with [`ScanError::BackendUnavailable`]
    /// off aarch64 or for non-f32 element types.
    Neon,
}

/// Channel-level threading policy. Parallel runs are bit-identical to
/// sequential ones (channels are fully independent; no cross-thread
/// reductions), enforced by tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Threading {
    /// Single-threaded.
    Sequential,
    /// Always use the rayon pool (respects `RAYON_NUM_THREADS`). Falls
    /// back to sequential when the `parallel` feature is disabled.
    Rayon,
    /// Rayon for large problems, sequential below the work threshold
    /// where scheduling overhead would dominate.
    Auto,
}

/// Backend + threading selection for [`selective_scan_with_options`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScanOptions {
    pub backend: Backend,
    pub threading: Threading,
}

impl Default for ScanOptions {
    fn default() -> Self {
        ScanOptions {
            backend: Backend::Auto,
            threading: Threading::Auto,
        }
    }
}

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
    /// The requested backend cannot run on this platform / element type.
    BackendUnavailable(Backend),
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
            ScanError::BackendUnavailable(b) => {
                write!(
                    f,
                    "backend {b:?} is unavailable on this platform/element type"
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

/// Run the selective scan with the best backend and threading for this
/// platform. Writes the gated output into `out` and, when requested, the
/// final SSM state into `last_state`.
pub fn selective_scan<T: Float>(
    dims: &ScanDims,
    input: &ScanInput<'_, T>,
    out: &mut [T],
    last_state: Option<&mut [T]>,
) -> Result<(), ScanError> {
    selective_scan_with_options(dims, input, out, last_state, ScanOptions::default())
}

/// Run the selective scan on an explicitly chosen backend with automatic
/// threading (for parity tests and benchmark ladders).
pub fn selective_scan_with_backend<T: Float>(
    dims: &ScanDims,
    input: &ScanInput<'_, T>,
    out: &mut [T],
    last_state: Option<&mut [T]>,
    backend: Backend,
) -> Result<(), ScanError> {
    selective_scan_with_options(
        dims,
        input,
        out,
        last_state,
        ScanOptions {
            backend,
            threading: Threading::Auto,
        },
    )
}

/// Run the selective scan with full backend + threading control.
pub fn selective_scan_with_options<T: Float>(
    dims: &ScanDims,
    input: &ScanInput<'_, T>,
    out: &mut [T],
    // `mut` is only exercised by the aarch64 NEON dispatch below
    #[cfg_attr(not(target_arch = "aarch64"), allow(unused_mut))] mut last_state: Option<&mut [T]>,
    opts: ScanOptions,
) -> Result<(), ScanError> {
    validate(dims, input, out, last_state.as_deref())?;
    let threading = opts.threading;
    match opts.backend {
        Backend::Scalar => {
            scalar::scan(dims, input, out, last_state, threading);
            Ok(())
        }
        Backend::Auto => {
            #[cfg(target_arch = "aarch64")]
            if try_neon(dims, input, out, &mut last_state, threading) {
                return Ok(());
            }
            scalar::scan(dims, input, out, last_state, threading);
            Ok(())
        }
        Backend::Neon => {
            #[cfg(target_arch = "aarch64")]
            if try_neon(dims, input, out, &mut last_state, threading) {
                return Ok(());
            }
            Err(ScanError::BackendUnavailable(Backend::Neon))
        }
    }
}

/// Route to the NEON implementation when `T` is f32. The `TypeId` check
/// proves `T == f32`, making the slice reinterpretations sound.
#[cfg(target_arch = "aarch64")]
fn try_neon<T: Float>(
    dims: &ScanDims,
    input: &ScanInput<'_, T>,
    out: &mut [T],
    last_state: &mut Option<&mut [T]>,
    threading: Threading,
) -> bool {
    use core::any::TypeId;
    if TypeId::of::<T>() != TypeId::of::<f32>() {
        return false;
    }
    fn cast<T: 'static>(s: &[T]) -> &[f32] {
        // SAFETY: caller verified T == f32; same pointer, same length.
        unsafe { core::slice::from_raw_parts(s.as_ptr().cast::<f32>(), s.len()) }
    }
    fn cast_mut<T: 'static>(s: &mut [T]) -> &mut [f32] {
        // SAFETY: as above.
        unsafe { core::slice::from_raw_parts_mut(s.as_mut_ptr().cast::<f32>(), s.len()) }
    }
    let input_f32 = ScanInput {
        u: cast(input.u),
        delta: cast(input.delta),
        a: cast(input.a),
        b: cast(input.b),
        c: cast(input.c),
        d_skip: input.d_skip.map(cast),
        z: input.z.map(cast),
        delta_bias: input.delta_bias.map(cast),
        delta_softplus: input.delta_softplus,
    };
    neon::scan(
        dims,
        &input_f32,
        cast_mut(out),
        last_state.as_deref_mut().map(cast_mut),
        threading,
    );
    true
}
