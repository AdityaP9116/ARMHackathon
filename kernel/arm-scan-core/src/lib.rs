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
    /// Walk the sequence backward in time: the state starts at zero at the
    /// END of the sequence and accumulates toward the start. Output for
    /// timestep `t` is still written at index `t`, and the pointwise D-skip
    /// and z-gate still apply at index `t` — only the recurrence's traversal
    /// order changes, never the layout.
    ///
    /// Equivalent to reversing the time axis of u/delta/b/c/z, running a forward
    /// scan, and reversing the output — but without materializing any of those
    /// copies. That equivalence is the definition, enforced by
    /// `reverse_matches_flip_forward_flip` in `tests/property.rs` (and,
    /// independently, by `tests/check_bidirectional_math.py` in numpy).
    ///
    /// On the scalar backend the two are **bit-identical**. On NEON they agree to
    /// ~1e-7, not bit-exactly: `discretize_chunk` and `epilogue_row` process 4
    /// timesteps at a time with a scalar tail, and the vector and tail branches
    /// evaluate softplus/SiLU by different means (NEON polynomial vs libm).
    /// Which branch a timestep takes depends on its array POSITION, so flipping
    /// the array moves timesteps across that boundary. That is a property of the
    /// existing forward kernel, not of `reverse`.
    ///
    /// `last_state` under reverse is the state after consuming `t == 0` — the
    /// state at the START of the sequence. It is not a resumable decode cache
    /// the way the forward scan's `last_state` is.
    ///
    /// This is the 1D half of the bidirectional / 2D cross-scan topologies; see
    /// TOPOLOGY_IMPLEMENTATION_PLAN.md.
    pub reverse: bool,
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
    last_state: Option<&mut [T]>,
    opts: ScanOptions,
) -> Result<(), ScanError> {
    selective_scan_with_state(dims, input, out, last_state, None, opts)
}

/// Like [`selective_scan_with_options`], but seeds the recurrence from a
/// caller-provided initial state `h0` (shape `(batch, dim, state)`, the same
/// layout as `last_state`) instead of zeros.
///
/// This is what makes the scan resumable: run a prefix, take its `last_state`,
/// feed it back as `h0` for the next segment, and the concatenated output is
/// identical to scanning the whole sequence at once. It is the kernel half of
/// streaming / autoregressive decode. Pass `None` for the default
/// zero-initialized behavior.
pub fn selective_scan_with_state<T: Float>(
    dims: &ScanDims,
    input: &ScanInput<'_, T>,
    out: &mut [T],
    // `mut` is only exercised by the aarch64 NEON dispatch below
    #[cfg_attr(not(target_arch = "aarch64"), allow(unused_mut))] mut last_state: Option<&mut [T]>,
    h0: Option<&[T]>,
    opts: ScanOptions,
) -> Result<(), ScanError> {
    validate(dims, input, out, last_state.as_deref())?;
    if let Some(h0) = h0 {
        let expected = dims.batch * dims.dim * dims.state;
        if h0.len() != expected {
            return Err(ScanError::BadLen {
                tensor: "h0",
                expected,
                got: h0.len(),
            });
        }
    }
    let threading = opts.threading;
    match opts.backend {
        Backend::Scalar => {
            scalar::scan(dims, input, h0, out, last_state, threading);
            Ok(())
        }
        Backend::Auto => {
            #[cfg(target_arch = "aarch64")]
            if try_neon(dims, input, h0, out, &mut last_state, threading) {
                return Ok(());
            }
            scalar::scan(dims, input, h0, out, last_state, threading);
            Ok(())
        }
        Backend::Neon => {
            #[cfg(target_arch = "aarch64")]
            if try_neon(dims, input, h0, out, &mut last_state, threading) {
                return Ok(());
            }
            Err(ScanError::BackendUnavailable(Backend::Neon))
        }
    }
}

/// Fused bidirectional scan: produce both the forward and backward outputs from
/// one set of inputs, computing the shared, direction-independent Pass A
/// (discretize + exp + input projection — ~85% of the work) **once** instead of
/// twice. `out_fwd`/`out_bwd` are `(batch, dim, len)`; `last_fwd`/`last_bwd` are
/// `(batch, dim, state)` and must both be `Some` or both `None`.
///
/// Semantically equal to calling [`selective_scan`] twice (once forward, once
/// with `reverse: true`) and is checked bit-for-bit against exactly that. The
/// backward output's `last_state` is the state after consuming `t == 0` (the
/// start of the sequence). No `h0`: both directions seed from zero.
/// `input.reverse` is ignored.
///
/// See `BIDIRECTIONAL_SPEEDUP_IDEAS.md §3.2` for why sharing Pass A is the win,
/// and `TOPOLOGY_IMPLEMENTATION_PLAN.md §3.2` — this is the 1D form of the SS2D
/// "read once, emit multiple directions" structure.
pub fn selective_scan_bidirectional<T: Float>(
    dims: &ScanDims,
    input: &ScanInput<'_, T>,
    out_fwd: &mut [T],
    out_bwd: &mut [T],
    #[cfg_attr(not(target_arch = "aarch64"), allow(unused_mut))] mut last_fwd: Option<&mut [T]>,
    #[cfg_attr(not(target_arch = "aarch64"), allow(unused_mut))] mut last_bwd: Option<&mut [T]>,
    opts: ScanOptions,
) -> Result<(), ScanError> {
    validate(dims, input, out_fwd, last_fwd.as_deref())?;
    let bdl = dims.batch * dims.dim * dims.len;
    if out_bwd.len() != bdl {
        return Err(ScanError::BadLen {
            tensor: "out_bwd",
            expected: bdl,
            got: out_bwd.len(),
        });
    }
    let bdn = dims.batch * dims.dim * dims.state;
    if let Some(lb) = last_bwd.as_deref() {
        if lb.len() != bdn {
            return Err(ScanError::BadLen {
                tensor: "last_bwd",
                expected: bdn,
                got: lb.len(),
            });
        }
    }
    if last_fwd.is_some() != last_bwd.is_some() {
        // The kernel requires both or neither; report it as a shape error
        // rather than panicking in the parallel driver.
        return Err(ScanError::BadLen {
            tensor: "last_bwd",
            expected: if last_fwd.is_some() { bdn } else { 0 },
            got: if last_bwd.is_some() { bdn } else { 0 },
        });
    }

    match opts.backend {
        Backend::Scalar => {
            scalar::scan_bidirectional(
                dims,
                input,
                out_fwd,
                out_bwd,
                last_fwd,
                last_bwd,
                opts.threading,
            );
            Ok(())
        }
        Backend::Auto => {
            #[cfg(target_arch = "aarch64")]
            if try_neon_bidir(
                dims,
                input,
                out_fwd,
                out_bwd,
                &mut last_fwd,
                &mut last_bwd,
                opts.threading,
            ) {
                return Ok(());
            }
            scalar::scan_bidirectional(
                dims,
                input,
                out_fwd,
                out_bwd,
                last_fwd,
                last_bwd,
                opts.threading,
            );
            Ok(())
        }
        Backend::Neon => {
            #[cfg(target_arch = "aarch64")]
            if try_neon_bidir(
                dims,
                input,
                out_fwd,
                out_bwd,
                &mut last_fwd,
                &mut last_bwd,
                opts.threading,
            ) {
                return Ok(());
            }
            Err(ScanError::BackendUnavailable(Backend::Neon))
        }
    }
}

/// NEON dispatch for the fused bidirectional scan — the `T == f32` analog of
/// [`try_neon`], returning `false` for other element types so the caller falls
/// back to scalar.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn try_neon_bidir<T: Float>(
    dims: &ScanDims,
    input: &ScanInput<'_, T>,
    out_fwd: &mut [T],
    out_bwd: &mut [T],
    last_fwd: &mut Option<&mut [T]>,
    last_bwd: &mut Option<&mut [T]>,
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
        reverse: input.reverse,
    };
    neon::scan_bidirectional(
        dims,
        &input_f32,
        cast_mut(out_fwd),
        cast_mut(out_bwd),
        last_fwd.as_deref_mut().map(cast_mut),
        last_bwd.as_deref_mut().map(cast_mut),
        threading,
    );
    true
}

/// Route to the NEON implementation when `T` is f32. The `TypeId` check
/// proves `T == f32`, making the slice reinterpretations sound.
#[cfg(target_arch = "aarch64")]
fn try_neon<T: Float>(
    dims: &ScanDims,
    input: &ScanInput<'_, T>,
    h0: Option<&[T]>,
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
        reverse: input.reverse,
    };
    neon::scan(
        dims,
        &input_f32,
        h0.map(cast),
        cast_mut(out),
        last_state.as_deref_mut().map(cast_mut),
        threading,
    );
    true
}
