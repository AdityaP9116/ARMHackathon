//! Mamba-3 SISO selective scan — types, validation, and backend dispatch.
//!
//! Semantics match `tests/reference/mamba3_ref.py`, which reproduces the
//! **official** `mamba3_siso_combined` Triton kernel to 4.47 bf16 ULP across
//! the ground-truth cases in `tests/golden/mamba3/` (captured from the real
//! `state-spaces/mamba3-siso-187m` checkpoint — see `MAMBA3_IMPLEMENTATION_PLAN.md`).
//!
//! # Why this is a separate module and not a flag on `selective_scan`
//!
//! Mamba-3 shares the *shape* of Mamba-1's recurrence (`h = a*h + b`) but
//! nothing else: a disjoint tensor set, a different memory layout, and a state
//! that is a **matrix per head** rather than a vector per channel. Overloading
//! one entry point would mean a signature half-ignored on every call with a
//! flag deciding which half. Same call the repo already made for the 2D scan.
//!
//! # The recurrence
//!
//! Per (batch, head), with `S` of shape `(dv, dqk)`:
//!
//! ```text
//!   lam_t   = sigmoid(trap_t)
//!   gamma_t = dt_t * lam_t                            weight of Bx_t at t
//!   scale_t = gamma_t + dt_{t+1} * (1 - lam_{t+1})    ...plus its weight at t+1
//!   alpha_t = exp(adt_t)                              adt <= 0
//!
//!   q_t = rope(q_t + q_bias, cos_t, sin_t)
//!   k_t = rope(k_t + k_bias, cos_t, sin_t)
//!
//!   y_t = alpha_t * (q_t . S^T)  +  (D + gamma_t * (q_t . k_t)) * v_t
//!   S   = alpha_t * S  +  scale_t * (v_t (x) k_t)
//!   out_t = y_t * silu(z_t)
//! ```
//!
//! `scale` collapsing the two contributions of `Bx_t` into one term carried by
//! the decay is what identifies this as the **paper's** trapezoidal form rather
//! than the community reimplementation's — confirmed against the official
//! kernel's own `Scale_store` buffer to 3.4e-8.
//!
//! # Layout contract (all row-major, fully contiguous)
//!
//! Chosen to match what the model and the goldens already emit, so **no
//! transposes are needed anywhere**. Note Mamba-3 is *time-major* where Mamba-1
//! is channel-major — which is why it needs `parallel::for_each_head` rather
//! than `for_each_channel`.
//!
//! ```text
//!   q, k            (batch, len, 1, dqk)      groups axis is 1 (SISO)
//!   v, z            (batch, len, heads, dv)   read strided per head — fine
//!   adt, dt, trap   (batch, heads, len)
//!   cos, sin        (batch, len, heads, dqk/2)
//!   q_bias, k_bias  (heads, dqk)
//!   d_skip          (heads,)
//!   out             (batch, heads, len, dv)   <-- HEAD-MAJOR, see below
//!   last_state      (batch, heads, dv, dqk)
//!   last_bx         (batch, heads, dqk)       the 2-tap carry — new in Mamba-3
//! ```
//!
//! **`out` is head-major while the inputs are time-major**, and that asymmetry
//! is deliberate. A head's outputs are strided through a `(batch, len, heads,
//! dv)` buffer, and a strided *mutable* view cannot be handed to a rayon worker
//! in safe code. Reads do not have that problem, so the inputs stay in the
//! model's native layout and only the output is permuted — once, in Python.
//! Head-major is also the better layout for this kernel regardless: the
//! recurrence is serial in `t` within a head, so writing contiguously per head
//! beats scattering across heads every timestep. See `parallel::for_each_head`.
//!
//! # Why `cos`/`sin` are inputs rather than computed here
//!
//! Upstream splits the rotation across two kernels: `angle_dt_fwd` accumulates
//! `theta = cumsum(tanh(angle) * PI * dt)`, and `mamba3_siso_fwd` consumes the
//! result. We mirror that split, so our goldens stay directly comparable and no
//! new NEON transcendental lands on the critical path. Fusing them is a
//! measured optimisation, not a prerequisite.

mod scalar;

use crate::{Backend, Float, ScanError, ScanOptions};

/// Which Mamba-3 block variant to run. `Mimo` is reserved so adding it later is
/// an addition rather than a refactor; it is rejected at validation today.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mamba3Variant {
    /// Single-input single-output — the published 187M/443M/893M/1.5B models.
    Siso,
}

/// Problem dimensions for the Mamba-3 scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mamba3Dims {
    pub batch: usize,
    pub heads: usize,
    /// Head dimension of `v`/`out` — the state matrix's row count.
    pub dv: usize,
    /// Head dimension of `q`/`k` — the state matrix's column count. Must be even
    /// (RoPE rotates lane pairs).
    pub dqk: usize,
    pub len: usize,
}

/// Borrowed input tensors. See the module docs for layouts.
pub struct Mamba3Input<'a, T> {
    pub q: &'a [T],
    pub k: &'a [T],
    pub v: &'a [T],
    /// `A * dt`, always <= 0 — the same precondition the NEON fast `exp` needs.
    pub adt: &'a [T],
    /// `dt`, post-softplus and therefore >= 0.
    pub dt: &'a [T],
    /// The trapezoid gate, **pre-sigmoid** — the kernel applies it, matching
    /// upstream, so a caller cannot accidentally apply it twice.
    pub trap: &'a [T],
    pub q_bias: &'a [T],
    pub k_bias: &'a [T],
    pub cos: &'a [T],
    pub sin: &'a [T],
    pub d_skip: Option<&'a [T]>,
    pub z: Option<&'a [T]>,
    /// Walk the sequence backward in time. Output for timestep `t` is still
    /// written at index `t` and the pointwise D-skip / z-gate still read index
    /// `t` — only the traversal order changes, exactly as `ScanInput::reverse`
    /// defines it for Mamba-1.
    ///
    /// **Mamba-3 makes this more than an index flip.** The trapezoid's second
    /// term reads the *next* timestep in scan order, so a forward scan pairs
    /// `t` with `t+1` and a backward scan pairs `t` with `t-1`. Getting that
    /// wrong is the highest-risk correctness item in this kernel; it is covered
    /// by flip-forward-flip equivalence in the property tests.
    ///
    /// This is the 1D half of both the bidirectional and the 2D cross-scan
    /// topologies — they are traversal orders over the same primitive.
    pub reverse: bool,
}

/// Errors specific to the Mamba-3 entry point. Shape problems reuse
/// [`ScanError`] so callers have one error type to handle.
pub type Mamba3Error = ScanError;

pub(crate) fn validate<T>(
    dims: &Mamba3Dims,
    input: &Mamba3Input<'_, T>,
    out: &[T],
    last_state: Option<&[T]>,
    last_bx: Option<&[T]>,
) -> Result<(), Mamba3Error> {
    let Mamba3Dims {
        batch,
        heads,
        dv,
        dqk,
        len,
    } = *dims;
    for (n, v) in [
        ("batch", batch),
        ("heads", heads),
        ("dv", dv),
        ("dqk", dqk),
        ("len", len),
    ] {
        if v == 0 {
            return Err(ScanError::ZeroDim(n));
        }
    }
    // RoPE rotates (2i, 2i+1) lane pairs, so an odd head dim has no meaning.
    if dqk % 2 != 0 {
        return Err(ScanError::BadLen {
            tensor: "dqk (must be even for RoPE lane pairs)",
            expected: dqk + 1,
            got: dqk,
        });
    }

    let blqk = batch * len * dqk; // q, k — groups axis is 1
    let blhv = batch * len * heads * dv; // v, z, out
    let bhl = batch * heads * len; // adt, dt, trap
    let blhr = batch * len * heads * (dqk / 2); // cos, sin
    let hqk = heads * dqk; // q_bias, k_bias

    let checks: [(&'static str, usize, usize); 10] = [
        ("q", input.q.len(), blqk),
        ("k", input.k.len(), blqk),
        ("v", input.v.len(), blhv),
        ("adt", input.adt.len(), bhl),
        ("dt", input.dt.len(), bhl),
        ("trap", input.trap.len(), bhl),
        ("q_bias", input.q_bias.len(), hqk),
        ("k_bias", input.k_bias.len(), hqk),
        ("cos", input.cos.len(), blhr),
        ("sin", input.sin.len(), blhr),
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
        ("d_skip", input.d_skip.map(<[T]>::len), heads),
        ("z", input.z.map(<[T]>::len), blhv),
        ("out", Some(out.len()), blhv),
        (
            "last_state",
            last_state.map(<[T]>::len),
            batch * heads * dv * dqk,
        ),
        ("last_bx", last_bx.map(<[T]>::len), batch * heads * dqk),
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

/// Run the Mamba-3 SISO scan with the best backend and threading for this
/// platform.
///
/// `last_state` (the `(dv, dqk)` matrix per head) and `last_bx` (the 2-tap
/// carry) must both be `Some` or both `None` — a resumed scan needs both, and
/// silently dropping one would produce a plausible but wrong continuation.
pub fn mamba3_scan<T: Float>(
    dims: &Mamba3Dims,
    input: &Mamba3Input<'_, T>,
    out: &mut [T],
    last_state: Option<&mut [T]>,
    last_bx: Option<&mut [T]>,
) -> Result<(), Mamba3Error> {
    mamba3_scan_with_options(
        dims,
        input,
        out,
        last_state,
        last_bx,
        ScanOptions::default(),
    )
}

/// Like [`mamba3_scan`], with explicit backend + threading control.
pub fn mamba3_scan_with_options<T: Float>(
    dims: &Mamba3Dims,
    input: &Mamba3Input<'_, T>,
    out: &mut [T],
    last_state: Option<&mut [T]>,
    last_bx: Option<&mut [T]>,
    opts: ScanOptions,
) -> Result<(), Mamba3Error> {
    validate(dims, input, out, last_state.as_deref(), last_bx.as_deref())?;
    if last_state.is_some() != last_bx.is_some() {
        return Err(ScanError::BadLen {
            tensor: "last_state and last_bx must both be Some or both None",
            expected: usize::from(last_state.is_some()),
            got: usize::from(last_bx.is_some()),
        });
    }
    match opts.backend {
        // NEON lands in M4; until then Auto resolves to scalar, which is
        // correct everywhere and is the permanent non-Arm fallback anyway.
        Backend::Scalar | Backend::Auto => {
            scalar::scan(dims, input, out, last_state, last_bx, opts.threading);
            Ok(())
        }
        Backend::Neon => Err(ScanError::BackendUnavailable(Backend::Neon)),
    }
}
