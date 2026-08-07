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
//!   last_bx         (batch, heads, dqk)       see the warning below
//! ```
//!
//! # `last_bx` does NOT make this scan resumable — read this before using it
//!
//! It holds `scale_T * k_T` from the final step of the traversal, and it is
//! written but never read: the recurrence needs no such carry, because
//! `scale_t` already folds both of `Bx_t`'s contributions into one weight.
//!
//! It is **not sufficient to resume a split sequence**, and Mamba-3 cannot be
//! made resumable by any carry of this shape. The reason is structural:
//!
//! ```text
//!   scale_t = dt_t*lam_t + dt_{t+1}*(1 - lam_{t+1})
//!                          ^^^^^^^^^^^^^^^^^^^^^^^^ the NEXT timestep
//! ```
//!
//! The trapezoid looks **forward**, so the last step of a segment cannot be
//! finished without the first step of the segment that follows. A resumable
//! Mamba-3 needs *lookahead into the next chunk* — an extra input — not a
//! carry out of the previous one. Mamba-1's `h0`/`last_state` contract does
//! not transfer.
//!
//! The field is kept because the plumbing is already through the C ABI and a
//! future chunked path will want somewhere to put its carry. Until that path
//! exists, treat a non-null `last_bx` as diagnostic output only. The Python
//! layer deliberately does not expose it.
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

mod mimo;
mod scalar;
mod tiled;

#[cfg(target_arch = "aarch64")]
pub(crate) use tiled::TILE;

use crate::{Backend, Float, ScanError, ScanOptions};

/// Which Mamba-3 block variant to run. Determined by the input, not chosen:
/// see [`Mamba3Input::variant`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mamba3Variant {
    /// Single-input single-output — the published `mamba3-siso-*` models.
    Siso,
    /// Rank-`r` multi-input multi-output — the published `mamba3-mimo-*`
    /// models. A **shared** state updated with `r` outer products per step, so
    /// `r` times the arithmetic on one state load. That ratio is the reason
    /// this variant is interesting on a CPU at all.
    Mimo,
}

/// The three per-(head, rank) projections that define a MIMO block.
///
/// Grouped into one struct so "all three or none" is a property of the type
/// rather than something validation has to check — passing two of them is not
/// a representable state.
///
/// All three are `(heads, rank, dv)`, row-major. They are *elementwise*
/// reweightings of the head dimension, not matmuls: upstream builds them as
/// `Psi`/`Zeta`/`Phi` and multiplies pointwise.
pub struct Mamba3Mimo<'a, T> {
    /// `Psi` — input projection. `x_r[p] = psi[h][r][p] * v[p]`.
    pub psi: &'a [T],
    /// `Zeta` — gate projection. The gate is `silu(z[p] * zeta[h][r][p])`,
    /// applied per rank **before** `phi` reduces the rank axis.
    pub zeta: &'a [T],
    /// `Phi` — output projection. Multiplies each rank's result, then the
    /// ranks are summed away.
    pub phi: &'a [T],
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
    /// MIMO rank. **1 for SISO**, and the two are not interchangeable even at
    /// `rank == 1`: the families rotate different lane pairs (see
    /// [`Mamba3Mimo`] and the `rope` functions). `rank > 1` requires the MIMO
    /// projections to be present.
    pub rank: usize,
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
    /// Present iff this is a MIMO block. See [`Mamba3Mimo`].
    pub mimo: Option<Mamba3Mimo<'a, T>>,
}

impl<T> Mamba3Input<'_, T> {
    /// Which variant these inputs describe. The caller does not choose it —
    /// supplying the projections *is* the choice.
    pub fn variant(&self) -> Mamba3Variant {
        if self.mimo.is_some() {
            Mamba3Variant::Mimo
        } else {
            Mamba3Variant::Siso
        }
    }
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
        rank,
    } = *dims;
    for (n, v) in [
        ("batch", batch),
        ("heads", heads),
        ("dv", dv),
        ("dqk", dqk),
        ("len", len),
        ("rank", rank),
    ] {
        if v == 0 {
            return Err(ScanError::ZeroDim(n));
        }
    }
    // A rank above 1 has no meaning without the projections that consume it,
    // and SISO has no place to put them. Reject the mismatch rather than
    // silently ignoring one side.
    if rank > 1 && input.mimo.is_none() {
        return Err(ScanError::BadLen {
            tensor: "rank > 1 requires the MIMO projections (psi/zeta/phi)",
            expected: 1,
            got: rank,
        });
    }
    // RoPE rotates (2i, 2i+1) lane pairs, so an odd head dim has no meaning.
    if dqk % 2 != 0 {
        return Err(ScanError::BadLen {
            tensor: "dqk (must be even for RoPE lane pairs)",
            expected: dqk + 1,
            got: dqk,
        });
    }

    // Every formula below is written with `rank` in it and reduces to the SISO
    // expression at rank 1, so there is one set of shape rules, not two.
    let blqk = batch * len * rank * dqk; // q, k — groups axis is 1
    let blhv = batch * len * heads * dv; // v, z, out
    let bhl = batch * heads * len; // adt, dt, trap
    let blhr = batch * len * heads * (dqk / 2); // cos, sin
    let hqk = heads * rank * dqk; // q_bias, k_bias
    let hrv = heads * rank * dv; // psi, zeta, phi

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

    let optional: [(&'static str, Option<usize>, usize); 8] = [
        ("d_skip", input.d_skip.map(<[T]>::len), heads),
        ("z", input.z.map(<[T]>::len), blhv),
        ("out", Some(out.len()), blhv),
        ("mimo psi", input.mimo.as_ref().map(|m| m.psi.len()), hrv),
        ("mimo zeta", input.mimo.as_ref().map(|m| m.zeta.len()), hrv),
        ("mimo phi", input.mimo.as_ref().map(|m| m.phi.len()), hrv),
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
    #[cfg_attr(not(target_arch = "aarch64"), allow(unused_mut))] mut last_state: Option<&mut [T]>,
    #[cfg_attr(not(target_arch = "aarch64"), allow(unused_mut))] mut last_bx: Option<&mut [T]>,
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
    // MIMO has one implementation today: the portable scalar path. It is
    // correct on every target and is what the goldens gate. Routing it here —
    // before the backend match — keeps `Backend::Neon` from reporting a
    // MIMO-shaped call as "NEON unavailable", which would be true but useless.
    // A blocked/NEON MIMO path is B3 work; until it exists, saying so plainly
    // beats silently running something slower than the caller asked for.
    if input.variant() == Mamba3Variant::Mimo {
        mimo::scan(dims, input, out, last_state, last_bx, opts.threading);
        return Ok(());
    }
    match opts.backend {
        // NEON lands in M4; until then Auto resolves to scalar, which is
        // correct everywhere and is the permanent non-Arm fallback anyway.
        // The naive scalar path is the oracle; the blocked path is the
        // structural twin of the NEON kernel and is what `Auto` should use once
        // NEON lands, so exercise it by default now to keep it honest.
        Backend::Scalar => {
            scalar::scan(dims, input, out, last_state, last_bx, opts.threading);
            Ok(())
        }
        Backend::Auto => {
            #[cfg(target_arch = "aarch64")]
            if try_neon(
                dims,
                input,
                out,
                &mut last_state,
                &mut last_bx,
                opts.threading,
            ) {
                return Ok(());
            }
            // Non-aarch64, or f64 (the NEON path is f32-only): the blocked
            // portable kernel, which is the same algorithm.
            tiled::scan(dims, input, out, last_state, last_bx, opts.threading);
            Ok(())
        }
        Backend::Neon => {
            #[cfg(target_arch = "aarch64")]
            if try_neon(
                dims,
                input,
                out,
                &mut last_state,
                &mut last_bx,
                opts.threading,
            ) {
                return Ok(());
            }
            Err(ScanError::BackendUnavailable(Backend::Neon))
        }
    }
}

/// Dispatch the f32 NEON kernel, or report that it does not apply.
///
/// The public API is generic over `Float` so tests can run the whole kernel in
/// f64, but NEON is f32-only. Mirrors `crate::try_neon` for the Mamba-1 path:
/// verify the type identity, then reinterpret the slices — same pointer, same
/// length, no copy.
#[cfg(target_arch = "aarch64")]
fn try_neon<T: Float>(
    dims: &Mamba3Dims,
    input: &Mamba3Input<'_, T>,
    out: &mut [T],
    last_state: &mut Option<&mut [T]>,
    last_bx: &mut Option<&mut [T]>,
    threading: crate::Threading,
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
    let input_f32 = Mamba3Input {
        q: cast(input.q),
        k: cast(input.k),
        v: cast(input.v),
        adt: cast(input.adt),
        dt: cast(input.dt),
        trap: cast(input.trap),
        q_bias: cast(input.q_bias),
        k_bias: cast(input.k_bias),
        cos: cast(input.cos),
        sin: cast(input.sin),
        d_skip: input.d_skip.map(cast),
        z: input.z.map(cast),
        reverse: input.reverse,
        // Always None here: dispatch routes MIMO to `mimo::scan` before the
        // backend match ever runs, so the NEON path only ever sees SISO. If a
        // NEON MIMO kernel lands, this is one of the two places to change.
        mimo: None,
    };
    crate::neon::mamba3::scan(
        dims,
        &input_f32,
        cast_mut(out),
        last_state.as_mut().map(|s| cast_mut(s)),
        last_bx.as_mut().map(|s| cast_mut(s)),
        threading,
    );
    true
}
