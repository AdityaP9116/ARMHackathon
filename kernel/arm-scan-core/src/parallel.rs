//! Channel-level parallelism.
//!
//! Every (batch, channel) pair of the selective scan is fully independent —
//! no shared mutable state, no reductions — so parallelizing across them is
//! deterministic: outputs are bit-identical to the sequential run no matter
//! the thread count or schedule (enforced by `tests/property.rs`).
//!
//! Work is dealt to rayon as disjoint `out`/`last_state` row chunks; each
//! worker gets its own scratch via `for_each_init`, so per-channel work
//! allocates nothing.

use crate::{Float, Threading};

/// Below this many total lane-steps (channels x len x state), thread spawn
/// and scheduling overhead beats the parallel win; stay sequential.
const PARALLEL_WORK_THRESHOLD: usize = 1 << 17;

pub(crate) fn should_parallelize(n_channels: usize, len: usize, state: usize) -> bool {
    n_channels >= 2 && n_channels * len * state >= PARALLEL_WORK_THRESHOLD
}

/// Run `f(scratch, channel_index, out_row, last_state_row)` for every
/// channel, sequentially or via rayon per `threading`.
pub(crate) fn for_each_channel<T, S, I, F>(
    len: usize,
    state: usize,
    out: &mut [T],
    last_state: Option<&mut [T]>,
    threading: Threading,
    init: I,
    f: F,
) where
    T: Float,
    S: Send,
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize, &mut [T], Option<&mut [T]>) + Sync + Send,
{
    let n_channels = out.len() / len;
    let parallel = match threading {
        Threading::Sequential => false,
        Threading::Rayon => cfg!(feature = "parallel"),
        Threading::Auto => cfg!(feature = "parallel") && should_parallelize(n_channels, len, state),
    };

    if !parallel {
        let mut scratch = init();
        match last_state {
            Some(ls) => {
                for (i, (o, l)) in out
                    .chunks_exact_mut(len)
                    .zip(ls.chunks_exact_mut(state))
                    .enumerate()
                {
                    f(&mut scratch, i, o, Some(l));
                }
            }
            None => {
                for (i, o) in out.chunks_exact_mut(len).enumerate() {
                    f(&mut scratch, i, o, None);
                }
            }
        }
        return;
    }

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        match last_state {
            Some(ls) => {
                out.par_chunks_exact_mut(len)
                    .zip_eq(ls.par_chunks_exact_mut(state))
                    .enumerate()
                    .for_each_init(&init, |s, (i, (o, l))| f(s, i, o, Some(l)));
            }
            None => {
                out.par_chunks_exact_mut(len)
                    .enumerate()
                    .for_each_init(&init, |s, (i, o)| f(s, i, o, None));
            }
        }
    }
    #[cfg(not(feature = "parallel"))]
    unreachable!("parallel=true requires the `parallel` feature");
}

/// Mamba-3's work heuristic. The per-timestep cost is the rank-1 state update
/// plus the output projection, `~3 * dv * dqk` FMAs — not the `state` lane-steps
/// [`should_parallelize`] counts, so it needs its own threshold rather than a
/// reinterpretation of the old one.
pub(crate) fn should_parallelize_heads(n_heads: usize, len: usize, dv: usize, dqk: usize) -> bool {
    n_heads >= 2 && n_heads.saturating_mul(len).saturating_mul(dv * dqk) >= PARALLEL_WORK_THRESHOLD
}

/// Run `f(scratch, head_index, out_row, last_state_row, last_bx_row)` for every
/// `(batch, head)` pair — the Mamba-3 equivalent of [`for_each_channel`].
///
/// # Why this exists instead of reusing `for_each_channel`
///
/// That driver derives `n_channels = out.len() / len`, which assumes **one
/// output scalar per (channel, timestep)**, and chunks the final state by a flat
/// `state`. Mamba-3 emits a `dv`-vector per (head, timestep) and carries a
/// `(dv, dqk)` matrix *plus* a `dqk` 2-tap vector. Neither shape is expressible
/// through the old signature. The bidirectional topology needed a sibling for
/// the same reason ([`for_each_channel_bidir`]); this follows that precedent.
///
/// # Why `out` is head-major here
///
/// The public Mamba-3 layout is time-major `(batch, len, heads, dv)`, matching
/// the goldens. A given head's outputs are therefore **strided** through that
/// buffer, and a strided mutable view cannot be handed to a rayon worker in safe
/// code — `chunks_exact_mut` only yields contiguous slices. So the kernel writes
/// `(batch, heads, len, dv)` and the Python layer permutes once.
///
/// That is not purely a safety concession: head-major is also the better layout
/// *for this kernel*, because the recurrence is serial in `t` within a head, so
/// writing contiguously per head beats scattering across heads every timestep.
/// The permute is O(output) and measurable; eliminating it is a later lever, not
/// a correctness issue.
///
/// Parallel output is bit-identical to sequential: `(batch, head)` pairs are
/// fully independent, with no shared mutable state and no cross-thread
/// reduction, so a worker owns disjoint `out` / `last_*` rows regardless of
/// scheduling.
#[allow(clippy::too_many_arguments)]
pub(crate) fn for_each_head<T, S, I, F>(
    len: usize,
    dv: usize,
    dqk: usize,
    out: &mut [T],
    last_state: Option<&mut [T]>,
    last_bx: Option<&mut [T]>,
    threading: Threading,
    init: I,
    f: F,
) where
    T: Float,
    S: Send,
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize, &mut [T], Option<&mut [T]>, Option<&mut [T]>) + Sync + Send,
{
    let row = len * dv;
    let n_heads = out.len() / row;
    let parallel = match threading {
        Threading::Sequential => false,
        Threading::Rayon => cfg!(feature = "parallel"),
        Threading::Auto => {
            cfg!(feature = "parallel") && should_parallelize_heads(n_heads, len, dv, dqk)
        }
    };

    // Collapse the carry pair to "have them" vs "don't"; the caller has already
    // been rejected if only one was supplied.
    let carry = match (last_state, last_bx) {
        (Some(ls), Some(lb)) => Some((ls, lb)),
        (None, None) => None,
        _ => unreachable!("mamba3: last_state and last_bx must both be Some or both None"),
    };

    if !parallel {
        let mut scratch = init();
        match carry {
            Some((ls, lb)) => {
                for (i, ((o, s), b)) in out
                    .chunks_exact_mut(row)
                    .zip(ls.chunks_exact_mut(dv * dqk))
                    .zip(lb.chunks_exact_mut(dqk))
                    .enumerate()
                {
                    f(&mut scratch, i, o, Some(s), Some(b));
                }
            }
            None => {
                for (i, o) in out.chunks_exact_mut(row).enumerate() {
                    f(&mut scratch, i, o, None, None);
                }
            }
        }
        return;
    }

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        match carry {
            Some((ls, lb)) => {
                out.par_chunks_exact_mut(row)
                    .zip_eq(ls.par_chunks_exact_mut(dv * dqk))
                    .zip_eq(lb.par_chunks_exact_mut(dqk))
                    .enumerate()
                    .for_each_init(&init, |s, (i, ((o, st), bx))| {
                        f(s, i, o, Some(st), Some(bx))
                    });
            }
            None => {
                out.par_chunks_exact_mut(row)
                    .enumerate()
                    .for_each_init(&init, |s, (i, o)| f(s, i, o, None, None));
            }
        }
    }
    #[cfg(not(feature = "parallel"))]
    unreachable!("parallel=true requires the `parallel` feature");
}

/// Like [`for_each_channel`], but for the fused bidirectional scan: each channel
/// produces TWO output rows (forward and backward) and, optionally, two final
/// states. Parallelism is still across channels — each channel computes the
/// shared Pass A once and both directions' recurrences — so a rayon worker owns
/// disjoint `out_fwd`/`out_bwd`/`last_*` rows and the output is schedule-
/// independent, exactly as the single-output driver guarantees.
///
/// `last_fwd` and `last_bwd` must both be `Some` or both `None`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn for_each_channel_bidir<T, S, I, F>(
    len: usize,
    state: usize,
    out_fwd: &mut [T],
    out_bwd: &mut [T],
    last_fwd: Option<&mut [T]>,
    last_bwd: Option<&mut [T]>,
    threading: Threading,
    init: I,
    f: F,
) where
    T: Float,
    S: Send,
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize, &mut [T], &mut [T], Option<&mut [T]>, Option<&mut [T]>) + Sync + Send,
{
    let n_channels = out_fwd.len() / len;
    let parallel = match threading {
        Threading::Sequential => false,
        Threading::Rayon => cfg!(feature = "parallel"),
        Threading::Auto => cfg!(feature = "parallel") && should_parallelize(n_channels, len, state),
    };

    // Collapse the four last-state cases to "have them" vs "don't".
    let last = match (last_fwd, last_bwd) {
        (Some(lf), Some(lb)) => Some((lf, lb)),
        (None, None) => None,
        _ => unreachable!("fused bidir: last_fwd and last_bwd must both be Some or both None"),
    };

    if !parallel {
        let mut scratch = init();
        match last {
            Some((lf, lb)) => {
                for (i, (((of, ob), lf), lb)) in out_fwd
                    .chunks_exact_mut(len)
                    .zip(out_bwd.chunks_exact_mut(len))
                    .zip(lf.chunks_exact_mut(state))
                    .zip(lb.chunks_exact_mut(state))
                    .enumerate()
                {
                    f(&mut scratch, i, of, ob, Some(lf), Some(lb));
                }
            }
            None => {
                for (i, (of, ob)) in out_fwd
                    .chunks_exact_mut(len)
                    .zip(out_bwd.chunks_exact_mut(len))
                    .enumerate()
                {
                    f(&mut scratch, i, of, ob, None, None);
                }
            }
        }
        return;
    }

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        match last {
            Some((lf, lb)) => {
                out_fwd
                    .par_chunks_exact_mut(len)
                    .zip_eq(out_bwd.par_chunks_exact_mut(len))
                    .zip_eq(lf.par_chunks_exact_mut(state))
                    .zip_eq(lb.par_chunks_exact_mut(state))
                    .enumerate()
                    .for_each_init(&init, |s, (i, (((of, ob), lf), lb))| {
                        f(s, i, of, ob, Some(lf), Some(lb))
                    });
            }
            None => {
                out_fwd
                    .par_chunks_exact_mut(len)
                    .zip_eq(out_bwd.par_chunks_exact_mut(len))
                    .enumerate()
                    .for_each_init(&init, |s, (i, (of, ob))| f(s, i, of, ob, None, None));
            }
        }
    }
    #[cfg(not(feature = "parallel"))]
    unreachable!("parallel=true requires the `parallel` feature");
}
