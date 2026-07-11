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
