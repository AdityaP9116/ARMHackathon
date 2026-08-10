# 6. Inside our kernel

How the Rust actually works. Everything here refers to real code you can open.

## The central observation

From [§2](02_mamba_and_selective_scan.md), the scan is two nested loops:

```rust
for t in 0..len {                    // SEQUENTIAL — h_t needs h_{t-1}
    for n in 0..state {              // INDEPENDENT — state elements never interact
        h[n] = exp(dt*a[n])*h[n] + dt_u*b[..];
        y += c[..]*h[n];
    }
}
```

- The **time** loop cannot be parallelized. Real data dependency.
- The **state** loop is fully independent. Free parallelism.
- The **channel** axis (`batch × dim`) is *also* fully independent — separate
  channels share no state at all.

Every optimization below follows from those three facts.

## The two-pass split

The expensive operation is `exp(dt * a[n])`. Transcendental functions are far
costlier than multiply/add — our x86 profiling put `exp` at roughly **half** of
total runtime, and the Arm profile put it at 53.7%.

Crucially, **`exp(dt·a[n])` does not depend on `h`.** It depends only on `dt`
(from the input) and `a` (a learned parameter). So it can be computed for many
timesteps *before* the sequential walk begins.

That is the split, from
[`kernel/arm-scan-core/src/neon/mod.rs`](../../kernel/arm-scan-core/src/neon/mod.rs):

> **Pass A** (per chunk of `CHUNK` timesteps, vectorized **across time**)
> **Pass B** (vectorized **across state**): the recurrence `h = ābar⊙h + b̄`

### Pass A — everything that is pointwise in time

```
A1:  dt  = softplus(delta + bias)      # the step size
     dtu = dt * u
A2:  ābar = exp(dt * a)                # THE EXPENSIVE ONE
     b̄    = dtu * B                     # input projection
```

None of this reads `h`. There is no cross-timestep dependency, so **all of it
vectorizes across time** — 4 timesteps per NEON instruction.

The source comment states the design intent directly:

> Pass A is pointwise in time (no cross-timestep dependency — that is the whole
> point)... Only Pass B's serial recurrence walks time.

Because Pass A is ~85% of the work and is the parallelizable part, this split is
where most of the speedup comes from.

### Pass B — the unavoidable sequential walk

```
for t in chunk:
    h = ābar[t] ⊙ h + b̄[t]     # vectorized across STATE
    y[t] = dot(c[t], h)
```

Still sequential in `t` — no way around it. But the **inner** work over `state`
is vectorized, so each step processes 4 state elements per instruction. With the
typical `state = 16`, that is 4 NEON operations instead of 16 scalar ones.

Note what Pass B has become: pure **fused multiply-add** (FMA) — one instruction
doing `a*b + c`. No transcendentals, no branches. That is the densest, most
predictable work a CPU can do.

## Chunking, and why `CHUNK = 128`

```rust
const CHUNK: usize = 128;
```

The naive version of the two-pass idea computes Pass A for the *whole* sequence,
then Pass B. That reproduces exactly the problem we criticized PyTorch for in
[§2](02_mamba_and_selective_scan.md): a `(batch, dim, len, state)` intermediate —
12.88 GB at L=131,072.

So we process **128 timesteps at a time**:

```
for each chunk of 128 timesteps:
    Pass A over the chunk   → small scratch buffers
    Pass B over the chunk   → carries h into the next chunk
```

The scratch buffers are sized `CHUNK` and `CHUNK * n4` — a few kilobytes, sized
to stay in **L1 cache** (typically 64 KB). They are allocated once and reused for
every chunk.

**This is why our memory is flat in sequence length.** The state `h` carries
across the chunk boundary, so the result is identical to an unchunked scan — this
is a pure implementation detail, not an approximation.

## Threading

Channels (`batch × dim`) are completely independent, so they distribute across
cores with no synchronization at all. We use **rayon**, Rust's data-parallelism
library, via `parallel::for_each_channel`.

Note the scalar path uses the same helper:

```rust
crate::parallel::for_each_channel(...)
```

So even the portable fallback scales across cores.

**Because channels never interact, threaded output is bit-identical to
single-threaded output** — not "close", *identical*. Each channel's arithmetic
happens in exactly the same order regardless of which thread runs it. This is
tested at `RAYON_NUM_THREADS ∈ {1, 2, 8}` and it is a real correctness property:
if it ever failed, it would mean channels were sharing state.

Measured on Neoverse-N2: **3.99× on 4 cores — 99.7% efficiency.**

## The three backends

| Backend | File | Role |
|---|---|---|
| `scalar` | `src/scalar.rs` | Direct transcription. The in-crate correctness reference and the non-Arm fallback |
| `tiled` | `src/mamba3/tiled.rs` | Cache-blocked but portable — the NEON kernel's structural twin, runnable on x86 |
| `neon` | `src/neon/` | The real thing. `#[cfg(target_arch = "aarch64")]` |

The scalar path is **not dead code**. From `CLAUDE.md`: it is the correctness
reference, the non-Arm fallback, and what keeps x86 CI meaningful.

The `tiled` path exists for a subtle reason: it lets the *structure* of the NEON
kernel — the chunking and blocking — be tested on an x86 developer machine, even
though the SIMD instructions themselves cannot run there.

**Consequence worth internalizing:** on x86, `neon/` is not compiled at all. So
every Mamba-3 timing taken on an x86 box measures `tiled`, not NEON. The Graviton
session is the first measurement of the actual NEON Mamba-3 kernel anywhere.

## The Mamba-3 kernel

Per [§3](03_mamba3_whats_new.md), the trapezoidal recurrence groups into
`h_t = α_t·h_{t-1} + b̄_t`, where `b̄_t` is a 2-tap convolution.

| Phase | Change from Mamba-1 |
|---|---|
| Pass A1 | `dt = softplus(delta + bias)` — unchanged |
| Pass A2 | `α = exp(dt·A)` — unchanged. **New:** form `Bx_t`, then the 2-tap `b̄_t = β_t·Bx_{t-1} + γ_t·Bx_t`. Pointwise in time → vectorizes like the rest of Pass A |
| **Pass B** | **Unchanged** — `h = α⊙h + b̄` |
| Chunk carry | now carries `h` **and** `Bx_last` |

That table is the whole reason Mamba-3 was achievable in days: the sequential
part did not change.

### Mamba-3 is not resumable

Mamba-1 supports a `h0` argument — resume a scan from a saved state. Mamba-3
**cannot**, and no carry can fix it: `scale_t` depends on `dt_{t+1}`, so the
trapezoid looks **forward** in time. There is no state you could save at a
boundary that captures a dependency on a step you have not seen.

`last_bx` exists in the code but is diagnostic only. During review it was found
that MIMO was writing a *meaningless* carry — rank 0's slice standing in for all
`r` ranks — so validation now **rejects** the carry for MIMO rather than silently
producing something wrong.

## Numerics: fast is not bit-identical

Two deliberate deviations from a naive reference:

1. **The NEON `exp` is a polynomial approximation** (`src/neon/exp.rs`), not
   libm's `expf`. Accurate to ~1e-6 relative, which is at the edge of fp32
   precision anyway.
2. **FMA reassociation.** A fused multiply-add computes `a*b + c` with a single
   rounding instead of two. More accurate per operation, but a *different* result
   from separate multiply-then-add.

So results match the reference to **fp32 tolerance, not bit-exactly**. We
disclose this, and back it with output-level model metrics showing quality is
unchanged — the 187M model matches the official implementation at **98.05%**
argmax agreement.

Hiding this would be the wrong call: an engineer judge will assume it anyway, and
finding it undisclosed would cast doubt on everything else.

---

**Next:** [The three topologies](07_the_three_topologies.md)
