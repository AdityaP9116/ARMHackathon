# 9. Benchmarking honestly

Benchmarks are the easiest thing in engineering to accidentally fake. This file
covers how we avoid it — and two occasions when we did not.

## The baseline decides everything

A speedup is meaningless without saying *over what*. Pick a weak baseline and any
number is achievable.

Every benchmark here reports **two**:

| Baseline | What it is | What it proves |
|---|---|---|
| `ref_eager` | the recurrence in plain PyTorch | what a CPU user gets today. Large speedups here are real but easy |
| **`ref_compile`** | **`torch.compile` of that same reference** | **the number that matters** |

From our own benchmark source:

> Large speedups against `ref_eager` are real but they are the "we fixed the
> unoptimized path" story, not the hard one. `ref_compile` — **this is the number
> that matters.** An Arm-engineer judge will discount the eager column and look
> straight at this one.

The gap is stark. On 2D, measured on Graviton4: **17.7–92.5×** against eager but
**14.25×** against `torch.compile` — and on the 187M model, **18.89×** against
eager collapses to **1.64–3.05×** against `torch.compile`. Both are true. Only
one is interesting, and quoting only the first is the kind of thing that ends a
submission's credibility.

### Why we can beat `torch.compile` at all

Per [§2](02_mamba_and_selective_scan.md): it cannot restructure a sequential
recurrence. It removes dispatch overhead and fuses pointwise work, but the chain
of dependent steps remains, and it cannot choose a chunked two-pass algorithm —
that is a human decision about numerics.

We also report its **compile time**, because for a sequential scan that cost *is*
the story: the graph grows with sequence length. Measured **59.9 s at L=256 →
532.8 s at L=2048**, paid before a single token is produced.

### A subtle honesty point

The reference is timed at **fp32, not f64**. Our oracle defaults to f64 for
accuracy; timing an f64 baseline against an fp32 kernel would inflate every
speedup by roughly the f64/f32 ratio.

That is your own thumb on the scale, and it would be very easy to do by accident.

## Method

From `bench/README.md`, applied everywhere:

- **Fixed thread count** — stated with every number
- **Warmup, then medians** — not means; medians reject outliers from scheduling
- **Pinned seeds** — same inputs every run
- **Host and git commit tagged into the output JSON**
- **Instance type and torch version stated alongside every number**

And: **correctness gates speed.** Every shape's output is diffed against the
reference *before* its timing is reported. A shape that fails is reported as
failed, not timed.

## Two times we got it wrong

Both are in the repo's history deliberately. They are the reason the rules above
exist.

### The contaminated run

An early benchmark ran with `reps=2` on a machine that was doing other things.
It reported a **0.50× "regression"** — the optimization apparently making things
twice as slow.

That number reached several documents before anyone re-ran it. A quiesced re-run
with `reps=3` gave **1.82×** — an improvement, not a regression.

Nothing was wrong with the code. The machine was busy.

> **Rule: a contended benchmark is void.** Check load average ≈ 0 first.

### The ratio between equals

A first version of the non-causal benchmark reported a single causal-vs-non-causal
ratio and got **0.88×** — non-causal apparently *faster* than causal, which is
impossible, since non-causal does strictly more work.

The cause was real and worth keeping: in 2D both quantities do near-identical
work (the cross-scan already runs both directions), so the ratio was **noise
between two equal things**.

The fix was structural, not cosmetic: separate the 1D and 2D cases — which is
where the actual finding lives — and **flag rows whose medians are under 1 ms as
dispatch-dominated**, so a reader knows when they are looking at measurement
overhead rather than a result.

## Publish unflattering rows

From `CLAUDE.md`:

> If a row is unflattering, publish it anyway.

The kernel's moat is that `torch.compile` cannot restructure a sequential
recurrence. That argument only lands if the numbers are clearly trustworthy — and
selective reporting is the fastest way to destroy that.

Current unflattering rows we publish:

- **MIMO is slow.** Correct everywhere, but scalar-path only — no NEON or blocked
  MIMO kernel. On Graviton4 it is **3.43× slower than SISO** at L=1024. Its ratio
  against the PyTorch baseline reaches 31–56×, and we deliberately **do not quote
  that**: the MIMO reference is itself ~10× slower than the SISO reference, so the
  ratio describes a pathological baseline rather than a fast kernel. The
  arithmetic-intensity argument for MIMO on CPU remains a **prediction, not a
  result** — which is exactly why the honest framing of Path B is coverage and
  correctness (**1.90 bf16 ULP through the C ABI**, tighter than our own PyTorch
  reference) rather than speed.
- **`TILE = 32` has never been swept.** It is a placeholder. It can only be tuned
  on Arm, since x86 does not execute that path.
- **No scalar → blocked → NEON ablation for Mamba-3.** It needs a backend selector
  through the torch op, which does not exist. Adding a flag that silently
  benchmarked one backend three times would be worse than its absence.

## The provisional rule

The repo classifies numbers by where they were measured:

| Source | Status |
|---|---|
| x86 dev box | **provisional** |
| shared 4-core CI runner | **provisional** |
| dedicated Arm instance (Graviton) | **headline** |

**As of Aug 11, 2026 the headline numbers exist**: `c8g.16xlarge`, Graviton4
(Neoverse-V2), 64 vCPU. See `README.md` for the tables and `bench/results/` for
the raw JSON.

Two of those measurements **reversed** what x86 had said, which is the whole
reason the provisional/headline distinction exists:

- the SS2D traversal-pair rewrite measured 1.80× on x86 and **0.96× — a
  regression — on 64 cores**, because the pair form halves the rayon rows
- the "is a fully fused 2D kernel worth building" verdict flipped from *no*
  (7.2–13.8% overhead) to *yes* (**46.1%**)

A number without its hardware attached is not a result.

## What gets measured on Graviton

| Bench | What it measures |
|---|---|
| `bench_mamba3.py` | the Mamba-3 kernel vs both baselines |
| `bench_ss2d_mamba3.py` | 2D Mamba-3 at real vision grid sizes |
| `bench_mamba3_noncausal.py` | causal vs non-causal — the novel comparison |
| `bench_mamba3_lm.py` | the real 187M model, SISO and MIMO |
| `run_baseline.sh` | Mamba-1 ladder, op-level, **core-scaling 1→64** |
| `bench_longctx.py` | the constant-memory claim |

The **core-scaling curve** is the headline chart for a Cloud-track entry, and it
is why the instance needs many cores rather than fast ones.

## Profiling — evidence, not vibes

Timing tells you *how fast*. Profiling tells you **why**, which is what makes a
"what would you do next" answer credible.

`perf` and **Arm Streamline** (Arm's own top-down profiler) give a phase
breakdown and hardware counters:

| Signal | Verdict |
|---|---|
| IPC ≈ 2.5–3+ | compute-bound, near the ceiling — ship it |
| IPC < 1 + high backend stalls | memory-bound — bf16 storage / better B/C blocking is the next lever |
| `vexpq_f32` dominates | `exp` is still the hot spot |
| transpose loop hot | B/C prep costs too much |

Measured on Graviton4 (Neoverse-V2): NEON **4.23×** over scalar, **+6.17×** from
rayon on top (26× total), `exp` at **47.7–48.2%** of runtime and the transpose at
**0.0–0.1%** — stable to within 0.5% across every shape tested.

That last pair is a genuine finding: `exp` dominates and the transpose is free,
which says exactly where the next optimization goes (bf16 storage, SVE2 `FEXPA`)
and where it must not.

Summed differently: **Pass A is ~72% of runtime.** That is the two-pass design's
founding premise, measured rather than asserted.

---

**Next:** [Code tour](10_code_tour.md)
