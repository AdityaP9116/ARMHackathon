# PROFILING_EXPLAINED — the plain-language version

This explains **what our profiling is, why it exists, and how to read it**, in
ordinary terms. For the exact commands, see [`PROFILING.md`](./PROFILING.md);
this doc is the mental model behind those commands.

---

## The problem it solves

The kernel does a lot of different work per step — building tables, computing
`exp`, running the recurrence, applying the gate. If we want it faster, the
worst thing we can do is *guess* which part is slow and optimize that. Guessing
wastes days polishing something that was never the bottleneck.

Profiling replaces the guess with a measurement: **it tells us exactly what
fraction of the time each part of the kernel takes.** Then we optimize the
biggest slice first.

## What we built: a "phase profiler"

Think of the kernel as an assembly line with six stations:

| Station | What it does |
|---|---|
| **transpose** | reshuffles two input tensors into a friendlier layout |
| **discretize** | turns the raw timestep into the form the math needs (a `softplus`) |
| **exp** | computes the state-decay factors (lots of `exp` calls) |
| **projection** | multiplies one input into the running values |
| **recurrence** | the actual step-by-step scan + a dot product |
| **epilogue** | applies the output gate (a `SiLU`) |

The profiler is a **copy of the kernel with a stopwatch at each station**
(`scan_profiled` in the code). It runs the real math, but times each station
separately and prints a table of how many nanoseconds — and what percentage —
each one took. The biggest percentage is the bottleneck.

Two things to know when reading it:
- **Read the percentages, not the raw nanoseconds.** The absolute times depend
  on the machine; the *split* between stations is the durable insight.
- It runs **single-threaded on purpose** — that makes the timing clean and the
  percentages exact. (The real kernel runs on many cores; that changes total
  speed but not which station dominates.)

## How we run it — three "tiers," cheapest first

All three are free. You almost always only need the first.

1. **Tier 1 — GitHub Actions (the one we use).** There's a workflow called
   **"Profile kernel."** It runs the profiler on a real Arm (Neoverse) machine
   in the cloud and hands back the table. You trigger it and read the result in
   the browser — nothing to install. This is what produced our numbers.

2. **Tier 0 — local assembly check.** If you have the Rust toolchain installed,
   a script dumps the actual machine instructions for the hottest functions, so
   we can spot obvious waste. Needs no Arm hardware. Optional.

3. **Tier 2 — Oracle Ampere (deep hardware counters).** A free Arm cloud box
   where a tool called `perf` reads the CPU's internal counters to distinguish
   "slow because of math" vs "slow because of memory." We only reach for this
   if Tier 1 leaves a question open. We haven't needed it.

## What we actually learned

The very first profile was decisive and consistent across every problem size:

- **`exp` alone is ~59% of the kernel's time.**
- Two other stations (`discretize` and `epilogue`) are *also* `exp`-family math
  (`softplus` and `SiLU` both contain an `exp`). Add them up and **~85% of the
  kernel's time is spent evaluating `exp`-like functions.**
- The `transpose` station — which we'd earlier *guessed* might be a big cost —
  is **0.1%**. Negligible. That guess was wrong, and the profiler caught it.

So the whole optimization plan pivoted to one thing: **make `exp` cheaper.**
That's what the first optimization did (a specialized, cheaper `exp` for the
main pass), and re-running the profiler confirmed `exp` dropped and the kernel
got ~8–9% faster overall — see [`OPTIMIZATION_LOG.md`](./OPTIMIZATION_LOG.md).

## The one-sentence summary

The phase profiler is a stopwatch on each stage of the kernel; it told us `exp`
is ~85% of the work, so `exp` is where every optimization should aim — and we
re-run it after each change to prove the slice we targeted actually shrank.
