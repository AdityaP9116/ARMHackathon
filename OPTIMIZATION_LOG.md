# OPTIMIZATION_LOG — kernel speed ladder

A running, measured record of each kernel optimization: what changed, why,
the measured impact, and the correctness status. Every entry obeys the
[`CLAUDE.md`](./CLAUDE.md) rule **correctness gates speed** — no speed number
is recorded until the golden gate (max_abs vs f64 reference < 1e-4) and the
scalar↔NEON parity check (< 1e-5) are green.

Measurement method: the phase profiler (`scan_profiled`, single-thread NEON)
and the criterion ladder, run on the GitHub `ubuntu-24.04-arm` runner via the
**Profile kernel** workflow. See [`PROFILING.md`](./PROFILING.md) for how, and
[`PROFILING_EXPLAINED.md`](./PROFILING_EXPLAINED.md) for the plain-language why.

---

## Baseline (commit `ce07edd`) — see [`BASELINE_REPORT.md`](./BASELINE_REPORT.md)

- vs `torch.compile`: **3.3–4.2×**; vs eager: **8.2–30.4×**.
- HF mamba-130m prefill: **1.9–2.1×**; greedy tokens identical.
- Kernel error vs f64 reference: **≤ 3.8e-6** (well inside the 1e-4 gate).

### Measured phase split (the profiler, single-thread NEON — where time goes)

| Phase | % of kernel time |
|---|---|
| **exp** (Pass A2, the 16 exps/timestep) | **~59%** |
| discretize (softplus = exp + log + divide) | ~19% |
| recurrence (FMA chain + C·h dot) | ~10.5% |
| epilogue (SiLU = exp + divide) | ~7% |
| projection (B multiply) | ~5% |
| transpose | ~0.1% |

**Conclusion:** the kernel is compute-bound on transcendental polynomial
evaluation — exp + softplus + SiLU are ~85% of runtime. Make exp cheaper.
(Full analysis: [`IMPROVEMENT_IDEAS.md`](./IMPROVEMENT_IDEAS.md) §1.0.)

---

## Step 1 — non-positive fast exp for Pass A2

**Branch:** `feature/faster-exp` · **Idea:** IMPROVEMENT_IDEAS §3.1

### What
Added `vexpq_f32_nonpos`, a variant of the NEON exp specialized for the scan's
always-non-positive argument `dt·A` (A < 0, post-softplus dt ≥ 0). It drops
three things the general exp only needs for positive / edge inputs:

1. the **upper overflow clamp** — impossible when x ≤ 0;
2. the **exact-zero underflow select** — deep-underflow lanes return ~1e-38
   instead of exactly 0, which is 30+ orders of magnitude under the 1e-4 gate
   and cannot be amplified by the contraction `h = abar·h + b̄` (abar < 1);
3. **one polynomial degree** (the P0 term, worth < ~1.3e-7 over the range).

Wired into Pass A2 in `channel_n16` and `channel_general`. The general
`vexpq_f32` (used by SiLU) and **every tolerance are unchanged** — this adds a
new function with its own accuracy test rather than loosening an existing one.

### Measured impact (profiler, vs baseline run, same runner class)

| Shape | exp phase | **total kernel** |
|---|---|---|
| L128 | −14.2% | −8.0% |
| L512 | −14.0% | −9.1% |
| L2048 | −14.1% | −8.5% |
| batch8 | −14.2% | −8.3% |

exp's share fell **59% → 55%**. Clean attribution: the non-exp phases moved
< 0.01% in absolute ns between runs (e.g. recurrence at L512: 2,594,201 →
2,594,209 ns), so the ~14% drop is the exp change, not runner noise.

### Correctness
Golden gate green (out_err unchanged at ~4e-6, ≪ 1e-4), scalar↔NEON parity
< 1e-5, new `vexpq_f32_nonpos` sweep < 1.5e-6 over [-104, 0]. Precondition
(dt ≥ 0 → argument ≤ 0) holds on every golden case, confirmed in
`gen_golden.py` (post-softplus, or the positive-timestep `no_softplus` case).

### Next candidates (informed by this result)
- Extend the same treatment to **softplus's exp** (part of the ~19% discretize).
- Now that the golden margin is known, evaluate dropping a **second degree**.
- **SVE FEXPA** exp (§3.2) — the larger swing, hits the whole ~85%.
