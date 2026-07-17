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

---

## Step 2 — reuse the fast exp in softplus

**Branch:** `feature/exp-round2` · **Idea:** IMPROVEMENT_IDEAS §3.1 / §10 item 2

### What
`vsoftplusq_f32` (the softplus in the ~19% discretize phase) computes
`exp(-|x|)`, whose argument is always ≤ 0, so it now uses the same
`vexpq_f32_nonpos` from Step 1 instead of the general `vexpq_f32`. One-line
change; no new function, no tolerance change.

### Measured impact (profiler, vs the Step-1 run)

| Shape | discretize phase | total kernel |
|---|---|---|
| L128 | −9.5% | −2.1% |
| L512 | −10.3% | −1.9% |
| L2048 | −6.2% | −1.4% |
| batch8 | −7.5% | −2.4% |

Clean attribution: the exp / recurrence / projection phases moved < ~1% in
absolute ns (untouched); only discretize fell. **Cumulative Step 1 + Step 2:
~10% faster total kernel vs baseline** (L512: 24.62M → 21.96M ns).

### Correctness
Golden gate green on the PR (tolerances unchanged). Adds only ~1e-7 of exp
error to softplus (nonpos exp ~6e-7 vs the previous ~5e-7); softplus's
component sweep bound is 2e-6. The sweep domain [-30, 30] never reaches the
nonpos deep-underflow region.

### Second-degree drop — deferred here, then reopened by the golden table
Initially deferred: `tiny`'s floor_bound (2.49e-6) looked too tight for the
~2.4e-6 error a degree-3 exp adds. But the PR's per-case golden table showed
`tiny` actually runs at `out_err = 1.978e-7` (parity 1.198e-7) — ~12× of
headroom — so a degree-3 Pass-A2 exp projects to ~1.2e-6, still ~2× under the
ceiling. **Now the next step** (branch `feature/exp-degree-cut`), keeping
softplus on the degree-4 exp to protect its own 2e-6 sweep bound.

---

## Cumulative vs baselines (op-level, measured on the PR CI, after Step 2)

`bench_op.py` on the GitHub arm64 runner (4-core, torch 2.13.0) — kernel vs
PyTorch eager and torch.compile, the fair baselines. Correctness in the same
run: kernel-vs-ref max_abs_err 1.9e-6 (L128) / 3.0e-6 (L512), well under 1e-4.

| Shape | kernel | vs eager | vs torch.compile | baseline (`ce07edd`) |
|---|---|---|---|---|
| B1 D768 L128 | 0.85 ms | 16.3× | **3.74×** | 0.96 ms / 3.3× |
| B1 D768 L512 | 2.87 ms | 23.2× | (compile skipped) | 3.27 ms / 4.2× |

The two exp steps made the kernel ~11–12% faster op-level and **widened the lead
over both baselines** (vs torch.compile 3.3× → 3.74× at L128; vs eager 14.4× →
16.3×). Shared-runner numbers are provisional per BASELINE_REPORT — headline
figures still need a dedicated Graviton instance.

---

## Step 3 — degree-3 exp for the Pass-A2 decay factor

**Branch:** `feature/exp-degree-cut` · **Idea:** IMPROVEMENT_IDEAS §3.1 (degree cut)

### What
Added `vexpq_f32_nonpos_fast` — the same non-positive exp as Step 1 but one more
degree lower (drops `P1` as well as `P0`), so ~2.7e-6 worst-case vs ~2.6e-7.
Pointed **only Pass A2** at it (`channel_n16`, `channel_general`, profiler). One
more FMA off the 56%-of-runtime exp phase. softplus stays on the degree-4
`vexpq_f32_nonpos` — its 2e-6 sweep bound is left untouched.

### Why it's now safe (it wasn't attempted at Step 2)
The PR-CI golden table gave the real budget. The binding case `tiny` runs at
`out_err = 1.978e-7` against a 2.487e-6 floor_bound (~12× headroom); with the
degree-3 exp its NEON-exp contribution ~4×'s, projecting to ~1.2e-6 — still
~2× under the ceiling. Every other case has 10–40× more room; parity projects
to ~1.1e-6 vs the 1e-5 gate. `tiny` is the case to watch.

### Measured impact (profiler, vs the Step-2 run)

| Shape | exp phase | total kernel |
|---|---|---|
| L128 | −8.9% | −5.2% |
| L512 | −9.2% | −4.9% |
| L2048 | −8.8% | −5.2% |
| batch8 | −8.9% | −5.3% |

Beat the ~4% projection. **Cumulative over Steps 1–3: exp phase −22%, total
kernel −15%** vs baseline (L512: 24.62M → 20.89M ns). Op-level vs baselines
(`bench_op.py`): kernel 0.82 ms / 2.77 ms (was 0.96 / 3.27), **24.1× vs eager,
3.7× vs torch.compile**.

### Correctness — green, comfortable margins
`vexpq_f32_nonpos_fast` sweep worst 3.353e-6 (< 4e-6 bound). Golden gate passes
every case; the errors rose as expected but stay 15–25× under each case's
floor_bound (e.g. `small` auto 3.15e-6 vs 4.8e-5; `extreme_delta` 1.83e-5 vs
4.19e-4). Notably the tight-floor cases (`tiny`, `edge_L1`) barely moved — they
are short sequences where exp accuracy hardly matters, while the exp-sensitive
cases all have generous floors. Proptest `f32_matches_f64` and parity (<1e-5)
green. kernel-vs-ref in bench_op 4–5e-6, ~20× under the 1e-4 gate.


---

## Step 4 — P0 batched 4-direction SS2D call (app-side; Jul 17)

All 4 cross-scan directions in ONE kernel call (`SS2DBlock.forward`) —
shared projections make it exact. Kernel calls/denoiser-forward 12→3;
per-NFE 542→23 ms (x86 scalar+rayon, 32×32 toy). Phase-C parity re-gated:
PASS (sampling parity 0.0). `bench_ss2d.py` at the real diffusion shapes:
flip/permute/projection overhead 21–25% → fused `selective_scan_2d`
justified per the 15% rule.

## Step 5 — P1-3 thread-local B/C plane cache (Jul 17, `3177ded`)

The transposed-plane workspace (2 × ~7.9 MB at L=123k) was
reallocated+zeroed each of the ~1,650 calls/reconstruction. Now cached
thread-locally keyed by (planes,len,n4,state); padding-zero invariant
holds on key hits (transpose never writes n ≥ state). No numerics change;
aarch64 clippy clean; arm gates via CI.
