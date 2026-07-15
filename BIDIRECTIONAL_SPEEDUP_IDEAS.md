# BIDIRECTIONAL_SPEEDUP_IDEAS — closing the gap to the GPU

**Status: research/ideas only — nothing here is implemented.** Written Jul 15, 2026,
from a read of the official Mamba CUDA kernel design, the CPU-vs-GPU bandwidth
reality, and the current NEON kernel at the post-merge state.

**Scope.** How to make the *bidirectional* scan faster on Arm, with the explicit
goal of approaching GPU single-stream performance. This is a sibling to
[`IMPROVEMENT_IDEAS.md`](./IMPROVEMENT_IDEAS.md) (general kernel optimization) and
does not repeat it — where an idea overlaps, it is cited, not restated. The
bidirectional-native ideas (§3) are new white space that neither document nor the
upstream repo covers.

Rules that bind every idea (per [`CLAUDE.md`](./CLAUDE.md)):
- **Correctness gates speed.** Every new path gets its own golden gate and error
  floor before it is benchmarked. The bidirectional definition
  (`tests/check_bidirectional_math.py`, `reverse_matches_flip_forward_flip`) is
  the reference any faster path must reproduce.
- **Benchmark honestly.** `torch.compile` is the baseline; medians after warmup;
  state the host. Current standing: **~5.2–5.6× vs torch.compile at L ≥ 512**, and
  torch.compile **OOMs at L=8192** on a 4-core arm64 box (see `BIDIRECTIONAL_LOG.md`).
- **Measure before building.** Every idea below names the measurement that
  justifies (or kills) it. The fused `reverse` flag is the cautionary tale: built
  on a ~2% assumption, worth building only because SS2D needed it anyway.

---

## 1. The reality: what "as fast as GPU" can and cannot mean

The gap on this op is **memory bandwidth, ~10–20×**, not compute:

| | bandwidth | source |
|---|---|---|
| Graviton4 (DDR5) | ~0.5 TB/s | [NextPlatform](https://www.nextplatform.com/compute/2024/09/19/aws-boosts-memory-capacity-on-graviton-4-compute/1642050) |
| H100 (HBM3) | ~4 TB/s | [Spheron](https://www.spheron.network/blog/hbm3e-vs-hbm4-vs-hbm4e-llm-inference-guide/) |
| B200 (HBM3e) | ~8 TB/s | same |

**The opening:** selective_scan on a GPU runs at **10–15% utilization** (vs 80–90%
for a transformer's tensor cores — [Mamba-2 co-design](https://medium.com/@danieljsmit/mamba2-the-hardware-algorithm-co-design-that-unified-attention-and-state-space-models-77856d2ac4f4)).
The GPU is *also* memory-bound here and is not using the hardware that makes it a
GPU. The scan is a diagonal recurrence, not a matmul; the tensor cores sit idle.

Consequences that set the whole agenda:

- **Concede batch throughput.** 10–20× bandwidth wins raw tokens/sec. Publish that
  losing row honestly (per `IMPROVEMENT_IDEAS.md §11`); do not chase it.
- **Chase single-stream latency and $/token.** At batch=1 the GPU's bandwidth is
  *stranded* — one sequence cannot fill it — and per-launch overhead dominates.
  This is exactly where bidirectional inference lives (non-causal, one sequence at
  a time). Parity here is real and defensible.
- **Own the workload the GPU cannot do at all.** Constant-memory scan at L=131k:
  a transformer's KV cache is multiple GB, and — measured — `torch.compile` cannot
  even *build the graph* at L=8192 on CPU. Not "faster"; a different class.

So the target is precise: **single-stream bidirectional latency within a small
factor of the GPU, at a fraction of the instance cost** — not a throughput race
we would lose.

---

## 2. What the official CUDA kernel does — and what we already have

Three techniques ([Mamba paper §D](https://arxiv.org/pdf/2312.00752),
[Princeton PLI](https://pli.princeton.edu/blog/2024/mamba-2-algorithms-and-systems)):

| GPU technique | Our status |
|---|---|
| **Fuse discretize + scan + gate in SRAM**, load A/B/C/Δ from HBM once, never materialize the `(B,D,L,N)` intermediate | ✅ **done** — two-pass NEON kernel keeps chunk scratch in L1, streams B/C once (`neon/mod.rs`) |
| **N=16 state resident in registers** | ✅ **done** — `channel_n16`, four q-registers |
| **Work-efficient parallel associative scan over the sequence length L** (Blelloch tree), so a single sequence still fills the machine | ❌ **not done** — we parallelize over batch×channel only. This is the one real gap, and §4 is about it. |

The takeaway: **structurally we already match the GPU's fusion strategy.** Our moat
over the `torch.compile` strawman *is* that fusion — the compiler cannot restructure
the recurrence, so it unrolls and dies. What we are missing is the GPU's answer to
the batch=1 problem, which is the L-parallel scan.

Best borrowable reference: **[`mamba-mini`](https://github.com/MzeroMiko/mamba-mini)**
— "the code closest to `selective_scan_cuda`," single-file, CPU+GPU, with the math
derivation. Read this for exact discretization/scan order, not the CUDA source
(which is hard to map onto NEON).

---

## 3. Bidirectional-native wins (white space — the GPU does not do these)

**The official Mamba repo has no bidirectional kernel.** Bidirectional is Vim /
Caduceus territory, and those just call the forward scan twice. So there is nothing
to "borrow" here — it is unclaimed, and there are two strong ideas.

### 3.1 Run forward and backward in parallel across cores ⭐ cheapest, do first

The two directions are **fully independent** — no data dependency between them. We
currently run them as two sequential kernel calls (`bidirectional_scan` →
`selective_scan` then `selective_scan(reverse=True)`). When `batch × dim` does not
saturate the core count — which is the *common* bidirectional case (B=1) — the
second direction waits while cores sit idle.

Dispatch forward to one rayon scope and backward to another, concurrently.
**Potentially ~2× on bidirectional latency, zero new math, zero numerics risk.**

- **Effort:** low. It is a scheduling change at the `bidirectional_scan` layer (run
  the two `selective_scan` calls on separate threads / a rayon `join`), or one
  level deeper: a kernel entry that takes both an `out_fwd` and `out_bwd` and
  parallelizes over `(channel × direction)` instead of `channel`, doubling the
  independent work units fed to rayon.
- **Measure first:** does B=1/D=768 actually leave cores idle on the target? On 4
  cores with 768 channels, no — already saturated, so this buys nothing there. On a
  **64-core Graviton at B=1**, 768 channels across 64 cores is ~12 rows/core, and a
  second direction is free parallelism. This is a *big-core-count* win; confirm the
  idle cores exist before building.
- **Risk:** ~0. Outputs are unchanged; only the schedule differs. Parity tests pass
  by construction (each direction is the existing, verified path).

### 3.2 Fuse both directions into one pass — SHARE the exp ⭐⭐ the real win

**This is the highest-value item in the whole document, and the roofline (§5)
is why.** The framing that matters is not bandwidth — it is *redundant compute*.

**The load-bearing fact:** Pass A (discretize + exp + projection) is **pointwise in
time and therefore direction-independent**. `dt = softplus(δ+bias)`,
`ābar = exp(dt·A)`, `b̄ = dt·u·B` are identical whether the sequence is scanned
forward or backward. Only Pass B (the recurrence `h = ābar·h + b̄`, `y = C·h`)
walks time and depends on direction.

But `bidirectional_scan` calls `selective_scan` **twice**, so it computes Pass A —
which the profiler measured at **83% of runtime** (exp 59% + discretize 19% +
projection 5%) — **twice**. We are duplicating the expensive transcendental work,
and the kernel is compute-bound on exactly that work (§5).

A fused kernel computes Pass A **once**, then runs Pass B (the cheap ~10% pure-FMA
recurrence) twice — once forward, once backward — from the shared, L1-resident
`ābar`/`b̄` chunk data. Using the measured phase split:

| | cost (1 forward scan = 100%) |
|---|---|
| current bidirectional (2 calls) | **200%** |
| fused: Pass A ×1 (83%) + Pass B ×2 (21%) + epilogue ×2 (14%) | **~117%** |
| **projected speedup** | **~1.7×** |

That is a **compute** saving — sharing the exp — not a bandwidth trick, and it rests
on already-measured profiler data, not a guess. (It also happens to halve input
traffic as a side effect, but §5 shows that is not the binding constraint.)

- **Strategic multiplier for SS2D:** SS2D's four directions scan the same grid with
  the same pointwise discretize+exp. Compute Pass A **once**, run Pass B **four
  times** (the four traversals) → the shared 83% is amortized across 4 directions
  instead of duplicated 4×. Projected saving on the shared part approaches **~3–4×**.
  Same kernel structure. **Build it at 1D here, reuse it at 2D.** This is the exact
  "read once, emit multiple directions" core of `TOPOLOGY_IMPLEMENTATION_PLAN.md §3.2`
  — the SS2D substrate, just as `reverse` was.
- **Effort:** medium. New core entry point with two output buffers and two `h`
  accumulators; Pass A unchanged and run once; Pass B run twice over the shared
  chunk scratch (forward index, then reversed index — the `reverse` logic already
  exists). `parallel.rs`, the transpose, discretize, and exp are all reused as-is.
- **Risk:** low-medium. New kernel surface → new golden gate, but each direction's
  math is the existing verified path, so it diffs bit-for-bit (scalar) / ~1e-7
  (NEON) against the two-call `bidirectional_scan`, exactly like `reverse` did.
- **Measure:** the ~1.7× projection assumes Pass A fully shares. Confirm the phase
  split still holds at the fused structure (the profiler, `PROFILING.md`), and that
  keeping two `h` register sets live does not spill (§3.3.1 of `IMPROVEMENT_IDEAS.md`
  notes 8 h-registers still fit the 32-register file at N=16).

---

## 4. The GPU-parity mechanism: parallel scan over L

**This is how the GPU wins single-stream, and it is the biggest borrow.** Covered as
a stretch in `IMPROVEMENT_IDEAS.md §4.3`; elevated here because the CPU-vs-GPU
research makes it *the* lever for the parity goal, not a nice-to-have.

The recurrence `h_t = ā_t · h_{t-1} + b̄_t` is an **associative scan**: the transition
`(ā, b̄)` composes as `(a,b) ∘ (a',b') = (a·a', a'·b + b')`. So it parallelizes with a
3-phase Blelloch scan across chunks:

1. **Per-chunk (parallel):** compose each chunk's timesteps into a single
   `(A_prod, B_comb)` transition. Our chunked structure already computes the pieces.
2. **Combine (tiny, sequential):** scan the `L/CHUNK` per-chunk carries — a handful
   of elements.
3. **Finalize (parallel):** re-run each chunk from its now-known entry state to emit
   outputs.

Cost: one extra elementwise pass over the data. Payoff: **full-machine parallelism
when B×D < cores** — i.e. the single-stream latency case (B=1 audio, ECG, genomics)
on a big Graviton, which is precisely the GPU-parity regime.

- **Bidirectional angle:** combines with §3.1/§3.2 — a bidirectional scan on a
  64-core box could parallelize over *both* directions *and* the L dimension,
  saturating the machine on a single sequence the way the GPU does.
- **Effort:** high (the real item on this list). But the chunked two-pass kernel
  already exposes the composed pairs, so the groundwork is laid.
- **Risk:** medium. Numerics: composing `A_prod` across a long chunk multiplies many
  `ā ∈ (0,1]` values → underflow toward 0, which is *fine* (it means the distant past
  doesn't affect the present — the `§7.2` underflow-cut observation in
  `IMPROVEMENT_IDEAS.md` is the same fact) but must be handled so it doesn't NaN.
  Gate hard against the sequential reference.

---

## 5. The roofline — done analytically, and it redirected §3.2

Computed from the code + the existing profiler split (confirm with PMU later, but
the conclusion is robust):

**Arithmetic intensity at the mamba shape** (B=1, D=768, L=512, N=16):
- main-memory traffic ≈ **6.4 MB** (u, delta, z, out ≈ 1.57 MB each; B/C/A are tens
  of KB)
- ≈ **85 MFLOP** (~13 FLOP per (b,d,l,n) point × 6.3M points)
- **intensity ≈ 13 FLOP/byte**

**Graviton4 machine balance** ≈ 5.5 TFLOP/s ÷ 0.5 TB/s ≈ **11 FLOP/byte**. We sit
right at the knee — but the exp is **transcendental**, and the profiler already
measured **~85% of runtime in exp/softplus/silu** vs ~10% in the recurrence. So in
practice **we are compute-bound on the transcendentals, not bandwidth-bound.**

**Why this matters, and what it changed:** the GPU is bandwidth-bound here because it
has *too much* compute (10–15% utilization). We are the mirror image — compute-bound
because SIMD gives us relatively little, and the exp is expensive. So **bandwidth
reduction is not our lever; cutting redundant compute is.** That is exactly why §3.2
was rewritten from "halve input bandwidth" (wrong axis) to "share the direction-
independent exp between the two scans" (the real, ~1.7× win).

**Still worth doing as a writeup metric:** report **% of the CPU's memory bandwidth**
achieved, mirroring the GPU's own 10–15%-utilization honesty. But treat it as a
number for the pitch, not as the optimization compass — the compass says compute.
When (if) compute is fully shared and parallelized and we *become* bandwidth-limited,
that is when `IMPROVEMENT_IDEAS.md §5.1` (fp16 plane storage) becomes the last lever.

---

## 6. Explicitly NOT worth borrowing

- **Mamba-2 / SSD matmul reformulation.** The GPU recasts the scan as matmuls to use
  tensor cores. On CPU with diagonal A and N=16 the plain scan is already near SIMD
  peak and the dual strictly *adds* FLOPs — `IMPROVEMENT_IDEAS.md §7.1` argues this
  and the research confirms it. Relevant only if Mamba-2 support is added, and even
  then as a measured rejection.
- **Warp-level primitives / shared-memory staging.** GPU-specific; the NEON analog
  (registers + L1) is already what we do.
- **Chasing batch throughput.** Conceded above; it is a bandwidth race we lose.

---

## 7. Priority shortlist

Re-ranked after the §5 roofline: we are **compute-bound on the exp**, so the winner
is the one that stops computing it twice.

| # | Idea | § | Effort | Expected effect | Risk |
|---|---|---|---|---|---|
| **1** | **Fused two-direction pass — share the exp** | **3.2** | **Med** | **~1.7× on bidirectional; ~3–4× shared-part for SS2D; the SS2D substrate** | **Low-med** |
| 2 | Fwd ∥ bwd across cores | 3.1 | Low | ~2× latency **only when cores are idle** (big-core, B=1) — needs a dedicated host to see | ~0 |
| 3 | L-parallel Blelloch scan | 4 | High | Single-stream GPU-parity; unlocks 64-core at B=1 | Med |
| 4 | %-bandwidth line for the writeup | 5 | Low | No speedup; pitch metric, not the compass | ~0 |

**Sequencing:** **#1 first** — it is the largest bidirectional win, it is directly
verifiable on the 4-core CI runner (unlike #2, which needs idle cores that only exist
on a big-core host), and it *is* the SS2D substrate, so it pays off twice exactly
like `reverse` did. Then #2 and #3 on a dedicated Graviton session where the core
count makes them visible. #4 rides along with any benchmark run.

Note #1 and #2 **compose** — #1 shares the exp; #2 then runs the two (now
exp-free) Pass-B recurrences on separate cores. And both feed SS2D.

**The honest north star:** we already match the GPU's *fusion*. The single-stream gap
closes by (a) not computing the transcendental Pass A twice for bidirectional / 4×
for SS2D (#1), and (b) parallelizing the one sequence across all the cores the GPU
would spread across threads (#2, #3). None of it needs the tensor cores the GPU is
not using anyway.

---

## Sources

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (§D, hardware-aware scan)](https://arxiv.org/pdf/2312.00752)
- [Princeton PLI — Mamba-2: Algorithms and Systems](https://pli.princeton.edu/blog/2024/mamba-2-algorithms-and-systems)
- [Tri Dao — State Space Duality (Mamba-2) Part III: the algorithm](https://tridao.me/blog/2024/mamba2-part3-algorithm/)
- [mamba-mini — CPU/GPU single-file selective scan closest to the CUDA kernel](https://github.com/MzeroMiko/mamba-mini)
- [Mamba-2 hardware-algorithm co-design (10–15% utilization figure)](https://medium.com/@danieljsmit/mamba2-the-hardware-algorithm-co-design-that-unified-attention-and-state-space-models-77856d2ac4f4)
- [AWS Graviton4 memory bandwidth (~0.5 TB/s DDR5)](https://www.nextplatform.com/compute/2024/09/19/aws-boosts-memory-capacity-on-graviton-4-compute/1642050)
- [GPU HBM bandwidth (H100 ~4 TB/s, B200 ~8 TB/s)](https://www.spheron.network/blog/hbm3e-vs-hbm4-vs-hbm4e-llm-inference-guide/)
