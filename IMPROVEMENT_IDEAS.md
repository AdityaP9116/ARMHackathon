# IMPROVEMENT_IDEAS — Kernel & Integration Optimization Backlog

**Status: research/ideas only — nothing here is implemented or committed to.**
Written Jul 14, 2026, from a deep read of the kernel at `ce07edd`, the measured
numbers in [`BASELINE_REPORT.md`](./BASELINE_REPORT.md), Neoverse
microarchitecture docs, and prior art (sources at the bottom).

Rules that bind every idea below (per [`CLAUDE.md`](./CLAUDE.md)):
- **Correctness gates speed.** Every new mode/path gets its own golden gate and
  recorded error floor before it is benchmarked. Never loosen a tolerance.
- **Measure before optimizing.** The diagnosis in §1 is inference from public
  latency/throughput tables; a PMU session (§9) confirms or reorders the list.
- **Disclose approximations.** Any mode that trades accuracy is opt-in, named,
  and shipped with measured output-level model metrics.

---

## 1. Diagnosis — where the cycles go today

Back-of-envelope from our own measurements: 25.08 ms single-thread at
D=1536, L=512 (Surface A) ≈ **~32 ns ≈ ~90 cycles per (channel, timestep)**.

Per timestep, the N=16 fast path issues roughly:

| Pass | Work | ~µops |
|---|---|---|
| A1 discretize (amortized) | softplus + mul, 4 t/iter | ~4 |
| A2 exps + input proj | 4 × `vexpq_f32` (~12 instr each) + 8 mul/store | ~55 |
| B recurrence + dot | 8 FMA + `vaddvq` + loads | ~14 |
| Epilogue (amortized) | d_skip FMA + SiLU (has a `vdivq`) | ~4 |
| Transpose/zeroing (amortized) | scalar strided loops, serial | varies |

On Neoverse-N1-class cores (Ampere A1, Graviton2: **2×128-bit SIMD pipes**)
that is ~35–50 cycles of pure SIMD issue → the kernel is **throughput-bound
and the exp pass dominates** (~half the instruction stream).

On Neoverse V2 (Graviton4 / `c8g`: **4×128-bit SIMD pipes**) issue throughput
doubles, so the binding constraint flips to **Pass B's loop-carried FMA chain**:
`h = vfmaq(bbar, abar, h)` has ~4-cycle FMLA latency but only ~2.5 cycles of
per-timestep throughput cost → ~40% idle on the critical path.

**Consequence: the optimization priorities differ per Graviton generation.**
That is both an engineering fact and a great line for the writeup.

The measured tables also point at the non-math overheads:
- Speedup sags exactly where transpose/workspace work grows: **B=8 → 8.2×**,
  **D=3072 → 11.2×** (the `bt`/`ct` planes are allocated, zeroed, and
  transposed serially before rayon starts).
- 4-core scaling is 3.88×, not 4.0× — the serial prologue is the Amdahl term.
- At long L the shared transposed planes (2 × 4·L·n4 bytes per plane) fall out
  of L2 and every channel re-streams them from DRAM (§5.2).

---

## 2. Zero-risk integration wins (Python/patch layer, no kernel math changes)

### 2.1 Kill the double transpose of B and C  ⭐ top priority
HF's `x_proj` emits B/C as `(B, L, N)` — which is *exactly* the time-major
layout the kernel builds internally. Today the path is:

```
x_proj → (B, L, N)  --transpose(1,2)-->  view  --_c()-->  full (B,N,L) copy
                     --kernel transpose-->  (len, n4) scratch   ← back where we started
```

Add a layout flag to the FFI (“b/c are time-major already”) and delete **two
full-tensor copies plus the kernel's entire transpose pass** on the HF path.
Zero numerics change. Helps op-level and e2e.

### 2.2 Native HF layout mode for u/delta/z too
`in_proj(...).transpose(1,2)` and `dt_proj(...).transpose(1,2)` are views that
`_c()` then materializes as full `(B,D,L)` copies; the gate `z` is a strided
chunk view, also copied. Options, in increasing effort:
- accept a per-tensor stride for `z` (the gate is only read once in the epilogue);
- accept `(B, L, D)` time-major u/delta and transpose per-chunk into scratch
  (same trick the kernel already uses for B/C) — a “zero `.contiguous()`
  anywhere” fast path for the HF forward.

### 2.3 Cache `A = -exp(A_log)` per layer
`patch.py:94` recomputes a D×N exp in torch + allocates, on **every forward of
every layer**. Cache on the module keyed by `A_log._version`. Same class of
fix: skip the `last_state` allocation when `return_last_state=False`
(`op.py` always allocates and writes it).

### 2.4 Accelerate decode, not just prefill  ⭐ moves the “total ×” column
Decode steps currently fall back to upstream by design, capping total
`generate()` speedup at the prefill share (1.20× at short prompts). Upstream's
decode step still does D×N exps + a pile of small tensor ops **per layer per
token in Python** — dispatch overhead dominates. Plan:
1. Add an **initial-state input `h0`** to the kernel (the missing half of the
   existing `last_state` output).
2. Patch the decode branch: one FFI call per layer per token (L=1, all D
   channels fused).
3. This also unlocks a **streaming API** — constant-memory processing of
   unbounded sequences via repeated chunked calls carrying `h` — the enabler
   for the audio/ECG/genomics “infinite stream on one Graviton” demo.

### 2.5 Fuse the depthwise causal conv1d (+ SiLU) into Rust
Kernel-size-4 depthwise conv is trivial NEON, takes a visible slice of the
patched prefill, and is *another* op with no optimized CPU path in this
ecosystem — widens the moat to “we own the whole Mamba mixer hot path except
the GEMMs.” Also enables the in-kernel rolling conv state for streaming decode.

### 2.6 Publish a “kernel + torch.compile together” row
`register_fake` means the op composes with compile. Kernel-inside-compiled-graph
should beat both baselines simultaneously — a devastating row against any
“you beat a strawman” critique. (Add a CI/bench assertion that no graph break
occurs, so a transformers/torch upgrade can't silently regress it.)

### 2.7 Patch breadth (ecosystem = Impact points)
- Patch **`mamba-ssm`'s** `selective_scan_ref`/interface for users who install
  the reference package on CPU.
- **`Mamba2Mixer`** (HF) / Mamba-2: scalar-A-per-head SSD models — the kernel
  for scalar A is *simpler* than the current one (one exp per timestep per
  head, not N) — one more “entire model family accelerated” bullet, covers
  Falcon-Mamba-2/Codestral-style checkpoints.
- Audit coverage for Jamba/Zamba/Bamba-style hybrids (they reuse the same
  mixer classes).

---

## 3. Kernel micro-optimizations (same algorithm, fewer cycles)

### 3.1 Cheapen `vexpq_f32` — it is ~half the single-thread cycles
Stacked, independently testable options:
1. **Domain-restrict**: the scan argument `dt·A` is always ≤ 0 (A<0, dt≥0).
   Fit the polynomial only for the actual domain and drop the `MAX_X` overflow
   clamp from the hot path (keep a checked general entry for other callers).
2. **Degree cut**: accuracy budget at the output is 1e-4; current exp is ~2 ulp
   (5e-7 rel). A degree-4 polynomial still lands ~1e-6 and cuts 2 of 6 serial
   FMAs. Re-record golden floors; keep the “small factor of `f32_max_abs_err`”
   criterion.
3. **Estrin's scheme** instead of Horner: halves the polynomial dependency
   depth at equal instruction count — pure latency win that helps whenever
   fewer than 4 exps are in flight.
4. **Table-assisted exp**: round k on `x·64/ln2` so the polynomial only covers
   1/64 of the range → degree-2 suffices (exactly glibc's SVE expf structure).
   On NEON the 2^(j/64) lookup is integer trickery or `vtbl`.
5. **Two accuracy modes**: `strict` (current) and `fast` (Schraudolph-style
   bit-trick exp + cubic mantissa correction, ~1e-5 rel — Malossi et al. show
   ~3–7× vs libm on SIMD). Both stay under the 1e-4 golden gate; publish
   model-metric parity for each. An honest extra rung for the ladder.

### 3.2 SVE2 `FEXPA` exp path (Graviton 3/4)  ⭐ WOW item
`FEXPA` performs the 2^(k/64) table lookup **in hardware**; glibc's SVE exp is
FEXPA + degree-2 polynomial — roughly half the instructions of our NEON exp.
- Graviton4 (V2): SVE2 at 128 bits — same lanes as NEON, win is pure
  instruction count on the dominant pass (est. 1.3–1.8× on Pass A2).
- Graviton3 (V1): SVE at **256 bits** — double the lanes on top. Worth one
  `c7g` benchmark session purely for the headline.
- Rust SVE intrinsics are nightly-only; a ~30-line inline-asm/`global_asm!`
  routine (or a tiny prebuilt `.S` object in the build) keeps everything else
  on stable Rust. Runtime-dispatched, NEON remains the fallback.
- Judge-facing framing: “first FEXPA-accelerated Mamba scan.”

### 3.3 Break the Pass B latency chain (the Graviton4 lever)
Two independent fixes that stack:
1. **Interleave 2 channels per Pass B sweep**: 8 h-registers live (still fits
   the 32-register file), zero extra FLOPs — the classic multi-accumulator
   chain-break from the Arm optimization guides. Requires Pass A to have both
   channels' `abar`/`bbar` chunks in scratch (they differ per channel — scratch
   doubles to ~34 KB, still L1-resident).
2. **Pairwise composition inside registers**: in Pass A (parallel-friendly),
   pre-compose adjacent timesteps under `(a,b)∘(a',b') = (a·a', a'·b + b')`;
   Pass B then runs L/2 iterations. Halves the serial chain for ~50% more
   Pass-A FLOPs; can be applied twice (4× shorter chain) if profitable.
   This is “the associative scan argument executed inside the register file” —
   excellent writeup material, and exactly the thing `torch.compile`
   structurally cannot do.

Expected effect: up to ~1.6× on Pass B on V2; ~nil on N1 (already
throughput-bound there) — frame as Graviton4-specific tuning.

### 3.4 Fuse the epilogue into Pass B; kill the per-timestep `vaddvq`
Accumulate 4 timesteps' dot products, `vpaddq`-tree them into one 4-lane y
vector, apply `d_skip` FMA + SiLU right there. Removes a full extra
read-modify-write pass over the output row (out is currently written by Pass B
then re-read and rewritten by `epilogue_row`) and replaces 4 cross-lane
reductions with a cheap tree.

### 3.5 Replace divisions with reciprocal-estimate + Newton
`vsiluq_f32` and the softplus Goldberg correction both use `vdivq_f32`
(low throughput, ~10+ cycle latency on these cores). `vrecpe` + 1–2 `vrecps`
Newton steps is the standard NEON idiom; verify the softplus correction still
meets its 2e-6 sweep bound (it may need the exact division — measure).

### 3.6 Vectorize the plane transpose
The `bt`/`ct` build is a scalar strided double loop. 4×4 in-register blocks
(`vld1q` ×4 + `vtrn`/`vzip` + `vst1q` ×4), or SVE gather loads, or fold the
transpose into Pass A per chunk (§5.2). Also consider **interleaving B and C
into one buffer** (`[b0..b15, c0..c15]` per t) so Pass A2/B read a single
forward stream instead of two.

### 3.7 Assembly audit (cheap, occasionally a jackpot)
`vexpq_f32` materializes ~11 vector constants (`vdupq_n_f32`) per call, ×4
calls per timestep. If LLVM fails to hoist them out of the A2 loop (register
pressure with a0–a3 live is real), that's dozens of redundant µops per
timestep. `cargo asm`/`objdump` the release build of `channel_n16`; check for
constant re-materialization and h-register spills. One hour, sometimes 20%.

### 3.8 Alignment & streaming micro-hygiene
- 64-byte-align scratch and the `bt`/`ct` planes (allocator API or manual
  offset) — avoids split-line loads on the hot streams.
- Software prefetch (`prfm pldl1keep`) of the next chunk's `bt`/`ct` rows
  during Pass B — the access pattern is perfectly predictable; HW prefetchers
  usually get this, so measure before keeping.
- Non-temporal stores (`stnp`) for `out` rows at very long L (written once,
  not re-read if §3.4 lands) — frees L1/L2 for the planes.
- Check false sharing at out-row boundaries between rayon workers (rows are
  `len·4` bytes; only matters for tiny L — likely a non-issue, verify once).

---

## 4. Memory & threading (fixes the regimes where the numbers sag)

### 4.1 Reuse workspace across calls + parallelize the transpose  ⭐
`bt`/`ct` are **allocated and zeroed fresh on every call, single-threaded**,
before rayon starts — at B=8/L=1024 that's ~16 MB of serial zero+transpose
per call. Almost certainly a chunk of the B=8 (8.2×) and D=3072 (11.2×) dips
and part of the 3.88×-not-4.0× scaling gap. Fixes:
- thread-local / call-cached workspace (grow-only, reused across layers and
  timesteps of `generate()`);
- rayon over planes (and over n-rows within a plane) for the transpose;
- skip the zero-fill except for the `n4` padding tail (the body is fully
  overwritten anyway).

### 4.2 Cache-block over L for long sequences  ⭐ protects the 131k demo
At L=8192 each transposed plane is 2×2 MB; all D channels re-stream it →
L2 misses → ~3 GB of DRAM traffic per call at D=1536. At the genomics length
(L=131k) a plane is 16 MB — hopeless without blocking. Restructure the loop
nest to:

```
for each L-chunk:                 # chunk of ~CHUNK timesteps
    (transpose that chunk of B/C once, hot in L1/L2)
    for each channel in a channel-block:
        Pass A + Pass B on this chunk   # h carried per channel (64 B each)
```

Per-channel carry state is tiny; `out` rows are written chunk-column-wise.
This keeps the ~21× plateau from degrading at exactly the sequence lengths the
best demos live at, removes the need for whole-plane buffers entirely (the
transpose becomes chunk-local scratch), and reduces cross-core L2/SLC
contention on 64-core `c8g` where all workers currently stream the same plane.

### 4.3 L-dimension parallel scan (3-phase Blelloch across chunks)
Per-chunk composed `(A_prod, B_comb)` computed in parallel → tiny sequential
combine of ~L/CHUNK carries → parallel finalize/output pass. Cost: one extra
elementwise pass. Enables full-machine scaling when **B×D < cores** — i.e.
single-stream latency (B=1 audio/ECG) on a big Graviton. This is the
documented stretch in `INTEGRATION_PLAN.md` Phase 3; the chunked structure
already exposes the composed pairs.

### 4.4 Thread hygiene & tuning
- Verify the FFI thread-count knob landed (plan §3.2 promised it); document
  the `torch.set_num_threads` / OMP interplay in the patched path.
- `with_min_len` / explicit chunk sizes on the rayon iterators to cut
  work-stealing overhead at large D.
- Pin workers (core affinity) for benchmark stability on shared runners.
- Autotune `PARALLEL_WORK_THRESHOLD` (currently `1<<17`) and `CHUNK`
  (currently 128, tuned for one L1 size — V2 has 64 KB L1D / 2 MB L2; make it
  a runtime pick keyed on detected core).
- NUMA: single-socket on the target instances today; re-check if `.metal`
  numbers are ever taken.

### 4.5 Allocator & pages
If workspace reuse (§4.1) lands, this mostly disappears; otherwise: mimalloc/
jemalloc as the cdylib allocator, and transparent-hugepage advice
(`madvise`) for multi-MB planes to cut dTLB pressure at long L.

---

## 5. Precision plays (opt-in, disclosed, gated)

### 5.1 fp16 storage for the streamed planes
Store `bt`/`ct` (and optionally `abar`/`bbar` scratch) as fp16, convert on
load (`vcvt_f32_f16`, single-µop pairs); compute stays f32. **Halves memory
traffic in exactly the bandwidth-bound regimes of §4.2.** All target cores
have FP16. Accuracy: ~1e-3 rel on B/C inputs vs. the 1e-4 output gate —
borderline by analysis, must be measured; ship only as opt-in `fast16` with
its own golden numbers and model-metric parity table. Rungs beyond:
`FMLAL` (fp16 mul, f32 accumulate) for the input-projection multiplies;
BF16/`BFDOT` for the C·h dot (higher accuracy risk; probably reject —
document the rejection).

### 5.2 The accuracy flex (opposite direction)
Optional f64 or Kahan-compensated accumulation for the y dot product (16
terms — negligible cost). Lets RESULTS.md show the kernel can be **more**
accurate than torch eager f32, not merely “within tolerance.” Cheap
credibility with numerics-literate judges.

---

## 6. Build & toolchain (boring, compounding)

1. **`-C target-cpu=neoverse-{n1,v1,v2}` builds with runtime dispatch**
   (function-pointer selection at init keyed on MIDR/`hwcaps` — no nightly
   needed). Currently the wheel is generic aarch64; per-core scheduling models
   help the exp/FMA interleaving. Options: fat cdylib with all variants, or
   per-arch wheels.
2. **`lto = "fat"`, `codegen-units = 1`, `opt-level = 3`** on the release
   profile of the cdylib (verify what's set today). Keep unwind (the FFI
   catches panics).
3. **PGO** on the bench workload — typically 5–15% on dispatch-heavy code;
   scriptable into `build_wheel.py`. Optionally **BOLT** the `.so`.
4. CI: build one wheel per target-cpu variant, run goldens through each.

---

## 7. Genuinely novel / out-of-the-box (including things to measure-and-reject publicly)

### 7.1 SSD / matmul duality on CPU (Mamba-2 style) — probably a documented rejection
Reformulate chunks as small GEMMs (the Mamba-2 “state space duality”
decomposition: intra-chunk matmuls + short inter-chunk scan). On GPUs this
wins because tensor cores make matmul FLOPs ~free. On CPU with diagonal A and
N=16, the plain scan is already near SIMD peak and the dual strictly adds
FLOPs (O(L·CHUNK) intra-chunk attention-like term vs. O(L·N) scan). The
winning move: benchmark it, show the crossover doesn't exist on 128-bit SIMD,
and write it up as a *considered-and-rejected* decision with numbers. Becomes
genuinely relevant only for Mamba-2 support (§2.7), where scalar-A makes the
matmul form natural.

### 7.2 Chunk-carry underflow cut (data-dependent free parallelism)
`abar ∈ (0,1]` always. Whenever a chunk's per-state product of `abar`
underflows to **exact f32 zero for all N states**, later chunks are *provably
independent* of everything before them — bit-exact, no approximation. Detect
with one running product + `vmaxvq` per chunk; when the carry is all-zero,
the L-parallel scheduler (§4.3) can cut the sequence there for free. Cute,
honest, and a lovely “the math hands us parallelism” aside for the writeup.
Data-dependent — never a headline claim, always an observed bonus.

### 7.3 Merged A2+B experiment
The two-pass split exists to keep exp latency out of the serial chain. But
with channel interleaving (§3.3.1) providing ILP, inline exp might hide in the
recurrence again — deleting the 16 KB/chunk scratch round-trip entirely. This
contradicts the current design, which is exactly why it should be a measured
A/B, not an assumption. Keep whichever wins per-core-generation.

### 7.4 Selectivity skipping
Where `softplus(delta) ≈ 0`: `abar ≈ 1`, `bbar ≈ 0` → the timestep is a
near-no-op for that channel. Branchless SIMD hates this and it's
data-dependent; note it, measure the delta distribution on real checkpoints
once, then (probably) document the rejection.

### 7.5 Reverse flag + fused SS2D cross-scan (application-critical, same theme)
From `APPLICATIONS.md`, restated here because they are also *kernel*
optimizations in the “kill redundant memory traffic” family:
- **`reverse` flag** — walk the sequence backward in place: deletes two
  full-tensor `torch.flip` copies for every bidirectional model (genomics,
  audio, ECG). ~Half a day.
- **Fused four-direction SS2D traversal** — read the patch grid once instead
  of materializing four permuted/flipped copies; the real white space (no CPU
  implementation exists anywhere). Also composes with §4.2's chunk-blocking.

### 7.6 Streaming/stateful API as a product feature
`h0` in + `last_state` out + rolling conv state (§2.4, §2.5) = process
unbounded sequences in constant memory with bit-identical results to the
one-shot call (prove it with a golden test: split any golden vector at random
boundaries, compare). This turns “linear time, constant memory” from a slide
claim into an API a judge can call.

### 7.7 Alternative baselines to beat (breadth of honesty)
Add rows nobody will think to demand, before someone does:
- ONNX Runtime CPU (if any Mamba export path works),
- `mamba.py` / pure-numpy path,
- llama.cpp-style Mamba implementations (ggml has one) — the strongest
  non-PyTorch CPU baseline in existence; beating or matching ggml's scan on
  the same box, from *inside* PyTorch, is a serious flex. If ggml wins a row,
  publish it anyway and analyze why (their layout has no Python/dispatch tax).

---

## 8. Overhead trims (small, add up)

- ctypes call path: prebuild the `ArmScanDims` struct + argtypes once; the
  per-call Python overhead only matters for L=1 decode (§2.4), where it
  matters a lot.
- `_c()` calls `.contiguous().float()` — `.float()` on an already-f32 tensor
  is a no-op but still a dispatcher round-trip ×9 tensors ×24 layers; guard
  with `if t.dtype is torch.float32`.
- `torch.empty_like` + `new_empty` per call → consider a cached output
  arena keyed by shape for the decode path (functional custom-op constraints
  permitting).
- `kernel_calls` dict increment per call — free, but move off the hot path if
  decode lands.
- The general-N path re-runs `vexpq` on padded lanes (harmless but wasted);
  only worth touching if a real model with N≠16 shows up (Mamba-2 heads).

---

## 9. Measure-first program (protects “benchmark honestly”)

> **Tooling now exists — see [`PROFILING.md`](./PROFILING.md).** The
> instrumented phase profiler (`scan_profiled`), the one-click free
> `Profile kernel` GitHub Actions workflow (Tier 1, real Neoverse), the
> Oracle A1 perf script (Tier 2), and the asm-audit script (Tier 0) are all
> built. Run Tier 1 first; its per-phase table reorders §10 below.

1. **PMU cycle accounting — do this FIRST, before any §3 kernel work.**
   The PMU (Performance Monitoring Unit) is the CPU's hardware counters
   (cycles, instructions, stalls, cache misses); a "session" is 1–2 hours of
   running the kernel under Linux `perf` on real Arm silicon:
   - `perf stat -e cycles,instructions,stalled-cycles-backend,l2d_cache_refill,l1d_cache_refill`
     on `bench_op.py` shapes → answers *compute-bound vs memory-bound* per shape.
   - `perf record` + `perf annotate` on `channel_n16` → shows which
     instructions eat the cycles (exp polynomial vs Pass-B FMA chain vs
     transpose).
   - **Why first:** the whole §10 priority order rests on the §1 diagnosis,
     which is *inferred* from instruction counts and published latency tables,
     not measured. If perf shows e.g. memory stalls dominating, §4
     (cache-blocking, workspace) jumps ahead of §3.1 (exp tuning) and weeks of
     polynomial work are skipped. The §2 integration wins (2.1–2.3) are exempt
     — they delete provably redundant copies and need no profiling.
   - **Where:** a real Arm Linux box with PMU access — Oracle Ampere A1 or a
     short Graviton spot session. Not GitHub shared runners (no PMU access),
     not the x86 dev box (NEON path never runs there). Ideally once on
     N1-class (A1) *and* once on Graviton4, since §1 predicts different
     bottlenecks per generation.
2. **Roofline sweep over L** (op-level GFLOP/s + measured bytes) showing the
   compute→bandwidth transition — motivates §4.2 with one chart judges will
   love, and quantifies the fp16 (§5.1) ceiling.
3. **Per-pass timers** behind a cargo feature so RESULTS.md can attribute the
   ladder rung-by-rung (A1/A2/B/epilogue/transpose).
4. **Criterion baselines in CI** — store per-shape medians as artifacts and
   fail on >X% regression, so integration work never silently eats kernel wins.
5. Every new mode (layout flag, fast-exp, fp16, h0, reverse, FEXPA) gets its
   own golden gate + recorded floor; parity matrix reruns under
   `RAYON_NUM_THREADS ∈ {1,2,8}`.
6. Core-scaling curve **to 16/32/64 cores** on `c8g` — the shared-plane
   contention (§4.2) will show up there first; fix before the headline run.

---

## 10. Priority shortlist

| # | Idea | § | Effort | Expected effect | Risk |
|---|---|---|---|---|---|
| 1 | B/C layout flag — kill double transpose/copies | 2.1 | Low | e2e prefill +10–25%; op-level too | ~0 |
| 2 | Workspace reuse + parallel transpose | 4.1 | Low | Fixes B=8 / D=3072 dips; closes scaling gap | ~0 |
| 3 | Decode-step kernel via `h0` input | 2.4 | Med | Total `generate()` × moves toward/past prefill × | Low |
| 4 | Cheaper exp (domain, degree-4, Estrin) | 3.1 | Med | 10–25% op-level (exp ≈ half the cycles) | Low (gated) |
| 5 | L cache-blocking | 4.2 | Med | Protects long-L/131k headline; 64-core scaling | Low |
| 6 | Pass B chain-breaking (interleave / compose) | 3.3 | Med | 10–30% op-level on Graviton4 specifically | Low |
| 7 | target-cpu builds + LTO + PGO | 6 | Low | 5–15% across the board | ~0 |
| 8 | SVE2 FEXPA exp | 3.2 | Med-Hi | up to ~1.5–1.8× on the exp pass; big WOW | Med (asm) |
| 9 | L-parallel chunk scan | 4.3 | High | Single-stream scaling story (B=1 demos) | Med |
| 10 | Conv1d+SiLU fusion | 2.5 | Med | e2e prefill; widens the moat | Low |
| 11 | fp16 plane storage (opt-in) | 5.1 | Med | Bandwidth-bound regimes; needs accuracy data | Med |
| 12 | A-cache + misc patch trims | 2.3, 8 | Low | e2e, esp. many-layer models | ~0 |

**The theme:** the inner math is already close to SIMD peak on 2-pipe cores.
The remaining big wins are (a) stop doing redundant memory work around the
kernel, (b) make exp cheaper, and (c) break the serial chain that 4-pipe
Graviton4 exposes. Items 1–3 alone would visibly move every table in
`BASELINE_REPORT.md` without touching a single polynomial coefficient.

---

## 11. Target numbers — what "exceptional" looks like

Aspirational but reachable-from-this-backlog goals for the submission. Every
row must be honestly measured under the CLAUDE.md benchmarking rules; if a
target is missed, publish the real number anyway.

| Stat | Today (BASELINE_REPORT) | Target | Backlog items that get us there |
|---|---|---|---|
| Op-level vs **torch.compile** | 3.3–4.2× | **≥5×, stretch 8×** | §3.1–3.4, §6 |
| Op-level vs eager | 8.2–30.4× (sags at B=8, D=3072) | **geomean ≥25×, no row < ~15×** | §2.1, §4.1 (fixes the sag), §3 |
| E2E prefill, mamba-130m | 1.9–2.1× | **≥3×** | §2.1–2.3, §2.5 |
| E2E **total** `generate()` | 1.20–2.03× | **≥2.5× at every prompt length** | §2.4 (decode kernel — the big mover) |
| Kernel ladder scalar→full | 13.3× (4 cores) | **~25× (4 cores)** | §3 collectively |
| Core scaling | 3.88×/4 | **≥90% efficiency at 32–64 cores on c8g** | §4.1, §4.2 |
| Long context | untested > 8k | **131k tokens, ~flat ×eager, constant memory** | §4.2, §7.6 |
| Max error vs f64 ref | ≤ 3.8×10⁻⁶ | **unchanged or better** | §5.2 (Kahan flex); everything else gated |

### The GPU question (framing + measured rows to add)

Physics first, because it is the best pitch line we have: **a full 64-core
Graviton4 is within striking distance of a T4-class GPU on paper for this op**
— ~5–6 TFLOP/s peak fp32 SIMD vs ~8 for a T4, comparable memory bandwidth,
and the selective scan is *not a matmul*, so the GPU's tensor cores don't
apply to it. CPU Mamba being ~100× slower than GPU Mamba today is a
**software gap, not a silicon gap** — and closing it is this project.
Use that line in the video and README.

What to chase and what to concede:

1. **Concede batch prefill throughput.** A well-fed GPU wins raw tokens/sec
   by 5–20× on the full model (the surrounding GEMMs *do* use tensor cores).
   Publish the losing row honestly — it buys credibility for the winning rows.
2. **Chase single-stream decode.** GPU decode of a 130m Mamba is
   kernel-launch-overhead bound (~100–200 tok/s regardless of silicon); a
   fused CPU decode step (§2.4) has no launch overhead. **Target: within
   1–3× of GPU single-stream decode speed at ~1/4 the instance cost →
   parity or better on $/token.** "CPU matches GPU per dollar on interactive
   decode" is the exceptional-but-defensible claim.
3. **Chase cost per unit work.** $/1M prefill tokens, $/1000 recon slices:
   target **within ~2–4× of a T4 on raw $/token**, winning on availability
   (no GPU quota, spot capacity everywhere, scales to zero, hardware that
   hospitals/bio clusters already own).
4. **The knockout row:** 131k-token genomics context in constant memory on a
   ~$0.15/hr instance. A transformer can't do that on CPU at any price, and
   on GPU the KV cache alone at 131k is multiple GB. Not "close to GPU" —
   a workload class where the GPU comparison doesn't survive.

Notes for the GPU measurement session:
- Measure it ourselves: one short `g4dn.xlarge` (T4) or `g5.xlarge` (A10G)
  session, ~$1–2 total — never cite third-party numbers, judges will check.
- Same model, same prompts, same seeds, same tokenizer; report instance type,
  torch/CUDA versions, and warmup protocol identically to the CPU rows.
- Record GPU utilization alongside — the underutilization at batch=1 *is*
  the argument, so show it, don't just assert it.

### Summary sentence to aim the whole submission at

> **5× past the compiler, 3× end-to-end, 90% scaling to 64 cores,
> GPU-per-dollar parity on interactive decode, and one workload the GPU
> can't touch — at 1e-4-gated accuracy throughout.**

That set is achievable from §2–§6, every number is honestly defensible, and
together it is a stronger story than a raw-speed race we would lose.

---

## Sources

- [Arm Neoverse V2 Software Optimization Guide](https://developer.arm.com/documentation/109898/latest/)
- [Arm Neoverse N2 Software Optimization Guide](https://developer.arm.com/documentation/109914/latest/)
- [Chips and Cheese — Arm's Neoverse V2 in AWS Graviton4](https://chipsandcheese.com/p/arms-neoverse-v2-in-awss-graviton-4) (4×128-bit SIMD pipes)
- [Chips and Cheese — Deep Diving Neoverse N1](https://chipsandcheese.com/p/deep-diving-neoverse-n1)
- [AWS Graviton getting-started: assembly optimization guide](https://github.com/aws/aws-graviton-getting-started/blob/main/arm64-assembly-optimization.md) (multi-accumulator FMLA chain-breaking)
- [Arm learning path — Optimize exponentials with FEXPA](https://learn.arm.com/learning-paths/servers-and-cloud-computing/fexpa/fexpa/)
- [glibc: SVE vector exp routines (FEXPA + degree-2 poly)](https://sourceware.org/pipermail/libc-alpha/2023-June/149127.html)
- [LLVM forum — vscale on Graviton4 = 128-bit SVE2](https://discourse.llvm.org/t/vscale-of-sve-vectors-on-aws-graviton-4-neoverse-v2/80597) (Graviton3/V1 = 256-bit)
- [Ash Vardanian — NEON → SVE2 on Graviton](https://ashvardanian.com/posts/aws-graviton-checksums-on-neon-vs-sve/)
- [Tri Dao — State Space Duality (Mamba-2) Part III: the algorithm](https://tridao.me/blog/2024/mamba2-part3-algorithm/)
- [Schraudolph — A Fast, Compact Approximation of the Exponential Function](https://nic.schraudolph.org/pubs/Schraudolph99.pdf)
- [Malossi et al. — Fast Exponential Computation on SIMD Architectures](https://wapco.e-ce.uth.gr/2015/papers/SESSION3/WAPCO_3_5.pdf) (bit-trick exp + polynomial correction)
