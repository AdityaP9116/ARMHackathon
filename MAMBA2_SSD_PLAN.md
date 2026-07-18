# MAMBA2_SSD_PLAN — a true Mamba-2 (SSD) → Mamba-3 Arm/NEON kernel for 2D workloads

**Written Jul 18, 2026. Final thorough-steps revision same day.** All external facts in this
doc were verified Jul 18 (searches + repo checks); each carries its source inline. Companion
to `SS2D_REPOSITIONING_PLAN.md` (which *rejected* SSD for the Aug-14 submission) and
`RESEARCH_TRIAGE_MAMBA2_2D.md` (which verified that rejection). This doc is the answer to
"what would it take to do it anyway, properly" — for 2D applications: segmentation, image
processing, and the diffusion backbone — now extended through **Mamba-3** (§5), whose
official kernels are GPU-DSL-only, leaving the CPU slot open.

---

## 0. The strategic gate (decide this before any code)

**What Mamba-2/SSD actually changes.** Mamba-1 (what every kernel in this repo implements):
diagonal `A ∈ (channels × d_state)`, per-element recurrence `h = exp(Δ·A)⊙h + ΔB·x` —
sequential, scan-shaped, SIMD-friendly. Mamba-2 (SSD): **scalar-per-head A**, heads with
`d_head` channels sharing `(a_t, B_t, C_t)`, and the state-space *duality*: token mixing is
a 1-semiseparable matrix computed chunk-wise as **dense matmuls** (intra-chunk
attention-like Gram products under a cumulative-decay mask) plus a tiny inter-chunk state
recurrence. Compute shifts from exp+FMA scan to ~90% GEMM.

**Consequences to accept explicitly:**

1. **The moat weakens.** Our torch.compile argument is "compilers can't restructure a
   sequential recurrence." SSD's dual form is matmuls — exactly what compilers/BLAS *are*
   good at. Expect the vs-torch.compile margin to shrink; the honest claim becomes
   "competitive, portable, pip-installable Rust SSD on Arm," not a structural win.
2. **Claims policy check:** llama.cpp already runs Mamba-2 models on CPU (generalized
   `ssm_scan`, GGUF runtime). The defensible novelty is **2D/cross-scan SSD on CPU** (VSSD
   is the vision-Mamba-2 precedent and is CUDA-bound) and **PyTorch-callable** SSD on Arm —
   run a fresh prior-art pass before writing any claim.
3. **Timeline:** ~2–3 weeks of careful work for M-track, +1–1.5 weeks for N-track (§5).
   **Not before Aug 14.** Build **additively** on `feature/ssd` — the Mamba-1 scan stays
   the shipped product; SSD/Mamba-3 become the post-submission second op. Nothing existing
   is converted or deleted (the scalar-path rule and the diffusion app keep working
   unchanged).
4. **Mamba-3 is a targeted extension (§5), not a watch item.** Verified Jul 18: ICLR 2026
   ([arXiv:2603.15569](https://arxiv.org/abs/2603.15569),
   [Together blog](https://www.together.ai/blog/mamba-3)); trapezoidal discretization,
   complex states ≡ data-dependent RoPE on B/C, MIMO decode. Its released kernels are
   Triton (SISO prefill, RoPE-fused), TileLang (MIMO prefill), and CuTe DSL (decode,
   gating+MIMO-fused) — **all GPU**; no CPU path exists and community requests for one are
   open ([mamba#809](https://github.com/state-spaces/mamba/issues/809)). M0's reference
   vendoring and M4's ABI are designed so the Mamba-3 block is an *addition*, not a
   rewrite. Vision-Mamba work (VSSD, EfficientViM) is still SSD-based today; §5's
   application gates account for that lag.

**The gate itself:** proceed only when (a) the Aug-14 submission is frozen or the schedule
has verified slack, and (b) a fresh prior-art pass (30 min: llama.cpp PRs, arXiv "SSD CPU",
"Mamba-2 SIMD", "Mamba-3 CPU") comes back clean. Record the pass in `PROJECT_CONCEPT.md`.

---

## 1. What we already have that carries over (the leverage)

| Asset | Reuse in SSD/Mamba-3 work |
|---|---|
| Golden methodology (f64 vendored ref → npz + manifest floors → independent numpy verifier → Rust gates) | Identical pipeline; only the reference function changes |
| FFI / ctypes loader / torch custom-op / wheels / CI matrix / bench harness | New entry point beside the old one; ABI bump per the versioning contract |
| `parallel.rs` dispatcher + thread-local workspace arena (P1-3) | Parallelize over `batch × heads (× directions)`; same arena pattern |
| `vexpq_f32` family | Computes `exp(segsum)` decay masks — segsum arguments are always ≤ 0 (cumulative sums of `log a`, `a ∈ (0,1)`), so `vexpq_f32_nonpos`/`_fast` apply directly, same precondition proof as Pass A2 |
| P0-1 batched 4-direction cross-scan machinery (`ss2d.py`, block layout) | The 2D wrapper is architecture-agnostic — SSD slots in behind the same `scan_fn` seam |
| Reverse flag / h0-resume / `last_state` plumbing | Same semantics needed for SSD chunk boundaries, Mamba-3's input carry, and bidirectional 2D |
| `matrixmultiply` 0.3.11 (verified present in `kernel/Cargo.lock`) | f32 GEMM baseline before any hand-rolled microkernel |
| EDM/CSI diffusion scaffolding + Route-A distillation plan | Backbone-agnostic; a Mamba-2 student distills from the same U-Net teacher |

---

## 2. Milestones — with steps

Standing rules apply to every step: correctness before speed; never loosen a tolerance;
parallel output bit-identical to sequential; every benchmark names its baseline, host,
torch version, and thread count.

### M0 — Ground truth (gate: goldens verified two ways)

1. Vendor [`mamba_ssm/modules/ssd_minimal.py`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/ssd_minimal.py)
   (exact path verified; it is Listing 1 of the Mamba-2 paper) into
   `tests/reference/ssd_minimal_ref.py`, pinned to a tagged `mamba_ssm` release; strip the
   torch-only decorators so it runs CPU-only, f64.
2. Write `tests/gen_golden_ssd.py`: cases over heads ∈ {4,8}, d_head ∈ {32,64},
   d_state ∈ {64,128}, chunk ∈ {32,64,128}, L including non-multiples of chunk, plus a
   fast-forget case (`a → 0⁺`, the decay-mask error driver) and a `groups>1`-style
   B/C-sharing case if the head layout supports it. Emit `tests/golden/ssd_*.npz` +
   manifest entries recording each case's `f32_max_abs_err` floor, same format as 1D.
3. **Numerics rule:** segsum is computed the stable way (within-chunk pairwise differences
   of `log a`), never "global cumsum then subtract" — at chunk 128 the naive form cancels
   catastrophically in f32.
4. Write `tests/verify_golden_ssd.py`: an independent numpy re-derivation (chunked math
   re-expressed differently — e.g., materialize the full 1-semiseparable matrix for small L
   and multiply) so the goldens don't rest on one implementation. Two-implementations rule,
   same as Phase 0.

**Exit:** every golden reproduced by both implementations to f64 agreement; floors recorded.

### M1 — Rust scalar SSD (gate: goldens at f32 floor)

1. New module `kernel/arm-scan-core/src/ssd/mod.rs` + `ssd/scalar.rs`; public entry
   `ssd_scan(dims: &SsdDims, input, out, state_carry, opts)` with
   `SsdDims { batch, heads, d_head, d_state, len, chunk }`.
2. Implement per (batch, head): segsum → decay mask `L_ij = exp(segsum_i − segsum_j)`
   (lower-triangular); Gram `G = C·Bᵀ`; `Y_intra = (G ⊙ L)·X`; chunk state
   `S = (decay-weighted B)ᵀ·X`; sequential inter-chunk carry
   `S_carry ← a_chunk·S_carry + S`; output `Y = Y_intra + (C·S_carry)·decay_in`.
   Generic over f32/f64 via the existing `Float` trait, mirroring `scalar.rs` style
   (clarity over speed — this is the in-crate oracle).
3. Port the golden harness: `tests/golden_ssd.rs` asserting `< 1e-4` **and** proximity to
   each case's recorded floor (not orders above), exactly like the 1D gate.
4. Property tests: chunked-vs-unchunked equivalence (chunk=L must equal chunk=32 result
   within f32 tolerance), h0-resume splicing (scan halves == scan whole).

**Exit:** all SSD goldens green on scalar, f32 errors near floors; property tests green.

### M2 — NEON (gate: parity vs scalar + goldens)

1. First pass: route the three GEMM-shaped ops (`G`, `Y_intra`, `S`) through
   `matrixmultiply::sgemm` (verified in-tree, v0.3.11); keep mask/segsum/decay in NEON
   using `vexpq_f32_nonpos`. Measure — this is the baseline that decides where hand-rolling
   pays.
2. Hand-roll only what profiling flags: `vfmaq_laneq_f32` panel microkernels (~8×12 f32)
   for the chunk-sized Gram products; layout chunk-major, head-contiguous so panels stream.
   Pass-A/B experience maps: Pass A ≈ mask/decay precompute, Pass B ≈ GEMM pipeline.
3. **Chunk size is a static tune, not a runtime policy:** sweep `Q ∈ {32,64,128}` per shape
   and fix the best. COREY ([arXiv:2604.10597](https://arxiv.org/abs/2604.10597)) measured
   static chunks matching/beating entropy-guided runtime scheduling end-to-end; dynamic
   chunking would also break rayon bit-identity.
4. Parity test NEON↔scalar (`≤` the 1D parity tolerance), per case.
5. **Graviton3/4 stretch:** BF16 `bfmmla` 2×2 tiles for Gram products (bf16 storage / f32
   accumulate; same disclosure discipline as the precision-plays section).

**Exit:** goldens + parity green; profile table (matrixmultiply vs hand-rolled) recorded in
`OPTIMIZATION_LOG.md`; chunk choice per shape documented.

### M3 — Threading + workspace (gate: rayon bit-identity)

1. Extend `parallel.rs` with a `for_each_head` driver: rayon over
   `batch × heads (× directions)`; thread-local arena (P1-3 pattern) for per-chunk panels
   (G, mask, S, scratch).
2. Intra-row parallelism over chunks is allowed **only** for the chunk-local phases; the
   inter-chunk carry stays a sequential loop per row — no tree reduction (FMA reassociation
   would break the bit-identity guarantee the entire suite treats as hard).
3. Bit-identity test at `RAYON_NUM_THREADS ∈ {1,2,8}`, per variant.

**Exit:** bit-identity green; thin-batch scaling curve (chunks-parallel) measured.

### M4 — FFI + Python (gate: goldens through the C ABI)

1. `arm-scan-ffi`: new `arm_scan_ssd_f32` entry with `#[repr(C)] SsdDims` + a
   **block-variant enum** (`SSD = 0`, reserved values for Mamba-3 SISO/MIMO — the
   §5 extensibility contract); overflow-checked sizes, `catch_unwind`, ABI version bump.
2. `python/arm_scan/_ffi.py`: `ssd_raw` binding; `op.py`: `torch.ops.arm_scan.ssd` custom
   op with fake kernel (torch.compile composability); `numpy_api.py` twin.
3. `tests/check_ffi.py` extended: SSD goldens through the real C ABI.
4. Block-selection seam: `use_arm_scan(module, block="scan_v1"|"ssd")` so app code selects
   per config — and so an HSM-style block (M7c) is selectable later without another ABI
   change.

**Exit:** goldens replay through the C ABI; torch op passes a compile-mode smoke test.

### M5 — 2D wrappers (gate: cross-scan parity on 2D goldens)

1. Reuse the batched 4-direction machinery verbatim behind the SSD op ("SS2D-SSD"): same
   view-building, `4B` stacking, merge — only the scan callee changes.
2. Add the **VSSD non-causal variant** ([ICCV 2025](https://arxiv.org/abs/2407.18559)) as a
   flag: causal mask dropped intra-chunk. Clear-eyed CPU framing: non-causal collapses the
   dual form to two dense GEMMs (linear-attention-shaped) — best fit for NEON matmul
   throughput, least defensible vs a `torch.compile`d baseline; M6 measures that trade
   honestly.
3. Grid blocking follows 2DMamba's tiling discipline
   ([CVPR 2025](https://arxiv.org/abs/2412.00678)): tiles sized from the target core's
   L1d/L2 so panels + carries stay resident (their SRAM argument, our cache hierarchy) —
   derived, not hardcoded.
4. 2D goldens: per-direction outputs checked *before* merge (isolates kernel bugs from
   merge bugs), non-square and non-multiple-of-4 grids included.

**Exit:** 2D goldens green per direction; VSSD flag verified against a non-causal reference.

### M6 — Honest benchmarks (gate: RESULTS rows, all baselines named)

1. Baselines: torch.compile'd `ssd_minimal` (the fair fight — expect a smaller margin;
   publish it), eager, and **our own Mamba-1 kernel at matched shapes**.
2. **The crossover curve is the headline science.** Don't presume SSD loses on CPU: at
   diffusion shapes (L≈123k) the Mamba-1 scan is bandwidth-bound (7.9 MB B/C planes per
   call) while SSD's chunked GEMMs carry higher arithmetic intensity. Measure the
   scan-vs-dual crossover over `(L, d_state, cores)` and publish the curve — no such CPU
   result exists in the literature; it upgrades this branch from "port" to "study."
3. arm64 CI first (provisional tags), Graviton `c8g` for headline rows; core-scaling curve
   per the existing three-surface discipline in `BASELINE_TEST_PLAN.md`.

**Exit:** RESULTS rows with named baselines; crossover curve plotted; unflattering rows
published.

### M7 — Applications

**(a) Segmentation/processing.** VSSD or VMamba-2-style backbone; Phase-A-style CPU
dry-run gate on a real checkpoint, then `patch()`-style integration + mIoU parity.
**Checkpoint reality (verified Jul 18):** [VSSD's repo](https://github.com/YuHengsss/VSSD)
publishes ImageNet *classification* weights; segmentation weights are **by author request
only**. Gate on weight availability *first*; fallback is classification top-1 parity (a
weaker but self-serve demo) or training the seg head ourselves. CUDA-coupling risk applies
(the gate that killed DH-Mamba).

**(b) Diffusion.** `SS2DBlock` variant with per-head scalar `A_log` + heads (localized
change; EDM contract untouched); Route-A distillation works identically since the teacher
is a U-Net.

**(c) Backbone-efficiency variant worth a flag.** EfficientViM's HSM-SSD
([CVPR 2025](https://arxiv.org/abs/2411.15241)) moves channel mixing into the compressed
hidden-state space — on CPU that directly attacks the projection GEMMs that dominate once
the scan is fast; natural "backbone v2" partner, selectable via M4's seam.

---

## 3. Risks

| Risk | Mitigation |
|---|---|
| Moat dilution vs torch.compile | Publish the scan-vs-dual comparison as a finding; keep Mamba-1 as headline until numbers say otherwise |
| Prior art (llama.cpp Mamba-2 CPU) | Fresh prior-art pass at the §0 gate; novelty lives in 2D/VSSD + PyTorch-callable + Mamba-3 |
| Timeline vs Aug 14 | Additive branch; zero changes to the submission path |
| GEMM perf on NEON is a deep well | `matrixmultiply` first; hand-tune only where profiled |
| VSSD checkpoint CUDA-coupling | Same hard Phase-A dry-run gate that killed DH-Mamba |
| VSSD downstream weights are request-only (verified Jul 18) | Gate on availability before committing M7a; classification-parity fallback |
| Mamba-3 spec/kernels still settling post-publication | Pin vendored references to tagged releases; goldens are the contract, not repo HEAD |

## 4. Sequencing

Post-submission (or if the main plan freezes early): M0–M1 week 1; M2–M4 week 2; M5–M7
week 3+; N0–N4 (§5) follow M0–M6 as deltas — they reuse the SSD chunked machinery, not a
parallel build. Every gate inherits the standing rules.

---

## 5. Mamba-3 extension (targeted; verified against the paper Jul 18, 2026)

**Why this is worth targeting, in one paragraph.** Mamba-3
([ICLR 2026](https://arxiv.org/abs/2603.15569); CMU/Princeton/Cartesia/Together) ships its
kernels as: SISO prefill in **Triton** (RoPE-fused), MIMO prefill in **TileLang**, decode in
**CuTe DSL** (gating+MIMO-fused) — **all GPU-only**. As of Jul 18, 2026 there is no CPU
kernel anywhere ([mamba#809](https://github.com/state-spaces/mamba/issues/809)) and no
official HF checkpoint yet
([mamba#860](https://github.com/state-spaces/mamba/issues/860)). That is the same
white-space shape this project already exploited for SS2D — first-mover on a brand-new
block. Claims discipline as always: "first CPU/Arm Mamba-3 kernel, to our knowledge,"
re-verified at kickoff.

### 5.1 What actually changes in the kernel (the three deltas)

**(a) Trapezoidal three-term recurrence.** Mamba-2/1 update is
`h_t = ā_t·h_{t-1} + Δ_t B_t x_t`. Mamba-3:

```
h_t = exp(Δ_t A_t)·h_{t-1} + (1−λ_t)·Δ_t·exp(Δ_t A_t)·B_{t-1} x_{t-1} + λ_t·Δ_t·B_t x_t
```

— equivalently a data-dependent **width-2 convolution on `B_t x_t` inside the recurrence**
(λ_t a learned convex weight). Kernel impact is localized to Pass A: compute
`v_t = Δ_t B_t x_t` as today, then blend `w_t = λ_t·v_t + (1−λ_t)·ā_t·v_{t-1}` — one extra
vector FMA per timestep plus a **one-element `v` carry across chunk boundaries** (and
across the h0/resume seam; the carry joins the state in the resumable-API contract). Pass B
is untouched: same `h = ā⊙h + w` FMA chain. Bidirectional Pass-A sharing survives with
per-direction `w` (the width-2 conv points the other way under `reverse` — same
`chunks_in_scan_order` treatment).

**(b) Complex states ≡ data-dependent RoPE on B/C.** The paper's own equivalence: apply
data-dependent rotations to B/C lane-pairs instead of complex arithmetic. On NEON: a
pointwise pass over even/odd lane pairs (`vfma`/`vfms` with swapped-pair operands via
`vrev64q_f32`), structurally like the existing epilogue; lives in Pass A next to the exp.
Rotation angles are cumulative and carry across chunks like the decay — the segsum
machinery reused. **Goldens note:** rotations are norm-preserving so floors should track
Mamba-2 cases; add a rapidly-varying-angle case to catch cumulative-angle f32 drift.

**(c) MIMO decode.** Rank-r input/output (outer-product state update instead of rank-1).
This raises arithmetic intensity exactly where the CPU story is weakest — decode, the
memory-bound regime (`IMPROVEMENT_IDEAS.md` §2.4) — so MIMO is the delta most likely to
*help* CPU throughput: more FLOPs per byte of state streamed. Implementation: a small
`(r × d_state)` panel FMA, reusing M2 microkernels at decode shapes.

### 5.2 Milestones (deltas on M0–M6; same gates, same discipline)

- **N0 — Reference + goldens.** Vendor the reference from the official repo's minimal/eager
  path (pin to a tagged release; if no eager reference exists, transcribe the paper's
  equations and verify once against their Triton output on a borrowed GPU). Goldens add:
  λ extremes (**λ=1 recovers exactly the Euler/Mamba-2 form — a free cross-check against
  M0's goldens**), rapidly-varying RoPE angles, MIMO r ∈ {1,2,4} (**r=1 must reproduce the
  SISO goldens bit-for-bit in f64 — another free gate**).
- **N1 — Scalar Mamba-3.** `ssd/scalar.rs` grows the three deltas behind the M4
  block-variant enum. Gate: goldens at f32 floor; λ=1 path bit-identical to M1 scalar.
- **N2 — NEON.** Pass-A additions ((a) one FMA + carry, (b) rotation pass); Pass B
  unchanged; MIMO panel FMA from M2 microkernels. Gate: NEON↔scalar parity per variant.
- **N3 — Threading/FFI/torch op.** Same bit-identity rule (sequential inter-chunk carry now
  carries `(h, v, angle)`), one ABI bump for variant enum + λ/rank params, fake kernel
  unchanged in shape.
- **N4 — Benchmarks.** The M6 crossover study gains a third curve (Mamba-2 dual vs Mamba-3
  trapezoidal at matched shapes — the paper claims better quality per state byte; we
  measure cost per token on CPU). Decode rows at r ∈ {1,4} quantify the MIMO
  arithmetic-intensity win on Graviton — the most quotable row.

### 5.3 What NOT to do (Mamba-3-specific)

- **No application milestone yet.** No public checkpoints (#860), no vision-Mamba-3 models,
  nothing to patch or distill from. N0–N4 are kernel + benchmark work; the application gate
  re-runs when checkpoints land. Do not train a Mamba-3 model to justify the kernel —
  that's the DH-Mamba lesson at larger scale.
- **No complex-arithmetic implementation.** Use the paper's own RoPE equivalence; complex
  NEON (interleaved re/im) doubles the surface for zero expressiveness gain.
- **No speculative Mamba-4-proofing beyond the variant enum.** The enum + separate entry
  points already bought extensibility; deeper abstraction now is speculation.

### 5.4 Risks (adds to §3)

| Risk | Mitigation |
|---|---|
| λ/RoPE cumulative-angle f32 drift at L≈10⁵ | N0's drift golden; chunk-local angle re-basing (same trick as stable segsum) |
| No checkpoints → no end-to-end validation | Kernel-level goldens + λ=1/r=1 equivalence gates carry correctness until checkpoints exist |
| Ecosystem adopts Mamba-3 slowly (vision still SSD) | N-track is additive; M-track serves today's models; both share ~90% of the machinery |
