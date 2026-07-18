# MAMBA3_KERNEL_PLAN — the full Mamba-3 Arm/NEON kernel in Rust (1D + 2D)

**Written Jul 18, 2026** (supersedes `MAMBA2_SSD_PLAN.md`, which now redirects here; history
preserved in git). All external facts verified Jul 18; sources inline. Companion to
`SS2D_REPOSITIONING_PLAN.md` (the Aug-14 submission plan — **unchanged by this doc**) and
`RESEARCH_TRIAGE_MAMBA2_2D.md`.

**The goal:** the first CPU/Arm kernel for **Mamba-3**
([ICLR 2026](https://arxiv.org/abs/2603.15569); CMU/Princeton/Cartesia/Together) — SISO
prefill, MIMO, and decode — pip-installable and PyTorch-callable, extended to 2D
(cross-scan and non-causal) for images and diffusion. Mamba-3's official kernels are
Triton (SISO prefill, RoPE-fused), TileLang (MIMO prefill), and CuTe DSL (decode) —
**all GPU-only**; community requests for a CPU path are open and unanswered
([mamba#809](https://github.com/state-spaces/mamba/issues/809)). The CPU slot for the
newest SSM block is empty. This plan fills it.

---

## 0. Strategy: why the road to Mamba-3 runs through the SSD substrate

This is the part that looks like a detour and isn't. Mamba-3's block is **the Mamba-2/SSD
chunked machinery plus three deltas**:

1. **Trapezoidal discretization** — a three-term recurrence, equivalent to a data-dependent
   width-2 convolution on the state input;
2. **Complex states** — implemented, per the paper's own equivalence, as data-dependent
   RoPE rotations on B/C (no complex arithmetic needed);
3. **MIMO** — rank-r state updates in place of rank-1, raising decode arithmetic intensity.

Everything else — chunking, segsum decay, Gram products, inter-chunk carry, the head
layout — **is the SSD substrate**. The paper's own kernel suite is organized the same way.
Two consequences we exploit:

- **Stage 1 (substrate) is not throwaway Mamba-2 work; it is 90% of Mamba-3.** And it
  serves today's real models (VSSD, EfficientViM — the vision-Mamba-2 ecosystem) while the
  Mamba-3 ecosystem grows checkpoints.
- **The reductions are free correctness gates:** λ=1 collapses Mamba-3's trapezoid to
  exactly the Euler/SSD form, and r=1 collapses MIMO to SISO — so Stage 2 is verified
  against Stage 1 bit-for-bit at those settings, the same oracle discipline the 1D kernel
  used (scalar↔NEON↔threaded).

**Scope guard — what does NOT migrate.** The Aug-14 submission (Mamba-1 selective scan +
SS2D + diffusion MRI app) is untouched: it serves the ecosystem that exists, has
checkpoints, and is the contest entry. There are **no public Mamba-3 checkpoints**
([mamba#860](https://github.com/state-spaces/mamba/issues/860)) and no vision-Mamba-3
models to patch; migrating the shipped kernel or the app to Mamba-3 today would trade a
working demo for an empty ecosystem. This plan runs on `feature/mamba3`, additively;
kickoff gate: submission frozen (or verified slack) + a fresh 30-minute prior-art pass
("Mamba-3 CPU", "SSD SIMD", llama.cpp PRs) recorded in `PROJECT_CONCEPT.md`.

**Claims (to-our-knowledge, re-verified at kickoff):** first CPU/Arm Mamba-3 kernel; first
PyTorch-callable SSD-family op on Arm; first 2D/non-causal SSD on CPU. Never "first Mamba
on CPU" (llama.cpp runs Mamba-2 GGUF models; BitMamba-2 exists — see README prior-art
table).

**Honesty about the moat:** the dual form is GEMM-shaped, which compilers and BLAS are good
at — expect a thinner vs-torch.compile margin than the Mamba-1 scan enjoys, and say so.
The compensating asset is the **crossover study** (Stage 3): scan-form vs dual-form vs
trapezoidal on CPU across `(L, d_state, cores)` — a result nobody has published.

---

## 1. Leverage — what the repo already provides

| Asset | Reuse |
|---|---|
| Golden methodology (f64 vendored ref → npz + recorded f32 floors → independent numpy verifier → Rust gates) | Identical pipeline for every stage; only reference functions change |
| FFI / ctypes / torch custom-op / wheels / CI matrix / bench harness | New entry points beside the old; ABI bumps per the versioning contract |
| `parallel.rs` + thread-local arena (P1-3) | Parallelism over `batch × heads (× directions)`; same arena pattern |
| `vexpq_f32_nonpos`/`_fast` | Decay masks: segsum arguments are always ≤ 0 (cumsums of `log a`, `a ∈ (0,1)`) — same precondition proof as Pass A2 |
| Reverse flag / h0-resume / `last_state` | SSD chunk carries, Mamba-3's `v`-carry and angle-carry, bidirectional 2D |
| Batched 4-direction SS2D machinery (`ss2d.py`) | 2D wrapper is block-agnostic; Mamba-3 slots behind the same `scan_fn` seam |
| `matrixmultiply` 0.3.11 (verified in `kernel/Cargo.lock`) | f32 GEMM baseline before any hand-rolled microkernel |
| EDM/CSI diffusion scaffolding | Backbone-agnostic; a Mamba-3 student could distill from the same U-Net teacher once the block is proven |

---

## 2. Stage 1 — the SSD substrate (milestones S0–S4)

Standing rules everywhere: correctness before speed; never loosen a tolerance; parallel
bit-identical to sequential; every number names its baseline, host, torch version, threads.

### S0 — Ground truth (gate: goldens verified two ways)

1. Vendor [`mamba_ssm/modules/ssd_minimal.py`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/ssd_minimal.py)
   (verified path; Listing 1 of the Mamba-2 paper) → `tests/reference/ssd_minimal_ref.py`,
   pinned to a tagged release, CPU-only f64.
2. `tests/gen_golden_ssd.py`: heads ∈ {4,8}, d_head ∈ {32,64}, d_state ∈ {64,128},
   chunk ∈ {32,64,128}, L non-multiples, a fast-forget case (`a → 0⁺`). Manifest records
   each case's `f32_max_abs_err` floor, same format as the 1D goldens.
3. **Numerics rule:** segsum via within-chunk pairwise differences of `log a` — never
   global-cumsum-then-subtract (catastrophic f32 cancellation at chunk 128).
4. `tests/verify_golden_ssd.py`: independent numpy re-derivation (materialize the full
   1-semiseparable matrix at small L and multiply). Two-implementations rule.

**Exit:** every golden reproduced by both implementations in f64; floors recorded.

### S1 — Rust scalar substrate (gate: goldens at f32 floor)

1. `kernel/arm-scan-core/src/ssd/{mod,scalar}.rs`; entry
   `ssd_scan(dims: &SsdDims, input, out, state_carry, opts)`,
   `SsdDims { batch, heads, d_head, d_state, len, chunk }`. **The block-variant enum is
   born here** (`Ssd = 0`, `Mamba3Siso = 1`, `Mamba3Mimo = 2` reserved) so Stage 2 is an
   addition, not a refactor.
2. Per (batch, head): segsum → lower-triangular decay mask `L_ij = exp(seg_i − seg_j)`;
   Gram `G = C·Bᵀ`; `Y_intra = (G ⊙ L)·X`; chunk state `S = (decay-weighted B)ᵀ·X`;
   sequential inter-chunk carry `S_carry ← a_chunk·S_carry + S`;
   `Y = Y_intra + (C·S_carry)·decay_in`. Generic f32/f64 over the `Float` trait; clarity
   over speed — this is the in-crate oracle.
3. `tests/golden_ssd.rs`: `< 1e-4` **and** near the recorded floor, per case.
4. Property tests: chunk=L ≡ chunk=32 (within f32 tolerance); h0-resume splice ≡ whole-scan.

### S2 — NEON (gate: NEON↔scalar parity + goldens)

1. Route `G` / `Y_intra` / `S` through `matrixmultiply::sgemm` first; keep segsum/mask in
   NEON with `vexpq_f32_nonpos`. Measure — the baseline that decides where hand-rolling pays.
2. Hand-roll only what profiling flags: `vfmaq_laneq_f32` panel microkernels (~8×12 f32),
   chunk-major head-contiguous layout so panels stream.
3. **Static chunk tune:** sweep `Q ∈ {32,64,128}` per shape, fix the best — per COREY
   ([arXiv:2604.10597](https://arxiv.org/abs/2604.10597)), static chunks match/beat
   entropy-guided runtime scheduling end-to-end; dynamic chunking would break bit-identity.
4. Graviton3/4 stretch: BF16 `bfmmla` 2×2 tiles for Gram products (bf16 storage / f32
   accumulate, disclosed).

### S3 — Threading + workspace (gate: rayon bit-identity at 1/2/8 threads)

1. `for_each_head` driver in `parallel.rs`; thread-local arena for per-chunk panels.
2. Chunk-parallel *only* for chunk-local phases; **inter-chunk carry stays sequential per
   row** — no tree reduction (FMA reassociation breaks bit-identity).

### S4 — FFI + Python (gate: goldens through the real C ABI)

1. `arm_scan_ssd_f32` with `#[repr(C)] SsdDims` + the variant enum; overflow-checked sizes;
   `catch_unwind`; ABI version bump.
2. `_ffi.ssd_raw`; `torch.ops.arm_scan.ssd` with fake kernel (torch.compile-composable);
   numpy twin; `tests/check_ffi.py` extended.
3. Block-selection seam: `use_arm_scan(module, block="scan_v1"|"ssd"|"mamba3")`.

---

## 3. Stage 2 — the Mamba-3 core (milestones T0–T3)

The three deltas, from §5.1 of the verified paper reading:

**(a) Trapezoidal three-term recurrence.**

```
h_t = exp(Δ_t A_t)·h_{t-1} + (1−λ_t)·Δ_t·exp(Δ_t A_t)·B_{t-1} x_{t-1} + λ_t·Δ_t·B_t x_t
```

≡ data-dependent width-2 convolution on `B_t x_t` inside the recurrence. Kernel cost:
compute `v_t = Δ_t B_t x_t` as today, blend `w_t = λ_t·v_t + (1−λ_t)·ā_t·v_{t-1}` — one
extra vector FMA per timestep + a **one-element `v` carry across chunk boundaries** (joins
`last_state` in the resumable contract). The recurrence core is unchanged: `h = ā⊙h + w`.
Under `reverse`, the width-2 conv points the other way — same `chunks_in_scan_order`
treatment as the 1D reverse flag.

**(b) Complex states ≡ data-dependent RoPE on B/C** (the paper's own equivalence — no
complex arithmetic). NEON: pointwise even/odd lane-pair rotations (`vfma`/`vfms` with
`vrev64q_f32`-swapped operands), placed next to the exp in the precompute pass. Angles are
cumulative → carried across chunks exactly like decay; reuse the segsum machinery.
Rotations are norm-preserving, so f32 floors should track the SSD cases; a
rapidly-varying-angle golden catches cumulative drift, mitigated by chunk-local angle
re-basing (same trick as stable segsum).

**(c) MIMO.** Rank-r state update = small `(r × d_state)` panel FMA (S2 microkernels at
decode shapes). This raises arithmetic intensity precisely in decode — the memory-bound
regime where the CPU story is weakest (`IMPROVEMENT_IDEAS.md` §2.4) — so it's the delta
most likely to *help* CPU throughput.

### T0 — Reference + goldens
Vendor the official repo's minimal/eager Mamba-3 path pinned to a tag (if none exists,
transcribe the paper's equations; verify once against their Triton output on a borrowed
GPU). Golden additions: λ extremes (**λ=1 must reproduce S0's SSD goldens — free
cross-check**), rapidly-varying RoPE angles, MIMO r ∈ {1,2,4} (**r=1 must reproduce SISO
bit-for-bit in f64 — free gate**).

### T1 — Scalar Mamba-3
The three deltas land in `ssd/scalar.rs` behind `Mamba3Siso`/`Mamba3Mimo` enum arms.
Gates: goldens at f32 floor; λ=1 bit-identical to S1.

### T2 — NEON Mamba-3
Precompute-pass additions ((a) one FMA + carry, (b) rotation pass); recurrence core
unchanged; MIMO panel FMA. Gate: NEON↔scalar parity per variant.

### T3 — Threading/FFI/torch op
Sequential inter-chunk carry now carries `(h, v, angle)`; one ABI bump for λ/rank params;
fake kernel unchanged in shape; bit-identity re-run per variant.

---

## 4. Stage 3 — 2D, benchmarks, applications (milestones U0–U2)

### U0 — 2D wrappers (gate: per-direction parity on 2D goldens, pre-merge)
1. Reuse the batched 4-direction machinery verbatim behind the new op — cross-scan
   Mamba-3 ("SS2D-M3").
2. **Non-causal flag** (VSSD-style, [ICCV 2025](https://arxiv.org/abs/2407.18559)): mask
   dropped intra-chunk → two dense GEMMs. Best NEON fit, thinnest vs-torch.compile margin;
   U1 measures the trade honestly.
3. Grid blocking per 2DMamba's tiling discipline
   ([CVPR 2025](https://arxiv.org/abs/2412.00678)): tile sizes derived from L1d/L2, not
   hardcoded.
4. 2D goldens include non-square and non-multiple-of-4 grids.

### U1 — Honest benchmarks (gate: RESULTS rows, all baselines named)
Baselines: torch.compile'd reference (fair fight — publish the thinner margin), eager, and
**our Mamba-1 kernel at matched shapes**. **Headline science: the three-way crossover** —
scan-form (Mamba-1) vs dual-form (SSD) vs trapezoidal (Mamba-3) across
`(L, d_state, cores)`, plus decode rows at r ∈ {1,4} quantifying the MIMO
arithmetic-intensity win on Graviton. No such CPU comparison exists in the literature.
arm64 CI provisional → Graviton `c8g` headline, per `BASELINE_TEST_PLAN.md` surfaces.

### U2 — Applications (gated on ecosystem reality, re-checked at the time)
- **No Mamba-3 application exists yet to port** — no public checkpoints (#860), no
  vision-Mamba-3. Do **not** train a model to justify the kernel (the DH-Mamba lesson).
  Re-run this gate when checkpoints land.
- **Nearest-term real apps are Mamba-2-family** (served by Stage 1): VSSD backbone —
  ImageNet classification weights are public, segmentation weights **by author request
  only** (verified Jul 18; [repo](https://github.com/YuHengsss/VSSD)) → gate on weight
  availability, fall back to classification-parity. EfficientViM/HSM-SSD
  ([CVPR 2025](https://arxiv.org/abs/2411.15241)) as the backbone-v2 partner via the
  block-selection seam.
- **Diffusion:** a per-head scalar-A `SS2DBlock` variant under the existing EDM contract
  distills from the same U-Net teacher — a Mamba-3 backbone experiment becomes possible
  the day T3 lands, but is research, not roadmap.

---

## 5. Risks

| Risk | Mitigation |
|---|---|
| Timeline vs Aug 14 | Everything on `feature/mamba3`, post-submission; §0 scope guard |
| Moat thinner vs torch.compile (GEMM-shaped dual form) | Publish the three-way crossover as the headline finding; Mamba-1 scan remains the shipped structural win |
| Mamba-3 spec/kernels still settling | Pin vendored references to tags; goldens are the contract, not repo HEAD |
| No Mamba-3 checkpoints → no end-to-end validation | λ=1 / r=1 reduction gates + goldens carry correctness until checkpoints exist |
| λ/RoPE cumulative-angle f32 drift at L≈10⁵ | Drift golden + chunk-local angle re-basing |
| GEMM perf on NEON is a deep well | `matrixmultiply` baseline first; hand-tune only where profiled |
| Vision ecosystem still SSD-based | Stage 1 serves it directly; Stage 2 is additive on top |
| Prior art appears before kickoff | Fresh prior-art pass at the §0 gate, recorded in the decision log |

## 6. Sequencing

Post-submission: S0–S1 week 1 · S2–S4 week 2 · T0–T3 week 3 · U0–U2 week 4+. Each stage's
gates inherit the standing rules. If the schedule only allows one stage, Stage 1 alone is
already shippable ("PyTorch-callable SSD on Arm") — the plan degrades gracefully.
