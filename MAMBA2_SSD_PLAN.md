# MAMBA2_SSD_PLAN — a true Mamba-2 (SSD) Arm/NEON kernel for 2D workloads

**Written Jul 18, 2026. Planning only.** Companion to `SS2D_REPOSITIONING_PLAN.md`
(which *rejected* SSD for the Aug-14 submission) and `RESEARCH_TRIAGE_MAMBA2_2D.md`
(which verified that rejection). This doc is the answer to "what would it take to do
it anyway, properly" — for 2D applications: segmentation, image processing, and the
diffusion backbone.

## 0. The strategic gate (decide this before any code)

**What Mamba-2/SSD actually changes.** Mamba-1 (what every kernel in this repo
implements): diagonal `A ∈ (channels × d_state)`, per-element recurrence
`h = exp(Δ·A)⊙h + ΔB·x` — sequential, scan-shaped, SIMD-friendly. Mamba-2 (SSD):
**scalar-per-head A**, heads with `d_head` channels sharing `(a_t, B_t, C_t)`, and the
state-space *duality*: the token mixing is a 1-semiseparable matrix computed chunk-wise
as **dense matmuls** (intra-chunk attention-like Gram products under a cumulative-decay
mask) plus a tiny inter-chunk state recurrence. Compute shifts from exp+FMA scan to
~90% GEMM.

**Consequences to accept explicitly:**
1. **The moat weakens.** Our torch.compile argument is "compilers can't restructure a
   sequential recurrence." SSD's dual form is matmuls — exactly what compilers/BLAS *are*
   good at. Expect the vs-torch.compile margin to shrink; the honest claim becomes
   "competitive, portable, pip-installable Rust SSD on Arm," not a structural win.
2. **Claims policy check:** llama.cpp already runs Mamba-2 models on CPU (generalized
   `ssm_scan`). The defensible novelty here is **2D/cross-scan SSD on CPU** (VSSD is the
   vision-Mamba-2 precedent and is CUDA-bound) and **PyTorch-callable** SSD on Arm — run
   a fresh prior-art pass before writing any claim.
3. **Timeline:** this is ~2–3 weeks of careful work. **Not before Aug 14** alongside the
   existing plan. Recommendation: build it **additively** on `feature/ssd` — the Mamba-1
   scan stays the shipped product; SSD becomes the post-submission (or stretch) second op.
   Everything below is written for that additive path; nothing existing is converted or
   deleted (the scalar path rule and the diffusion app keep working unchanged).

## 1. What we already have that carries over (the leverage)

| Asset | Reuse in SSD work |
|---|---|
| Golden methodology (f64 vendored ref → npz → independent verifier → Rust gates) | Identical pipeline; only the reference function changes |
| FFI / ctypes loader / torch custom-op / wheels / CI matrix / bench harness | New entry point beside the old one; ABI bump per contract |
| `parallel.rs` dispatcher + thread-local workspace arena (P1-3) | Parallelize over `batch × heads (× directions)`; same arena pattern |
| `vexpq_f32` family | Computes `exp(segsum)` decay masks |
| P0-1 batched 4-direction cross-scan machinery (`ss2d.py`, block layout) | The 2D wrapper is architecture-agnostic — SSD slots in behind the same seam |
| Reverse flag / h0 initial state | Same semantics needed for SSD chunk boundaries and bidirectional 2D |
| EDM/CSI diffusion scaffolding + Route-A distillation plan | Backbone-agnostic; a Mamba-2 student distills from the same U-Net teacher |

## 2. Milestones

**M0 — Ground truth (gate: goldens verified two ways).**
Vendor `ssd_minimal_discrete` from `state-spaces/mamba` (`modules/ssd_minimal.py`) as
the reference; generate goldens over heads ∈ {4,8}, d_head ∈ {32,64}, d_state ∈ {64,128},
chunk ∈ {32,64,128}, L non-multiples included; independent numpy verifier re-derives
them (same two-implementations rule as Phase 0).

**M1 — Rust scalar SSD (gate: goldens at f32 floor).**
`kernel/arm-scan-core/src/ssd/scalar.rs`: segsum (cumulative log-decay), per-chunk:
`G = C·Bᵀ` Gram, causal decay mask multiply, `Y_intra = M·X`; chunk state
`S = (decay-weighted B)ᵀ·X`; inter-chunk scalar recurrence over `S`; output
`Y = Y_intra + C·S_carry`. Generic f32/f64 like `scalar.rs` today.

**M2 — NEON (gate: parity vs scalar + goldens).**
The hot op is now small GEMM, so: start with the `matrixmultiply` crate (already in the
tree as a dev-dep transitively) for correctness+baseline, then hand-roll 8×8/4×12 FMLA
microkernels only where profiling says it pays. Reuse `vexpq` for masks; layout: chunk-
major, head-contiguous so Gram panels stream. The existing chunked Pass-A/B experience
maps directly (Pass A ≈ mask/decay precompute, Pass B ≈ GEMM pipeline).

**M3 — Threading + workspace (gate: rayon bit-identity).**
Rayon over `batch × heads × directions`; P1-3-style thread-local arena for the per-chunk
panels (G, mask, S). Note SSD parallelizes over chunks *within* a row too (intra-chunk
matmuls are independent) — better thin-batch scaling than the 1D scan had.

**M4 — FFI + Python (gate: goldens through the C ABI).**
`arm_scan_ssd_f32` entry (ABI bump), numpy + torch op with fake kernel
(torch.compile composability), `use_arm_scan`-style seam so blocks can select
scan-v1 or ssd per config.

**M5 — 2D wrappers (gate: cross-scan parity on 2D goldens).**
Reuse the batched 4-direction machinery verbatim behind the SSD op ("SS2D-SSD").
Add the **VSSD non-causal variant** (vision Mamba-2: causal mask dropped intra-chunk)
as a flag — that's the segmentation-relevant mode and the strongest 2D-novelty claim.

**M6 — Honest benchmarks (gate: RESULTS rows, all baselines named).**
vs torch.compile'd `ssd_minimal` (the fair fight — expect a smaller margin; publish it),
vs eager, and **vs our own Mamba-1 kernel at matched shapes** (the scientifically
interesting row: scan-form vs dual-form on CPU, per d_state). arm64 CI first, Graviton
for headline.

**M7 — Applications.**
(a) **Segmentation/processing:** VSSD or VMamba-2-style backbone; Phase-A-style CPU
dry-run gate on a real checkpoint (VSSD ships them; CUDA-coupling risk applies), then
`patch()`-style integration + mIoU parity.
(b) **Diffusion:** `SS2DBlock` variant with per-head scalar `A_log` + heads (localized
change; EDM contract untouched); Route-A distillation works identically since the
teacher is a U-Net.

## 3. Risks

| Risk | Mitigation |
|---|---|
| Moat dilution vs torch.compile | Publish the scan-vs-dual comparison as a finding; keep Mamba-1 as headline until numbers say otherwise |
| Prior art (llama.cpp Mamba-2 CPU) | Fresh prior-art pass before claims; novelty lives in 2D/VSSD + PyTorch-callable |
| Timeline vs Aug 14 | Additive branch; zero changes to the submission path |
| GEMM perf on NEON is a deep well | `matrixmultiply` first; hand-tune only where profiled |
| VSSD checkpoint CUDA-coupling | Same hard Phase-A dry-run gate that killed DH-Mamba |

## 4. Sequencing

Post-submission (or if the main plan freezes early): M0–M1 week 1; M2–M4 week 2;
M5–M7 week 3+. Every gate inherits the standing rules: correctness before speed,
never loosen a tolerance, all numbers name their baseline.
