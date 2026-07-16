# MAMBA_DIFFUSION_MRI_PLAN — Arm-accelerated Mamba diffusion for MRI reconstruction

**Status: planning only — no code written or modified.** Written Jul 16, 2026.

Companion to [`APPLICATIONS.md`](APPLICATIONS.md) (topology/app brainstorm),
[`TOPOLOGY_IMPLEMENTATION_PLAN.md`](TOPOLOGY_IMPLEMENTATION_PLAN.md) (how SS2D gets fast in Rust),
and [`PROJECT_CONCEPT.md`](PROJECT_CONCEPT.md) (the decision log this plan proposes to amend).

---

## 0. TL;DR

Today the repo ships an Arm-optimized `selective_scan` (NEON + chunked + rayon) and proposes to
prove it on MRI via **MambaRecon** — a *discriminative* unrolled network that maps undersampled
k-space to an image in one shot. This document plans a different, higher-ceiling application in the
same MRI slot: a **diffusion-based** reconstructor whose denoiser backbone is a **2D-cross-scan
(SS2D) Mamba**, trained and sampled under the **NVIDIA EDM** framework, and applied to MRI using the
**UT CSI Lab (Jon Tamir)** posterior-sampling scaffolding. The whole sampling loop runs on **Arm CPU
(Graviton)**, with our kernel accelerating the SS2D scan that dominates every one of the (many)
denoiser forward passes.

Why this is the stronger story: a discriminative recon calls the network **once**; a diffusion
reconstruction calls it **tens-to-hundreds of times** (one or two forward passes per sampling step).
That turns the selective-scan hot path from "run once" into "run 50–500×," which is exactly the
regime where a fast CPU scan compounds — and it directly attacks the claim that diffusion recon
*must* be GPU-bound. "Diagnostic-quality MRI reconstruction from a diffusion prior, on a CPU box"
is a materially bigger WOW than one-shot recon, and it lands on the same SS2D kernel work the
topology plan already scopes.

The one honest tension, stated up front: **there is no public pretrained Mamba-backbone EDM MRI
checkpoint.** The current project deliberately rules out training. Section 8 treats this as the
central risk and lays out three concrete ways to get a runnable prior without a from-scratch GPU
campaign (distillation from Tamir's existing U-Net EDM MRI weights is the recommended one).

---

## 1. How this extends the current project (and what it changes)

| Dimension | Current plan (MambaRecon) | This plan (Mamba-diffusion) |
|---|---|---|
| Model class | Discriminative unrolled recon | Generative diffusion prior + data consistency |
| Network calls per reconstruction | 1 (or a few unroll steps) | 18–256 denoiser evaluations (sampler steps × 1–2 NFE) |
| Kernel hot path invoked | Once per image | Once **per sampling step** — the multiplier we want |
| "Full" reconstruction | n/a (always conditioned) | Unconditional / fully-sampled (R=1) prior sampling & denoising |
| "Partial" reconstruction | The whole task | Posterior sampling from undersampled k-space (R=2,4,6,8) |
| Checkpoint availability | Published (IXI) | **None public — must train or distill** (Section 8) |
| Kernel topology exercised | SS2D | SS2D (identical) — **no new kernel surface beyond the topology plan** |
| Quality gate | PSNR/SSIM/NMSE vs. reference | PSNR/SSIM/NMSE **+ FID** (generative prior earns a distributional metric) |

The kernel contribution is unchanged: this still lives or dies on the **SS2D fused cross-scan**
described in [`TOPOLOGY_IMPLEMENTATION_PLAN.md`](TOPOLOGY_IMPLEMENTATION_PLAN.md) §3. What changes is
the *application wrapper* around it: EDM sampling loop + MRI forward operator instead of a
single-shot unrolled net. Everything the topology plan says about correctness gates, the
transpose-then-reuse-row-scan strategy, and the Python-first (`ss2d.py`) unfused path applies here
verbatim.

**Decision-log impact:** `PROJECT_CONCEPT.md` currently locks "Training: None." Adopting this plan
means reopening that decision (Section 8). Do not silently contradict the decision log — if this is
adopted, amend `PROJECT_CONCEPT.md` in the same change, per `CLAUDE.md`'s "don't leave two docs
disagreeing" rule.

---

## 2. Background: what EDM gives us, and what is backbone-agnostic

NVIDIA's **EDM** ("Elucidating the Design Space of Diffusion-Based Generative Models," Karras et al.
2022; `NVlabs/edm`) is the framework Tamir's MRI diffusion work is built on. Its value here is that
it cleanly separates three things from the network architecture, so we can drop a Mamba backbone in
without touching the diffusion math:

**Preconditioning.** EDM wraps a raw network `F_θ` as
`D_θ(x;σ) = c_skip(σ)·x + c_out(σ)·F_θ(c_in(σ)·x; c_noise(σ))`, with the four coefficients fixed
functions of the noise level σ and the data scale `σ_data`. This is **backbone-agnostic** — `F_θ`
can be a U-Net or a Mamba, the wrapper is identical.

**Training loss.** EDM samples σ per-example from a log-normal (`P_mean=-1.2, P_std=1.2`), weights
the denoising MSE by `λ(σ)=(σ²+σ_data²)/(σ·σ_data)²`, and trains `D_θ` to denoise. Again independent
of what `F_θ` is.

**Sampler.** A deterministic 2nd-order **Heun** ODE sampler over a σ schedule
(`σ_max≈80, σ_min≈0.002, ρ=7`), 18–40 steps typical, optionally stochastic (`S_churn/S_min/S_max`).
NFE ≈ `2·steps − 1`. **This loop is where our kernel earns its keep**: every step calls `D_θ`, every
`D_θ` call runs the SS2D backbone, every SS2D backbone is dominated by selective scans.

The practical consequence: **we reuse EDM's precond + loss + sampler unmodified and only replace the
U-Net with an SS2D-Mamba `F_θ`.** That is a small, well-scoped surface, and it is exactly how the
Mamba-diffusion vision literature (DiM, ZigMa, DiffuSSM, DiMSUM) already operates — they swap the DiT/U-Net
backbone for a Mamba and keep the diffusion scaffolding.

---

## 3. CSI Lab (Tamir) references — what to take from each

The UT Computational Sensing and Imaging Lab (`github.com/utcsilab`) has the most directly reusable
MRI-diffusion scaffolding available, and it is already EDM-based. Map:

| Repo | What it is | What we reuse |
|---|---|---|
| [`ambient-diffusion-mri`](https://github.com/utcsilab/ambient-diffusion-mri) (ICLR 2025) | **Primary reference.** EDM-based multi-coil MRI diffusion; trains on Fourier-subsampled k-space at R=2/4/6/8; ships 9 pretrained U-Net checkpoints (FastMRI brain, 384×384, 2-channel complex). | The entire MRI wrapper: multi-coil forward operator, sensitivity-map handling, k-space mask/acceleration machinery, `train.py`/`prior.py`/`solve_inverse_adps.py` structure, FID eval, FastMRI preprocessing → `ksp_brainMRI_384.zip`. **This is the scaffolding; we change only the backbone.** |
| [`utcsilab/edm`](https://github.com/utcsilab/edm) | Their fork of NVIDIA EDM. | Confirms the exact EDM version/precond variants their MRI code expects; our Mamba `F_θ` must satisfy the same `Precond` interface. |
| [`csgm-mri-langevin`](https://github.com/utcsilab/csgm-mri-langevin) (Jalal et al., NeurIPS 2021) | Annealed-Langevin posterior sampling for CS-MRI ("Robust CS-MRI with Deep Generative Priors"). | The canonical **measurement-consistency** formulation for MRI (A^H A data term + score prior) — the math behind "partial" reconstruction; a fallback sampler if EDM/A-DPS integration is fiddly. |
| [`gsure-diffusion-mri`](https://github.com/utcsilab/gsure-diffusion-mri) (MRM 2025) | Self-supervised (GSURE) denoising for multi-coil recon — train a prior *without* fully-sampled ground truth. | De-risks the data problem: a route to a usable prior if fully-sampled FastMRI ground truth is a bottleneck. |
| [`score-diffusion-training`](https://github.com/utcsilab/score-diffusion-training) / [`-sampling`](https://github.com/utcsilab/score-diffusion-sampling) | Generic score-model train/sample pipelines for inverse problems. | Reference patterns for the train/sample split and config structure; not on the critical path if we go pure-EDM. |
| [`deep-jsense`](https://github.com/utcsilab/deep-jsense), [`Nufft_Torch`](https://github.com/utcsilab/Nufft_Torch) | Sensitivity-map estimation (J-Sense) and PyTorch NUFFT. | MRI physics utilities for the forward operator if we go beyond Cartesian masks (non-Cartesian trajectories). Likely out of scope for MVP; note for later. |

**The key realization:** `ambient-diffusion-mri` is a fork of `NVlabs/edm` with (a) an MRI data
loader, (b) a multi-coil forward operator, (c) an "ambient"/supervised precond, and (d) posterior
samplers. **Only its `--arch` backbone (`ddpmpp` U-Net) is what we replace.** Everything else — the
part that makes it *MRI*, not generic image gen — we inherit.

---

## 4. Architecture: the SS2D-Mamba denoiser under EDM

The network `F_θ` we build is a Mamba/SS2D image backbone, following the established recipe from the
Mamba-diffusion vision papers, adapted to satisfy EDM's precond interface and to route its scans
through our kernel:

- **Patchify** the (noised) 2-channel complex image into a token grid `(B, D, H, W)` — the same
  grid shape our `ss2d_scan(...)` contract expects.
- **Stacked SS2D blocks** (VMamba-style): each block runs the four-direction cross-scan (rows
  fwd/back, cols fwd/back) that [`TOPOLOGY_IMPLEMENTATION_PLAN.md`](TOPOLOGY_IMPLEMENTATION_PLAN.md)
  §3 already plans, plus the usual gating/conv/MLP. **These scans are the hot path.**
- **σ conditioning** via EDM's `c_noise(σ)` embedding, injected per block (adaLN-style or additive),
  exactly as the U-Net consumes it — no change to EDM's side.
- **Unpatchify** back to image space; EDM's precond wraps the whole thing into `D_θ`.

Design choices to pin during planning (each has a clean Mamba-diffusion precedent):

1. **Scan pattern.** Plain 4-direction SS2D (VMamba/DiM) vs. zigzag continuity (ZigMa) vs.
   hybrid spatial-frequency (DiMSUM). Recommend **plain SS2D first** — it maps 1:1 onto the kernel
   work already scoped and is the cleanest correctness target. Fancier scans are a later variable.
2. **U-Net vs. isotropic.** DiM/DiffuSSM use a U-Net-shaped stack of Mamba blocks (multiscale);
   isotropic (DiT-like) is simpler. Recommend **U-Net-shaped** to match `ambient-diffusion-mri`'s
   multiscale inductive bias for MRI and to reuse its `--cres` config surface.
3. **Complex data.** MRI is complex; Tamir's models carry it as 2 real channels (`img_channels=2`).
   Keep that convention — it means zero change to the real-valued kernel.

Crucially, **none of this adds kernel surface beyond SS2D.** The Mamba block's scan is the same
`selective_scan` / `ss2d_scan` the repo already has (Python-rearranged today, fused-in-Rust per the
topology plan). The diffusion backbone is "SS2D called many times inside a sampling loop."

---

## 5. "Partial" vs "full" reconstruction — precise definitions

The user asked for **both partial and full**. In diffusion-MRI terms these are two distinct sampling
modes over the *same* trained prior:

**Full reconstruction (fully-sampled / R=1).** The unconditional or lightly-conditioned regime:
sample from the prior, or denoise a fully-sampled acquisition. Uses EDM's plain Heun sampler with no
(or trivial) data-consistency term. This is the "does the generative prior produce diagnostic-quality
anatomy at all" check, and it is the cleanest first milestone because it exercises the full
sampler×backbone×kernel loop **without** the MRI forward-operator complications. Maps to
`ambient-diffusion-mri`'s `prior.py` path.

**Partial reconstruction (undersampled / R>1) — the clinical task.** Given undersampled multi-coil
k-space `y = M·F·S·x` (M = sampling mask, F = Fourier, S = coil sensitivities), recover `x` by
**posterior sampling**: interleave EDM denoising steps with a **data-consistency / measurement-guidance**
step (DPS-style likelihood gradient, or A-DPS as in `ambient-diffusion-mri`, or annealed-Langevin
measurement term as in `csgm-mri-langevin`). Sweep acceleration R=2/4/6/8 to match Tamir's evaluation
grid. Maps to `solve_inverse_adps.py`.

Both modes call `D_θ` the same number of times per step; **partial adds a cheap forward-operator
evaluation (FFT + mask + coil combine) per step**, which is already fast on Arm (Arm Compute
Library / pocketfft) and is explicitly out of scope as an optimization target (`PROJECT_CONCEPT.md`
rules the FFT step a black box). So from the kernel's perspective, partial and full are the same hot
loop; only the wrapper differs. That is a tidy story: **one prior, one kernel, two reconstruction
modes.**

---

## 6. Where the Arm kernel plugs in — and why the CPU story is strong here

The sampling loop is, in pseudocode terms, `for step in schedule: x = heun_update(D_θ, x, σ)`, and
`D_θ` = EDM-precond around the SS2D-Mamba. Every `D_θ` call streams the token grid through a stack of
four-direction selective scans. So per reconstruction the kernel is invoked
`(#blocks) × (4 directions) × (2 Heun sub-steps) × (#steps)` times — hundreds to thousands of scan
calls for a single image. **This is the multiplier that makes the Arm optimization matter.**

Why this beats the discriminative framing on every rubric axis:

- **Technical (40):** the scan is now inside a tight, repeatedly-invoked loop, so kernel-level wins
  (NEON, chunking, rayon over batch×channel×direction) compound across NFEs instead of amortizing
  over one call. The fused SS2D traversal — the one op with no CPU implementation anywhere — is
  exercised at maximum leverage.
- **WOW (25):** "diffusion-prior MRI reconstruction on a CPU" is a stronger visual and economic claim
  than one-shot recon. Diffusion recon is *assumed* to require a GPU; showing diagnostic-quality R=4/8
  reconstruction running on Graviton is the memorable demo.
- **Impact (20):** diffusion posterior sampling is the modern SOTA family for accelerated MRI (Tamir,
  Jalal, Chung/Ye). Making it CPU-affordable is a real deployment argument (hospital CPU boxes, batch
  offline recon), not a toy.
- **DX (15):** unchanged — still a pip-installable kernel; the diffusion wrapper is a separate demo
  app that imports it.

**Honesty guardrails (per `CLAUDE.md`).** Absolute CPU diffusion-sampling latency is high — hundreds
of NFEs is not free on any CPU. The claim must be framed correctly and benchmarked honestly:
(a) lead with **per-NFE / per-scan speedup vs. `torch.compile`**, the fair baseline, so the kernel's
contribution is isolated from the sampler's cost; (b) report **end-to-end wall-clock and $/reconstruction
on Graviton vs. a GPU baseline**, stating the NFE count and sampler; (c) lean on **low-NFE samplers**
(EDM Heun at 18–35 steps; note deterministic ODE needs far fewer NFEs than ancestral) to keep the
end-to-end number defensible; (d) never quote a speedup without naming the baseline and the sigma
schedule. If the end-to-end CPU number is unflattering in absolute terms, publish it with the
per-scan speedup and the cost table — the moat is that `torch.compile` cannot restructure a
sequential recurrence, and that argument only lands with trustworthy numbers.

---

## 7. Integration with the existing kernel & topology plan

No new kernel primitives beyond what `TOPOLOGY_IMPLEMENTATION_PLAN.md` §3 already scopes. The
dependency chain:

1. **`ss2d.py` (Python-rearranged, unfused)** — §3.1 of the topology plan. Build the four permuted
   views, stack into the batch dim, one `selective_scan` call, split & merge. **This is enough to
   stand up the entire Mamba-diffusion MRI demo end-to-end and get honest numbers** before any new
   Rust lands. Do this first.
2. **Measure** the unfused SS2D overhead at the *diffusion* workload's real grid size and NFE count
   (topology plan §3.4 open question #2). Because the loop calls SS2D hundreds of times, the
   flip/permute copy traffic is paid hundreds of times too — so this workload is the strongest
   justification yet for the fused traversal. Measure before committing the week.
3. **Fused `selective_scan_2d` in Rust** — §3.2. Transpose-then-reuse-row-scan; new C ABI entry
   point; `ss2d.py` internals swap to it. Same four correctness gates (golden-vs-f64, NEON-vs-scalar,
   rayon bit-identity, real-C-ABI replay).
4. **Threading leverage is higher here.** Parallelism is over `batch × channel × direction`, and a
   diffusion sampler often runs a **batch of samples/seeds** together (Tamir's code batches seeds),
   giving even more independent rows for rayon — a good core-scaling curve on Graviton.

The kernel stays fp32; the diffusion backbone runs fp32 on CPU. (Tamir trains fp16 on GPU; inference
precision is a separate knob — fp32 CPU inference is the safe correctness baseline, and a
BF16-storage/fp32-accumulate experiment on Graviton4 is already a listed stretch.)

---

## 8. The hard part: data & checkpoints (the training tension)

**No public Mamba-backbone EDM MRI checkpoint exists.** The current project rules out training for
good reason (GPU + time). This is the single decision that determines whether this plan is a Week-4
demo or a research project. Three routes, cheapest first:

**Route A — Distill Tamir's U-Net EDM prior into an SS2D-Mamba (recommended).**
`ambient-diffusion-mri` ships 9 pretrained U-Net EDM checkpoints on FastMRI brain. Train the Mamba
`F_θ` to match the U-Net denoiser's outputs (denoising-target or output-distillation) across the σ
schedule. This needs GPU hours but **far fewer than from-scratch**, reuses their exact data pipeline
and precond, and gives an apples-to-apples "same prior, Mamba backbone" comparison — a clean
scientific framing. Risk: distillation quality; budget a modest GPU spend (spot hours).

**Route B — Train small, from scratch, on a reduced problem.**
Lower resolution (e.g. 128–192px), single-coil or RSS-combined, IXI or a FastMRI subset. Produces a
*proof-of-concept* prior good enough for the demo and the parity story, not SOTA FID. Aligns with the
roadmap's "phantom/shareable demo" philosophy. Risk: quality may be visibly below the U-Net baseline;
frame honestly as "backbone feasibility on Arm," not "new SOTA."

**Route C — Self-supervised (GSURE) prior, no fully-sampled ground truth.**
Use `gsure-diffusion-mri`'s training objective with the Mamba backbone. Sidesteps the fully-sampled
data bottleneck. Highest novelty, highest risk; likely a stretch, not MVP.

**Shareable, no-credentials fallback (independent of A/B/C).** Keep a **synthetic phantom** track
(Shepp–Logan / brain-web style) so the end-to-end pipeline — sampler + Mamba backbone + kernel — is
runnable by a judge with no FastMRI credentials and no checkpoint, mirroring the repo's existing
`make validate` philosophy. This proves the *kernel-in-the-loop* claim even if the clinical
checkpoint isn't shareable.

**Recommendation:** Route A for the headline (real FastMRI, real quality gate, clean comparison) +
the synthetic phantom track for reproducibility. Decide the GPU budget explicitly before committing —
this is the amendment `PROJECT_CONCEPT.md` needs.

---

## 9. Phased plan (planning only — no code yet)

Sequenced to get an end-to-end, honestly-benchmarked pipeline standing before any new Rust, per the
repo's "correct-and-unfused first, fuse only if measurement justifies it" discipline.

**Phase A — Reference study & feasibility gate.** Clone and CPU-dry-run `ambient-diffusion-mri`:
confirm its EDM precond variant, the exact `F_θ` interface a backbone must satisfy, the data format
(`ksp_brainMRI_384.zip`), and the posterior sampler. Confirm one U-Net checkpoint loads and samples
on CPU (however slowly). Pin the SS2D backbone recipe (DiM/VMamba-style, §4). **Gate: interface +
data + checkpoint understood; backbone recipe locked.**

**Phase B — Backbone bring-up (GPU-side, small).** Stand up the SS2D-Mamba `F_θ` under EDM precond;
verify it trains/denoises on a toy target; pick Route A/B/C for the prior. **Gate: a `D_θ` that
denoises, satisfying EDM's interface.**

**Phase C — Kernel-in-the-loop on CPU (unfused SS2D).** Route the backbone's scans through
`arm_scan` via the Python-rearranged `ss2d.py`. Run **full** (R=1) sampling end-to-end on Arm CPU;
confirm output parity between the kernel path and the reference scan (the topology plan's gates).
**Gate: full reconstruction runs on Arm CPU through our kernel, output-parity verified.**

**Phase D — Partial reconstruction.** Add the MRI data-consistency/posterior step (A-DPS or Langevin)
around the CPU sampler; sweep R=2/4/6/8; compute PSNR/SSIM/NMSE (+FID for the prior). **Gate: R=4
reconstruction at a defensible quality metric on Arm CPU.**

**Phase E — Measure & (maybe) fuse.** Benchmark the unfused SS2D at the diffusion workload's real
grid/NFE against `torch.compile`; build the $/reconstruction-vs-GPU table; core-scaling curve over
batched seeds. If copy overhead dominates (it likely does at hundreds of NFEs), implement fused
`selective_scan_2d` (topology plan §3.2) and re-measure. **Gate: honest RESULTS rows + decision on
fusion.**

**Phase F — Demo, video, writeup.** Side-by-side (undersampled zero-filled vs. diffusion-reconstructed)
on Graviton; the phantom track for reproducibility; Devpost writeup; reconcile `PROJECT_CONCEPT.md`.

This maps onto the existing Week-4/5 window (2D variant + benchmarks + stretch) — but note Phases A/B
add real work the MambaRecon plan didn't have (a backbone + a prior). If the calendar can't absorb
the training, the fallback is the current discriminative MambaRecon plan; **this plan is a bet on a
higher ceiling, and the decision point is the GPU/training budget in Section 8.**

---

## 10. Risk register (specific to this plan)

| Risk | Severity | Mitigation |
|---|---|---|
| No pretrained Mamba-diffusion MRI checkpoint | **High** | Route A distillation from Tamir's U-Net weights; small-scale Route B; phantom track for reproducibility (Section 8). |
| Training/distillation needs GPU the project scoped out | **High** | Explicit, bounded spot-GPU budget; decide before committing; MambaRecon remains the fallback app. |
| Absolute CPU diffusion latency looks bad | Medium | Lead with per-scan/per-NFE speedup vs. `torch.compile`; low-NFE Heun; $/recon vs. GPU cost table; frame honestly (§6 guardrails). |
| SS2D fused kernel eats a week | Medium | Unfused `ss2d.py` path stands up the whole demo first; fuse only if §Phase-E measurement justifies it. |
| `ambient-diffusion-mri` bundles CUDA-only ops / won't run on CPU | Medium | Same gate that killed DH-Mamba — CPU-dry-run in Phase A before building on it; force reference scan path. |
| Posterior-sampling data-consistency is finicky | Medium | `csgm-mri-langevin` annealed-Langevin as a proven fallback formulation; start with **full** (R=1) sampling which needs no data term. |
| Complex-data / coil-combine correctness bugs | Low–Med | Inherit Tamir's exact 2-channel + sensitivity-map handling; don't reinvent the forward operator. |
| Scope creep vs. Aug 14 deadline | **High** | Phase A is a hard go/no-go; if it slips, revert to MambaRecon. This is explicitly a higher-ceiling *alternative*, not a mandate. |

---

## 11. Success criteria

1. An SS2D-Mamba denoiser `F_θ` satisfying EDM's precond interface, producing a working `D_θ`.
2. **Full** (R=1) diffusion sampling running end-to-end on Arm CPU with our kernel in the loop,
   output-parity-verified against the reference scan.
3. **Partial** (R=2/4/6/8) posterior reconstruction on Arm CPU at a defensible PSNR/SSIM/NMSE (+FID),
   reproducing Tamir's evaluation grid on the same data.
4. Honest benchmarks: per-scan/per-NFE speedup vs. `torch.compile`, end-to-end wall-clock, and
   $/reconstruction on Graviton vs. a GPU baseline — every number naming its baseline, NFE count,
   and sigma schedule.
5. A synthetic-phantom track a judge can run with no FastMRI credentials and no checkpoint, proving
   the kernel-in-the-loop claim reproducibly.
6. `PROJECT_CONCEPT.md` amended (training decision reopened) so the decision log and this plan agree.

---

## 12. Open questions to resolve before committing

1. **Does `ambient-diffusion-mri` run its sampler on CPU at all** (even slowly), and is its precond
   the stock EDM one or a custom "ambient" variant our `F_θ` must match exactly?
2. **Distillation feasibility (Route A):** how many GPU-hours to distill a usable Mamba prior from the
   U-Net checkpoint, and does the distilled prior hold quality at R=4/8 posterior sampling?
3. **NFE floor:** what is the smallest Heun step count that keeps R=4 reconstruction diagnostic —
   this sets the end-to-end CPU number more than any kernel optimization does.
4. **Does the unfused `ss2d.py` copy overhead dominate** at the diffusion grid size × hundreds of
   NFEs (it probably does — which is the argument *for* the fused kernel)?
5. **Is this the MRI slot, or a fourth item?** Per `APPLICATIONS.md`'s topology-ladder trio, MRI is
   already the SS2D slot; this plan is a *replacement* for the MambaRecon framing of that slot, not
   an addition. Confirm before reworking the decision log.

---

## Reference links

- **CSI Lab (Tamir):** [ambient-diffusion-mri (ICLR 2025)](https://github.com/utcsilab/ambient-diffusion-mri) ·
  [csgm-mri-langevin (NeurIPS 2021)](https://github.com/utcsilab/csgm-mri-langevin) ·
  [gsure-diffusion-mri (MRM 2025)](https://github.com/utcsilab/gsure-diffusion-mri) ·
  [utcsilab/edm](https://github.com/utcsilab/edm) ·
  [score-diffusion-training](https://github.com/utcsilab/score-diffusion-training)
- **EDM:** [NVlabs/edm](https://github.com/NVlabs/edm) · Karras et al., "Elucidating the Design Space
  of Diffusion-Based Generative Models," NeurIPS 2022 ([arXiv:2206.00364](https://arxiv.org/abs/2206.00364))
- **Mamba-diffusion backbones:** [DiM](https://arxiv.org/abs/2405.14224) ·
  [ZigMa](https://arxiv.org/abs/2403.13802) · [DiffuSSM](https://arxiv.org/abs/2311.18257) ·
  DiMSUM (NeurIPS 2024) · [VMamba / SS2D](https://github.com/MzeroMiko/VMamba)
- **In-repo:** [`TOPOLOGY_IMPLEMENTATION_PLAN.md`](TOPOLOGY_IMPLEMENTATION_PLAN.md) §3 (SS2D) ·
  [`APPLICATIONS.md`](APPLICATIONS.md) (MRI slot) · [`PROJECT_CONCEPT.md`](PROJECT_CONCEPT.md) (decision log to amend)
