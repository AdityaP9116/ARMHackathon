# MRI_DIFFUSION_IMPLEMENTATION_PLAN — engineering plan, component by component

**Status: planning only — no code written or modified.** Written Jul 16, 2026.

The **strategy** and the case for this direction live in
[`MAMBA_DIFFUSION_MRI_PLAN.md`](MAMBA_DIFFUSION_MRI_PLAN.md). This document is the **engineering
plan**: the concrete interfaces, file-level changes, data flow, and milestone sequence needed to
build an Arm-CPU Mamba-diffusion MRI reconstructor. It is grounded in the actual source of the three
things it composes:

- **`arm_scan`** — this repo's kernel (`selective_scan` 1D done; SS2D per
  [`TOPOLOGY_IMPLEMENTATION_PLAN.md`](TOPOLOGY_IMPLEMENTATION_PLAN.md) §3).
- **NVIDIA EDM** (`NVlabs/edm`) — `EDMPrecond`, the network interface, `EDMLoss`, the Heun sampler.
- **UT CSI Lab** (`utcsilab/ambient-diffusion-mri`, a fork of `giannisdaras/ambient-diffusion`,
  itself a fork of `NVlabs/edm`) — the MRI data pipeline, multi-coil forward operator, and the
  `prior.py` / `solve_inverse_adps.py` sampling entry points.

> **Doc-conflict note (unchanged from the strategy doc):** adopting this reopens
> `PROJECT_CONCEPT.md`'s "Training: None" decision. Amend the decision log in the same change if this
> is adopted.

---

## 1. The whole system on one page

```
                         ┌─────────────────────────────────────────────┐
   undersampled          │             SAMPLING LOOP (CPU)             │
   multi-coil k-space ──▶│  for σ in EDM Heun schedule (18–35 steps):  │
   y = M·F·S·x           │    x̂ = D_θ(x; σ)          ← denoiser call    │
                         │    x  = heun_update(x, x̂, σ)                 │
   sensitivity maps S ──▶│    x  = data_consistency(x, y, M,F,S)  ⟵ partial only
                         └───────────────────────┬─────────────────────┘
                                                 │  D_θ = EDMPrecond(F_θ)
                                                 ▼
                         ┌─────────────────────────────────────────────┐
                         │   F_θ  =  SS2D-Mamba backbone (new)          │
                         │   stacked Mamba blocks; each block runs a    │
                         │   4-direction cross-scan over the token grid │
                         └───────────────────────┬─────────────────────┘
                                                 │  every scan call
                                                 ▼
                         ┌─────────────────────────────────────────────┐
                         │   arm_scan.ss2d_scan(...)  →  selective_scan │
                         │   NEON + chunked + rayon on Arm CPU          │
                         └─────────────────────────────────────────────┘
```

The kernel is the innermost box, invoked
`(#blocks) × (4 directions) × (~2 Heun sub-steps) × (#steps)` times per reconstruction — the
multiplier that makes the Arm optimization matter. Everything above the backbone (precond, loss,
sampler, MRI operator, data) is **inherited from EDM + CSI**, not rebuilt. **The only genuinely new
model code is the SS2D-Mamba backbone `F_θ`; everything else is glue and reuse.**

---

## 2. The contract: what EDM demands of a backbone

This is the single most important interface in the project, and it is small. From `NVlabs/edm`
`training/networks.py`, `EDMPrecond` wraps a raw network and calls it as:

```python
# inside EDMPrecond.forward(x, sigma, class_labels=None, ...)
c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
c_out  = sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
c_in   = 1 / (sigma_data**2 + sigma**2).sqrt()
c_noise = sigma.log() / 4
F_x = self.model((c_in * x), c_noise.flatten(), class_labels=class_labels)  # ← our backbone
D_x = c_skip * x + c_out * F_x                                              # denoised estimate
```

So the SS2D-Mamba backbone must expose **exactly** the same call signature the stock `SongUNet` /
`DhariwalUNet` do:

```python
class MambaSS2DNet(torch.nn.Module):          # must be @persistence.persistent_class
    def __init__(self, img_resolution, in_channels, out_channels,
                 label_dim=0, augment_dim=0, **backbone_kwargs): ...
    def forward(self, x, noise_labels, class_labels=None, augment_labels=None):
        # x: (B, in_channels, H, W)  — already scaled by c_in
        # noise_labels: (B,) — this is c_noise = ln(σ)/4
        # returns F_x: (B, out_channels, H, W)
        ...
```

Three hard requirements fall out of the EDM source, each a real constraint, not a nicety:

1. **`@persistence.persistent_class`.** EDM pickles the *class definition* alongside weights
   (`dnnlib`/`torch_utils.persistence`). Our backbone module must carry that decorator or checkpoints
   won't round-trip — and Route-A distillation (loading their U-Net, saving our Mamba) depends on it.
2. **σ conditioning via `noise_labels`.** EDM passes `c_noise = ln(σ)/4` as a `(B,)` tensor and
   expects the network to embed it (their U-Net uses `PositionalEmbedding`/`FourierEmbedding` →
   `map_layer0/1` → per-block affine). We reuse **exactly** EDM's `PositionalEmbedding` +
   two-layer MLP to produce an embedding vector, then inject it into each Mamba block (adaLN-style
   scale/shift, mirroring `UNetBlock`'s `affine`).
3. **Shape-preserving, `img_channels=2`.** MRI is complex, carried as 2 real channels
   (`--img_channels=2` throughout the CSI code). In/out channels are both 2; the backbone is
   real-valued, so the kernel needs no complex support.

Match this contract and `EDMPrecond`, `EDMLoss`, the Heun sampler, and the CSI posterior samplers all
work **unmodified**. That is the leverage of building on EDM.

---

## 3. Component 1 — the SS2D-Mamba backbone `F_θ` (the only new model)

The recipe follows the published Mamba-diffusion vision backbones (DiM, ZigMa, DiffuSSM) adapted to
EDM's interface and to route scans through `arm_scan`.

### 3.1 Block design

A U-Net-shaped stack (to match `ambient-diffusion-mri`'s `--cres` multiscale bias and reuse its skip
structure), where each residual block is an **SS2D-Mamba block** instead of a conv+attention
`UNetBlock`:

- **Norm + σ-conditioning:** GroupNorm, then adaLN scale/shift from the EDM σ-embedding (reuse
  `UNetBlock.affine` pattern exactly).
- **Patch/token view:** treat the `(B, D, H, W)` feature map as the token grid the cross-scan
  consumes; keep a light depthwise conv for local mixing (DiM/VMamba "local feature enhancement").
- **SS2D cross-scan:** the four traversals (rows fwd/back, cols fwd/back) via `arm_scan`
  (§4). This is the block's compute core and the kernel hot path.
- **Gating + projection:** SiLU gate and output projection (standard Mamba block tail).
- **Residual + skip-scale:** `np.sqrt(0.5)` skip scaling, mirroring EDM's `UNetBlock` for training
  stability.

Down/up-sampling between resolutions reuses EDM's `Conv2d(down=True/up=True)` resample — no need to
reinvent it.

### 3.2 Design decisions to lock (each has precedent)

| Decision | Recommended default | Why / alternative |
|---|---|---|
| Scan pattern | Plain 4-direction SS2D (VMamba) | Maps 1:1 to the kernel work already scoped; ZigMa zigzag / DiMSUM spatial-freq are later variables, not MVP. |
| Macro-arch | U-Net-shaped (multiscale) | Matches CSI `--cres`; reuses skip plumbing; better MRI inductive bias than isotropic DiT-style. |
| σ-embedding | EDM `PositionalEmbedding` + 2-layer MLP | Identical to the U-Net so distillation targets align; zero new math. |
| Attention | None (pure SSM), or 1 attn block at lowest res | Pure-SSM is the cleaner "Mamba does it" story; a single low-res attention block is a cheap quality insurance if needed. |
| Data channels | 2 (complex→real) | CSI convention; keeps kernel real-valued. |
| `sigma_data` | Re-estimate on MRI data | EDM default 0.5 is for natural images; MRI normalization differs — measure it from the training set (CSI's `--norm=2` path). |

### 3.3 How it registers into the CSI training code

`ambient-diffusion-mri/train.py` selects the backbone via `--arch` (`ddpmpp`→SongUNet,
`adm`→DhariwalUNet). The integration is a **new `--arch=ss2dmamba` branch** that constructs
`MambaSS2DNet` with the same `img_resolution/in_channels/out_channels/label_dim` wiring the existing
arches get. No change to `EDMPrecond`, `EDMLoss`, or `training_loop.py`. This is the minimal,
surgical insertion point — one dispatch branch.

---

## 4. Component 2 — kernel integration (the hot path)

The backbone's cross-scan calls `arm_scan`, staged exactly as
[`TOPOLOGY_IMPLEMENTATION_PLAN.md`](TOPOLOGY_IMPLEMENTATION_PLAN.md) §3 sequences it:

**Stage 1 — unfused `ss2d.py` (ship first, no new Rust).** Given the block's `(B, D, H, W)` grid,
build the four permuted/flipped views, stack along batch → `4B`, **one** `selective_scan` call, split
and merge. This stands up the entire diffusion pipeline end-to-end on Arm CPU and produces honest
numbers before any new kernel work. **Everything in §§3,5–8 can be built and demoed on this path.**

**Stage 2 — measure.** Because the sampler calls the backbone hundreds of times, the per-call
flip/permute copy traffic is paid hundreds of times. This is the strongest justification yet for
fusing — but it must be *measured* at the diffusion workload's real grid size and NFE count against
`torch.compile`, per the "benchmark honestly" rule, before committing the week.

**Stage 3 — fused `selective_scan_2d` in Rust (only if Stage 2 justifies it).** Transpose-then-
reuse-row-scan; new C ABI entry point; `ss2d.py` internals swap to it behind the same public
signature. Same four correctness gates as the 1D kernel (golden-vs-f64, NEON-vs-scalar, rayon
bit-identity at `RAYON_NUM_THREADS ∈ {1,2,8}`, real-C-ABI replay).

**Threading leverage is unusually good here.** Rayon parallelizes over `batch × channel × direction`,
and the CSI samplers already batch multiple seeds/samples (`--batch`, `--seeds`) — so a diffusion
reconstruction has many independent rows, which makes a strong core-scaling curve on Graviton's high
core counts. That curve is a headline Cloud-track artifact.

**Precision:** kernel and backbone run fp32 on CPU. (CSI trains `--fp16=True` on GPU; inference
precision is a separate knob. fp32 CPU inference is the correctness baseline; a
BF16-storage/fp32-accumulate pass on Graviton4 is a listed stretch, not MVP.)

**`torch.compile` composability:** the SS2D op must register a fake/meta kernel (as the 1D op already
does) so the backbone composes under `torch.compile` — necessary because `torch.compile` is the fair
baseline for every number.

---

## 5. Component 3 — EDM training, precond, loss (inherited)

Reused from EDM/CSI with **no changes beyond the `--arch` branch**:

- **Preconditioning:** `EDMPrecond` (or CSI's `AmbientPrecond` variant — confirm in Phase A which the
  MRI checkpoints use; the "ambient" method wraps EDM precond with a corruption model). Our backbone
  slots into `self.model` unchanged.
- **Loss:** `EDMLoss` — σ ~ LogNormal(`P_mean=-1.2, P_std=1.2`), weight
  `λ(σ)=(σ²+σ_data²)/(σ·σ_data)²`. Backbone-agnostic.
- **Training loop:** `training/training_loop.py` — EMA, `training_options.json`, `stats.jsonl`,
  checkpoint pickling (why §2's persistence decorator matters). Unchanged.

The only substantive training question is *where the weights come from*, which is Component 6.

---

## 6. Component 4 — MRI forward operator & data (inherited from CSI)

The part that makes this *MRI* and not generic image-gen. All inherited from
`ambient-diffusion-mri/utils` + `dataset_tool.py`; we do **not** reimplement MRI physics.

- **Forward operator** `A = M·F·S`: coil sensitivity maps `S`, 2D Fourier `F`, undersampling mask
  `M`. Used in the data-consistency step of posterior sampling and in "ambient" training.
- **Data:** preprocessed FastMRI multi-coil brain, packaged as `ksp_brainMRI_384.zip` via
  `dataset_tool.py` following EDM's dataset format. 384×384, 2-channel complex.
- **Sensitivity maps:** estimated per-slice (ESPIRiT/J-Sense; CSI's `deep-jsense` /
  `Nufft_Torch` are the lab's tooling if we need to regenerate them).
- **FFT stays a black box** — already fast on Arm (Arm Compute Library / pocketfft); explicitly *not*
  an optimization target per `PROJECT_CONCEPT.md`. The data-consistency step is cheap relative to a
  full backbone forward pass, so it does not move the CPU-latency story.

**FastMRI access** requires registration (approval can take days — start day 1). The credential-free
**synthetic phantom** track (§10) mirrors the repo's `make validate` philosophy and lets a judge run
the whole pipeline with no dataset and no checkpoint.

---

## 7. Component 5 — sampling: full and partial, on CPU

Both modes are the EDM Heun sampler around our CPU `D_θ`; they differ only in the data term.

### 7.1 Full reconstruction (R=1 / unconditional) — `prior.py` path

Plain EDM deterministic Heun sampler (σ_max≈80, σ_min≈0.002, ρ=7, 18–35 steps), no data-consistency.
This is the **first end-to-end milestone**: it exercises sampler × backbone × kernel with none of the
MRI-operator complications, and proves the generative prior produces coherent anatomy. Maps to
`prior.py`. Deliverable: prior samples on Arm CPU, output-parity-verified between the kernel scan path
and the reference scan.

### 7.2 Partial reconstruction (R=2/4/6/8) — `solve_inverse_adps.py` path

The clinical task. Interleave EDM denoising steps with a **measurement-consistency** step:

- **A-DPS** (CSI's method): likelihood guidance from `y = A·x` folded into the sampler, with
  `--l_ss` (likelihood step size), `--inference_R`, `--training_R`, `--num_steps`, `--S_churn`.
- **Fallbacks if A-DPS integration is fiddly:** DPS-style `∇ log p(y|x)` gradient, or
  `csgm-mri-langevin`'s annealed-Langevin measurement term (a proven, self-contained formulation).

Sweep R ∈ {2,4,6,8} to reproduce Tamir's evaluation grid. The backbone/kernel call count per step is
identical to the full case; partial just adds one cheap `A`/`Aᴴ` evaluation per step.

**One prior, one kernel, two reconstruction modes** — the tidy story the user asked for ("partial and
full").

### 7.3 The honest CPU-latency framing (non-negotiable, per `CLAUDE.md`)

Hundreds of NFEs on a CPU is not free. Frame and benchmark it correctly:
(a) **lead with per-scan / per-NFE speedup vs. `torch.compile`** so the kernel's contribution is
isolated from the sampler's inherent cost; (b) report **end-to-end wall-clock and $/reconstruction on
Graviton vs. a GPU baseline**, always naming NFE count and σ schedule; (c) push **low-NFE** samplers
(deterministic Heun needs far fewer NFEs than ancestral) to keep the end-to-end number defensible;
(d) publish unflattering rows anyway — the moat is that `torch.compile` cannot restructure a
sequential recurrence.

---

## 8. Component 6 — checkpoints & training (the critical-path risk)

**No public Mamba-backbone EDM MRI checkpoint exists.** This decides whether the project is a Week-4
demo or a research campaign. Routes, cheapest first (detail in
[`MAMBA_DIFFUSION_MRI_PLAN.md`](MAMBA_DIFFUSION_MRI_PLAN.md) §8):

- **Route A (recommended) — distill CSI's U-Net EDM prior into the Mamba backbone.** Their 9
  checkpoints (FastMRI brain, incl. a supervised R=1 EDM model) are the teacher. Train `MambaSS2DNet`
  to match the U-Net denoiser output across σ. Fewer GPU-hours than from-scratch, reuses their exact
  precond/data, and yields a clean "same prior, Mamba backbone" comparison. The persistence decorator
  (§2) makes loading the teacher and saving the student mechanical.
- **Route B — train small from scratch** at reduced resolution (128–192px), single-coil/RSS, IXI or a
  FastMRI subset. Proof-of-concept quality; aligns with the "shareable demo" philosophy.
- **Route C — GSURE self-supervised prior** (`gsure-diffusion-mri`), no fully-sampled ground truth.
  Highest novelty/risk; stretch.

**Recommendation:** Route A for the headline + the synthetic-phantom track for reproducibility.
**Decide the GPU budget explicitly before committing** — this is the `PROJECT_CONCEPT.md` amendment.
Training/distillation is GPU work (a bounded spot-instance spend); *inference and all benchmarking* is
the Arm-CPU story.

---

## 9. Correctness & parity gates (same discipline as the kernel)

Layered so a failure localizes to one component — mirroring the kernel's five-layer/four-gate template
in [`TOPOLOGY_IMPLEMENTATION_PLAN.md`](TOPOLOGY_IMPLEMENTATION_PLAN.md) §1.6:

1. **Kernel level (exists):** golden-vs-f64 `< 1e-4`, NEON-vs-scalar, rayon bit-identity — already
   enforced for 1D; extended to SS2D per the topology plan.
2. **Op level:** `ss2d_scan` output parity against the reference SS2D rearrangement (the topology
   plan's per-direction golden check), so a backbone bug can't hide behind a scan bug.
3. **Block level:** one SS2D-Mamba block through `arm_scan` vs. the same block through a pure-PyTorch
   reference scan — token-identical within fp32 tolerance.
4. **Backbone level:** full `F_θ` through the kernel vs. through the reference scan — same output for
   the same weights; this is the "kernel didn't change the answer" gate.
5. **Model level:** `D_θ` denoising parity, then **sampling parity** — a full reconstruction with the
   kernel path vs. the reference path must agree to fp32 tolerance (not bit-exact; the NEON exp
   polynomial is disclosed-approximate).
6. **Application level:** PSNR/SSIM/NMSE (+FID for the prior) between the kernel-path reconstruction
   and the reference-path reconstruction — quality parity at identical NFE/σ schedule.

**Never loosen a tolerance to pass** — the kernel's whole credibility rests on that rule.

---

## 10. Benchmarking plan

Same three-surface rigor as `BASELINE_TEST_PLAN.md`, applied to the diffusion workload:

- **Baselines:** `torch.compile` (the fair fight, lead with it) and eager fallback (color). Ablate the
  kernel ladder (scalar → +NEON → +chunked → +rayon) at the backbone's real scan shapes.
- **Op/NFE metrics:** per-scan and per-`D_θ`-call time, kernel vs. baselines, medians after warmup,
  fixed thread counts, pinned seeds, stated torch version + instance type.
- **End-to-end metrics:** wall-clock per reconstruction at fixed NFE; core-scaling curve over batched
  seeds; peak memory (the constant-state SSM story vs. attention's growth).
- **The economic headline:** **$/reconstruction on Graviton (`c8g`) vs. a GPU baseline** — the
  "$4,000 GPU not required" argument, with the NFE count and quality metric stated so it's honest.
- **Quality:** PSNR/SSIM/NMSE at R=2/4/6/8 + FID for the prior, reproducing Tamir's grid.

Every number names its baseline, NFE count, and σ schedule. Unflattering rows ship.

---

## 11. Proposed repo layout (the new demo app)

Keep the kernel crate untouched; add the diffusion app as a separate top-level module that *imports*
`arm_scan`, so the pip-installable kernel stays clean and general:

```
apps/mri_diffusion/
  backbone/            MambaSS2DNet (the only new model code) + σ-embedding reuse
  edm/                 thin vendored/patched EDM: EDMPrecond, EDMLoss, Heun sampler (or import CSI's)
  mri/                 forward operator, sensitivity maps, data-consistency step (from CSI utils)
  sampling/            full (prior) + partial (A-DPS / Langevin) CPU sampling entry points
  data/                dataset_tool wrapper; synthetic-phantom generator (credential-free)
  distill/             Route-A: load CSI U-Net teacher, train Mamba student
  bench/               diffusion-workload benchmarks (extends repo bench/ harness)
  tests/               block/backbone/model/sampling parity gates (§9)
  README.md            run/validate instructions; phantom path needs no data or checkpoint
```

`ss2d.py` / `selective_scan_2d` stay in `python/arm_scan/` (kernel side, per the topology plan) — the
app depends on the kernel, not vice-versa.

---

## 12. Phased milestones (detailed, each with an exit gate)

Sequenced so an end-to-end, honestly-benchmarked pipeline stands **before** any new Rust and **before**
any large training spend. Fits the existing Week-4/5 window; Phases A–B add work the MambaRecon plan
didn't have, so Phase A is a hard go/no-go.

**Phase A — Reference study & CPU feasibility gate (days, not weeks).**
Clone `ambient-diffusion-mri`; CPU-dry-run it. Confirm: (1) which precond the MRI checkpoints use
(stock `EDMPrecond` vs. `AmbientPrecond`); (2) the exact backbone call signature and `training_options.json`
wiring; (3) one checkpoint loads and `prior.py` samples on CPU, however slowly; (4) the data format and
whether a checkpoint or the FastMRI pipeline is reachable. Lock the SS2D backbone recipe (§3.2).
**Gate: precond + interface + data + a runnable checkpoint understood. If a checkpoint won't run on CPU
at all, fall back to discriminative MambaRecon and stop here.**

**Phase B — Backbone bring-up + prior (GPU-side, bounded).**
Implement `MambaSS2DNet` satisfying §2; verify it denoises under `EDMPrecond` on a toy target; execute
the chosen prior route (A/B/C). **Gate: a `D_θ` that denoises and pickles/loads via EDM persistence.**

**Phase C — Kernel-in-the-loop, full reconstruction on CPU (unfused SS2D).**
Route the backbone's scans through `arm_scan` via `ss2d.py`. Run R=1 Heun sampling end-to-end on Arm
CPU. Pass the §9 backbone- and sampling-level parity gates (kernel path vs. reference scan).
**Gate: full reconstruction runs on Arm CPU through our kernel, output-parity verified.**

**Phase D — Partial reconstruction + quality.**
Add the data-consistency step (A-DPS, or Langevin fallback); sweep R=2/4/6/8; compute PSNR/SSIM/NMSE
(+FID). **Gate: R=4 reconstruction at a defensible metric on Arm CPU, quality-parity vs. reference
scan path confirmed.**

**Phase E — Measure & (maybe) fuse.**
Benchmark unfused SS2D at the diffusion grid × NFE vs. `torch.compile`; build the core-scaling curve
and the $/reconstruction-vs-GPU table. If copy overhead dominates (likely, at hundreds of NFEs),
implement fused `selective_scan_2d` (topology §3.2) and re-measure. **Gate: honest `RESULTS.md` rows +
a fusion decision backed by measurement.**

**Phase F — Demo, video, writeup.**
Side-by-side (zero-filled undersampled vs. diffusion reconstruction) on Graviton; the phantom track for
reproducibility; `make validate`; Devpost writeup; reconcile `PROJECT_CONCEPT.md`. **Gate: submittable.**

---

## 13. Risk register (implementation-specific)

| Risk | Severity | Mitigation |
|---|---|---|
| No pretrained Mamba-diffusion MRI checkpoint | **High** | Route A distillation from CSI U-Net weights; Route B small-scale; phantom track for reproducibility. |
| Training/distillation needs GPU the project scoped out | **High** | Bounded spot-GPU budget, decided up front; MambaRecon fallback; inference/benchmarks stay CPU. |
| CSI code bundles CUDA-only ops / won't run on CPU | **High** | Phase-A CPU-dry-run is the go/no-go — same gate that killed DH-Mamba; force reference scan path before building. |
| `AmbientPrecond` ≠ stock `EDMPrecond`, breaking the clean interface | Medium | Confirm in Phase A; our backbone targets whichever `self.model` interface the checkpoint's precond uses (both call the network identically). |
| Absolute CPU diffusion latency looks bad | Medium | Per-scan/per-NFE speedup lead; low-NFE Heun; $/recon-vs-GPU framing (§7.3). |
| SS2D fused kernel eats a week | Medium | Unfused `ss2d.py` stands up the whole demo; fuse only if Phase-E measurement justifies it. |
| Data-consistency/posterior sampling instability | Medium | Start with R=1 (no data term); Langevin fallback from `csgm-mri-langevin`; tune `--l_ss`. |
| σ_data / normalization mismatch on MRI vs. EDM defaults | Low–Med | Re-estimate `sigma_data` from the training set; inherit CSI `--norm` handling. |
| Persistence/pickle incompatibility blocks distillation | Low–Med | Decorate the backbone `@persistence.persistent_class` from day one; test teacher-load + student-save early. |
| Scope creep vs. Aug 14 | **High** | Phase A hard gate; MambaRecon fallback loses only the extra ceiling, not the submission. |

---

## 14. Decisions needed before coding starts

1. **Adopt this over MambaRecon for the MRI slot, or run MambaRecon as the safe baseline first?**
   (Recommendation: Phase A decides; keep MambaRecon as the fallback, don't build both in parallel.)
2. **GPU budget for Route-A distillation** — the explicit `PROJECT_CONCEPT.md` amendment.
3. **Backbone recipe** (§3.2 defaults) — confirm plain-SS2D, U-Net-shaped, pure-SSM (or one attn block).
4. **Which precond** the target checkpoints use (Phase A finding) — sets the exact `self.model` interface.
5. **Is the MRI slot allowed to be quality-parity-only** (no headline speedup claim) if the CPU
   end-to-end number is weak — keeping the kernel speedup as the headline and the recon as the WOW demo?

---

## 15. References

- **EDM:** [`NVlabs/edm`](https://github.com/NVlabs/edm) — `training/networks.py` (`EDMPrecond`,
  `SongUNet`, `DhariwalUNet`, the backbone contract); `generate.py` (Heun sampler); Karras et al.,
  NeurIPS 2022 ([arXiv:2206.00364](https://arxiv.org/abs/2206.00364)).
- **CSI Lab:** [`ambient-diffusion-mri`](https://github.com/utcsilab/ambient-diffusion-mri) (ICLR 2025;
  `train.py`/`prior.py`/`solve_inverse_adps.py`/`dataset_tool.py`, `--arch` dispatch, 9 checkpoints) ·
  [`csgm-mri-langevin`](https://github.com/utcsilab/csgm-mri-langevin) (Langevin fallback) ·
  [`gsure-diffusion-mri`](https://github.com/utcsilab/gsure-diffusion-mri) (Route C) ·
  [`deep-jsense`](https://github.com/utcsilab/deep-jsense) / [`Nufft_Torch`](https://github.com/utcsilab/Nufft_Torch) (MRI physics).
- **Mamba-diffusion backbones:** [DiM](https://arxiv.org/abs/2405.14224) ·
  [ZigMa](https://arxiv.org/abs/2403.13802) · [DiffuSSM](https://arxiv.org/abs/2311.18257) ·
  [VMamba / SS2D](https://github.com/MzeroMiko/VMamba).
- **Mamba-MRI baselines (for comparison, not backbone):**
  [MambaRecon](https://github.com/yilmazkorkmaz1/MambaRecon) (WACV 2025) ·
  [DH-Mamba](https://github.com/XiaoMengLiLiLi/DH-Mamba).
- **In-repo:** [`MAMBA_DIFFUSION_MRI_PLAN.md`](MAMBA_DIFFUSION_MRI_PLAN.md) (strategy) ·
  [`TOPOLOGY_IMPLEMENTATION_PLAN.md`](TOPOLOGY_IMPLEMENTATION_PLAN.md) §3 (SS2D kernel) ·
  [`PROJECT_CONCEPT.md`](PROJECT_CONCEPT.md) (decision log to amend) ·
  [`BASELINE_TEST_PLAN.md`](BASELINE_TEST_PLAN.md) (benchmark surfaces).
```
