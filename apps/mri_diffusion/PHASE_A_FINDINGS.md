# PHASE A FINDINGS — reference study & CPU feasibility gate

**Verdict: GO.** All four gate items pass (MRI_DIFFUSION_IMPLEMENTATION_PLAN §12
Phase A). Jul 16, 2026; evidence gathered on the Windows x86 dev box; reference
clone at `..\reference\ambient-diffusion-mri`, checkpoints at `..\reference\models\`.

## 1. Precond (plan open question #4) — RESOLVED, best case
`train.py`: `--precond=ambient` → `network_kwargs.class_name =
'training.networks.EDMPrecond'` — **"ambient" changes only the loss**
(`AmbientLoss`). All 9 checkpoints (supervised R=1 and ambient R=2–8) wrap the
**stock EDMPrecond**. Verified in the R=1 `training_options.json`
(`class_name: training.networks.EDMPrecond`, `use_fp16: false`, resolution 384,
`augment_dim: 7`, SongUNet ddpmpp, `gated` convolutions — a CSI addition to
stock EDM, relevant for Route-A distillation parity).

## 2. Backbone interface — CONFIRMED VERBATIM
`EDMPrecond.forward`: `F_x = self.model((c_in·x).to(dtype), c_noise.flatten(),
class_labels=..., **model_kwargs)`. Backbone selection = `globals()[model_type]`
inside `networks.py` + config-driven `construct_class_by_name` from
`training_options.json` → our integration is one `model_type='MambaSS2DNet'`
injection + one `--arch=ss2dmamba` branch. `prior.py` rebuilds nets from
training_options and copies params; snapshots are also directly usable
(persistence embeds class source).

## 3. Data / checkpoints — REACHABLE
All 9 checkpoints: public UT Box zip (1.7 GB, no credentials), layout
`ambient/R={2,4,6,8}`, `edm/{supervised_R=1, l1_recon_R={2,4,6,8}}`, each with
`network-snapshot.pkl` + `training_options.json`. FastMRI raw data needs NYU
registration (start early; not needed for prior sampling or the phantom track).
Shapes: 2-channel complex; **the 384→320 width crop is hardcoded** in BOTH
`prior.py`'s sampler (`latents[:,:,:,0:320]`) and `EDMLoss`
(`images[:,:,:,32:352]`) — models genuinely operate at 384×320. Ambient (R>1)
models take 4 input channels (image+mask) and need sensitivity maps even for
prior sampling; **the R=1 supervised model is the clean 2-channel path** and the
Phase-C target.

## 4. CPU dry-run — PASS (with numbers)
CUDA coupling is mild: hardcoded `.cuda()` calls + a `device='cuda'` default
arg; **no custom CUDA kernels**. A minimal stock-Heun script (scratch;
mirrors prior.py's math incl. the 320 crop) loaded the R=1 EMA directly from
the pickle and sampled on CPU:
`EDMPrecond(SongUNet) 65.5M params → (1,2,384,320), finite, 7 NFE = 55.7 s,
per-NFE median 8.8 s` (32-thread x86 i9; torch 2.11). Implication for the NFE
floor: 35-step Heun (69 NFE) ≈ 10 min/reconstruction unoptimized on x86 —
the number the Mamba backbone + arm_scan must attack.
Environment notes: their `dnnlib`/`torch_utils` import `s3fs` and `wandb`
unconditionally (must be pip-installed even for CPU inference); numpy-2
incompatibilities exist only in plotting utilities, not the model path.

## 5. Backbone recipe — LOCKED (plan §3.2 defaults confirmed)
Plain 4-direction SS2D (VMamba), U-Net-shaped, EDM PositionalEmbedding + MLP
σ-embedding with adaLN injection, pure-SSM (no attention) first,
`img_channels=2`, σ_data re-estimated on MRI data at training time.
Implemented in Phase B as `apps/mri_diffusion/backbone/mamba_ss2d.py` with the
scan behind a swappable `scan_fn` seam (`torch_scan.py` reference now,
`arm_scan` in Phase C).

## Go/no-go
**GO** — every Phase-A exit criterion met; no fallback to MambaRecon needed.
Route A/B/C (prior weights) and the GPU budget remain the user's §14 decisions;
Route A (distill from `edm/supervised_R=1`) is recommended and now de-risked:
the teacher loads and runs.
