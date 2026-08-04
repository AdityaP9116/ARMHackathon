# ARMHackathon — Arm-Optimized Selective-Scan for Vision & Diffusion Mamba

**Arm Create: AI Optimization Challenge 2026 — Cloud AI track**

Mamba state-space models are spreading from language into vision and diffusion — VMamba, MambaRecon, DiM, ZigMa, DiffuSSM — and every one of those research models lives in PyTorch, where the efficient selective-scan is **CUDA-only**. On CPU it falls back to an unoptimized sequential loop. Worse, the **multi-directional 2D cross-scan (SS2D)** that vision- and diffusion-Mamba models are built on has **no fast CPU implementation anywhere** — VMamba ships it as a CUDA-only extension, and CPU users are left with the slow pure-PyTorch reference.

This project ships an **Arm-optimized `selective_scan` for the PyTorch ecosystem, written in Rust** (chunked/associative scan + NEON SIMD + rayon multi-core), as a pip-installable drop-in — and doubles down on the part nobody else has: **the SS2D cross-scan on CPU**, proven by running a **diffusion-based MRI reconstruction** (an EDM diffusion prior with an SS2D-Mamba denoiser) end-to-end on Arm cloud CPUs (AWS Graviton). Diffusion recon calls the denoiser 18–256 times per image, so every scan-level win compounds — exactly the workload "everyone knows" needs a GPU.

## Prior art — what exists, and what doesn't

We checked, so the "first" claims don't rest on faith:

| Prior work | What it is | What it doesn't do |
|---|---|---|
| [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) | CPU `ssm_scan` for GGUF-converted language Mamba/Mamba-2 | Not callable from PyTorch; requires converting the model into its own runtime; 1D language models only — no SS2D |
| [BitMamba-2](https://engrxiv.org/preprint/view/6680) | 1.58-bit quantized Mamba-2, custom C++ engine, ARM NEON | Quantized custom engine, not a reusable kernel; not PyTorch; 1D only |
| [mamba.rs](https://github.com/LaurentMazare/mamba.rs), [Candle](https://github.com/huggingface/candle), [flawedmatrix/mamba-ssm](https://github.com/flawedmatrix/mamba-ssm) | Standalone Rust Mamba inference engines | Full-model runtimes, not a scan kernel; no PyTorch interop; 1D only |
| [mamba.py](https://github.com/alxndrTL/mamba.py), [mamba-mini](https://github.com/MzeroMiko/mamba-mini) | Pure-PyTorch parallel-scan implementations | No SIMD, no Arm tuning; memory-hungry parallel form; no SS2D kernel |
| [VMamba](https://github.com/MzeroMiko/VMamba) | The SS2D reference — as a **CUDA-only** extension | No CPU kernel at all; CPU falls back to the slow pure-PyTorch reference |
| [2DMamba](https://arxiv.org/abs/2412.00678) (CVPR 2025) | Hardware-aware 2D selective scan with SRAM tiling — the GPU-side precedent for tiled 2D scanning | **GPU-only** (CUDA); intrinsic-2D semantics, not the VMamba cross-scan existing checkpoints use; no CPU or PyTorch-CPU path |

**What this project claims — precisely.** Not "first Mamba on Arm" (see above). To the best of our knowledge it is: (1) the **first Arm-optimized `selective_scan` exposed as a PyTorch custom op** — a drop-in for existing PyTorch checkpoints, no model conversion; (2) the **first fast CPU implementation of the SS2D multi-directional cross-scan** on any architecture; (3) the **first diffusion-prior MRI reconstruction demonstrated on Arm CPU**.

## Why it matters

- **SSMs run in linear time with constant memory** — ideal for enormous imaging sequences (a 384×320 slice is a 122k-token scan per direction) and for cost-effective CPU serving.
- **The scan is the hot path**, and in the ecosystem where research models actually live it has zero CPU optimization. `torch.compile` cannot close this gap: it cannot restructure a sequential recurrence.
- **Diffusion posterior sampling is the modern SOTA family for accelerated MRI**, assumed GPU-bound because the denoiser runs tens-to-hundreds of times per image. That multiplier is precisely where a fast CPU scan compounds — turning "diagnostic-quality reconstruction on a hospital CPU box or a cheap Graviton fleet" into a real deployment story.

## What's here

| Component | Description |
|---|---|
| Rust `selective_scan` kernel | Chunked two-pass scan + fused discretization, NEON intrinsics (stable Rust), specialized non-positive `exp` paths, rayon threading over independent channels; scalar reference path is the in-crate oracle, the non-Arm fallback, and what keeps x86 CI meaningful |
| SS2D path | `arm_scan.ss2d.ss2d_scan` takes a `(B, D, H, W)` token grid and runs VMamba's 4 directions as **two traversal-order pairs** on the fused bidirectional kernel, so the direction-independent Pass A (discretize + `exp`, ~85% of kernel time) is computed twice per block instead of four times, with no `torch.flip` copies. A single fused `selective_scan_2d` owning all four directions in Rust remains the identified next lever ([`TOPOLOGY_IMPLEMENTATION_PLAN.md`](./TOPOLOGY_IMPLEMENTATION_PLAN.md) §3.2) — currently **not** justified by its own >15% overhead rule, see below |
| PyTorch custom-op bridge | `torch.library` op with registered fake kernel (composes with `torch.compile`), C-ABI FFI with panic containment, `arm_scan.patch()` for HF `transformers` Mamba |
| MRI diffusion app | `apps/mri_diffusion`: SS2D-Mamba denoiser under NVIDIA EDM preconditioning, built on UT CSI Lab's [`ambient-diffusion-mri`](https://github.com/utcsilab/ambient-diffusion-mri) scaffolding; full (R=1) and undersampled (R=2–8) posterior reconstruction on CPU |
| Correctness suite | Golden vectors vs. vendored upstream reference + an independent numpy re-derivation; NEON↔scalar parity; rayon output bit-identical to sequential at any thread count; goldens replayed through the real C ABI — **no dataset required**, runs on any arm64 machine, Apple Silicon included |
| Benchmark harness | Op-level and end-to-end, against **both** eager and `torch.compile`; medians after warmup, pinned threads/seeds, host- and SHA-tagged JSON; every kernel optimization logged with measured attribution in [`OPTIMIZATION_LOG.md`](./OPTIMIZATION_LOG.md) |

## Results so far (provisional — headline numbers land on Graviton)

Measured on shared GitHub `ubuntu-24.04-arm` CI runners (4-core, torch 2.13); provisional per [`BASELINE_TEST_PLAN.md`](./BASELINE_TEST_PLAN.md), to be replaced by dedicated Graviton (`c8g`) numbers in [`bench/results/RESULTS.md`](./bench/results/RESULTS.md):

| Benchmark | vs eager | vs `torch.compile` |
|---|---|---|
| Op-level, B1 D768 L128 | 24.1× | **3.7×** |
| HF mamba-130m prefill, end-to-end (greedy tokens identical) | ~2× | — |

Kernel error vs the f64 reference ≤ 5e-6 across all golden cases (gate: 1e-4). The 2D
cross-scan goldens land at ~1.0× each case's recorded f32 error floor — i.e. the kernel is
as accurate as upstream's own float32 evaluation, not merely inside the gate.

**SS2D, structural (dev box, x86 scalar backend — directional only, not a speed claim):**
running the four directions as two traversal pairs instead of four forward scans measures
**1.77–1.82× (geomean 1.80×)** on block total across the four production shapes
(384×320 @ 96ch and 192×160 @ 192ch, seed-batch 1 and 4), and **1.81–1.90×** on scan time
alone. Both sides of that ratio use the same kernel, so it is attributable to the
restructuring, not to the backend. Non-scan time (flips, permutes, projections) falls from
**21–25% to 7.2–13.8%**. That worst case of 13.8% sits *just* under this repo's own >15%
bar for justifying a fully fused `selective_scan_2d`, so that kernel is currently **not**
justified — but only marginally, and the verdict gets re-taken on Arm where the cache
behaviour differs.

Still to land, format locked: **every number above re-measured on a named Graviton
instance** (nothing in this repo has yet run on dedicated Arm hardware), diffusion
end-to-end per-NFE latency and $/reconstruction, PSNR/SSIM/NMSE parity at R=2–8,
core-scaling curve, and mamba-130m generality rows. Unflattering rows get published too.

## Quick validation (≈5 minutes, any arm64 machine, no data downloads)

```bash
git clone https://github.com/AdityaP9116/ARMHackathon && cd ARMHackathon
make validate   # builds the kernel; runs the 1D + 2D golden, parity and FFI
                # gates; checks the SS2D cross-scan against its oracle and
                # samples a diffusion reconstruction through the kernel; then
                # prints an op-level microbenchmark table for this machine
make demo       # the side-by-side reconstruction image, phantom track
```

Needs a Rust toolchain and `pip install -r requirements-dev.txt` (CPU torch). The
kernel-only gates run on `requirements.txt` alone — numpy, no torch — and the SS2D and
diffusion gates skip themselves with a message rather than failing if torch is absent.

Works on AWS Graviton, Oracle Ampere, Raspberry Pi 5, and Apple Silicon Macs. Correctness validation never requires fastMRI credentials or an AWS account; the MRI demo keeps a synthetic Shepp–Logan phantom track for the same reason.

## Numerics, disclosed

NEON `exp` polynomials and FMA reassociation mean results match the f64 reference to fp32 tolerance, not bit-exactly. Every golden case records its f32 error floor and every kernel change is gated against it; end-to-end, patched HF mamba-130m produces token-identical greedy output, and the MRI app carries a PSNR/SSIM/NMSE parity gate at identical output quality.

## Data

The phantom track is synthetic and credential-free — `make demo` needs no download.

Real-data results use **fastMRI knee single-coil** (NYU Langone). That data is *not*
redistributed here and cannot be: its Data Sharing Agreement forbids redistributing the
dataset or the download links, and limits use to internal research and education. Request
access at <https://fastmri.med.nyu.edu/>, then see `tools/prepare_fastmri.py`.

> Knoll et al., *fastMRI: A Publicly Available Raw k-Space and DICOM Dataset of Knee Images
> for Accelerated MR Image Reconstruction Using Machine Learning*, Radiology: Artificial
> Intelligence, 2020; and Zbontar et al., *fastMRI: An Open Dataset and Benchmarks for
> Accelerated MRI*, arXiv:1811.08839.
>
> Data used in the preparation of this work were obtained from the NYU fastMRI Initiative
> database. NYU fastMRI investigators provided data but did not participate in analysis or
> writing of this work. A listing of NYU fastMRI investigators, subject to updates, can be
> found at fastmri.med.nyu.edu.

## Status

Kernel + 1D PyTorch integration landed and measured. SS2D runs on the fused bidirectional
kernel with per-direction 2D goldens and a parity gate against the four-separate-scans
oracle; the diffusion app samples end-to-end on CPU through the kernel, credential-free
(Phase A feasibility: **GO** — [`apps/mri_diffusion/PHASE_A_FINDINGS.md`](./apps/mri_diffusion/PHASE_A_FINDINGS.md)).

Open, and stated plainly: **no number in this repo has been measured on dedicated Arm
hardware yet** — everything is x86 or a shared 4-core CI runner, and is labelled as such.
There is also no trained diffusion prior; the app currently reconstructs with a small
prior trained in-process on synthetic phantoms, which demonstrates the pipeline rather
than reconstruction quality. Both are the active work, sequenced in
[`SUBMISSION_ENDGAME_PLAN.md`](./SUBMISSION_ENDGAME_PLAN.md).

Decision history: [`PROJECT_CONCEPT.md`](./PROJECT_CONCEPT.md). Engineering plans:
[`TOPOLOGY_IMPLEMENTATION_PLAN.md`](./TOPOLOGY_IMPLEMENTATION_PLAN.md),
[`MRI_DIFFUSION_IMPLEMENTATION_PLAN.md`](./MRI_DIFFUSION_IMPLEMENTATION_PLAN.md).

## License

MIT — see [`LICENSE`](./LICENSE) (also set in the repository About sidebar, per contest rules).
