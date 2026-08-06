# ARMHackathon — Arm-Optimized Selective-Scan for the PyTorch Mamba Ecosystem

**Arm Create: AI Optimization Challenge 2026 — Cloud AI track**

Mamba state-space models run in **linear time with constant memory** — which is exactly what
a CPU is good at, and exactly what a transformer's growing KV cache is not. But every Mamba
model in PyTorch calls a selective-scan that is **CUDA-only**. On CPU it falls back to an
unoptimized sequential loop, and `torch.compile` cannot fix it: it cannot restructure a
sequential recurrence, only unroll it into a graph that gets slower to build as the sequence
grows.

This project ships an **Arm-optimized `selective_scan` written in Rust** (chunked scan +
NEON SIMD + rayon), as a pip-installable PyTorch drop-in — and covers **all three scan
topologies** the Mamba ecosystem actually uses, each proven on a real workload:

| Topology | What it powers | Our demonstration |
|---|---|---|
| **1D unidirectional** | causal language models | **Long context on a CPU** — constant memory at any length, where `torch.compile` can't even build the graph |
| **1D bidirectional** | non-causal sequences | **Speech enhancement** — noisy audio in, clean audio out, measured as real-time factor |
| **2D cross-scan (SS2D)** | vision & diffusion Mamba | **Diffusion MRI reconstruction** — the scan runs 18–256× per image, so kernel wins compound |

Three topologies, three demos, one kernel. That is the claim: **the kernel is general**, not a
point optimization.

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

**What this project claims — precisely.** Not "first Mamba on Arm" (see above). To the best of our knowledge it is: (1) the **first Arm-optimized `selective_scan` exposed as a PyTorch custom op** — a drop-in for existing checkpoints, no model conversion; (2) the **first fast CPU implementation of the SS2D multi-directional cross-scan** on any architecture; (3) the **first diffusion-prior MRI reconstruction demonstrated on Arm CPU**.

All three rest on the same kernel. The topology table above is the argument that it is general rather than tuned to one model.

## Why it matters

- **SSMs run in linear time with constant memory** — ideal for enormous imaging sequences (a 384×320 slice is a 122k-token scan per direction) and for cost-effective CPU serving.
- **The scan is the hot path**, and in the ecosystem where research models actually live it has zero CPU optimization. `torch.compile` cannot close this gap: it cannot restructure a sequential recurrence.
- **Diffusion posterior sampling is the modern SOTA family for accelerated MRI**, assumed GPU-bound because the denoiser runs tens-to-hundreds of times per image. That multiplier is precisely where a fast CPU scan compounds — turning "diagnostic-quality reconstruction on a hospital CPU box or a cheap Graviton fleet" into a real deployment story.

## What's here

| Component | Description |
|---|---|
| Rust `selective_scan` kernel | Chunked two-pass scan + fused discretization, NEON intrinsics (stable Rust), specialized non-positive `exp` paths, rayon threading over independent channels; scalar reference path is the in-crate oracle, the non-Arm fallback, and what keeps x86 CI meaningful |
| Bidirectional path | `arm_scan.bidirectional_scan` emits **both time directions from one call**, computing the direction-independent Pass A (discretize + `exp`, ~85% of kernel time) **once** instead of twice. Measured **6.4–9× vs `torch.compile`** on Arm — the strongest result in the project |
| SS2D path | `arm_scan.ss2d.ss2d_scan` takes a `(B, D, H, W)` token grid and runs VMamba's 4 directions as **two traversal-order pairs** on that same fused bidirectional kernel, so Pass A is computed twice per block instead of four times, with no `torch.flip` copies. A single fused `selective_scan_2d` in Rust remains the identified next lever ([`docs/archive/TOPOLOGY_IMPLEMENTATION_PLAN.md`](./docs/archive/TOPOLOGY_IMPLEMENTATION_PLAN.md) §3.2) — currently **not** justified by its own >15% overhead rule, see below |
| Streaming state | `h0` in, `last_state` out — process an unbounded stream with **flat memory**, verified bit-exact through the C ABI |
| PyTorch custom-op bridge | `torch.library` op with registered fake kernel (composes with `torch.compile`), C-ABI FFI with panic containment, `arm_scan.patch()` for HF `transformers` Mamba |
| MRI diffusion app | `apps/mri_diffusion`: SS2D-Mamba denoiser under NVIDIA EDM preconditioning, built on UT CSI Lab's [`ambient-diffusion-mri`](https://github.com/utcsilab/ambient-diffusion-mri) scaffolding; full (R=1) and undersampled (R=2–8) posterior reconstruction on CPU |
| Correctness suite | Golden vectors vs. vendored upstream reference + an independent numpy re-derivation; NEON↔scalar parity; rayon output bit-identical to sequential at any thread count; goldens replayed through the real C ABI — **no dataset required**, runs on any arm64 machine, Apple Silicon included |
| Benchmark harness | Op-level and end-to-end, against **both** eager and `torch.compile`; medians after warmup, pinned threads/seeds, host- and SHA-tagged JSON; every kernel optimization logged with measured attribution in [`OPTIMIZATION_LOG.md`](./docs/archive/OPTIMIZATION_LOG.md) |

## Results so far (provisional — headline numbers land on Graviton)

Measured on shared GitHub `ubuntu-24.04-arm` CI runners (4-core, torch 2.13); provisional per [`BASELINE_TEST_PLAN.md`](./docs/archive/BASELINE_TEST_PLAN.md), to be replaced by dedicated Graviton (`c8g`) numbers in [`bench/results/RESULTS.md`](./bench/results/RESULTS.md):

| Benchmark | vs eager | vs `torch.compile` |
|---|---|---|
| Op-level, 1D unidirectional | 16.0× | **3.71×** |
| **1D bidirectional** (fused, Pass A shared) | 26.9–42.9× | **6.39–8.99×** |
| HF mamba-130m prefill, end-to-end (greedy tokens identical) | ~2× | — |

**The compile-time wall.** `torch.compile` unrolls the recurrence into an L-step graph, so its
*one-time compile* grows with sequence length — 59.9 s at L=256, 137.5 s at L=512, 254.3 s at
L=1024, **532.8 s at L=2048** — and L=4096/8192 had to be capped rather than measured. Our
kernel runs in constant memory at any L. That is the long-context argument as a measurement
rather than an assertion.

Ablation ladder on Neoverse-N2: NEON **4.03–4.08×** over scalar, threading **3.99× on 4
cores** (99.7% scaling efficiency). Single-core time is **53.7% `exp`**, 0.1% transpose —
see [`bench/ARM_FIRST_LOOK.md`](./bench/ARM_FIRST_LOOK.md).

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
[`TOPOLOGY_IMPLEMENTATION_PLAN.md`](./docs/archive/TOPOLOGY_IMPLEMENTATION_PLAN.md),
[`MRI_DIFFUSION_IMPLEMENTATION_PLAN.md`](./MRI_DIFFUSION_IMPLEMENTATION_PLAN.md).

## License

MIT — see [`LICENSE`](./LICENSE) (also set in the repository About sidebar, per contest rules).
