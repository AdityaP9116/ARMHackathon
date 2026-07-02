# ARMHackathon — Arm-Optimized Selective-Scan for MRI Reconstruction

**Arm Create: AI Optimization Challenge 2026 — Cloud AI track**

Mamba state-space models are taking over long-sequence workloads, but every research checkpoint lives in the PyTorch / [`mamba-ssm`](https://github.com/state-spaces/mamba) ecosystem, where the efficient **selective-scan is CUDA-only**. On CPU it falls back to an unoptimized sequential loop — and the **multi-directional 2D scans** used by vision-Mamba models (including every Mamba MRI-reconstruction network) have **no fast CPU implementation anywhere**.

This project ships the **first Arm-optimized selective-scan for the PyTorch ecosystem — written in Rust** — combining a chunked (associative) scan, NEON SIMD, and multi-core threading, and proves it on **MRI reconstruction (fastMRI)**: diagnostic-quality reconstruction, practical and cheap, on Arm cloud CPUs (AWS Graviton).

## Why it matters

- **State-space models** run in linear time with constant memory, unlike transformers whose KV cache grows with sequence length — ideal for the enormous sequences in medical imaging and for cost-effective CPU serving.
- **The selective scan is the hot path**, and in the ecosystem where research models actually live it has zero CPU optimization. That gap is this project's contribution — packaged as a reusable, pip-installable kernel that benefits *any* Mamba PyTorch model, not just ours.
- **MRI reconstruction** is a real, high-value DSP workload where Mamba is actively adopted (MambaRecon, DH-Mamba, DDMamba), with clean quality metrics (PSNR/SSIM/NMSE on fastMRI) — so speedups can be verified at *identical output quality*.

## What's here

| Component | Description |
|---|---|
| Rust `selective_scan` kernel | Chunked/associative scan + fused discretization, NEON intrinsics (stable Rust), rayon multi-threading; 1D and vision-style 2D/bidirectional variants; optional SVE2 path (nightly) |
| PyTorch custom-op bridge | Thin C-ABI shim registering the Rust kernel as a drop-in replacement for the CPU fallback |
| Correctness suite | Unit + property-based tests vs. the reference `selective_scan_ref` on random tensors — **no dataset required**; runs on any arm64 machine, Apple Silicon included |
| Benchmark harness | Stock CPU fallback **and** `torch.compile` baselines vs. optimized: seq-length sweep, core-scaling curve, peak memory, Arm Performix profiles |
| Quality validation | PSNR/SSIM/NMSE parity check on fastMRI, plus a fully shareable **synthetic Shepp–Logan phantom** demo path (fastMRI data cannot be redistributed) |
| MRI demo | Side-by-side reconstruction (stock vs. optimized) with latency and cost counters, running on a Graviton instance |

## Quick validation (≈5 minutes, any arm64 machine, no data downloads)

> Placeholder — lands with the kernel. The intent is fixed:

```bash
git clone https://github.com/AdityaP9116/ARMHackathon && cd ARMHackathon
make validate   # builds the kernel, runs correctness tests vs. selective_scan_ref,
                # then prints an op-level microbenchmark table for this machine
```

Works on AWS Graviton, Oracle Ampere, Raspberry Pi 5, and Apple Silicon Macs. Headline benchmark numbers in [`RESULTS.md`] are measured on AWS Graviton4 (`c8g`).

## Results (to be filled — format locked)

| Benchmark | Stock CPU fallback | torch.compile | **This kernel** | Speedup |
|---|---|---|---|---|
| Op-level, L=4096 | — | — | — | — |
| End-to-end recon / slice | — | — | — | — |
| PSNR / SSIM (parity gate) | ref | — | — | Δ ≤ tolerance |
| $ / 1,000 slices (c8g vs GPU) | — | n/a | — | — |
| mamba-130m tokens/sec (generality) | — | — | — | — |

## Status

Early development. See [`PROJECT_CONCEPT.md`](./PROJECT_CONCEPT.md) for the full plan and architecture decisions, and [`ROADMAP.md`](./ROADMAP.md) for the build/test plan and week-by-week schedule.

## Setup / Run / Validate

_Learning-path-style numbered instructions (setup → run → validate → results) land with the kernel and harness. Correctness validation will never require fastMRI credentials or an AWS account._

## License

MIT — see [`LICENSE`](./LICENSE) (also set in the repository About sidebar, per contest rules).
