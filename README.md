# ARMHackathon — Arm-Optimized Selective-Scan for MRI Reconstruction

**Arm Create: AI Optimization Challenge 2026 — Cloud AI track**

The efficient **selective-scan** at the heart of Mamba state-space models is **CUDA-only**. On CPU, the reference `mamba-ssm` library falls back to a sequential loop that runs roughly **70× slower than GPU**. This project implements the **first Arm-optimized (NEON/SVE2) `selective_scan` kernel for CPU — written in Rust** — and demonstrates it on **MRI reconstruction (fastMRI)**, making high-quality medical image reconstruction practical and cheap on Arm cloud servers (AWS Graviton).

## Why it matters

- **State-space models** run in linear time with constant memory, unlike transformers whose KV cache grows with sequence length — ideal for the long sequences in medical imaging and for cost-effective CPU serving.
- The **selective-scan is the hot path** and has **no existing Arm CPU optimization** — that gap is this project's contribution.
- **MRI reconstruction** is a real, high-value DSP workload where Mamba is actively being adopted (MambaRecon, DH-Mamba, DDMamba), with a clean quality metric (PSNR/SSIM/NMSE on fastMRI).

## What's here

| Component | Description |
|---|---|
| Rust `selective_scan` kernel | NEON intrinsics (stable Rust); optional SVE2 path (nightly) |
| PyTorch custom-op bridge | Thin C-ABI shim registering the Rust kernel as a drop-in replacement for the CPU fallback |
| Benchmark harness | Baseline (stock CPU fallback) vs. optimized, with Arm Performix numbers |
| Quality validation | PSNR/SSIM/NMSE parity check on a fastMRI sample |
| MRI reconstruction demo | End-to-end reconstruction running on a Graviton instance |

## Status

Early development. See [`PROJECT_CONCEPT.md`](./PROJECT_CONCEPT.md) for the full plan, architecture decisions, model choices, and open questions.

## Setup

_Build/run/validate instructions on an Arm64 environment will be added as the kernel and harness land._

## License

MIT — see [`LICENSE`](./LICENSE).
