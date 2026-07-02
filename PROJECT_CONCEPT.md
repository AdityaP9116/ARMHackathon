# PROJECT CONCEPT — Decisions & Rationale

**Repo:** https://github.com/AdityaP9116/ARMHackathon
**Competition:** [Arm Create: AI Optimization Challenge](https://arm-ai-optimization-challenge.devpost.com/) — Deadline **Aug 14, 2026**
**Track:** Cloud AI

> This document is the **decision log** — what we chose, what we rejected, and why.
> For the pitch and deliverables see [`README.md`](./README.md); for the build/test plan and schedule see [`ROADMAP.md`](./ROADMAP.md). This file avoids duplicating those.

---

## The decision in one paragraph

We ship the **first Arm-optimized `selective_scan` for the PyTorch/`mamba-ssm` ecosystem, written in Rust** (chunked/associative scan + NEON + multi-core threading), packaged as a **reusable, pip-installable kernel** that accelerates *any* Mamba model on Arm CPU, and we prove it on **MRI reconstruction**. The efficient selective-scan is CUDA-only; on CPU it falls back to an unoptimized sequential loop, and the multi-directional 2D scans used by vision-Mamba models have no fast CPU path anywhere. That absence is the contribution.

Why this scores: it's a hand-written Arm kernel for an op with *zero* existing CPU optimization (Tech, 40), a reusable artifact that benefits the whole ecosystem (Impact, 20), "cheap medical image reconstruction on CPU" (WOW, 25), with reproducible free-tier validation and CI (DX, 15).

---

## Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Track | Cloud AI | Graviton CPU serving; "cheap long-context/large-image inference on Arm" story |
| Target op | `selective_scan` (1D core) + VMamba-style 2D cross-scan (SS2D) | The hot path; CUDA-only today; the 2D variant is what the MRI model actually calls |
| Language | Rust — stable `core::arch::aarch64` NEON | Memory-safe, differentiating; same instructions as C, zero perf penalty |
| Integration | Rust `cdylib` (C-ABI) → `pybind11` glue → PyTorch custom op + Python shim | Swaps the op without touching checkpoint or model source |
| **Primary model** | **MambaRecon** (WACV 2025) | **Has published pretrained checkpoints** — see verification below |
| Fallback model | U-Mamba (segmentation) | Only if the recon checkpoint path fails; kernel work is unchanged |
| Correctness gate | Diff vs. the model's own reference scan (`*_ref`) | Makes correctness mechanical, not judgment |
| Baselines | Stock CPU fallback **and** `torch.compile` | Pre-empts the "you beat a strawman" critique |
| SVE2 | Stretch only, on nightly Rust | SVE2 intrinsics are nightly/perma-unstable; NEON MVP stays on stable |
| Training | None | Use a published checkpoint; training is out of scope |

---

## Model verification (checked Jul 2, 2026)

**MambaRecon — VIABLE, selected as primary.** Official WACV 2025 repo ([yilmazkorkmaz1/MambaRecon](https://github.com/yilmazkorkmaz1/MambaRecon)), MIT-licensed. Ships **pretrained checkpoints** (Google Drive) and the IXI dataset link, so we skip training entirely. Bundles its own CUDA `causal-conv1d` + `mamba` (installed via `setup.py`) and is ~21% CUDA / 8% C++ — so it will **not** run on CPU as-shipped. That is not a blocker: it is exactly the gap we fill. CPU path = force the pure-PyTorch reference scan as the slow baseline → confirm quality parity → swap in the Rust kernel.

**DH-Mamba — REJECTED.** ([XiaoMengLiLiLi/DH-Mamba](https://github.com/XiaoMengLiLiLi/DH-Mamba)) Has **no pretrained checkpoints** ("No releases published"), README states the code is still being "sorted"/under review, and it carries the same CUDA-only VMamba dependency without weights to offset it. Using it would force a from-scratch training run (GPU + time we won't spend).

---

## Open technical flags (resolve in Week 1)

These came out of the model check and the review of the current README/ROADMAP:

1. **Use the *right* reference for the parity gate.** MambaRecon is built on VMamba and bundles its **own forks** of `mamba`/`causal-conv1d`. Ground-truth must be *that* model's reference scan (VMamba SS2D reference), not vanilla `mamba-ssm`'s `selective_scan_ref` — otherwise the parity check compares against the wrong math.
2. **Confirm the exact op path.** VMamba may use a *fused* CUDA cross-scan kernel (harder to fall back to on CPU) rather than a plain 1D `selective_scan` called 4×. Pin down directions, `d_state`, `d_model`, and whether a clean pure-PyTorch reference exists for the fused variant.
3. **Dataset must match the checkpoint.** MambaRecon's checkpoint/commands are **IXI (brain)**, not fastMRI. Run the quality-parity gate on whatever data the checkpoint was trained on (likely IXI); don't assume fastMRI, or PSNR/SSIM will look broken. The synthetic Shepp–Logan phantom path stays as the shareable, no-credentials demo.
4. **Don't hard-depend on Oracle's free A1.** The Always-Free Ampere A1 is often un-provisionable (regional capacity). Keep GitHub Actions arm64 + Apple Silicon as the real daily driver so dev is never blocked.

---

## What we ruled out (and why)

- **ESP32 sensor** — Xtensa, not Arm → ineligible.
- **Reimplementing BitNet/ternary NEON kernels** — already shipped in Microsoft's `bitnet.cpp`; the judges are Arm engineers who know it.
- **Two-track medical device system** — the two chips don't share code (Cortex-M has no NEON/SVE); straddling tracks halves depth.
- **DH-Mamba as the model** — no pretrained weights, incomplete code (see verification).
- **Training a model from scratch** — a separate, much larger project that would dilute the kernel contribution.
- **Optimizing the FFT / k-space step** — already fast on Arm (Arm Compute Library, pocketfft); not white space. Treat as a black box.
- **Rust SVE2 for the MVP** — nightly-only; MVP is stable-Rust NEON, SVE2 is a documented stretch.

---

## Success criteria

1. Rust `selective_scan` (1D + 2D cross-scan) exposed as a PyTorch custom op, correct vs. the model's reference within fp32 tolerance.
2. End-to-end MRI reconstruction on Graviton with our op swapped in, at quality parity (PSNR/SSIM/NMSE), measurably faster than both the stock fallback and `torch.compile`.
3. Reusable, pip-installable artifact that also generalizes (e.g. mamba-130m tokens/sec).
4. Reproducible: public repo, MIT license, arm64 CI, free-tier `make validate` needing no dataset or AWS account.
5. <3 min demo video showing the side-by-side reconstruction on Graviton.

---

## Reference links

- [Contest overview & rules](https://arm-ai-optimization-challenge.devpost.com/) · [Track details](https://arm-ai-optimization-challenge.devpost.com/details/trackdetails)
- [MambaRecon repo](https://github.com/yilmazkorkmaz1/MambaRecon) · [paper (WACV 2025)](https://arxiv.org/abs/2409.12401)
- [DH-Mamba repo (rejected)](https://github.com/XiaoMengLiLiLi/DH-Mamba)
- [VMamba (SS2D cross-scan)](https://github.com/MzeroMiko/VMamba)
- [mamba-ssm — `selective_scan` (CUDA-only op)](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/pdf/2312.00752)
