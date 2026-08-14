# PROJECT CONCEPT — Decisions & Rationale

**Repo:** https://github.com/AdityaP9116/Arm-Scan
**Competition:** [Arm Create: AI Optimization Challenge](https://arm-ai-optimization-challenge.devpost.com/) — Deadline **Aug 14, 2026**
**Track:** Cloud AI

> This document is the **decision log** — what we chose, what we rejected, and why.
> For the pitch and deliverables see [`README.md`](../../README.md); for the historical build/test plan and schedule see [`ROADMAP.md`](../archive/ROADMAP.md). This file avoids duplicating those.

---

## AMENDED Aug 6–7, 2026 — the Mamba-3 pivot happened

**This supersedes the Aug 4 scope below, including its "Rejected" row on Mamba-3.**
That row said pivoting "trades a measured submission for an unmeasured one" and left the
door open with: *"a half-day spike may show our existing scan can serve the Mamba-3 SISO
recurrence with only a 2-tap change to Pass A."*

The spike happened, and it was **half right in a way worth recording**. The 2-tap is indeed
one extra term and Pass B's *equation* is unchanged — but Mamba-1's state is 16 floats living
in four NEON registers, which is the entire reason that kernel is fast, while Mamba-3's is a
`dv x dqk` **matrix** (8–32 KB). It could not be a flag on the existing scan; it needed its
own kernel family. See `MAMBA3_KERNEL_WORKPLAN.md` §14.2.

**What changed the decision:** a prior-art sweep (verified repo by repo, Aug 6) moved two of
the three demo slots.

| Slot | Aug 4 plan | What the sweep found | Now |
|---|---|---|---|
| 1D bidirectional | Speech enhancement (SEMamba) | SEMamba is *outer* bidirectional — separate weights per direction — so our fused kernel **cannot help it**. VideoMamba later turned out identical. And `burn-mamba` already ships bidirectional Mamba-3 | **No novelty claim.** Kept as a near-free capability |
| 2D cross-scan | Diffusion MRI | The app's quality gate has never passed and there is no trained prior. Meanwhile **no CPU implementation of any 2D Mamba-3 exists anywhere** | **2D Mamba-3 is the headline.** MRI **demoted, not deleted** — still CI-gated as the only end-to-end SS2D exercise |
| 1D unidirectional | Long context | Unchanged, and now stronger: Mamba-3 has published weights *and* an authoritative oracle | **The evidence track** |

**Locked as of Aug 7:** two kernel families in one library (Mamba-1 shipped and measured;
Mamba-3 shipped and gated to 4.47 bf16 ULP against ground truth captured from the official
GPU kernels). They are **not** cross-compatible — Mamba-1 uses a per-state-element decay
vector, Mamba-3 a scalar decay per head.

**Also rejected on the way, with reasons:** the Mamba-2/SSD substrate route (target Mamba-3
directly instead — the substrate detour buys correctness gates we already have); MIMO and the
chunked dual form (the dual form is GEMM-shaped and would trade away the sequential-recurrence
moat); and a resumable Mamba-3 decode path, which is **structurally impossible via a carry** —
`scale_t` depends on `dt_{t+1}`, so the trapezoid looks forward.

**Standing caution:** three separate research digests handed to this project asserted Mamba-3
connections that did not exist (MFil-Mamba, Akasha 2, a VNCT code release). Verify against the
repo or the paper before a claim enters a document.

---

## Superseded — Final scope (Aug 4, 2026) — three topologies, three demos

**This supersedes the paragraph below**, which locked MRI as *the* single application. It
does not change the kernel work or the claims; it changes what we demonstrate them on.

`APPLICATIONS.md` made the argument and we are now acting on it: *"Three demos that each
exercise a different scan topology prove the **kernel itself** is general — which is the
claim we actually want the judges to believe (Impact, 20 pts)."*

| Topology | Demo | Why this one |
|---|---|---|
| 1D unidirectional | **Long context** on a language model | A *capability* claim, not a speed delta: constant memory at any L, while `torch.compile`'s compile time goes 59.9 s → 532.8 s from L=256 → 2048 and had to be capped past that. It is also Mamba's actual reason to exist. |
| 1D bidirectional | **Speech enhancement** | Our **strongest-measured** topology (6.39–8.99× vs `torch.compile` on Arm) had **no application at all**. It is also the only demo a judge can *hear*, its metric (real-time factor) needs no explanation, and Arm CPUs are literally where speech enhancement ships — earbuds, hearing aids, phones. |
| 2D cross-scan | **Diffusion MRI reconstruction** | Already built and gated. Highest ceiling, and the workload where kernel wins compound (18–256 denoiser calls per image). |

**Rejected, with reasons:**

- **Pivoting the kernel to Mamba-3.** Checkpoints *do* now exist (`state-spaces/mamba3-*`,
  arXiv 2603.15569 — the "no public checkpoints" line in `docs/roadmap/MAMBA3_KERNEL_PLAN.md`
  is out of date), and there is still **no CPU implementation anywhere**, so it is genuinely
  attractive. But it is a different recurrence, and with 10 days left and zero dedicated-Arm
  numbers banked, it trades a measured submission for an unmeasured one. It stays the
  roadmap — strengthened by the fact that the SSD substrate is 90% of it and the λ=1 / r=1
  reductions give free correctness gates against what we already verify. **Open:** a
  half-day spike may show our existing scan can serve the Mamba-3 SISO recurrence with only
  a 2-tap change to Pass A, which would make it cheap enough to reconsider.
- **VMamba classification as the 2D demo.** Cleaner metric, lower integration risk — kept
  explicitly as the de-risked substitute if MRI quality does not materialise.
- **Text generation as the 1D demo.** Same model, weaker framing. Long context is the same
  work with a far better story.

## The decision in one paragraph

**(Amended Jul 17, 2026 — see "Prior-art verification" and the amended rows below.)** We ship the **first Arm-optimized `selective_scan` for the PyTorch ecosystem, written in Rust** (chunked/associative scan + NEON + multi-core threading), packaged as a **reusable, pip-installable kernel** — and we double down on the **SS2D multi-directional cross-scan**, the one variant with no fast CPU implementation anywhere, proven on **diffusion-based MRI reconstruction** (EDM prior + SS2D-Mamba denoiser, per `MRI_DIFFUSION_IMPLEMENTATION_PLAN.md`). 1D language Mamba on CPU is contested space (llama.cpp, BitMamba, Rust engines); the PyTorch drop-in and SS2D are not. Diffusion recon invokes the scan hundreds of times per image, which is the regime where kernel wins compound — and it directly attacks "diffusion recon must be GPU-bound."

Why this scores: a hand-written Arm kernel for an op with *zero* existing CPU implementation (Tech, 40), a reusable artifact that benefits the whole PyTorch Mamba ecosystem (Impact, 20), "diffusion-prior medical image reconstruction on a CPU" (WOW, 25), with reproducible free-tier validation and CI (DX, 15).

---

## Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Track | Cloud AI | Graviton CPU serving; "cheap long-context/large-image inference on Arm" story |
| Target op | `selective_scan` (1D core) + VMamba-style 2D cross-scan (SS2D) | The hot path; CUDA-only today; the 2D variant is what the MRI model actually calls |
| Language | Rust — stable `core::arch::aarch64` NEON | Memory-safe, differentiating; same instructions as C, zero perf penalty |
| Integration | Rust `cdylib` (C-ABI) → `pybind11` glue → PyTorch custom op + Python shim | Swaps the op without touching checkpoint or model source |
| **Primary application** | **SS2D-Mamba diffusion MRI recon** (EDM + `ambient-diffusion-mri` scaffolding) — *amended Jul 17, 2026; Phase A gate passed (GO), see `apps/mri_diffusion/PHASE_A_FINDINGS.md`* | Diffusion's 18–256 denoiser calls/image maximize kernel leverage; strictly bigger WOW; exercises the same SS2D surface |
| Fallback model | MambaRecon (WACV 2025, published checkpoints) | Superseded as primary by the diffusion plan; remains the fallback if the prior-training route slips (verification below still valid) |
| Correctness gate | Diff vs. the model's own reference scan (`*_ref`) | Makes correctness mechanical, not judgment |
| Baselines | Stock CPU fallback **and** `torch.compile` | Pre-empts the "you beat a strawman" critique |
| SVE2 | Stretch only, on nightly Rust | SVE2 intrinsics are nightly/perma-unstable; NEON MVP stays on stable |
| Training | ~~None~~ → **Reopened (Jul 17, 2026):** bounded distillation/small-scale training for the diffusion prior (Route A recommended: distill Tamir's U-Net EDM checkpoint into the SS2D-Mamba; Route B small-scale fallback) | No public Mamba-backbone EDM MRI checkpoint exists; `MRI_DIFFUSION_IMPLEMENTATION_PLAN.md` §8 scopes the routes and the explicit GPU budget decision |

---

## Prior-art verification (checked Jul 17, 2026) — what we may and may not claim

A thorough novelty check (web + issue trackers) found real prior art for **1D language Mamba on CPU/Arm**: llama.cpp/ggml has a CPU `ssm_scan` (Mamba and Mamba-2, GGUF runtime, partially vectorized — RISC-V vector landed 2026); BitMamba-2 runs 1.58-bit Mamba-2 on ARM NEON in a custom C++ engine; mamba.rs / Candle / flawedmatrix are standalone Rust engines; mamba.py/mamba-mini are pure-PyTorch parallel scans. **Therefore: never claim "first Mamba on Arm CPU" or "first Mamba in Rust."**

What survives, verified: **no NEON/SIMD-optimized `selective_scan` exposed to PyTorch exists** (drop-in, no model conversion — nothing comparable found); **no fast CPU SS2D cross-scan exists anywhere** (VMamba ships CUDA-only; edge efforts are FPGA accelerators or distill-to-ONNX workarounds); **no diffusion-Mamba model has a CPU deployment path** (DiM/ZigMa/DiffuSSM are GPU-only in practice). These three are the claims, stated with "to the best of our knowledge," with the prior-art table published in the README so judges see we did the search.

**Consequences for positioning:** SS2D + the diffusion application are the moat and get the engineering focus; 1D language rows (mamba-130m) are kept as *generality* evidence, not the headline.

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
