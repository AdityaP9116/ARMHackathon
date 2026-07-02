# ARM AI Optimization Challenge — Project Concept (v2)

**Repo:** https://github.com/AdityaP9116/ARMHackathon
**Competition:** [Arm Create: AI Optimization Challenge](https://arm-ai-optimization-challenge.devpost.com/) — Deadline **Aug 14, 2026, 4:00 PM PDT**
**Track:** Cloud AI
**Status:** Concept locked on **Arm-optimized selective-scan for MRI reconstruction**; model/checkpoint verification is the Week-1 gate (see [Open Decisions](#open-decisions))

---

## One-line pitch

Every Mamba research model — including every Mamba MRI-reconstruction model — lives in the PyTorch / `mamba-ssm` ecosystem, whose efficient **selective-scan is CUDA-only**; on CPU it collapses to an unoptimized sequential loop, and the **multi-directional 2D scans used by vision-Mamba models have no fast CPU implementation anywhere**. We ship the **first Arm-optimized (NEON, multi-threaded) selective-scan for the PyTorch ecosystem — written in Rust** — and prove it on **MRI reconstruction (fastMRI)**, making diagnostic-quality reconstruction practical and cheap on Arm cloud CPUs (AWS Graviton).

> **Claim discipline.** We deliberately do *not* claim "the first CPU selective scan, period" — llama.cpp/ggml has an `ssm_scan` for GGUF-ported language Mamba models, and the judges (Arm engineers) will know it. What genuinely does not exist, and what we build:
> 1. an optimized scan **for PyTorch/`mamba-ssm`**, where all research checkpoints actually live (no vision/medical Mamba has a GGUF port), and
> 2. optimized **bidirectional / four-directional 2D scans** (the VMamba-style `SS2D`/`cross_scan` pattern used by MRI recon models) — unsupported on CPU by anyone.
>
> This sharpened claim survives judge scrutiny and ties the kernel directly to the MRI demo.

---

## The core strategy

Optimize the operation that actually matters — the selective scan — with a **three-layer optimization story** (algorithm + SIMD + threading), proven by **fair, measurable before/after benchmarks** on a real, high-value workload (MRI reconstruction), and packaged so a judge can validate it in **five minutes on any arm64 machine**.

Rubric mapping:

- **Technological Implementation (40 pts)** — a hand-written Arm kernel for an op with no CPU optimization in its ecosystem; algorithmic (chunked scan) + SIMD (NEON) + parallel (multi-core) layers, each profiled with Arm Performix; positioned as a **KleidiAI-style micro-kernel** for the SSM scan.
- **Potential Impact (20 pts)** — a reusable, **pip-installable** kernel (`pip install`, prebuilt aarch64 wheels) that benefits every Mamba-based PyTorch model, plus a tokens/sec generalization benchmark on a language Mamba.
- **WOW factor (25 pts)** — "cost-effective medical image reconstruction on cheap Arm CPU," shown as a **live side-by-side demo** (stock fallback vs. our kernel reconstructing the same slice on the same Graviton instance) with a **dollars-per-1,000-slices** cost line vs. a GPU instance.
- **Developer/User Experience (15 pts)** — the judges are **Arm Developer Evangelists** who author Learning Paths for a living. We overdeliver here: Learning-Path-style docs, a zero-data 5-minute validation path, green arm64 CI badges, and first-run support on Apple Silicon.

Guiding principles:

- **Optimize an operation, not a whole model.** Swap the slow `selective_scan` CPU fallback for our kernel; leave the published network untouched so quality-parity benchmarking stays trivial.
- **Benchmark fairly and religiously.** Same model, same server, our kernel vs. the stock fallback *and* vs. a hardened baseline (`torch.compile` on the reference), so no judge can call it a strawman. All headline numbers measured by us, on our hardware, via Arm Performix — no borrowed figures.
- **Ship a reusable artifact.** MIT license, public repo, PyPI wheels, reproducible build/run/validate on any arm64 box.
- **Narrow the demo (MRI), keep the kernel general.** The kernel accelerates any Mamba model; MRI is the compelling proof point; a small language-Mamba tokens/sec table proves generality.

---

## Why state-space models + why MRI

**Why SSMs:** transformers scale quadratically in compute and their KV cache grows with every token — the killer on CPU. State-space models (Mamba) run in **linear time** with a **fixed-size memory state** — constant memory, cheap long context. Images and volumes are enormous sequences, so Mamba's linear-time long-range modeling is a natural fit and an active research direction for MRI.

**Why MRI reconstruction specifically:** a genuine, high-value signal-processing problem where Mamba is being actively adopted (MambaRecon, DH-Mamba, DDMamba, PAS-Mamba, MambaRoll — all benchmarked on fastMRI, several with public code), with clean quality metrics (PSNR/SSIM/NMSE).

**The open lane (sharpened):** research Mamba models are locked into `mamba-ssm`, whose CPU path is an unoptimized sequential Python-level loop; vision-Mamba 2D scans have **no** fast CPU implementation at all. We close that gap for Arm — the platform where CPU inference is cheapest.

---

## Target: the selective-scan kernel

| Item | Decision |
|---|---|
| **Track** | Cloud AI |
| **Benchmark hardware** | AWS Graviton4 (Neoverse V2, NEON + SVE2-128, BF16) — `c8g`; rented only for final numbers |
| **Dev hardware (free)** | Oracle Cloud Always-Free Ampere A1 (Neoverse N1, NEON) + GitHub Actions free arm64 runners + Apple Silicon if available (see ROADMAP.md) |
| **Target op** | `selective_scan` (the recurrent SSM core), incl. the bidirectional/4-directional 2D variants used by vision-Mamba blocks |
| **Runtime / integration** | PyTorch **C++ custom op / extension** that drop-in replaces the `mamba-ssm` CPU fallback |
| **Optimization layers** | (1) **Algorithmic:** chunked/associative scan — the linear recurrence is associative (the Mamba-2/SSD insight), turning the time dimension from latency-bound to throughput-bound; (2) **SIMD:** NEON across `d_state` (typically 16 → four `float32x4` lanes) and channels, with discretization (`exp(Δ·A)`, `Δ·B·x`) **fused into the scan loop** to cut memory traffic; (3) **Parallel:** rayon threading across batch × channel groups with a published **core-scaling curve** (1→64 cores) — the Cloud-track story |
| **Precision** | fp32 default (quality gate); documented BF16-storage/fp32-accumulate experiment on Neoverse V2 as stretch |
| **ISA** | NEON baseline (stable Rust) → SVE2 path as documented nightly stretch |
| **Language** | **Rust** — `core::arch::aarch64` NEON intrinsics (stable since 1.61), bridged via a thin C-ABI shim (see [Kernel language](#kernel-language-rust)) |
| **Baselines (two)** | (a) stock `mamba-ssm` sequential CPU fallback; (b) the same reference under `torch.compile` — beating a hardened baseline by less is worth more than beating a strawman by more. Plus a written note on why ggml's `ssm_scan` is inapplicable to this model class |
| **Speed/mem metrics** | Op-level speedup across sequence lengths; end-to-end reconstruction latency/throughput; core-scaling curve; peak memory; **$ per 1,000 slices** on `c8g` vs. a GPU instance |
| **Quality metrics** | PSNR / SSIM / NMSE on fastMRI vs. reference output (correctness gate: parity within tolerance) |
| **Profiling** | Arm Performix (Arm Performance Studio, free) — the contest page explicitly promotes it; include profile screenshots showing where cycles went before/after |
| **FFT guardrail** | Do **not** optimize the FFT/k-space step — Arm FFT is already fast. Black box. |

---

## Model choice (Mamba-based MRI)

The kernel is model-agnostic; candidates differ in maturity, checkpoint availability, and **scan structure** (which now matters, since 2D-scan coverage is part of the headline claim).

| Model | Task | Notable for | Code/weights | Scan structure | Risk |
|---|---|---|---|---|---|
| **MambaRecon** | Reconstruction | +0.72 dB PSNR over prior best on fastMRI | Public repo | Verify (likely SS2D-style) | Medium (verify CPU checkpoint) |
| **DH-Mamba** | Reconstruction | Dual-domain hierarchical SSM | Public repo | Verify | Medium |
| **DDMamba** | Reconstruction | Dual-domain + Fourier fusion | Paper (Wiley) | Verify | Medium/High |
| **U-Mamba / SegMamba** | Segmentation | Mature, easier CPU inference | Public repo + weights | Uni/bi-directional | Low (fallback) |

**Selection criteria (Week-1 gate):** (1) checkpoint downloads and produces a reference reconstruction on CPU; (2) scan structure our kernel can cover cleanly (uni/bi/4-dir, `d_state`, `d_model`); (3) fastMRI eval script works. **Reconstruction stays primary** (higher WOW, sharper DSP story); **U-Mamba segmentation is the safety fallback**. All routes exercise the same kernel, so the core work is preserved either way.

**Known trap:** `mamba-ssm` (and `causal-conv1d`) historically **fail to pip-install without CUDA** — the build compiles CUDA extensions. Verify on arm64 in Week 1. If true, it strengthens the story ("our package makes Mamba models runnable on CPU at all") but changes the integration design: we either patch `mamba-ssm` or ship a standalone importable module the model code falls back to.

---

## Kernel language (Rust)

Rust for memory safety and differentiation — the surrounding ecosystem (`mamba-ssm`, PyTorch, ggml) is all C/C++; a safe Rust kernel stands out, and performance is a wash (`core::arch::aarch64` NEON intrinsics emit the same instructions as C).

**Toolchain reality:**

- **NEON = stable Rust** (`core::arch::aarch64`, stable since 1.61). MVP needs no nightly.
- **SVE2 = nightly-only** (perma-unstable intrinsics). Stretch path only, documented as such.

**Path A — Rust kernel + thin C-ABI shim (the plan).** Compile the scan to a `cdylib` with `extern "C"`; register it as a PyTorch custom op via minimal C++/`pybind11` glue passing raw tensor pointers + strides. The published MRI checkpoint and the reference PyTorch model stay untouched — we only swap the hot op, so quality-parity benchmarking stays trivial. Cost: an FFI boundary (dtype/stride/contiguity handling) to get right — mitigated by a contiguity-normalizing wrapper on the Python side and property-based tests on the Rust side.

**Path B — full Rust inference in Candle: CUT.** Reimplementing the model in Candle risks numerical mismatch that muddies the quality gate, and the FFI story is already differentiated. Revisit only if the MVP lands weeks early (it won't).

> **Fallback:** if the timeline tightens, plain C/C++ intrinsics remain the zero-FFI option. Rust buys differentiation + safety at the cost of the bridge.

---

## Developer experience plan (judges are evangelists — overdeliver here)

- **PyPI package with prebuilt aarch64 wheels** (via `maturin`): `pip install <name>` and any Mamba PyTorch model gets a fast CPU scan. This is the reusable artifact for the Impact criterion.
- **Learning-Path-style README**: numbered *setup → run → validate → results*, the exact format the judges author at learn.arm.com. One prize is being featured on the Arm Community Blog — write docs that could be that post.
- **Zero-data 5-minute validation path**: `make validate` runs correctness tests vs. `selective_scan_ref` on random tensors + a microbenchmark on **any arm64 machine — including a judge's Apple Silicon MacBook**. No fastMRI credentials, no AWS account, no downloads.
- **arm64 CI**: GitHub Actions free arm64 runners for public repos run the correctness suite on every push — green badges on real Arm hardware.
- **fastMRI redistribution trap handled**: fastMRI data cannot be redistributed, so (a) correctness/benchmarks need no data; (b) the MRI demo includes a **synthetic Shepp–Logan phantom k-space generator** (fully shareable) plus one-page fastMRI access instructions; (c) the results table and demo video carry the real-data proof so judges never have to reproduce it.

---

## The narrowed purpose

**Cheap, high-quality MRI reconstruction on Arm cloud CPU.** A Mamba reconstruction network takes undersampled (accelerated) k-space and recovers a diagnostic-quality image. Today that runs on expensive GPUs; on CPU it is impractically slow because the scan has no optimized CPU kernel. Our Arm kernel closes that gap.

**Money-shot benchmark:** *Same Graviton instance — the stock CPU selective-scan takes T seconds per reconstruction; our kernel takes T/X at identical PSNR/SSIM, for $A per 1,000 slices vs. $B on a GPU instance.* Delivered as a **live side-by-side Gradio demo** (undersampled input left, reconstruction filling in right, latency + cost counters) — the centerpiece of the 3-minute video.

> **Scope guardrails:**
> - The contribution is the **kernel**, demonstrated via MRI — not a new model. Use a **published checkpoint**; no fastMRI training runs.
> - Do **not** optimize the FFT.
> - Measure the "~70× slower than GPU" folklore ourselves; only our own numbers go in the writeup.

---

## What we ruled out (and why)

- **ESP32 sensor** — Xtensa, not Arm → ineligible.
- **Reimplementing BitNet/ternary NEON kernels** — already shipped in Microsoft's `bitnet.cpp`; judges know it.
- **Claiming "first CPU selective scan, period"** — ggml/llama.cpp has one for GGUF language models; claim sharpened to PyTorch ecosystem + 2D scans instead.
- **Two-track medical device system** — the chips don't share code; straddling tracks halves depth.
- **Rust SVE2 for the MVP** — nightly-only; documented stretch.
- **Candle full-Rust port (old Path B)** — numerical-mismatch risk threatens the quality gate; cut.
- **Training a model from scratch** — dilutes the kernel contribution.
- **Optimizing the FFT** — already fast on Arm; not white space.

---

## Success criteria

1. A working, correct Arm `selective_scan` kernel (incl. the demo model's 2D scan variant) exposed as a PyTorch custom op, validated bit-tolerance-close to `selective_scan_ref`.
2. A credible benchmark suite: op-level speedup across sequence lengths, end-to-end recon latency, **core-scaling curve**, peak memory, and **$/1,000 slices** vs. the stock fallback *and* a `torch.compile` baseline, with Performix profiles.
3. Reconstruction quality (PSNR/SSIM/NMSE) preserved within tolerance vs. reference on fastMRI.
4. Reproducible + reusable: public repo, MIT license (visible in the GitHub About sidebar — a stated rules requirement), PyPI aarch64 wheels, arm64 CI, zero-data 5-minute validation on any arm64 machine.
5. Generalization proof: tokens/sec before/after on a small language Mamba (e.g., mamba-130m) — the contest page lists tokens/sec as a metric they look for.
6. <3 min demo video (no copyrighted music, shows the project running on the actual instance) with the live side-by-side reconstruction.

---

## Deliverables checklist

- [ ] Public repo, MIT license in `LICENSE` **and** the GitHub About sidebar
- [ ] Rust NEON `selective_scan` kernel (MVP): chunked scan + fused discretization + rayon threading
- [ ] 2D/bidirectional scan variant matching the demo model's block structure
- [ ] C-ABI shim + PyTorch custom-op wrapper replacing the CPU fallback (with contiguity/stride handling)
- [ ] Correctness suite: unit + property-based tests vs. `selective_scan_ref` (random tensors, zero data needed)
- [ ] arm64 CI (GitHub Actions) running the correctness suite
- [ ] Benchmark harness: stock fallback + `torch.compile` baselines, seq-length sweep, core-scaling curve, Performix profiles
- [ ] Quality-validation script (PSNR/SSIM/NMSE on fastMRI sample) + synthetic Shepp–Logan phantom demo path
- [ ] $/1,000-slices cost comparison (c8g vs. GPU instance)
- [ ] Generalization benchmark: mamba-130m tokens/sec before/after
- [ ] PyPI package with prebuilt aarch64 wheels (`maturin`)
- [ ] Gradio side-by-side demo on Graviton
- [ ] Learning-Path-style README: overview, setup, run, validate, results
- [ ] <3 min demo video (YouTube/Vimeo, no copyrighted audio)
- [ ] Devpost writeup: overview, functionality, setup, why it wins
- [ ] SVE2 nightly path *(stretch — ranked below PyPI wheels and the tokens/sec table, which score on more criteria)*

---

## Open decisions

1. **Specific model/checkpoint** — MambaRecon vs. DH-Mamba, decided by the Week-1 gate (CPU checkpoint runs + scan structure coverable). U-Mamba fallback pre-verified in parallel.
2. **Integration shape** — patch `mamba-ssm` vs. standalone importable package, decided by the Week-1 CUDA-install finding.
3. **Stretch ordering** — locked as: PyPI wheels → tokens/sec generalization → BF16 experiment → SVE2. Candle port is cut.

---

## Reference links

- [Contest overview & rules](https://arm-ai-optimization-challenge.devpost.com/)
- [Track details](https://arm-ai-optimization-challenge.devpost.com/details/trackdetails)
- [MambaRecon: MRI Reconstruction with Structured State Space Models](https://arxiv.org/html/2409.12401v1)
- [DH-Mamba (repo)](https://github.com/XiaoMengLiLiLi/DH-Mamba)
- [DDMamba: dual-domain Mamba for multi-modal MRI reconstruction](https://onlinelibrary.wiley.com/doi/10.1002/mrm.70148?af=R)
- [MambaRoll](https://arxiv.org/html/2412.09331v1)
- [mamba-ssm selective_scan interface (CUDA-only op)](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py)
- [Mamba paper](https://arxiv.org/pdf/2312.00752)
- [Mamba-2 / SSD (associative-scan formulation)](https://arxiv.org/abs/2405.21060)
- [Arm Performance Studio / Performix](https://developer.arm.com/Tools%20and%20Software/Arm%20Performance%20Studio)
- [KleidiAI](https://gitlab.arm.com/kleidi/kleidiai)
