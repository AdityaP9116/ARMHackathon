# ARM AI Optimization Challenge — Project Concept

**Repo:** https://github.com/AdityaP9116/ARMHackathon
**Competition:** [Arm Create: AI Optimization Challenge](https://arm-ai-optimization-challenge.devpost.com/) — Deadline **Aug 14, 2026**
**Track:** Cloud AI
**Status:** Concept locked on **Arm-optimized selective-scan for MRI reconstruction**; specific model/checkpoint being finalized (see [Open Decisions](#open-decisions))

---

## One-line pitch

The efficient **selective-scan** at the heart of Mamba state-space models is **CUDA-only** — on CPU it falls back to a sequential loop that is ~70× slower than GPU. We write the **first Arm-optimized (NEON/SVE2) `selective_scan` kernel for CPU** and demonstrate it on **MRI reconstruction (fastMRI)**, making high-quality medical image reconstruction practical and cheap on Arm cloud servers (Graviton).

---

## The core strategy

Optimize the operation that actually matters — the selective scan — with **custom low-level Arm kernels**, proven by **measurable before/after benchmarks** on a real, high-value workload (MRI reconstruction). This maps directly onto the contest rubric:

- **Technological Implementation (40 pts)** — a hand-written Arm kernel for an op with *zero* existing CPU optimization.
- **Potential Impact (20 pts)** — a reusable kernel that benefits every Mamba-based model, contributed as a PyTorch custom op.
- **WOW factor (25 pts)** — "cost-effective medical image reconstruction on cheap Arm CPU."
- **Developer/User Experience (15 pts)** — clean repo, reproducible benchmarks, clear docs.

Guiding principles:

- **Optimize an operation, not a whole model.** Replace the slow `selective_scan` CPU fallback with a fast Arm kernel; leave the rest of the network untouched.
- **Benchmark religiously.** Same model, same server, our kernel vs. the stock CPU fallback, numbers via Arm Performix.
- **Ship a reusable artifact.** MIT/Apache, public repo, reproducible build/run steps.
- **Narrow the demo (MRI), keep the kernel general.** The kernel accelerates any Mamba model; MRI is the compelling proof point.

---

## Why state-space models + why MRI

**Why SSMs:** transformers scale *quadratically* in compute and their KV cache *grows with every token*, which is the killer on CPU. State-space models (Mamba) run in **linear time** with a **fixed-size memory state** — constant memory, cheap long context. Images and volumes are enormous sequences, so Mamba's linear-time long-range modeling is a natural fit and is now an active research direction for MRI.

**Why MRI reconstruction specifically:** it is a genuine, high-value signal-processing (DSP) problem where Mamba is being actively adopted (MambaRecon, DH-Mamba, DDMamba, PAS-Mamba, MambaRoll — all benchmarked on fastMRI, several with public code). It gives us a concrete, impressive demo and a clean quality metric (PSNR/SSIM/NMSE).

**The open lane:** the efficient `selective_scan` is CUDA-only. On CPU, `mamba-ssm` uses a slow sequential fallback (~70× slower than an A100). **There is no Arm-optimized CPU selective scan.** That absence is our contribution.

---

## Target: the selective-scan kernel

| Item | Decision |
|---|---|
| **Track** | Cloud AI |
| **Hardware** | AWS Graviton4 (Neoverse V2) — NEON + SVE2 (128-bit); spin up `c8g`/`r8g` |
| **Target op** | `selective_scan` (the recurrent SSM core in Mamba blocks) |
| **Runtime / integration** | PyTorch **C++ custom op / extension** replacing the `mamba-ssm` CPU fallback |
| **ISA** | NEON baseline → SVE2 path as stretch |
| **Language** | **Rust** — `core::arch::aarch64` NEON intrinsics (stable), bridged into the PyTorch op via a thin C-ABI shim (see [Kernel language](#kernel-language-rust)) |
| **Baseline** | Stock `mamba-ssm` sequential CPU `selective_scan` on the same Graviton instance |
| **Speed/mem metrics** | Op-level speedup; end-to-end reconstruction latency/throughput; peak memory; cost per volume |
| **Quality metrics** | PSNR / SSIM / NMSE on fastMRI vs. reference (correctness gate) |
| **FFT guardrail** | Do **not** optimize the FFT/k-space step — Arm FFT (ACL, pocketfft) is already fast. Treat it as a black box. |

---

## Model choice (Mamba-based MRI)

The kernel is model-agnostic; these differ mainly in maturity and whether a runnable checkpoint exists.

| Model | Task | Notable for | Code/weights | Risk |
|---|---|---|---|---|
| **MambaRecon** | Reconstruction | +0.72 dB PSNR over prior best on fastMRI | Public repo | Medium (verify CPU checkpoint) |
| **DH-Mamba** | Reconstruction | Dual-domain hierarchical SSM for MRI | Public repo | Medium |
| **DDMamba** | Reconstruction | Dual-domain + Fourier fusion, multi-modal | Paper (Wiley) | Medium/High |
| **U-Mamba / SegMamba** | Analysis (segmentation) | Mature medical-imaging Mamba, easier CPU inference | Public repo + some weights | Low (fallback option) |

**Recommended:** **Reconstruction as primary** (higher WOW, sharper DSP story) using MambaRecon or DH-Mamba — whichever has a checkpoint confirmed to run on CPU. **Segmentation (U-Mamba) is the safety fallback** if no recon checkpoint pans out. All routes exercise the same `selective_scan` kernel, so the core work is preserved either way.

---

## Kernel language (Rust)

We write the kernel in **Rust** for memory safety and as a differentiation/WOW angle — the surrounding ecosystem (`mamba-ssm`, PyTorch, ggml) is all C/C++, so a safe Rust kernel stands out. Performance is a wash: Rust `core::arch::aarch64` NEON intrinsics emit the same instructions as C.

**Toolchain reality:**

- **NEON = stable Rust** (`core::arch::aarch64`, stable since 1.61). The MVP kernel needs no nightly.
- **SVE2 = nightly-only** (intrinsics are perma-unstable). The stretch SVE2 path requires a nightly toolchain — acceptable, but documented.

**Path A — Rust kernel + thin C-ABI shim (default).** Compile the scan to a `cdylib` with `extern "C"`, and register it as the PyTorch custom op via a minimal C++/`pybind11` glue layer that passes raw tensor pointers + strides. Keeps the published MRI checkpoint and the reference PyTorch model untouched — we only swap the hot op, so quality-parity benchmarking stays trivial. Cost: an FFI boundary (dtype/stride/contiguity handling) to get right.

**Path B — full Rust inference in Candle (stretch).** HuggingFace's [Candle](https://github.com/huggingface/candle) has a CPU-first Mamba example and loads `safetensors` weights. Port the recon model into Candle and implement the scan as a native Candle op — an all-Rust, no-FFI, no-Python story with strong WOW. Cost: reimplement the model architecture in Rust and map checkpoint weights; higher risk of numerical mismatch muddying the quality metric. **Only attempt if the Path A MVP lands early.**

> **Fallback:** if the timeline tightens, plain C/C++ intrinsics remain the zero-FFI option, since ATen/`mamba-ssm` are already C++. Rust buys differentiation + safety at the cost of the bridge.

---

## The narrowed purpose

**Cheap, high-quality MRI reconstruction on Arm cloud CPU.** A Mamba reconstruction network takes undersampled (accelerated) k-space and recovers a diagnostic-quality image. Today that runs on expensive GPUs; on CPU it is impractically slow because the scan has no CPU kernel. Our Arm kernel closes that gap.

**Money-shot benchmark:** *Same Graviton instance — the stock CPU selective-scan takes T seconds per reconstruction; our Arm kernel takes T/X, at the same PSNR/SSIM, for a fraction of GPU cost.*

> **Scope guardrails:**
> - The contribution is the **kernel**, demonstrated via MRI — not a new model. Use a **published checkpoint**; do not start a full fastMRI training run (needs GPU + time and would eat the deadline).
> - Do **not** optimize the FFT. Keep the kernel work on the selective scan.

---

## What we ruled out (and why)

- **ESP32 sensor** — Xtensa, not Arm → ineligible.
- **Reimplementing BitNet/ternary NEON kernels** — already shipped in Microsoft's `bitnet.cpp`; the judges are Arm engineers who know it.
- **Two-track medical device system** — the two chips don't share code (Cortex-M has no NEON/SVE); straddling tracks halves depth.
- **Rust SVE2 for the MVP** — SVE2 intrinsics in Rust are nightly-only, so the MVP uses stable-Rust NEON; SVE2 is a documented nightly stretch (see [Kernel language](#kernel-language-rust)).
- **Training a model from scratch** — a separate, much larger project that would dilute the kernel contribution.
- **Optimizing the FFT** — already fast on Arm; not white space.

---

## Success criteria

1. A working, correct Arm `selective_scan` kernel exposed as a PyTorch custom op on Graviton.
2. A credible benchmark table: op-level and end-to-end speedup, memory, and cost vs. the stock CPU fallback (via Performix).
3. Reconstruction quality (PSNR/SSIM/NMSE) preserved within tolerance vs. reference.
4. Reproducible: public repo, MIT/Apache license, README with build/run/validate steps on Arm.
5. <3 min demo video showing MRI reconstruction running on the Graviton instance.

---

## Deliverables checklist

- [ ] Public repo with MIT or Apache 2.0 license (visible in About)
- [ ] Rust NEON `selective_scan` kernel (MVP) + optional SVE2 path on nightly (stretch)
- [ ] C-ABI shim + PyTorch custom-op wrapper replacing the CPU fallback (Path A)
- [ ] Benchmark harness (baseline fallback vs. optimized) with Performix numbers
- [ ] Quality-validation script (PSNR/SSIM/NMSE on a fastMRI sample)
- [ ] MRI reconstruction demo on Graviton
- [ ] README: overview, setup, run, validate, results table
- [ ] <3 min demo video (YouTube/Vimeo)
- [ ] Devpost writeup: overview, functionality, setup, why it wins

---

## Rough plan (MVP → stretch)

- **Step 0 (de-risk first):** confirm a Mamba MRI recon repo (MambaRecon / DH-Mamba) has a checkpoint that runs on CPU. If not, fall back to U-Mamba segmentation.
- **MVP:** Rust NEON `selective_scan` kernel bridged into the PyTorch op (Path A), beating the stock CPU fallback, with a benchmark + quality-parity table on one model.
- **+1:** Add the SVE2 path (nightly).
- **+2:** Show the kernel generalizes (second Mamba model, e.g. a text SSM or a segmentation model).
- **+3:** Polish the MRI demo, video, and writeup.

---

## Open decisions

1. **Reconstruction vs. analysis** — recon (recommended, higher WOW) vs. segmentation (safer).
2. **Specific model/checkpoint** — MambaRecon vs. DH-Mamba (pending CPU-checkpoint verification).
3. **Scope line** — how far past the NEON MVP we commit (SVE2 on nightly, second model, Candle/Path B all-Rust port).
4. **Compute for any training** — do we have GPU access if a from-scratch or fine-tune step becomes unavoidable? (Goal: avoid this entirely.)

---

## Reference links

- [Contest overview & rules](https://arm-ai-optimization-challenge.devpost.com/)
- [Track details](https://arm-ai-optimization-challenge.devpost.com/details/trackdetails)
- [MambaRecon: MRI Reconstruction with Structured State Space Models](https://arxiv.org/html/2409.12401v1)
- [DH-Mamba: dual-domain hierarchical SSM for MRI reconstruction (repo)](https://github.com/XiaoMengLiLiLi/DH-Mamba)
- [DDMamba: dual-domain Mamba for multi-modal MRI reconstruction](https://onlinelibrary.wiley.com/doi/10.1002/mrm.70148?af=R)
- [MambaRoll: physics-driven autoregressive SSM for medical image reconstruction](https://arxiv.org/html/2412.09331v1)
- [mamba-ssm selective_scan interface (CUDA-only op)](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/pdf/2312.00752)
