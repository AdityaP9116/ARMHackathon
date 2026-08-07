# APPLICATIONS — Brainstorm (nothing locked yet)

> **DECIDED (Jul 17, 2026):** the SS2D slot went to **Mamba-diffusion MRI reconstruction** (`MRI_DIFFUSION_IMPLEMENTATION_PLAN.md`); MambaRecon is the fallback. This file is the honest pre-decision brainstorm, kept as history. Decision log: `PROJECT_CONCEPT.md`.


**Status: open brainstorm, not a decision.** Written Jul 13, 2026.

The goal is to pick **three applications** that showcase the Arm selective-scan kernel across clearly different fields.

Nothing here has been verified for checkpoint availability or CPU viability yet — that gate comes after we agree on a shortlist.

> **Doc conflict to reconcile later.** `PROJECT_CONCEPT.md` currently locks MambaRecon/MRI as *the* application, while `INTEGRATION_PLAN.md` Phase 7 says the app is still open (it floats audio/ECG/RF). Whatever we choose here supersedes both, and those two docs get reconciled at that point.

---

## The axis worth optimizing for

Any three demos prove "it runs in three places."

Three demos that each exercise a **different scan topology** prove the *kernel itself* is general — which is the claim we actually want the judges to believe (Impact, 20 pts).

There are three topologies in the Mamba world:

| Topology | Where it shows up | Kernel status |
| --- | --- | --- |
| **1D unidirectional** | Causal language / audio generation | **Built.** This is what ships today. |
| **1D bidirectional** | Non-causal sequences: DNA, ECG, audio enhancement, sensor streams | **Correct today via rearrangement**; a fused in-kernel version is an optimization, not a prerequisite. |
| **2D cross-scan (SS2D)** | Vision Mamba: four traversals (rows fwd/back, cols fwd/back) over a patch grid | **Correct today via rearrangement**; the fused traversal is where the real Rust work (and the real white space) lives. |

See [What each topology actually costs](#what-each-topology-actually-costs) below — this is less work than it looks.

There's a second axis worth holding in mind: **what makes the linear-time claim visceral.**

A transformer's KV cache grows with sequence length; Mamba's state does not. That argument is abstract at L=512 and undeniable at L=131,072. So at least one of the three should live at an absurd sequence length.

---

## What each topology actually costs

**The important correction: bidirectional and SS2D are not new kernels.** Both are the *same* recurrence run over **rearranged views** of the data, so both are reachable today, correctly, with the op we already have. What we would actually build in Rust is the **fusion** — doing the rearrangement inside the kernel so we stop paying for the copies.

That splits every topology into two independent pieces of work:

| Topology | Get it **correct** | Make it **fast** |
| --- | --- | --- |
| 1D unidirectional | Done | Done (NEON + chunked + rayon) |
| 1D bidirectional | Python: `torch.flip` around the existing op, combine the two outputs. **Zero new Rust.** | Rust: a `reverse` flag so the kernel walks the sequence backward in place — kills two full-tensor copies. Half a day. |
| 2D cross-scan (SS2D) | Python: build the four permuted/flipped views of the patch grid, stack them into the batch dim, **one call to the existing op**, then merge. A day of plumbing. | Rust: a fused four-direction traversal that reads the grid once instead of materializing four copies. The week-long item — and the actual white space, since no CPU implementation of this exists anywhere. |

Two consequences worth internalizing:

**1. We can stand up all three applications end-to-end before touching the kernel again.** Correct output and honest speedup numbers for every app come first; *then* we optimize whichever topology shows the most headroom. The kernel work stops being a prerequisite and becomes a follow-on, which drains most of the risk out of the third slot.

**2. The story gets better, not worse.** "We built three kernels" is weak. "One kernel core, three scan topologies, and here is the fused-traversal work that made the 2D case fast" is what an Arm engineer judge would actually respect — it's a claim about generality *and* a claim about depth.

The one thing to watch: the rearrangement copies are not free. For SS2D the flip/permute traffic could plausibly dominate the scan itself at small resolutions, which would make the unfused version look unimpressive. That is a reason to measure early, not a reason to build the fused path first.

---

## Candidate applications

### Language — long-context prefill

**Topology:** 1D unidirectional  ·  **Kernel work:** none

mamba-130m through 2.8b via HF `transformers`, on CPU.

**Pitch:** dollars per million prompt-tokens on Graviton versus a GPU, with no KV cache to pay for.

Boring, and that is precisely its virtue. It is the "every Mamba model in the ecosystem gets faster" pillar, it needs zero new kernel work (`bench/bench_e2e.py` already runs it and asserts token-identical output), and a judge can reproduce it on their own laptop in five minutes.

**Effectively already in the bag.**

---

### Genomics — DNA foundation models

**Topology:** 1D bidirectional  ·  **Kernel work:** none to be correct; ~half a day to fuse the reverse pass

Caduceus-class models operate at **131k-token contexts**.

**Pitch:** at 131k tokens, a transformer on CPU is not slow — it is *impossible*. This is where the linear-time, constant-memory argument stops being a slide and becomes a knockout.

Genomics also genuinely cares about CPU inference: these are batch and offline pipelines running on commodity cloud fleets, not interactive GPU serving.

Concrete demo options: variant-effect prediction, or a promoter/enhancer classification sweep.

---

### Medical imaging — MRI reconstruction

**Topology:** 2D cross-scan (SS2D)  ·  **Kernel work:** ~a day of Python to be correct; ~a week to fuse the traversal

MambaRecon-style: undersampled k-space in, diagnostic image out, with a PSNR/SSIM/NMSE parity gate.

**Pitch:** a hospital could run this on a CPU box instead of renting GPUs.

Highest WOW on the list, and medical imaging *looks* good on video — which matters a lot for a 3-minute pitch.

Also the highest risk: SS2D has to be built, and the research repos bundle CUDA forks that must first be forced onto a CPU reference path.

---

### Audio — speech enhancement / separation

**Topology:** 1D bidirectional  ·  **Kernel work:** none (shares the genomics path)

Noisy WAV in, clean WAV out.

**Pitch:** real-time factor — audio-seconds processed per wall-second. A gorgeous, intuitive metric.

This is the **only** demo where a judge can *hear* the result rather than read a number, which makes it a strong candidate purely on the grounds of being sensorially different from everything else.

Known risk: the audio Mamba checkpoints are research-grade and CUDA-coupled.

---

### Biosignals — ECG / EEG

**Topology:** 1D bidirectional  ·  **Kernel work:** none (shares the genomics path)

Multi-hour recordings mean long sequences. Arrhythmia detection over a 24-hour Holter recording is a clean framing.

**Pitch:** the deployment story is *naturally* CPU — hospital servers and edge boxes, not GPU racks.

Weak spot: strong public checkpoints are scarce, which risks pulling us into training. The roadmap explicitly rules that out of scope.

---

### Time-series — anomaly detection / forecasting on telemetry

**Topology:** 1D  ·  **Kernel work:** none

**Pitch:** boringly commercial — which *is* the Impact argument. This is what cloud fleets actually run.

Millions of metric streams, embarrassingly parallel across channels, which happens to be exactly the dimension our rayon threading already exploits.

Lowest WOW on the list; highest "a real company would deploy this on Monday."

---

### RF / spectrum sensing

**Topology:** 1D  ·  **Kernel work:** none

SDR signal classification over long IQ streams.

Genuinely different field, very long sequences, and **Arm CPUs are the actual deployment target** in that world — which is a nice, on-theme argument to put in front of Arm engineer judges.

---

### Vision — plain image classification (VMamba / Vim)

**Topology:** 2D cross-scan  ·  **Kernel work:** same as MRI (shares the SS2D path)

The same SS2D work as MRI, but with a cleaner metric (top-1 accuracy) and far lower integration risk.

The story is duller, though: nobody is impressed by ImageNet on a CPU.

**Best thought of as the de-risked fallback for the MRI slot**, not as a first choice.

---

## Trios worth arguing about

### A. The topology ladder — *current favorite*

**Language (1D causal) → Genomics @131k (1D bidirectional) → MRI (2D cross-scan)**

Three fields, three scan shapes, one kernel core underneath all of them.

The narrative writes itself: each application forced the kernel to grow a new capability, and it did. That is a far better answer to "is this actually reusable?" than three parallel demos of the same code path.

**Upside:** strongest technical and impact story; hits an extreme sequence length; covers the one op (SS2D) that has no CPU implementation anywhere.

**Downside:** the *fused* SS2D traversal is the only item on this entire list that could eat a week. But per [What each topology actually costs](#what-each-topology-actually-costs), the correct-but-unfused version is a day, so this trio can be stood up end-to-end long before that week is spent — and if it never gets spent, we still have three working applications.

---

### B. The sensory trio

**Language → Audio enhancement → MRI**

Optimizes for the *video*: you read text, you hear denoised speech, you see a reconstructed scan. Three human senses in three minutes is a memorable pitch, and WOW is worth 25 points.

**Upside:** by far the most watchable submission.

**Downside:** weaker on the "linear time is magic" argument, since none of these reach extreme sequence lengths.

---

### C. The cloud-economics trio

**Language → Genomics → Time-series telemetry**

Every one of these is a real batch workload that someone is currently overpaying a GPU to run. Strongest Impact story, cleanest cost tables, and it is **all 1D** — meaning zero risky kernel work, and everything ships.

**Upside:** safest; nothing can slip.

**Downside:** the least visually exciting submission of the three. And this is a hackathon.

---

## Current instinct

**Trio A (the topology ladder), with Audio held as the swap-in if SS2D slips.**

Language and genomics are locked under every scenario — they are cheap, and they carry the ecosystem argument and the long-context argument respectively.

So the third slot is really a single bet:

| Third slot | The bet |
| --- | --- |
| MRI reconstruction | Highest ceiling |
| Audio enhancement | Most watchable |
| VMamba classification | De-risked SS2D |
| Time-series telemetry | Safest; ships no matter what |

---

## Open questions (deliberately not answered here)

1. **Do the genomics and vision checkpoints actually download and run on CPU**, and what exact scan does each one call (directions, `d_state`, `d_model`)? This is the same gate that killed DH-Mamba — a few hours of work, and cheap insurance.

2. **Does the rearrangement overhead swamp the unfused SS2D path?** Building the four permuted/flipped views costs real memory traffic, and at small patch grids it could plausibly dominate the scan itself — which would make the correct-but-unfused version look unimpressive. Measure this early; it decides whether the fused traversal is a nice-to-have or a must.

3. **Is one of the three allowed to be a quality-parity demo only** (no headline speedup claim), to keep the benchmark story focused?
