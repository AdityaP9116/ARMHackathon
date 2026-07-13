# APPLICATIONS — Brainstorm (nothing locked yet)

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
| **1D bidirectional** | Non-causal sequences: DNA, ECG, audio enhancement, sensor streams | Modest extension — forward pass plus a reversed pass. The two directions are independent, so it *adds* thread parallelism rather than costing any. |
| **2D cross-scan (SS2D)** | Vision Mamba: four traversals (rows fwd/back, cols fwd/back) over a patch grid | Real work. Also the one that **nobody has on CPU at all** — the biggest white space. |

There's a second axis worth holding in mind: **what makes the linear-time claim visceral.**

A transformer's KV cache grows with sequence length; Mamba's state does not. That argument is abstract at L=512 and undeniable at L=131,072. So at least one of the three should live at an absurd sequence length.

---

## Candidate applications

### Language — long-context prefill

**Topology:** 1D unidirectional  ·  **New kernel work:** none

mamba-130m through 2.8b via HF `transformers`, on CPU.

**Pitch:** dollars per million prompt-tokens on Graviton versus a GPU, with no KV cache to pay for.

Boring, and that is precisely its virtue. It is the "every Mamba model in the ecosystem gets faster" pillar, it needs zero new kernel work (`bench/bench_e2e.py` already runs it and asserts token-identical output), and a judge can reproduce it on their own laptop in five minutes.

**Effectively already in the bag.**

---

### Genomics — DNA foundation models

**Topology:** 1D bidirectional  ·  **New kernel work:** small

Caduceus-class models operate at **131k-token contexts**.

**Pitch:** at 131k tokens, a transformer on CPU is not slow — it is *impossible*. This is where the linear-time, constant-memory argument stops being a slide and becomes a knockout.

Genomics also genuinely cares about CPU inference: these are batch and offline pipelines running on commodity cloud fleets, not interactive GPU serving.

Concrete demo options: variant-effect prediction, or a promoter/enhancer classification sweep.

---

### Medical imaging — MRI reconstruction

**Topology:** 2D cross-scan (SS2D)  ·  **New kernel work:** significant

MambaRecon-style: undersampled k-space in, diagnostic image out, with a PSNR/SSIM/NMSE parity gate.

**Pitch:** a hospital could run this on a CPU box instead of renting GPUs.

Highest WOW on the list, and medical imaging *looks* good on video — which matters a lot for a 3-minute pitch.

Also the highest risk: SS2D has to be built, and the research repos bundle CUDA forks that must first be forced onto a CPU reference path.

---

### Audio — speech enhancement / separation

**Topology:** 1D bidirectional  ·  **New kernel work:** small (shared with genomics)

Noisy WAV in, clean WAV out.

**Pitch:** real-time factor — audio-seconds processed per wall-second. A gorgeous, intuitive metric.

This is the **only** demo where a judge can *hear* the result rather than read a number, which makes it a strong candidate purely on the grounds of being sensorially different from everything else.

Known risk: the audio Mamba checkpoints are research-grade and CUDA-coupled.

---

### Biosignals — ECG / EEG

**Topology:** 1D bidirectional  ·  **New kernel work:** small

Multi-hour recordings mean long sequences. Arrhythmia detection over a 24-hour Holter recording is a clean framing.

**Pitch:** the deployment story is *naturally* CPU — hospital servers and edge boxes, not GPU racks.

Weak spot: strong public checkpoints are scarce, which risks pulling us into training. The roadmap explicitly rules that out of scope.

---

### Time-series — anomaly detection / forecasting on telemetry

**Topology:** 1D  ·  **New kernel work:** none

**Pitch:** boringly commercial — which *is* the Impact argument. This is what cloud fleets actually run.

Millions of metric streams, embarrassingly parallel across channels, which happens to be exactly the dimension our rayon threading already exploits.

Lowest WOW on the list; highest "a real company would deploy this on Monday."

---

### RF / spectrum sensing

**Topology:** 1D  ·  **New kernel work:** none

SDR signal classification over long IQ streams.

Genuinely different field, very long sequences, and **Arm CPUs are the actual deployment target** in that world — which is a nice, on-theme argument to put in front of Arm engineer judges.

---

### Vision — plain image classification (VMamba / Vim)

**Topology:** 2D cross-scan  ·  **New kernel work:** significant (same as MRI)

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

**Downside:** SS2D is the only item on this entire list that could eat a week.

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

2. **How much SS2D work is really involved** once we look at a concrete model's forward pass? Four independent 1D scans over permuted views, or a fused traversal that needs its own kernel?

3. **Is one of the three allowed to be a quality-parity demo only** (no headline speedup claim), to keep the benchmark story focused?
