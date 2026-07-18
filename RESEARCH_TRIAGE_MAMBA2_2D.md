# RESEARCH_TRIAGE_MAMBA2_2D — external research review, verified & triaged

**Written Jul 18, 2026.** A user-provided research survey ("Advanced ARM NEON Kernel
Optimization for Mamba-2 Architectures in 2D Processing Domains via Rust") proposed a set of
techniques. This document (1) verifies every substantive reference, (2) corrects where the
survey's framing doesn't match this repo's actual kernel, and (3) triages each technique into
adopt / already-planned / rejected-with-reasons / unverified — with concrete deltas to
[`SS2D_REPOSITIONING_PLAN.md`](SS2D_REPOSITIONING_PLAN.md) where something changes.

**The one-sentence outcome:** the survey validates the plan we already have (cache-blocked
tiling, kernel fusion, static chunk tuning) and contributes two concrete implementation details
(a `vld4q`-based 4×4 tile transpose for P1-6, and a CHUNK retune for P1-5) plus several strong
new citations for the "measured rejections" section of the writeup — but its two headline
architectural recommendations (VSSD non-causal core, Mamba-2 SSD block structure) would
**dissolve this project's moat**, not enhance it, and are rejected for the submission with
reasoning below.

---

## 1. Reference verification

Checked Jul 18, 2026. Several are past the assistant's training window and were verified by search.

| Reference | Status | What it actually is |
|---|---|---|
| Mamba-2 / SSD (Dao & Gu) | **Real** | State-space duality; chunked block-matmul formulation; scalar-times-identity A |
| VMamba, Vim, PlainMamba, LocalMamba, ZigMa | **Real** | Multi-directional / trajectory 1D-scan vision Mambas |
| [VSSD](https://arxiv.org/abs/2407.18559) | **Real** | Non-causal SSD for vision: drops causal mask, single global hidden state |
| [2DMamba](https://arxiv.org/abs/2412.00678) (CVPR 2025) | **Real** | Intrinsic 2D selective SSM with a hardware-aware **GPU** operator (2D tiling in SRAM); giga-pixel WSI |
| [EfficientViM / HSM-SSD](https://arxiv.org/abs/2411.15241) (CVPR 2025) | **Real** | Channel mixing moved into the compressed hidden-state space |
| [Mamba2D](https://arxiv.org/abs/2412.16146), [V2M](https://arxiv.org/abs/2410.10382) | **Real** | Natively 2D recurrences (state depends on both row- and column-neighbors) |
| [COREY](https://arxiv.org/abs/2604.10597) | **Real — but see §2** | Entropy-guided runtime chunk scheduling for selective-scan kernels |
| [FairyFuse](https://arxiv.org/abs/2604.20913) | **Real** | Fused ternary (multiplication-free) LLM inference kernels on CPU |
| BitMamba-2 (NEON port) | **Real** | Already in our prior-art table (README) |
| [BAST-Mamba](https://arxiv.org/abs/2207.03927), AuM, ConMamba | **Real** | Audio-domain Mambas — not our application |
| "Dual-Axis SSD Scanner (DASS)" | **Unverified** | Could not locate under this name; nearest real work is dual-path/tri-axis video SSDs (e.g. SurgicalMamba). Treat the specific claims as unsourced. |
| "State Space Conditioning (SSC)" (σ-gated A matrix) | **Unverified** | Could not locate as a named, published technique. The idea (condition A on timestep) is plausible but treat as speculative. |
| "2D-CrossScan framework" (cross-directional subtraction, corner-initiated multi-path) | **Unverified as named** | Resembles a mixture of real ideas (VMamba CSM, V2M); no single citable source found. |

Rule per `CLAUDE.md` claims policy: nothing from the Unverified rows gets cited or built on.

---

## 2. Corrections — where the survey doesn't match this repo

These matter because applying "Mamba-2 kernel advice" naively to this codebase would target the
wrong math.

**(a) Our kernel is a Mamba-1-style selective scan, not SSD.** The survey's central object is
Mamba-2's SSD block (scalar-times-identity A, chunked BMM decomposition). This repo implements
the **diagonal-A selective scan** (`d_state=16` per channel) — which is what VMamba-style SS2D,
MambaRecon, and our locked MRI backbone (`mamba_ss2d.py`) actually use. The four-phase SSD
chunking pipeline the survey describes is not our kernel's structure and can't be bolted onto a
diagonal-A model without retraining on a different architecture. (§4 discusses whether we'd
*want* the SSD form on CPU: no — see the rejection.)

**(b) COREY's own conclusion is the opposite of the survey's pitch.** The survey sells
entropy-guided chunk scheduling as a latency win. The paper itself reports that when routed into
a live scan kernel, **"the best static chunk outperforms all entropy-guided and proxy
schedulers"** end-to-end; entropy scheduling only matched a static oracle at kernel level. The
actionable content for us is therefore *static chunk tuning* (§3, A2) — which also avoids
data-dependent control flow that would complicate our rayon bit-identity guarantee.

**(c) "Mamba is entirely memory-bandwidth bound" is not what our profiler measured.** At the 1D
language shapes (L≤2048), our kernel is **compute-bound on transcendentals** — exp+softplus+SiLU
≈ 85% of single-thread time (`OPTIMIZATION_LOG.md`). The survey's claim becomes true for us only
at the SS2D diffusion shapes (L≈123k, B/C planes ~7.9 MB × 2, multi-core) — which is exactly why
P1-5 cache-blocking is next in the plan. The right posture is "verify the regime, per shape,"
not a blanket assumption; §3 A4 adds the roofline check.

**(d) The Hillis-Steele in-register prefix sum solves a problem our kernel doesn't have.** The
survey proposes `vextq`+`vaddq` lane-scans to parallelize the recurrence across *time* lanes.
Our Pass B deliberately vectorizes across *state* (16 lanes in four q-registers), leaving time
scalar — and the recurrence is ~10.5% of single-thread runtime. A lane-wise affine scan
(composing `(a,b)` pairs) would add multiplies to save a fraction of an already-small phase; it
is the in-register cousin of the Blelloch-over-chunks idea already rejected in
`IMPROVEMENT_IDEAS.md` §4.3. Not adopted; noted as a possible latency-hiding experiment only if
Graviton profiling shows Pass B FMA-latency-bound at high core counts (§3, A5).

**(e) `vld3q/vld4q` de-interleaving mostly targets interleaved (RGB) layouts we don't have** —
all tensors here are planar. But `vld4q_f32`'s "load 4 vectors + de-interleave" behavior *is* a
4×4 transpose in one instruction family, which is directly useful — see A1.

---

## 3. Adopt — concrete, mapped to the existing plan

**A1. `vld4q_f32`-based 4×4 tile transpose for P1-6 (`transpose.rs`).** The topology plan
specified "vtrn/vzip-style shuffles"; the survey's pointer is better: `vld4q_f32` loads four
consecutive vectors and de-interleaves in one go, so a 4×4 f32 tile transpose is
`vld4q_f32` + four `vst1q_f32` (or `vst4q` on the store side for the inverse). Fewer shuffle
µops than an explicit vtrn/vzip ladder, and cleanly cache-blockable into the 32×32 tiles P1-5
wants. → Design note recorded for P1-6; no plan reorder.

**A2. Static CHUNK retune at SS2D shapes (fold into P1-5).** `CHUNK = 128` was sized for L≤2048
1D scans to keep ~17 KB of scratch in L1. At L=123k with streamed B/C planes, the L1 working set
per chunk step is different (plane rows stream through, scratch competes with them). COREY's
finding says a well-chosen *static* chunk captures ~all of the available win — so sweep
`CHUNK ∈ {64,128,256,512}` at the real shapes on arm64 as part of the P1-5 cache-blocking work
(one criterion run per value; pick per-shape or single best; document in `OPTIMIZATION_LOG.md`).

**A3. 2DMamba's tiling precedent, cited for the P1-7 fused-kernel design.** 2DMamba's operator
tiles the 2D grid into SRAM-resident blocks and runs horizontal+vertical scans locally with
carries aggregated across tiles — on GPU. It's the closest published validation of exactly the
"tile into fast memory, scan both axes, carry across tiles" structure our fused
`selective_scan_2d` plans for L1. Adopt as a **citation and design cross-check** (their tile
size is chosen so H×W tile + carries fit fast memory; ours should be derived from L1d the same
way). Note our semantics differ — VMamba cross-scan is four *flattened* 1D scans, not their
intrinsic 2D recurrence — so their algorithm doesn't transplant; only the blocking discipline
does.

**A4. Roofline check at SS2D shapes (extends `PROFILING.md`).** One profiler run at
`B=4(dirs)·seeds, D=96, L=122880`: phase split + a simple achieved-GB/s estimate against the
runner's STREAM-ish ceiling. Decides whether P1-5's win is latency (cache misses in Pass B) or
bandwidth (plane streaming), which changes what P1-7 should fuse first. Cheap: the profiling
workflow already exists.

**A5. (Conditional) Pass-B latency-chain experiment.** Only if A4 on Graviton shows Pass B
stalled on FMA latency at high core counts: interleave 2 channels per Pass-B loop (2 independent
h-register sets) to hide the 4-cycle FMA chain — `IMPROVEMENT_IDEAS.md` §3.3, unchanged. The
survey's lane-scan variant stays rejected per §2(d).

**A6. README prior-art table: add 2DMamba.** It's the strongest "someone did fast 2D scans —
on GPU" row; listing it (and noting SRAM tiling as GPU-side precedent our CPU work parallels)
strengthens, not weakens, the "no fast **CPU** SS2D exists" claim and preempts a judge finding
it independently. One row, same format as the others.

---

## 4. Rejected for the submission — with the reasoning on record

These are the survey's two headline recommendations, and both would un-make the project.

**R1. VSSD / NC-SSD as the core operator.** Real paper, real results — and a trap for *this*
project. VSSD removes the causal mask and collapses the per-token recurrence into a single
global aggregation: mathematically that turns the sequential scan into dense matmul-shaped work.
On CPU, dense matmuls are already served by oneDNN/ACL and `torch.compile` handles them fine —
**the entire moat of this project is that a sequential recurrence is the one thing
`torch.compile` cannot restructure.** Adopting VSSD would (a) delete the op our kernel exists to
accelerate, (b) move the workload onto ground where mature BLAS libraries already win, and
(c) orphan the drop-in story for the existing VMamba-family ecosystem (checkpoints are causal
cross-scan models). Correct uses instead: one honest paragraph in the Devpost/`RESULTS.md`
("why not just go non-causal?" — because we accelerate the models that exist), and a
future-work note. If time permits, a measured VSSD-style global-aggregation row in the
benchmark table would be a strong "we checked the alternative" flex — a candidate for
`IMPROVEMENT_IDEAS.md` §7.7's alternative-baselines list.

**R2. Mamba-2 SSD block structure (chunked BMM) on CPU.** Already flagged as a probable
documented rejection in `IMPROVEMENT_IDEAS.md` §7.1; the survey strengthens the case for
*writing the rejection well* rather than adopting: SSD's 2–8× wins come from feeding tensor
cores with BMMs — a hardware class CPUs don't have (NEON FMA throughput is the same units either
way). Our diagonal-A scan at `d_state=16` keeps h in four registers and streams everything else;
the SSD form would add FLOPs to buy parallelism we already get across channels via rayon. And
architecturally: the app's backbone (and every SS2D checkpoint in the wild) is diagonal-A —
adopting SSD means retraining onto a different block, spending the schedule's scarcest resource.
Keep §7.1 as measure-and-reject if time allows; otherwise cite this analysis.

**R3. Ternary/sub-byte quantization (FairyFuse-style, BitMamba-2-style).** Real and impressive —
and orthogonal to the deadline: it requires a quantized checkpoint (ours is an fp32 distilled
prior gated on PSNR/SSIM parity) and a second numerics story. Stays a post-hackathon direction;
BitMamba-2 already sits in the prior-art table, FairyFuse can join the future-work paragraph.

**R4. Entropy-guided scheduling (COREY).** Rejected on the paper's own end-to-end evidence
(§2b); its static-tuning corollary is adopted as A2. Data-dependent chunk sizes would also
threaten the rayon bit-identity property that our test suite treats as a hard guarantee.

**R5. New scan trajectories (ZigMa zigzag, corner-initiated multi-path, etc.).** Already
excluded by the plan ("no new scan pattern before Aug 14"); the backbone recipe is locked and
trained (or in training). Trajectory changes are model changes, not kernel changes.

**R6. σ-conditioned A / "SSC", DASS decomposition.** Unverified as published work (§1). The
underlying idea overlaps with our existing adaLN σ-conditioning; modulating A directly would be
an interesting *training-side* ablation someday, but nothing in it changes the kernel, and we
don't build on unsourced claims.

**R7. HSM-SSD (EfficientViM).** Real, but it's a backbone architecture change (channel mixing in
hidden-state space) with no effect on the scan kernel's contract, and our backbone is locked and
mid-training. Future-work: a "backbone v2" that pairs HSM-style projection thinning with this
kernel would be a legitimately novel CPU-efficiency paper — after Aug 14.

---

## 5. Future-work paragraph (for the Devpost writeup, when we get there)

The kernel's natural roadmap after the challenge, in order of leverage: (1) a native 2D
recurrence entry point serving the intrinsic-2D generation of models (Mamba2D, V2M,
2DMamba-style semantics — wavefront parallelism over anti-diagonals is new kernel surface no
CPU has); (2) a VSSD/NC-SSD comparison study — causal-scan CPU kernel vs. non-causal matmul
formulations at matched quality; (3) sub-byte paths (BitMamba-2 / FairyFuse-style ternary
`vbslq` selection) for edge Arm targets; (4) Mamba-3 support as its block structure stabilizes
in the ecosystem. Each reuses this repo's correctness harness (goldens, parity gates,
bit-identity threading) unchanged — which is the quiet argument that the harness, not any one
kernel, is the durable contribution.

---

## 6. Net effect on the active plan

`SS2D_REPOSITIONING_PLAN.md` §5 ordering is **unchanged** — the survey independently converges
on the same top items (cache-blocked tiling = P1-5, fusion = P1-7). Deltas: A1 (vld4q design
note) and A2 (CHUNK sweep) fold into P1-6 and P1-5 respectively; A4's roofline run precedes
P1-5; A6 adds one README row; R1/R2 give `RESULTS.md` and the writeup their
"alternatives considered" section. Nothing new jumps the queue, which is itself a useful
result: the plan survives contact with a fresh literature sweep.
