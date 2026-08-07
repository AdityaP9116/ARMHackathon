# THREE_PATHS_INTEGRATION — the three demonstrations, scoped

**Written Aug 7, 2026.** Execution plan for the three paths chosen in
[`MAMBA3_IMPLEMENTATION_PLAN.md`](MAMBA3_IMPLEMENTATION_PLAN.md) §0, now that the Mamba-3
kernel itself is complete and gated (Stages 0–5, M0–M8).

**The constraint that governs all three:** our Mamba-3 kernel runs only Mamba-3 models, and
**exactly one family has public weights** — `state-spaces/mamba3-*`. Everything else in the
Mamba ecosystem (VideoMamba, VMamba, PointMamba, Mamba-3VL, MM2CT) is an earlier generation
and belongs to the Mamba-1 kernel. That is why the three paths below are *architectures*
within one family rather than three unrelated models: it is the honest maximum.

| | Path A — 1D SISO | Path B — MIMO | Path C — 2D |
|---|---|---|---|
| Role | **credibility** | **second real-weights path** | **novelty** |
| Weights | ✅ 187M–1.5B | ✅ 187M/444M/894M | ❌ none exist |
| Oracle | ✅ official kernels | ✅ official kernels *(if capturable)* | ❌ **our own reference only** |
| Accuracy claim | ✅ token-level | ✅ | ❌ **impossible** |
| Speed claim | ✅ | ✅ | ✅ |
| Kernel work | none | **rank-r update** | seam widening |
| Risk | model plumbing | **TileLang on Blackwell** | no authoritative oracle |

---

## Path A — 1D SISO end to end ✅ *(Stage 6 — DONE, Aug 7 2026)*

**Goal:** run the real `state-spaces/mamba3-siso-187m` on Arm CPU through our kernel, and
prove it is the same model by matching logits.

**Status: complete and gated.** The 187M checkpoint loads into `apps/mamba3_lm/` and runs on
CPU with the recurrence on our kernel.

| Gate | Result |
|---|---|
| `tests/check_mamba3_block.py` — mixer vs the real block, real weights | **1.36 bf16 ULP** (bound 16) |
| `tests/check_mamba3_model.py` — logits vs the real model | **98.05%** argmax, drift 4.5e-3 of logit range |
| Reference vs **itself**, across processes | **98.83%** — the floor; see below |

Three findings worth carrying forward:

1. **There is no `conv1d` in Mamba-3.** Mamba-1 and Mamba-2 both open the mixer with a short
   depthwise convolution; Mamba-3 dropped it. The mixer has exactly **8** parameters. Do not
   add one back by analogy — the checkpoint has no weights for it.
2. **`A` is data-dependent**, not a learned `A_log`. It comes out of `in_proj` and through
   `heavy_tail`, so there is no `A` parameter to load.
3. **Ground truth is not reproducible across processes.** Two invocations of the *official*
   model disagree on up to 5/256 argmax positions at ~2.9e-3 relative logit drift, while two
   forwards *within* one process are bit-identical. That localises the cause to
   `triton.autotune` choosing a config by timing. **Our 98.05% therefore sits inside the
   reference's own noise band, not below it** — the gate cites the measured floor rather than
   comparing against an unattainable 100%.

The model gate is deliberately **not** in CI (357 MB download); `check_mamba3_block.py` is the
cheap proxy that runs on every push, carrying the real layer-0/1 weights inside the golden.

**Why it cannot be done by importing upstream:** `mamba_ssm.modules.mamba3` imports Triton /
TileLang / CuTe and asserts when they are missing; the package will not even install without
`nvcc`. So the block is reimplemented in plain PyTorch, and only the scan goes to our kernel —
exactly the discipline `python/arm_scan/patch.py` already follows for HF Mamba-1: *transcribe
the non-scan code verbatim, replace only the recurrence.*

### What has to be rebuilt, verified against the real checkpoint

`model_shape.json` (146 params) makes this mechanical rather than guesswork. Confirmed
arithmetic — `d_in_proj = 3432` reconciles exactly:

```
d_model 768 · expand 2 → d_inner 1536 · headdim 64 → nheads 24
d_state 128 · ngroups 1 · rope_fraction 0.5 → num_rope_angles 32

in_proj (3432, 768) splits as:
  z 1536 | x 1536 | B 128 | C 128 | dt 24 | A 24 | trap 24 | angles 32
```

Per layer: `norm` (RMSNorm 768) → mixer → `norm2` (768) → gated MLP
(`fc1` 3072 = 2×1536, `fc2` 1536→768). Top level: `embedding` (128256, 768), `norm_f`,
**tied** LM head.

Inside the mixer, in order:

1. `in_proj`, split as above.
2. `_A = -heavy_tail(dd_A)`, clamped to `≤ -A_floor` (1e-4), where
   `heavy_tail(x) = 1 + x` for `x ≥ 0` and `1 / (1 - x)` otherwise. **This is not a standard
   activation — transcribe it, do not substitute softplus.**
3. `DT = softplus(dd_dt + dt_bias)`; `ADT = _A * DT`.
4. `B_norm` / `C_norm` — RMSNorm over `d_state`, applied **before** the kernel.
5. The angle pre-pass (`arm_scan.angles_to_cos_sin`, already built).
6. **The scan** → `arm_scan.mamba3_scan`.
7. `out_proj`.

**Two traps already identified.** `B_bias`/`C_bias` are `(24, 1, 128)` — `(nheads, mimo_rank,
d_state)` — and must be squeezed to the kernel's `(heads, dqk)`; and the checkpoint calls them
`B_bias`/`C_bias` while the kernel signature calls them `Q_bias`/`K_bias`.

### Files *(all written)*

```
apps/mamba3_lm/block.py     the Mamba3 mixer in plain PyTorch, scan -> arm_scan
apps/mamba3_lm/model.py     embedding, layers, RMSNorm, gated MLP, tied head
apps/mamba3_lm/load.py      checkpoint -> our modules, driven by model_shape.json
tests/check_mamba3_block.py the mixer gate (cheap, in CI)
tests/check_mamba3_model.py the logits gate (needs the checkpoint)
bench/bench_mamba3_lm.py    prefill / long-context, vs eager and torch.compile
```

### Gates, and why they are shaped this way

- **Mixer vs the real block first.** An end-to-end logits mismatch says "something is wrong
  somewhere in twelve identical layers"; this says *which of the eight parameters or six steps*
  is wrong, on layer 0, in one forward. Proven to discriminate by negative control — injecting
  the three most likely bugs (B/C swapped, `Q_bias`/`K_bias` swapped, `heavy_tail` replaced by
  softplus) moves it from **0.75** ULP to **170.8 / 177.8 / 91.3**, all far outside the bound.
- **Logits gate on argmax + subset + logsumexp**, not on a raw logit tensor (131 MB).
- **The argmax test is on *explained* flips, not on a rate.** A bare agreement rate says
  nothing about whether flipped tokens were coin tosses or confident errors, and 98.05% against
  a 0.98 threshold is a coin flip away from red. The real test is that every disagreement has a
  top-2 margin the measured drift can account for: median margin at disagreeing positions is
  **0.0115** against **0.3809** at agreeing ones — a 33× gap. Unexplained flips: **0**.

### What it demonstrates

Not "N× faster" — **"this model cannot run on a CPU-only machine at all, and we made it run."**
The baseline is not upstream (there is no upstream CPU path); it is the PyTorch recurrence a CPU
user would have to write, benchmarked **in fp32, not the f64 oracle** — timing an f64 baseline
against an fp32 kernel would inflate every number here.

First measurements, `bench/bench_mamba3_lm.py`, 187M, 16 threads:

| L | our kernel | PyTorch eager | speedup | `torch.compile` | vs compiled | compile time |
|---|---|---|---|---|---|---|
| 128 | 125.7 ms | 341.6 ms | 2.72× | 174.5 ms | 1.39× | 35.5 s |
| 256 | 253.0 ms | 627.1 ms | 2.48× | 404.1 ms | 1.60× | 60.2 s |
| 512 | 389.2 ms | 719.7 ms | 1.85× | *(skipped)* | — | — |
| 1024 | 693.7 ms | 2538.8 ms | 3.66× | *(skipped)* | — | — |

**These are x86 numbers and are not the claim.** This box is not the target, not quiesced, and
the run-to-run spread (1.85× at L=512 against 3.66× at L=1024) is wide enough that only the
order of magnitude should be read. They also exercise the **scalar/blocked** path — NEON is not
compiled on x86 — so they are, if anything, a floor. Graviton is still the gap.

Compile time is itself part of the story and is why the compiled column stops at 256: 35.5 s →
60.2 s across one doubling of `L`, consistent with the Mamba-1 path's measured 59.9 s → 532.8 s
from L=256 → 2048. A sequential scan gives `torch.compile` a graph that grows with the sequence.

### Risks

All three risks flagged before the work turned out to be resolvable by reading, not guessing:

| Risk | How it actually resolved |
|---|---|
| `fused_add_norm` / residual-in-fp32 semantics differ subtly | Upstream ships the unfused branch too, and it is the same computation — `layer_norm_fn(..., prenorm=True)` is *defined* as `(norm(h+res), (h+res).float())`. Fusion is a memory optimisation. `residual_in_fp32=True` is kept because that one does change results |
| Tied embeddings applied wrongly | Not inferred — **verified**: the checkpoint's `lm_head.weight` and `backbone.embedding.weight` are the *same tensor object* (`is` → True). `load.py` refuses to drop the alias if they ever differ |
| Gated-MLP gate/value order flipped | Not a 50/50 after all — upstream is `y, gate = y.chunk(2, -1)` then `y * silu(gate)`, i.e. **value first**. Read, not guessed |

---

## Path B — MIMO *(kernel work; gated on a capture probe)*

**What MIMO is, concretely:** the axis our validation currently rejects.

```python
b, length, groups, dqk = q.shape
if groups != 1:                       # <- this axis IS mimo_rank
    raise ValueError("SISO expects one B/C group ...")
```

SISO updates the state with a **rank-1** outer product per step,
`S = α·S + scale·(v ⊗ k)`. MIMO uses **r** pairs — a rank-r update — with the input projected
down to `r` streams (`mimo_x`), a *shared* state, and projection back up (`mimo_o`).

**Why it is worth doing:** decode is memory-bandwidth bound — the whole 8–32 KB state is loaded
to perform very little arithmetic. MIMO does `r`× the work on the same load (~4× arithmetic
intensity), which is precisely the regime where a CPU is weakest. It is the Mamba-3 change most
likely to *help* us rather than merely be ported. And its weights already exist.

### Step B0 — capture probe *(1 hour, hard go/no-go, do this first)*

MIMO runs on **TileLang**, not Triton, and this box is consumer Blackwell. Open upstream issues:
**#994** (TileLang contiguity for MIMO), **#985** (MIMO fails at `seqlen=1`), **#990**
(consumer-Blackwell shared memory — already bit us once at `chunk_size ≥ 128`).

Probe: drive `mamba3_mimo_combined` at two shapes and capture inputs→outputs.
**If it does not run, Path B stops here** and the effort moves to Path C. Do not begin kernel
work on the assumption that ground truth will be available.

### Steps B1–B4 *(≈2.5 days, only if B0 is green)*

| | Work |
|---|---|
| **B1** | Extend the reference: `mamba3_ref.py` gains `rank`, state update sums `r` rank-1 terms. Gate: `r=1` reproduces the SISO goldens **bit-for-bit** — a free correctness check, exactly like λ=1 collapsing the trapezoid |
| **B2** | `Mamba3Dims` gains `rank`; validation accepts `groups == rank`; scalar, tiled and NEON inner loops sum over `r`. The tile loop is unchanged in shape — `r` is an outer accumulation |
| **B3** | **ABI 6 → 7** (new field in `ArmMamba3Dims`), Python loader bump, `mamba3_scan(rank=...)` |
| **B4** | `apps/mamba3_lm` gains the `mimo_x` / `mimo_z` / `mimo_o` projections; run `mamba3-mimo-187m` end to end |

**Gates:** `r=1` bit-identical to SISO; MIMO goldens at the bf16 bound; NEON↔scalar parity;
thread bit-identity; logits vs a MIMO `model_forward.npz`.

---

## Path C — 2D *(the novelty; ~2–3 days for the causal half)*

**No weights exist, and we are not training any.** That is a decision, not a gap — the plan's
strongest standing rule is *"do not train a model to justify the kernel"* (the DH-Mamba
lesson). A half-converged checkpoint produces visibly bad output and damages the demo.

**What that does and does not cost us.** Correctness and throughput are both
**weight-independent**: a scan over random weights performs identical arithmetic in an
identical memory pattern, and correctness is checked against a *reference implementation*, not
against task performance. So we can ship everything except an accuracy number.

| Claim | Available? |
|---|---|
| Kernel computes the right thing | ✅ |
| Throughput vs `torch.compile` at real vision grids | ✅ |
| **Causal vs non-causal comparison** | ✅ — **the novel result** |
| "Same accuracy, faster" / "we ran a real vision model" | ❌ |

### C1 — causal cross-scan *(the cheap half)*

`ss2d.py`'s `grid_to_views` / `views_to_grid` / merge are **pure layout ops with no recurrence
knowledge** — the best-designed extension point in the repo. The only change is widening the
`scan_pair` seam, whose signature is currently Mamba-1's parameter list, to a tensor-bundle
form (or a parallel `ss2d_scan_mamba3` sharing the same helpers).

Then `mamba3_scan_pair` — already built — supplies both traversal directions, and the four
directions are two pairs over `(rows, cols)`. **Zero new kernel code.**

**Gates:** per-direction 2D goldens (square, non-square, `H`/`W` not multiples of 4) generated
from our reference and independently re-derived in numpy; pair-vs-oracle parity at
`RAYON_NUM_THREADS ∈ {1,2,8}`.

### C2 — non-causal, VNCT-style *(the expensive half)*

Dropping causality removes the recurrence entirely: the intra-chunk term becomes **two dense
GEMMs** plus 2D RoPE. This is a *second kernel*, sharing packaging and tests but almost no
compute code, and `matrixmultiply` is currently only a transitive dev-dependency.

**Be honest about the moat here.** GEMMs are what BLAS and compilers are good at, so expect a
much thinner margin against `torch.compile` than the scan enjoys. That is the finding, not a
failure — and publishing both formulations side by side is what converts it into a result.

### The oracle caveat, stated plainly

For 1D we captured ground truth from official kernels. **For 2D nothing authoritative exists** —
VNCT's code is unreleased. We validate our Rust against *our own PyTorch reading of the paper*,
which proves the kernel implements the reference, **not** that the reference implements VNCT as
its authors intended. This must appear in the writeup, not be buried.

---

## Sequencing

| Order | Item | Why here |
|---|---|---|
| ~~1~~ | ~~**Path A**~~ | ✅ **done Aug 7** — the only accuracy evidence the project can produce |
| 2 | **B0 capture probe** | 1 hour, hard go/no-go, and it needs the GPU box — settle it before planning around it |
| 3 | **C1 causal 2D** | Smallest item, zero new kernel code, unblocks the headline claim |
| 4 | **B1–B4** *(if B0 green)* | Second real-weights path |
| 5 | **C2 non-causal** | Highest novelty, thinnest moat — last |

**Not sequenced here, and still the largest gap: dedicated-hardware numbers.** Every timing in
this repo is x86 or a shared 4-core runner. All five items above are laptop work; the Graviton
session is not, and none of them substitutes for it.

## What each path may claim

- **A:** first PyTorch-callable NEON Mamba-3 running the official checkpoint on Arm CPU, at
  token-level fidelity, constant memory in `L`.
- **B:** the first CPU MIMO Mamba-3 — and the arithmetic-intensity argument tested rather than
  asserted.
- **C:** first CPU implementation of any 2D Mamba-3, plus the causal-vs-non-causal comparison,
  which nobody has published for any Mamba generation. **Correctness and throughput only.**

Never: "first Mamba-3 on CPU" or "in Rust" (`mamba-rs`, `burn-mamba`, `mamba.c`); anything about
bidirectional Mamba-3 (`burn-mamba` ships it); any accuracy result for 2D.
