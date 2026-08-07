# MAMBA3_IMPLEMENTATION_PLAN — a CPU/Arm kernel for Mamba-3, all three topologies

**Aug 6, 2026. 8 days to the deadline.** This is the plan of record for the
decision to target Mamba-3 specifically, rather than retrofitting the existing
Mamba-1 kernel. It supersedes the Mamba-3 sections of
[`SUBMISSION_ENDGAME_PLAN.md`](SUBMISSION_ENDGAME_PLAN.md).

---

## 0. What we are building, and why it is less work than it sounds

**Target:** a Rust selective-scan kernel implementing **Mamba-3 SISO**, exposed
as a PyTorch custom op, covering all three scan topologies, running the official
`state-spaces/mamba3-siso-187m` checkpoint on Arm CPU.

Two structural facts make this tractable:

**The recurrence is our Pass B plus one term.** Mamba-3's "three-term
recurrence" is:

```
h_t = α_t·h_{t-1} + β_t·(B_{t-1}x_{t-1}) + γ_t·(B_t x_t)
```

Group the last two: `h_t = α_t·h_{t-1} + b̄_t`, where `b̄_t` is a **2-tap
convolution over the Bx stream**. Our kernel already computes exactly
`h = ābar⊙h + b̄` — Pass B is **unchanged**. All the new work lands in Pass A,
which is pointwise in time and therefore fully vectorisable.

**The topology layer does not know about the recurrence.** `ss2d_scan` →
`scan_pair` → `bidirectional_scan` → *a scan primitive*. Add a Mamba-3
primitive and bidirectional + SS2D compose on top. "All three topologies" is
not 3× the work.

**Not in scope:** MIMO (rank-*R*, different state shape), the SSD/dual form,
and training. SISO inference only.

---

## Stage 0 — Ground truth ✅ **DONE (Aug 6, 2026)**

> **Delivered:** 10 cases across 7 output shapes (L ∈ {1, 63, 64, 128, 255,
> 256, 2048}, batch 1–2, d_state 64/128) in `tests/golden/mamba3/`, plus
> `model_forward.npz` and `model_shape.json`; 19 MB; replay verified with
> numpy alone on a machine with no torch/mamba_ssm/CUDA.
>
> **Two findings that change later stages.** (a) `mamba-ssm` must be installed
> **from git main, not PyPI** — the release both rejects the published Mamba-3
> checkpoints and predates PR #997, which fixes *silent* forward-pass
> corruption on Blackwell. (b) The kernel is **mixed precision with no flag**:
> `Q/K/V/Trap/Angles/Z` → bf16, `ADT/DT` fp32 "for stability",
> `Q_bias/K_bias/D` fp32 as parameters, output bf16. Stages 2–3 must mirror
> that split, and Stage 1's tolerance follows from it (below). Full detail in
> [`HANDOFF.md`](HANDOFF.md).

Nothing downstream can start without a trustworthy oracle, and **we do not have
one**. `mamba_ssm/modules/mamba3.py` has no CPU path, and the community
`mamba3-pytorch` implements a *different* recurrence — measured 1.06 max_abs
disagreement with the paper on O(1) states, with the structural difference
(decay on the previous term) accounting for 0.363 on its own. No gate
remapping reconciles them.

**Run:** `tools/capture_mamba3_goldens.py` on the 5090. It wraps the official
kernel entry point (discovered by searching `mamba_ssm`, not hardcoded),
records inputs → outputs as `.npz`, and saves full-model logits for a fixed
prompt.

**Risk:** the kernels are CuTe/TileLang and the repo says *"only tested on
H100."* A 5090 is Blackwell. **Fallback:** Colab A100/L4.

**Exit gate:** ≥6 golden cases with inputs and outputs, plus `model_forward.npz`.
Commit them. **The GPU is never needed again.**

---

## Stage 1 — Paper-faithful reference *(CPU, ~½ day)*

**File:** `tests/reference/mamba3_ref.py`

Implement the recurrence in plain PyTorch, then **run it against the Stage-0
goldens.** This is what resolves the ambiguity: whichever formulation reproduces
the official kernel's outputs is the real one.

Three outcomes:
- Paper form matches → proceed, record it.
- Community form matches → proceed with that, record the correction.
- **Neither matches** → reverse-engineer from the captured tensors. We have
  inputs *and* outputs, so the coefficients are recoverable by fitting on a
  short sequence. Budget +1 day if this happens.

**Exit gate — CORRECTED Aug 6, 2026.** The original "< 1e-4 at f64" is
**unsatisfiable**: the kernel emits bf16 (~0.4% relative epsilon), four orders
of magnitude above that bound. Holding to it would mean hunting a bug that does
not exist. The replacement is *tighter* than a loose absolute tolerance:

> Round the f64 reference output to bf16; require agreement with the golden to
> ~1 ULP of bf16, on every case.

Achievable because Stage 0 records inputs **post-cast**, so the reference is fed
exactly what the kernel consumed. Per-tensor true dtypes are in
`tests/golden/mamba3/manifest.json`.

---

## Stage 2 — Scalar Rust *(~1 day)*

**File:** `kernel/arm-scan-core/src/mamba3.rs`

Direct transcription of the recurrence — clarity over speed. This is the
in-crate oracle and the non-Arm fallback, exactly as `scalar.rs` is today.

- New inputs: `lambda` (the trapezoidal gate, per (batch, head, t)), and the
  `Bx` 2-tap. `ScanDims` gains nothing — the state layout `(batch, dim, state)`
  already matches Mamba-3's `(B, H·P, D)`.
- `parallel.rs` is **untouched**: channels remain independent, so rayon
  bit-identity holds by construction.

**Exit gate:** scalar output matches the goldens near each case's recorded f32
floor, replayed via `tests/verify_golden_mamba3.py`.

---

## Stage 3 — NEON *(~1–1.5 days — the real engineering)*

**File:** `kernel/arm-scan-core/src/neon/mamba3.rs`

Extend the chunked two-pass pipeline. What changes:

| Phase | Change |
|---|---|
| Pass A1 | `dt = softplus(delta + bias)` — unchanged |
| Pass A2 | `α = exp(dt·A)` — unchanged. **New:** form `Bx_t`, then the 2-tap `b̄_t = β_t·Bx_{t-1} + γ_t·Bx_t`. Pointwise in time → vectorises across 4 timesteps like the existing code |
| **Pass B** | **Unchanged.** `h = α⊙h + b̄`, vectorised across state |
| Epilogue | `out = y + D·x` — unchanged (Mamba-3 has no `z` gate here) |
| Chunk carry | now carries `h` **and** `Bx_last` across the boundary |

**RoPE and BCNorm stay in Python for now** — both are pointwise and cheap
relative to the scan. Fuse later only if measurement justifies it. This mirrors
the repo's existing discipline: correctness path first, fusion second.

**Exit gate:** NEON↔scalar parity ≤ 3e-7; rayon bit-identical at
`RAYON_NUM_THREADS ∈ {1,2,8}`.

---

## Stage 4 — FFI and torch op *(~½ day)*

- `arm_scan_mamba3_scan_f32` in `arm-scan-ffi`, same discipline: null checks,
  overflow-checked sizing, `catch_unwind`. Bump `arm_scan_abi_version()`.
- `python/arm_scan/mamba3.py`: `torch.library` custom op + registered fake
  kernel so it composes with `torch.compile`.

**Exit gate:** goldens replayed through the **real C ABI**, not just Rust tests.

---

## Stage 5 — The other two topologies *(~½ day — cheap by construction)*

- **Bidirectional:** add `reverse` to the Mamba-3 entry point. The 2-tap
  reverses with the traversal; `Bx_prev` becomes `Bx_next`. Then
  `bidirectional_scan` works unchanged.
- **SS2D:** point `ss2d_scan`'s `scan_pair` at the Mamba-3 pair function.
  **Zero new orchestration** — that layer is recurrence-agnostic.

**Exit gate:** Mamba-3 2D goldens per direction; pair-vs-oracle parity at
1/2/8 threads.

---

## Stage 6 — CPU model path *(~1–1.5 days)*

**Directory:** `apps/mamba3_lm/`

`mamba_ssm`'s `Mamba3` cannot run on CPU and will not install without `nvcc`, so
the block is reimplemented in plain PyTorch: input projections, RoPE angle
computation, BCNorm, the gated MLP, the residual/norm plumbing, embedding and
head — with the scan routed to our kernel. Then load the published 187M weights
into it.

**Exit gate:** logits match `model_forward.npz` from Stage 0 within fp32
tolerance. That is the end-to-end proof that we run *the real model*.

---

## Stage 7 — The three demonstrations *(~1 day)*

| Topology | Demo | Evidence |
|---|---|---|
| 1D unidirectional | **Long context** on Mamba-3 187M | constant memory at 128k; `torch.compile` compile-time wall |
| 1D bidirectional | Mamba-3 run both directions | fused kernel: Pass A shared across directions |
| 2D cross-scan | Mamba-3 SS2D over a token grid | first 2D Mamba-3 on any CPU |

Topologies 2 and 3 have **no pretrained Mamba-3 models**, so they are *kernel
capability* demonstrations with correctness gates and measured throughput — the
same standing our SS2D work has today.

---

## Stage 8 — Graviton *(~3 hours)*

`bench/GRAVITON_SESSION.md`, extended with the Mamba-3 rows.

---

## Schedule, honestly

| Day | Work |
|---|---|
| 1 | Stage 0 (you, GPU) + Stage 1 |
| 2 | Stage 2 |
| 3–4 | Stage 3 |
| 4 | Stage 4 + 5 |
| 5–6 | Stage 6 |
| 7 | Stage 7 + **Graviton** |
| 8 | Video, writeup, submit |

**That is 8 days of work in 8 days, with zero slack**, and it assumes Stage 0
succeeds on the first try and Stage 1 finds a matching recurrence.

## Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Kernels don't build on Blackwell | Stage 0 blocked | Colab A100/L4 |
| Neither recurrence matches | +1 day | Reverse-engineer from captured tensors |
| Model integration overruns (weight naming, RoPE details) | +1–2 days | Descope to kernel-only demo against goldens |
| No time for video/writeup | **Submission incomplete** | See below |

## Descope ladder — cut in this order

1. **SS2D on Mamba-3** — we already have SS2D on Mamba-1, gated and measured
2. **Bidirectional on Mamba-3** — same
3. **Model integration** (Stage 6) — demo the kernel against goldens instead of
   running the checkpoint. Weakens the claim from "runs the model" to
   "implements the operator, verified against the official kernel"
4. **All of Mamba-3** — the Mamba-1 submission is complete, measured, and CI-green
   today. It remains the floor.

## What stays true regardless

The existing kernel, its three topologies, the MRI app and every measured number
are untouched by this work. Mamba-3 is **additive**. If it lands, it leads the
submission; if it does not, nothing is lost but the days spent.
