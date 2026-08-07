# HANDOFF — read this first when resuming on a new machine or in a new session

**Written Aug 6, 2026.** Deadline **Aug 14, 4:00 PM PDT — 8 days.**

This exists because the working context of a session does not survive a reboot
or a fresh clone. Everything below is state that was in someone's head or in a
chat log, not in the code. Start here, then read
[`MAMBA3_IMPLEMENTATION_PLAN.md`](MAMBA3_IMPLEMENTATION_PLAN.md).

---

## The one-line status

The Mamba-1 submission is **complete, measured, and CI-green** — that is the
floor and it is safe. The live work is an **additive** Mamba-3 kernel.
**Stage 0 is DONE (Aug 6)**; the GPU is no longer needed. Stage 1 (a
paper-faithful CPU reference, checked against those goldens) is the live task
and needs nothing but a laptop.

## Stage 0 — DONE, Aug 6, 2026

`tests/golden/mamba3/` holds **10 golden cases across 7 output shapes**
(L ∈ {1, 63, 64, 128, 255, 256, 2048}, batch 1–2, d_state 64/128), captured
from `mamba3_siso_combined` driven by the real `state-spaces/mamba3-siso-187m`
checkpoint plus an edge-shape sweep, alongside `model_forward.npz` and
`model_shape.json`. Verified to replay under a Python with **no torch, no
mamba_ssm, no CUDA** — so the GPU really is never needed again.

### Reproducing the environment (read before touching a GPU box)

```bash
python3 -m venv ~/venv-arm && source ~/venv-arm/bin/activate
pip install --upgrade pip wheel setuptools ninja packaging numpy
pip install torch --index-url https://download.pytorch.org/whl/cu130   # triton comes with it
MAMBA_SKIP_CUDA_BUILD=TRUE pip install --no-build-isolation \
    "git+https://github.com/state-spaces/mamba.git@main"
python tools/capture_mamba3_goldens.py --out tests/golden/mamba3
```

Three things that are not obvious and cost real time to find:

1. **Install from git, NOT PyPI.** The released wheel `2.3.2.post1`
   (2026-05-09) fails twice over: `create_block` only accepts
   `["Mamba1","Mamba2"]` so it **rejects every published Mamba-3 checkpoint**,
   and it predates upstream **PR #997** (merged 2026-07-22) which fixes
   **silent forward-pass corruption** in `mamba3_siso_fwd_kernel` on Blackwell
   (SM100/103/120 — `num_stages` 2 or 3 returns wrong output with no error).
   Ground truth captured through that bug would have validated every downstream
   Rust kernel against garbage *and looked green doing it*. The capture script
   now refuses to run without the fix; it inspects the source, because patched
   and unpatched installs both report version `2.3.2.post1`.
2. **`MAMBA_SKIP_CUDA_BUILD=TRUE` is required**, and then
   `import mamba_ssm` fails on `No module named 'selective_scan_cuda'` —
   that is *Mamba-1's* CUDA extension, hard-imported by `__init__.py` and
   never used by Mamba-3. Do not install a CUDA toolkit to satisfy it: the
   system nvcc must be ≥12.8 to emit `sm_120` at all. Write a stub instead,
   one that **raises on every attribute access** so it can never silently
   corrupt a capture. (Ours lives in the venv, not the repo:
   `~/venv-arm/lib/python3.14/site-packages/selective_scan_cuda.py`.)
3. **The kernel is mixed precision with no flag to change it.**
   `Q/K/V/Trap/Angles/Z` are cast to bf16 on entry; `ADT/DT` stay fp32 "for
   stability"; `Q_bias/K_bias/D` stay fp32 as model parameters; the output is
   bf16. The goldens therefore record inputs **post-cast** — the values the
   kernel actually consumed, not the ones we handed it. Recording pre-cast
   fp32 would make a CPU reference diverge by an amount that *compounds over
   the sequence* and reads as a kernel bug. **Our Rust kernel must mirror this
   precision split.**

### Why Stage 0 could not run on Windows

Established by inventory, not assumption:

| | Finding |
|---|---|
| `mamba-ssm` | **0 Windows wheels for any version.** sdist only |
| `triton` (official) | **0 Windows wheels.** The unofficial `triton-windows` fork ships cp310/cp311 only; the host is cp312 |
| `causal-conv1d` | source only |
| `cargo test` | **has never run on the Windows host** — active toolchain is `windows-msvc` with no MSVC linker; the `windows-gnu` fallback lacks `dlltool` |

That last row is the more serious finding and is easy to miss: **our primary
correctness gate has only ever run in CI.** `tools/setup_linux.sh` fixes it as
a side effect, independently of whether Mamba-3 works out.

**Do not install the CUDA toolkit preemptively.** PyTorch's cu128 wheels bundle
a CUDA runtime, Triton JIT-compiles through its own LLVM rather than shelling
out to `nvcc`, and `mamba-ssm` wants `nvcc` only for extensions the Triton SISO
path does not use (`MAMBA_SKIP_CUDA_BUILD=TRUE` skips them). It is 3 GB against
a problem you may not have.

---

## Why Stage 0 exists at all — the finding that drives everything

**There is no trustworthy Mamba-3 oracle, and the obvious substitute is wrong.**

`mamba_ssm/modules/mamba3.py` has **no CPU path** — it imports Triton/TileLang/
CuTe kernels and asserts if they are missing. The community reimplementation
[`rishikksh20/mamba3-pytorch`](https://github.com/rishikksh20/mamba3-pytorch)
runs on CPU, but **implements a different recurrence from the paper.** Measured
by `tools/check_mamba3_recurrence.py` on O(1) states, so this is not rounding:

```
community vs paper, same gate ........................ max_abs 1.06
best gate remapping (1 - gate/2) ..................... max_abs 0.145
structural difference alone (decay on the prev term) . max_abs 0.363
```

The paper carries the state decay on the **previous** input term; the community
version does not. **No gate remapping reconciles them.** We cannot tell which is
right from outside — and it does not matter, because the only authoritative
answer is what the official kernels compute, since that is what the published
checkpoints were trained against. Hence: capture from the official kernels.

Re-run that check any time with:

```bash
python tools/check_mamba3_recurrence.py
```

---

## Decisions already made — do not relitigate

- **Target Mamba-3 specifically**, not a Mamba-1 retrofit. The user's call, on
  novelty grounds. **All three topologies** must run on Mamba-3, including 1D
  unidirectional.
- **SISO inference only.** MIMO, the SSD/dual form, and training are out of scope.
- **The Mamba-1 work is untouched and remains the floor.** Mamba-3 is additive:
  if it lands it leads the submission; if not, nothing is lost but the days.
- Prior art exists and is in the README table — `silvermpx/mamba-rs` is
  Mamba-3 SISO in Rust on CPU. **Never claim "first Mamba-3 on CPU."** The
  defensible claim is *first PyTorch-callable, NEON-optimized* Mamba-3 scan.

---

# The full implementation plan

Nine stages. Reproduced here in full so this file is self-contained;
[`MAMBA3_IMPLEMENTATION_PLAN.md`](MAMBA3_IMPLEMENTATION_PLAN.md) is the same
plan and either may be read alone.

## Why this is less work than it sounds

Two structural facts make a Mamba-3 kernel tractable in the time left:

**The recurrence is our existing Pass B plus one term.** Mamba-3's three-term
recurrence is

```
h_t = α_t·h_{t-1} + β_t·(B_{t-1}x_{t-1}) + γ_t·(B_t x_t)
```

Group the last two: `h_t = α_t·h_{t-1} + b̄_t`, where `b̄_t` is a **2-tap
convolution over the Bx stream**. Our kernel already computes exactly
`h = ābar⊙h + b̄`, so **Pass B is unchanged**. All new work lands in Pass A,
which is pointwise in time and therefore fully vectorisable.

**The topology layer is recurrence-agnostic.** The call chain is
`ss2d_scan → scan_pair → bidirectional_scan → a scan primitive`. Add a Mamba-3
primitive and bidirectional + SS2D compose on top for free. "All three
topologies" is **not** 3× the work.

## Stage 0 — Ground truth ✅ DONE (Aug 6, 2026)

`tools/capture_mamba3_goldens.py` wraps the official kernel entry point —
discovered by **searching** `mamba_ssm`, not hardcoded, since the import path
has moved between releases — records inputs → outputs as `.npz`, saves a slim
full-model forward artifact, and dumps config + parameter shapes.

**Delivered:** 10 cases / 7 shapes, `model_forward.npz`, `model_shape.json`;
19 MB total; replay verified with numpy alone. Exit gate is now checked by the
script itself, which exits non-zero if unmet — the first run exited `0` having
met none of it, which is exactly the kind of false green this repo keeps
finding.

**The H100-only risk was real but landed well:** Triton JIT compiled and ran
correctly on sm_120, and TileLang/CuTe imported too. The actual Blackwell
hazard was not a build failure but PR #997's *silent numerical corruption* —
see the install notes above. Compiling and running is not computing correctly.

**`model_forward.npz` is deliberately not the raw logits.** Full 128k-vocab
logits are 58 MB compressed, past GitHub's warning threshold and ~20× every
other golden in this repo combined. It stores argmax + a seeded 512-id vocab
subset + logsumexp (0.49 MB) — and the logsumexp still depends on all 128k
logits, so an error confined to ids outside the subset is still caught.

## Stage 1 — Paper-faithful reference *(CPU, ~½ day)*

**File:** `tests/reference/mamba3_ref.py`

Implement the recurrence in plain PyTorch, then **run it against the Stage-0
goldens.** This is what resolves the paper-vs-community ambiguity: whichever
formulation reproduces the official kernel's output is the real one.

Three outcomes:
- Paper form matches → proceed, record it.
- Community form matches → proceed with that, record the correction.
- **Neither matches** → reverse-engineer from the captured tensors. We have
  inputs *and* outputs, so the coefficients are recoverable by fitting on a
  short sequence. **Budget +1 day if this happens.**

**Exit gate — CORRECTED Aug 6.** The original wording, "reference reproduces
every golden to < 1e-4 at f64", **cannot be satisfied and must not be used**:
the kernel emits **bf16**, whose relative epsilon is ~0.4% — four orders of
magnitude above 1e-4. Chasing that gate would mean hunting a bug that does not
exist. The honest gate, which is *tighter* than a loose absolute bound:

> Round the f64 reference output to bf16 and require agreement with the golden
> to ~1 ULP of bf16, on every case.

The goldens make this achievable because inputs are recorded **post-cast**, so
the reference is fed exactly the values the kernel consumed. `manifest.json`
carries the true per-tensor dtype for every case — never guess it.

## Stage 2 — Scalar Rust *(~1 day)*

**File:** `kernel/arm-scan-core/src/mamba3.rs`

Direct transcription of the recurrence — clarity over speed. This becomes the
in-crate oracle and the non-Arm fallback, exactly as `scalar.rs` is today.

- New inputs: `lambda` (the trapezoidal gate, per (batch, head, t)) and the `Bx`
  2-tap. `ScanDims` needs nothing new — the state layout `(batch, dim, state)`
  already matches Mamba-3's `(B, H·P, D)`.
- `parallel.rs` is **untouched**: channels stay independent, so rayon
  bit-identity holds by construction.

**Exit gate:** scalar output matches the goldens near each case's recorded f32
floor, replayed via `tests/verify_golden_mamba3.py`.

## Stage 3 — NEON *(~1–1.5 days — the real engineering)*

**File:** `kernel/arm-scan-core/src/neon/mamba3.rs`

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

## Stage 4 — FFI and torch op *(~½ day)*

- `arm_scan_mamba3_scan_f32` in `arm-scan-ffi`, same discipline as the existing
  entry point: null checks, overflow-checked sizing, `catch_unwind`. Bump
  `arm_scan_abi_version()`.
- `python/arm_scan/mamba3.py`: `torch.library` custom op **plus a registered
  fake kernel**, so it composes with `torch.compile`.

**Exit gate:** goldens replayed through the **real C ABI**, not just Rust tests.

## Stage 5 — The other two topologies *(~½ day — cheap by construction)*

- **Bidirectional:** add `reverse` to the Mamba-3 entry point. The 2-tap
  reverses with the traversal — `Bx_prev` becomes `Bx_next`. Then
  `bidirectional_scan` works unchanged.
- **SS2D:** point `ss2d_scan`'s `scan_pair` at the Mamba-3 pair function.
  **Zero new orchestration.**

**Exit gate:** Mamba-3 2D goldens per direction; pair-vs-oracle parity at
1/2/8 threads.

## Stage 6 — CPU model path *(~1–1.5 days)*

**Directory:** `apps/mamba3_lm/`

`mamba_ssm`'s `Mamba3` cannot run on CPU and will not install without `nvcc`, so
the block is reimplemented in plain PyTorch: input projections, RoPE angle
computation, BCNorm, the gated MLP, residual/norm plumbing, embedding and head —
with the scan routed to our kernel. Then load the published 187M weights into it.
**`model_shape.json` from Stage 0 is what makes the weight-name mapping
mechanical rather than guesswork.**

**Exit gate:** logits match `model_forward.npz` within fp32 tolerance. That is
the end-to-end proof that we run *the real model*.

## Stage 7 — The three demonstrations *(~1 day)*

| Topology | Demo | Evidence |
|---|---|---|
| 1D unidirectional | **Long context** on Mamba-3 187M | constant memory at 128k; `torch.compile` compile-time wall |
| 1D bidirectional | Mamba-3 run both directions | fused kernel: Pass A shared across directions |
| 2D cross-scan | Mamba-3 SS2D over a token grid | first 2D Mamba-3 on any CPU |

Topologies 2 and 3 have **no pretrained Mamba-3 models**, so they are *kernel
capability* demonstrations with correctness gates and measured throughput — the
same standing our SS2D work has today.

## Stage 8 — Graviton *(~3 hours)*

`bench/GRAVITON_SESSION.md`, extended with the Mamba-3 rows. **This is the
existential gap — see below. Do not let it slip behind Mamba-3 work.**

## Schedule, honestly

| Day | Work |
|---|---|
| 1 | Stage 0 (GPU) + Stage 1 |
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
| No time for video/writeup | **Submission incomplete** | Descope ladder below |

## Descope ladder — cut in this order

1. **SS2D on Mamba-3** — we already have SS2D on Mamba-1, gated and measured
2. **Bidirectional on Mamba-3** — same
3. **Model integration (Stage 6)** — demo the kernel against goldens instead of
   running the checkpoint. Weakens the claim from "runs the model" to
   "implements the operator, verified against the official kernel"
4. **All of Mamba-3** — the Mamba-1 submission is complete, measured and
   CI-green today. **It remains the floor.**

---

## Live gaps, in priority order

1. **No dedicated-Arm numbers at all.** Every figure in the repo is x86 or a
   shared 4-core CI runner. **The Graviton session is still unbooked.** This is
   the existential gap — a Cloud AI track submission with no Graviton numbers.
   It is now the *only* item on this list that cannot be done from a laptop,
   and the schedule below puts it on day 7. That is the wrong order: nothing
   else here is unrecoverable, and this is.
2. ~~Stage 0 blocked~~ — **done Aug 6.** Stage 1 is live and needs no GPU.
3. **Phase D quality gate has never passed.** Fully diagnosed in
   `docs/archive/PHASE_D_DIAGNOSIS.md` — **read it before touching the sampler.**
   An oracle denoiser reconstructs to 151 dB, so the sampler/FFT/mask/DC are
   exact. **Do not "fix" it by relaxing the assertion.**
4. **No video, no Devpost writeup.** Submit Aug 12–13, not at 3:50 PM on the 14th.

## Rules that have bitten us

- **Never loosen a tolerance to make a test pass.** Find the bug.
- **Benchmark quiesced.** A contaminated run (reps=2 under load) once produced a
  phantom "0.50× regression" that was written into several docs before a clean
  re-run gave 1.82×. Fixed thread counts, pinned seeds, medians after warmup.
- **Watch for vacuous gates.** Phase C's parity check compared an untrained net
  whose zero-init `out_proj`/`head` made the output independent of the scan — it
  reported `max_abs 0.0` and looked green. If a parity number is *exactly* zero,
  suspect the harness.

## Security constraints that remain in force

- The fastMRI download email contains **signed URLs with `AWSAccessKeyId` and
  `Signature` — these are credentials.** They must never appear in the repo, a
  commit, an issue, the Devpost writeup, or demo-video terminal scrollback.
  `.gitignore` blocks `/data/`, `*.h5`, `*.tar.xz`, `knee_singlecoil_*`,
  `brain_multicoil_*`.
- fastMRI Data Sharing Agreement: no redistribution of data **or links**,
  internal research/education only, no commercial monetisation, cite Knoll et al.

---

## Repo state at time of writing

- Branch `feature/ss2d-fused-bidirectional`, **27 commits ahead of `main`**
  (125 total), PR #9 open — **the branch, not `main`, is the current work**
- CI **green on all 7 jobs**
- Working tree clean, nothing unpushed

## Measured results that already exist

| Result | Number |
|---|---|
| SS2D via the fused bidirectional kernel | **1.77–1.82×** (geomean 1.80×) |
| 1D unidirectional vs `torch.compile` | 3.71× |
| 1D bidirectional vs `torch.compile` | 6.39–8.99× |
| Long context, L=131,072 | ours **4.60 s**; reference needs **12.88 GB**, not attempted |
| NEON vs scalar (Neoverse-N2) | 4.03–4.08× |
| Threading, 4 cores | 3.99× (99.7% efficiency) |

Reproduce the long-context row with `python bench/bench_longctx.py`
(needs `psutil`).
