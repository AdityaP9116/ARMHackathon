# CLAUDE.md — working guidelines for this repo

> **Resuming in a fresh session, or on a different machine? Read
> [`HANDOFF.md`](HANDOFF.md) first.** It carries the state that lives in a chat
> log rather than in the code: what is blocked and why, the full nine-stage
> Mamba-3 plan, decisions already settled, and the traps that have already cost
> us time. This file is the standing rules; that one is where we currently are.

## The goal

Win the **[Arm Create: AI Optimization Challenge](https://arm-ai-optimization-challenge.devpost.com/)** — **Cloud AI track**. Deadline **Aug 14, 2026, 4:00 PM PDT**. Every decision should be read against the judging rubric: Technical (40), WOW (25), Impact (20), Developer Experience (15).

The contribution: the **first Arm-optimized `selective_scan` for the PyTorch/Mamba ecosystem, written in Rust** (NEON + chunked scan + rayon), shipped as a pip-installable drop-in that makes *any* Mamba model faster on Arm CPU — covering **all three scan topologies**, each proven on a real workload running on Graviton.

| Topology | Kernel | Demonstration |
|---|---|---|
| 1D unidirectional | 3.71× vs `torch.compile` | **Long context** — constant memory at any L, where `torch.compile`'s compile time explodes (59.9 s → 532.8 s, L=256 → 2048) |
| 1D bidirectional | **6.39–8.99×** vs `torch.compile` | **Speech enhancement** — audible before/after, real-time factor |
| 2D cross-scan (SS2D) | 1.80× over the four-scan formulation | **Diffusion MRI reconstruction** — 18–256 denoiser calls per image |

**The argument is generality**: three topologies, three unrelated workloads, one kernel. A
point optimization would not survive that.

## Where things stand (as of Aug 4, 2026 — 10 days out)

**Sequencing lives in one place now: [`SUBMISSION_ENDGAME_PLAN.md`](SUBMISSION_ENDGAME_PLAN.md).**
It supersedes the week tables in `ROADMAP.md` §4 and `docs/archive/SS2D_REPOSITIONING_PLAN.md` §7,
which assumed the Jul 20 / Jul 27 weeks happened. They did not.

The kernel is built and measured. `docs/archive/INTEGRATION_PLAN.md` Phases 0–6 landed (goldens,
scalar+NEON chunked kernel, rayon, C ABI, torch custom op + `arm_scan.patch()`, wheels,
arm64/macOS/x86 CI, benchmark harnesses, `docs/archive/BASELINE_REPORT.md` with CI-provisional Arm
numbers). 1D bidirectional + resumable-state (h0) kernels exist.

**The demonstrations are DECIDED — one per topology** (rationale in `APPLICATIONS.md`:
*"three demos that each exercise a different scan topology prove the kernel itself is
general"*):

- **1D unidirectional → long context.** Not "text generation" — the *capability* claim.
  Constant memory at any L, against a `torch.compile` whose compile time explodes with
  sequence length. Cheapest of the three: `bench_e2e.py` already exists.
- **1D bidirectional → speech enhancement.** The only demo a judge can *hear*, measured as
  real-time factor. Fills the slot for our **strongest-measured** topology, which currently
  has no application at all. Known risk: audio Mamba checkpoints are research-grade and
  CUDA-coupled — resolve with a spike before committing.
- **2D cross-scan → diffusion MRI.** Built and gated; see
  `MRI_DIFFUSION_IMPLEMENTATION_PLAN.md`. VMamba classification is the de-risked
  substitute if MRI quality does not come together (`APPLICATIONS.md`).

Landed Aug 4 (the SS2D/app hole-closing pass):

- **SS2D runs as two traversal PAIRS, not four forward scans** — `arm_scan.ss2d.ss2d_scan`
  now drives the already-shipped fused bidirectional kernel, so Pass A (discretize + exp,
  ~85% of kernel time) is computed twice per block instead of four times, and the four
  `torch.flip` copies are gone. `SS2DBlock._cross_scan_legacy` is retained as the oracle.
- **Measured** (x86 dev box, quiesced, reps=3): **1.77–1.82× (geomean 1.80×)** on block
  total at the production shapes, 1.81–1.90× on scan time alone; non-scan overhead fell
  **21–25% → 7.2–13.8%**. That flips the P1-7 verdict to *not justified* (15% rule), though
  only marginally at the worst shape — re-take it on Arm.
- **2D goldens exist** (`tests/gen_golden_2d.py` / `verify_golden_2d.py`): per-direction,
  independently re-derived in numpy, replayed through the real C ABI, landing at ~1.0× each
  case's recorded f32 floor.
- **App tests are reproducible**: the hardcoded `C:\Users\Adity\...` reference path is gone,
  and `apps/mri_diffusion/edm_min.py` re-derives EDM preconditioning/loss from the published
  equations so the judge path needs no CC-BY-NC-SA clone. `ADM_REF` still selects the CSI
  classes when working against their checkpoints.
- **CI gates the app now** (`mri-app` job), and `make validate` exercises SS2D + diffusion,
  not just the 1D kernel. `apps/mri_diffusion/demo.py` produces the video's artifact.
- **Fixed a false-green gate:** Phase C's parity check compared an untrained net whose
  zero-init `out_proj`/`head` made the output independent of the scan — it reported
  max_abs 0.0 for that reason. It now activates those layers first.

Not done — the half that wins or loses the competition:

1. **No dedicated-hardware numbers at all** (Ampere/Graviton). Every figure in the repo is
   x86 or a shared 4-core runner. This is the existential gap; book the session.
2. **No trained prior.** Route A/B + GPU budget still open
   (`MRI_DIFFUSION_IMPLEMENTATION_PLAN.md` §14); the endgame plan recommends cutting Route-A
   distillation for a small-scale prior.
3. **Phase D's quality gate has never passed** — R=4 reconstruction is 2.75 dB *worse* than
   zero-filled against a >1 dB-better bar. **Fully diagnosed in
   [`PHASE_D_DIAGNOSIS.md`](docs/archive/PHASE_D_DIAGNOSIS.md); read it before touching the sampler.** An
   oracle denoiser reconstructs to 151 dB, so the sampler/FFT/mask/DC are exact — the causes
   are the evaluation data (smooth bumps, where zero-filled is already near-optimal), a mask
   that delivers effective R=2.67 when labelled R=4, a sampler σ_max=80 far outside the
   prior's trained support, and a gate that hides the kernel-parity check behind a
   model-quality assertion. **Do not "fix" it by relaxing the assertion.**
4. **No <3-min video, no Devpost writeup.** Submit Aug 12–13.

## Rules of engagement

**Claims policy (never over-claim).** Real prior art exists for 1D Mamba on CPU/Arm
(llama.cpp `ssm_scan`, BitMamba-2, mamba.rs, Candle). Never claim "first Mamba on Arm."
The three defensible to-our-knowledge claims: (1) first SIMD `selective_scan` callable
from PyTorch as a drop-in; (2) first fast CPU SS2D cross-scan; (3) first diffusion-prior
MRI reconstruction on CPU. Cite the prior-art table in `README.md`.

**Correctness gates speed. Always.** Every optimization layer must reproduce the previous layer's output within tolerance before anyone benchmarks it. The acceptance criterion is fixed: for every `tests/golden/*.npz`, `max_abs(out_kernel - out_f64) < 1e-4`, and a correct f32 kernel lands within a small factor of that case's recorded `f32_max_abs_err` floor — not orders of magnitude above it. Never loosen a tolerance to make a test pass; find the bug.

**Benchmark honestly.** `torch.compile` is the baseline that matters, not just the eager fallback — a "we beat a strawman" critique from an Arm engineer judge is fatal. Report medians after warmup, fixed thread counts, pinned seeds, and state the instance type and torch version alongside every number. If a row is unflattering, publish it anyway. The kernel's moat is that `torch.compile` cannot restructure a sequential recurrence; that argument only lands if the numbers are clearly trustworthy.

**Numerics are approximate, and we say so.** The NEON `exp` polynomial and FMA reassociation mean results match the reference to fp32 tolerance, not bit-exactly. Disclose it, and back it with an output-level model metric showing quality is unchanged.

**Keep `unsafe` where it lives.** All raw pointers stay in `arm-scan-ffi`; `unsafe` in `arm-scan-core` is confined to isolated NEON blocks with a SAFETY comment. Panics are caught at the C boundary and returned as error codes.

**The scalar path is not dead code.** It is the in-crate correctness reference, the non-Arm fallback, and what keeps x86 CI meaningful. Don't delete or let it rot.

**Free tier first.** Develop and test on GitHub Actions arm64 runners, Apple Silicon, or Oracle Ampere A1. Rent Graviton (`c8g`) only for headline numbers and the video — budget ~$5–20 total, script the setup, terminate the instance after each session.

## Repo map

```
kernel/arm-scan-core/    Rust kernel: scalar.rs (reference), neon/ (exp, math, chunked scan),
                         parallel.rs (rayon over B×D), float.rs (f32/f64 abstraction)
kernel/arm-scan-ffi/     cdylib, C ABI, one entry point. All raw-pointer handling.
python/arm_scan/         _ffi.py (ctypes loader), op.py (torch custom_op), patch.py (HF monkeypatch),
                         numpy_api.py (torch-free path)
tests/                   gen_golden.py, verify_golden.py (independent), golden/*.npz,
                         reference/selective_scan_ref.py (vendored ground truth), check_*.py
bench/                   bench_op.py (kernel vs eager vs torch.compile), bench_e2e.py (mamba-130m generate)
.github/workflows/ci.yml arm64 + macOS + x86: fmt, clippy, tests, golden-through-C-ABI, wheels, bench
```

Docs live in three places now — root is judge-facing only:
- **root** — `README.md` (pitch), `PROJECT_CONCEPT.md` (decision log), `ROADMAP.md`,
  `APPLICATIONS.md` (why these three demos), `MRI_DIFFUSION_IMPLEMENTATION_PLAN.md`,
  `SUBMISSION_ENDGAME_PLAN.md` (live schedule, archive after Aug 14).
- **`docs/archive/`** — the working record: build logs, superseded plans, measurement
  history. Kept because it is real evidence of how decisions were made; not judge-facing.
- **`docs/roadmap/`** — post-submission programs (Mamba-3 / SSD).

What each root doc is for — **keep them non-duplicative**:
- `README.md` — the pitch and the deliverables (what a judge reads first).
- `PROJECT_CONCEPT.md` — the decision log: what we chose, what we rejected, why.
- `ROADMAP.md` — schedule, compute strategy, risk register.
- `INTEGRATION_PLAN.md` — the engineering plan, phase by phase.

When a decision changes, update the decision log — don't leave two docs disagreeing. (They currently disagree about the application; that's a bug, not a feature.)

## Commands

```bash
cd kernel && cargo test --release        # goldens, property tests, parity (scalar↔NEON↔threaded)
cd kernel && cargo clippy --all-targets -- -D warnings && cargo fmt --check   # CI enforces both
cd kernel && cargo build --release -p arm-scan-ffi && cargo bench             # kernel ladder
python tests/check_ffi.py                # goldens through the real C ABI
python tests/verify_golden.py            # independent re-derivation of the goldens
python bench/bench_op.py [--quick]       # kernel vs eager vs torch.compile
python bench/bench_e2e.py                # mamba-130m generate(), patched vs unpatched
python scripts/build_wheel.py            # platform-tagged wheel
```

Run correctness under multiple thread counts (`RAYON_NUM_THREADS ∈ {1,2,8}`) — parallel output must be bit-identical to sequential.

## What "done" looks like for the submission

1. Public MIT repo (license visible in the GitHub About sidebar — a contest rule), green arm64 CI.
2. A `make validate` path that a judge can run on their own MacBook or an arm64 box in ~5 minutes, with **no dataset and no AWS account**.
3. `RESULTS.md` with the full ladder (scalar → +NEON → +chunked → +rayon), both baselines, on a named Graviton instance, plus a core-scaling curve.
4. Quality parity on the application, measured, at identical output quality.
5. A <3-minute demo video shot on Graviton. No copyrighted music.
6. Devpost writeup. **Submit Aug 12–13 — not at 3:50 PM on the 14th.**

Anything that doesn't move one of those six forward is a distraction this late.
