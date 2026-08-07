# CLAUDE.md — working guidelines for this repo

> **Resuming in a fresh session, or on a different machine? Read
> [`HANDOFF.md`](HANDOFF.md) first.** It carries the state that lives in a chat
> log rather than in the code: what is blocked and why, the full nine-stage
> Mamba-3 plan, decisions already settled, and the traps that have already cost
> us time. This file is the standing rules; that one is where we currently are.

## The goal

Win the **[Arm Create: AI Optimization Challenge](https://arm-ai-optimization-challenge.devpost.com/)** — **Cloud AI track**. Deadline **Aug 14, 2026, 4:00 PM PDT**. Every decision should be read against the judging rubric: Technical (40), WOW (25), Impact (20), Developer Experience (15).

The contribution: **hand-written Arm/NEON selective-scan kernels in Rust**, exposed as PyTorch
custom ops — for the deployed **Mamba-1** ecosystem and for **Mamba-3** (ICLR 2026), whose
official kernels are Triton/TileLang/CuTe and have **no CPU path at all**.

**Two kernels, one library, deliberately separate.** Mamba-3's state is a matrix per head where
Mamba-1's is a vector per channel, and their tensor sets are disjoint — one entry point serving
both would be half-ignored on every call. They share threading, the C ABI, packaging, CI and the
whole correctness harness. **They are not cross-compatible**: Mamba-1 uses a per-state-element
decay vector, Mamba-3 a scalar decay per head, so no input massaging bridges them.

## Where things stand (Aug 7, 2026)

**Sequencing lives in [`MAMBA3_IMPLEMENTATION_PLAN.md`](MAMBA3_IMPLEMENTATION_PLAN.md);
file-level execution in [`MAMBA3_KERNEL_WORKPLAN.md`](MAMBA3_KERNEL_WORKPLAN.md).** Earlier
schedules (`ROADMAP.md`, `SUBMISSION_ENDGAME_PLAN.md`, `APPLICATIONS.md`) are in
`docs/archive/` — read for history, not for what to do next.

**Mamba-1: complete.** Goldens, scalar + NEON chunked kernel, rayon, C ABI, torch op,
`arm_scan.patch()`, wheels, three-platform CI. 1D · fused bidirectional · SS2D cross-scan, the
last measured at 1.80× over the four-scan formulation.

**Mamba-3: kernel complete, gated (Stages 0–5, M0–M8).** Ground truth captured from the
official GPU kernels; reference, Rust scalar, cache-blocked and NEON kernels, C ABI at v6, and
the torch op all reproduce it to **4.47 bf16 ULP** — the floor bf16 output quantisation allows.
NEON verified on `linux-arm64` and `macos-arm64` in CI.

**The three paths.** 1D unidirectional earns credibility (the only one with published weights
*and* an authoritative oracle). 2D earns the novelty — no CPU implementation of any 2D Mamba-3
exists. Bidirectional is a near-free capability with **no novelty claim**: `burn-mamba` ships
bidirectional Mamba-3, and both candidate applications (SEMamba, VideoMamba) turned out to use
*outer* bidirectional — separate weights per direction — which our fused kernel cannot help.

Not done, in priority order:

1. **No dedicated-hardware numbers, for anything.** Every timing is x86 or a shared 4-core
   runner. This is the existential gap for a Cloud-track entry and no kernel work substitutes.
2. **2D cross-scan not yet wired to the Mamba-3 primitive.** `ss2d_scan`'s machinery is
   recurrence-agnostic; the `scan_pair` seam still speaks Mamba-1's parameter list.
3. **The real 187M checkpoint has not been run end to end** (Stage 6).
4. **Performance is unmeasured and untuned.** `TILE = 32` in the blocked kernel is a placeholder
   that has never been swept, and there is no phase profile for Mamba-3.

**Demoted, not abandoned:** the SS2D-Mamba diffusion MRI app (`apps/mri_diffusion/`). It stays
CI-gated because it is the only end-to-end exercise of the SS2D kernel, but it is off the
critical path and its quality gate has never passed — see
[`docs/archive/PHASE_D_DIAGNOSIS.md`](docs/archive/PHASE_D_DIAGNOSIS.md).

## Rules of engagement

**Claims policy (never over-claim).** Prior art is real and was verified repo by repo, not by
search summary — the table is in `README.md` and in `MAMBA3_IMPLEMENTATION_PLAN.md` §1.

May claim, to the best of our knowledge: (1) first Arm/NEON `selective_scan` exposed as a
**PyTorch custom op**; (2) first fast CPU **SS2D cross-scan**; (3) first **PyTorch-callable,
NEON-optimised Mamba-3** scan; and, once wired, (4) first **CPU implementation of a 2D
Mamba-3** plus the causal-vs-non-causal comparison, which is the novel *result*.

**Never claim:** "first Mamba on Arm/CPU" (llama.cpp, BitMamba-2, mamba.rs, Candle);
"first Mamba-3 on CPU" or "in Rust" (`mamba-rs`, `burn-mamba`, `mamba.c`); **anything about
bidirectional Mamba-3** (`burn-mamba` ships it); any accuracy result for 2D or bidirectional
Mamba-3 (no weights exist); or the old "first diffusion-prior MRI reconstruction on CPU" — that
app is demoted and its quality gate has never passed.

**Verify before repeating.** Three research digests handed to this project asserted Mamba-3
connections that did not exist (MFil-Mamba, Akasha 2, and a VNCT code release). Check the actual
repo or paper before a claim enters a document.

**Correctness gates speed. Always.** Every optimization layer must reproduce the previous layer's output within tolerance before anyone benchmarks it. The acceptance criterion is fixed: for every `tests/golden/*.npz`, `max_abs(out_kernel - out_f64) < 1e-4`, and a correct f32 kernel lands within a small factor of that case's recorded `f32_max_abs_err` floor — not orders of magnitude above it. Never loosen a tolerance to make a test pass; find the bug. **The one documented exception is
Mamba-3**, whose ground truth is a bf16-emitting GPU kernel: 1e-4 is unsatisfiable there, so its
gate is bf16 ULPs at tensor scale. That is a different instrument, not a relaxed bound — see
`tests/verify_golden_mamba3.py`.

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
tests/                   golden_inputs.py (torch-free draws + case table), gen_golden.py,
                         verify_golden.py (independent, numpy-only), golden/*.npz,
                         reference/selective_scan_ref.py (vendored ground truth), check_*.py
bench/                   bench_op.py, bench_e2e.py, bench_ss2d.py, bench_mamba3.py
                         (all vs eager AND torch.compile; correctness gates speed)
.github/workflows/ci.yml arm64 + macOS + x86: fmt, clippy, tests, golden-through-C-ABI, wheels, bench
```

Docs live in three places — **root is judge-facing and current only**:

- **root** — `README.md` (pitch), `PROJECT_CONCEPT.md` (decision log),
  `MAMBA3_IMPLEMENTATION_PLAN.md` (the plan), `MAMBA3_KERNEL_WORKPLAN.md` (execution),
  `HANDOFF.md` (session state).
- **`docs/archive/`** — the working record: superseded plans, build logs, measurement history,
  diagnoses. Kept because how a decision was reached is itself evidence. **Not** judge-facing,
  and **not** a source of instructions — a plan in here has been superseded by definition.
- **`docs/roadmap/`** — programs deferred past the current push.

**One sequencing table, and only one.** It is in `MAMBA3_IMPLEMENTATION_PLAN.md`. The repo
previously carried schedules in six documents at once and they disagreed; when a plan changes,
move the old one to `docs/archive/` rather than leaving two versions live.

The kernel also grew a second family. `kernel/arm-scan-core/src/mamba3/` (types, dispatch,
naive scalar oracle, cache-blocked) plus `neon/mamba3.rs` are Mamba-3; everything else is
Mamba-1 and is **not** modified by that work.

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
