# CLAUDE.md — working guidelines for this repo

## The goal

Win the **[Arm Create: AI Optimization Challenge](https://arm-ai-optimization-challenge.devpost.com/)** — **Cloud AI track**. Deadline **Aug 14, 2026, 4:00 PM PDT**. Every decision should be read against the judging rubric: Technical (40), WOW (25), Impact (20), Developer Experience (15).

The contribution: the **first Arm-optimized `selective_scan` for the PyTorch/Mamba ecosystem, written in Rust** (NEON + chunked scan + rayon), shipped as a pip-installable drop-in that makes *any* Mamba model faster on Arm CPU — and proven on a real application running on Graviton.

## Where things stand (as of Jul 13, 2026)

The kernel is built. `INTEGRATION_PLAN.md` Phases 0–6 are landed:

- Golden-vector ground truth vs. the vendored upstream reference + an independent numpy verifier.
- `arm-scan-core` (scalar + NEON chunked two-pass) and `arm-scan-ffi` (C ABI, the only crate with raw pointers).
- `torch.library` custom op + `arm_scan.patch()` for HF `transformers` Mamba.
- Platform-tagged wheels, arm64/macOS/x86 CI, op-level and end-to-end benchmark harnesses.

Not done — and this is the half that wins or loses the competition:

1. **The application is undecided.** `PROJECT_CONCEPT.md` locks MambaRecon (MRI); `INTEGRATION_PLAN.md` Phase 7 floats an audio/ECG/RF pivot. **Resolve this first — it blocks the 2D scan, the demo, and the video.**
2. **No 2D/bidirectional (SS2D) cross-scan** — required if the application stays vision-Mamba/MRI.
3. **No measured numbers.** No `RESULTS.md`; the README results table is empty; nothing has run on Graviton.
4. **README promises that don't exist** — `make validate` (no Makefile).
5. **No demo, no <3-min video, no Devpost writeup.**

## Rules of engagement

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

Docs, and what each is for — **keep them non-duplicative**:
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
