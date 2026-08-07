# Kernel Baseline Test Plan

**Goal:** establish a rigorous, reproducible baseline of (a) kernel
correctness and (b) measured performance improvement on simple benchmarks,
before Phase 7 — with an explicit strategy for the constraint that the dev
machine is an x86-64 Windows box that cannot natively execute a single NEON
instruction.

---

## 1. The Windows problem, stated precisely

The kernel has two code paths: **scalar** (portable, runs anywhere) and
**NEON** (aarch64-only). This machine (Intel i9, 32 threads, Windows 11) can:

| Testable on this box natively | NOT testable natively |
|---|---|
| Scalar backend: full correctness + performance | NEON instruction execution |
| rayon threading: correctness AND the best scaling study we have access to (32 cores!) | NEON performance |
| Entire Python/FFI/HF integration stack (scalar) | Arm cache/memory behavior |
| aarch64 *compilation* (cross-check + clippy) | |

So the plan splits every activity into three execution surfaces:

- **Surface W (Windows, native)** — scalar-backend correctness/perf, thread
  scaling, integration stack. Immediate, interactive, free.
- **Surface Q (Windows, QEMU emulation)** — real aarch64 NEON binaries run
  locally under emulation via Docker Desktop/WSL2 + binfmt. Valid for
  CORRECTNESS ONLY (bit-accurate instruction semantics); timing under
  emulation is meaningless and will never be reported as a number.
- **Surface A (real Arm hardware)** — GitHub Actions arm64 runners (free,
  4 cores, already wired) for automated numbers; Oracle Ampere A1 free tier
  and optionally AWS Graviton for dedicated-instance headline numbers.

Every claim in the final results table is tagged W/Q/A so nobody can
mistake an emulated or shared-runner number for a headline number.

## 2. Surface Q setup — running the NEON test suite locally

New capability this plan adds: execute the full aarch64 test suite on the
Windows box.

1. Prereq: Docker Desktop with WSL2 backend (one-time install).
2. Register the aarch64 emulator:
   `docker run --privileged --rm tonistiigi/binfmt --install arm64`
3. Run the whole Rust suite in an arm64 container:
   ```
   docker run --rm --platform linux/arm64 -v <repo>:/w -w /w/kernel \
       rust:1 cargo test --release -- --nocapture
   ```
   This executes the REAL NEON kernel (QEMU translates the instructions),
   including all 16 golden cases, the exp/log/softplus sweeps, and the
   NEON-vs-scalar parity tests — locally, before any push.
4. Same container runs `check_ffi.py` for the aarch64 cdylib if desired
   (python:3.12 arm64 image + built .so).

Deliverable: `scripts/test_arm64_local.ps1` wrapping steps 2–3.
Value: kills the current "push and wait 8 minutes for CI" loop for NEON
correctness; CI remains the authority, QEMU becomes the fast local filter.
Fallback if Docker is unavailable: CI stays the only NEON correctness
surface (current state — workable, just slower).

## 2.5 Surface A options — free Arm compute + Arm profiling tools

Ranked for this project:

1. **Oracle Cloud Always Free Ampere A1** — 4 OCPU Neoverse-N1 + 24 GB,
   permanently free, dedicated. The primary baseline machine (matches the
   ROADMAP). Card required at signup; never charged on the free tier.
2. **GitHub Actions arm64 runners** — free, automated, already wired into
   CI; shared/noisy, so numbers stay labeled provisional.
3. **GCP trial ($300 credits)** — Axion (C4A) Arm VMs; optional second
   microarchitecture column. Azure ($200, Cobalt) adds less since GH
   runners are Cobalt already.
4. **AWS Graviton (~$5–20)** — the plan's one optional paid reproduction.
5. **Android phone via Termux** — real Arm silicon in hand; edge-cred demo
   only (big.LITTLE scheduling makes timing noisy).

**Arm Performance Studio (Streamline)** is a profiler, not compute: its
gator daemon on the Ampere instance gives per-core utilization,
instruction mix, and cache-miss profiles of the kernel — added to the
Ampere session as an optional profiling pass (Arm-branded evidence of
NEON pipeline saturation for the submission). **Arm Virtual Hardware** is
simulated Cortex-M-class boards — wrong tool here (same "no emulated
timing" rule as QEMU).

## 3. Correctness baseline (the "is it right" half)

Already standing (re-run and record as part of the baseline):
- 16 golden cases vs float64 PyTorch reference — errors at/below torch's
  own f32 floor (all three surfaces run these).
- Independent numpy f64 verifier agrees with the torch reference to ~1e-15.
- HF mamba-130m: patched logits within 2e-6 relative, greedy tokens
  identical; vendored reference reproduces HF's mixer bit-exactly.
- NEON-vs-scalar parity ≤ 3e-7 scale-relative on all cases (A).
- Sequential-vs-rayon bit-identity (proptest, both backends).
- 4M-point exp sweep (~1 ulp), log/softplus/silu sweeps (~2e-7).

New for this baseline campaign:
1. **Extended fuzz soak (W + Q):** one-off `PROPTEST_CASES=20000` run of
   the property suites (minutes, not hours). Catches rare shape/value
   interactions the 256-case CI default can't.
2. **Long-sequence stress (W + A):** L ∈ {8k, 32k, 128k} single case each,
   f32 vs f64 drift measured — documents error growth with sequence length
   (expected: bounded by the exp decay, not linear).
3. **Denormal/edge audit (Q + A):** a targeted test with delta*A arguments
   straddling the underflow select (-87.3), confirming NEON-vs-scalar
   agreement in the region where our exp flushes to zero.
4. **Miri pass on the scalar path (W):** `cargo +nightly miri test` for the
   core crate with NEON cfg'd out — UB detection for the safe-Rust claim.
   (Miri cannot execute NEON intrinsics; scalar-only, documented as such.)

## 4. Performance baseline (the "how much faster" half)

### 4.1 Comparison targets ("improvement relative to WHAT")

| Baseline | Why it matters | Surface |
|---|---|---|
| PyTorch eager reference scan | what every CPU user actually hits today (HF slow path structure) | W + A |
| torch.compile of the reference | the fair fight; the one judges should trust | A (and W if MSVC cl configured; else documented-unavailable) |
| Kernel scalar_seq | our own rung 0 — isolates NEON and threading contributions | W + A |
| Kernel neon_seq / neon_par | the product | A |

### 4.2 Benchmark matrix

**Tier 0 — Rust criterion ladder** (exists; extend):
- backends × threading × shapes, plus a NEW **thread-scaling sweep**:
  `RAYON_NUM_THREADS ∈ {1,2,4,8,16,32}` at the mamba-130m shape.
  On the 32-core Windows box this is the strongest scaling study available
  to us anywhere (CI runners have 4 cores) — scalar backend scaling on W,
  NEON scaling on A up to 4.
- Report elements/s so W and A numbers are comparable per-core.

**Tier 1 — op level, Python** (exists; extend `bench_op.py`):
- **Sequence-length sweep:** L ∈ {64, 128, 256, 512, 1024, 2048, 4096, 8192}
  at B=1, D=768 — produces the O(L) curve and the kernel-vs-eager gap as a
  function of L (the pitch's core plot).
- **Channel sweep:** D ∈ {256, 768, 1536, 3072} at L=512 — shows threading
  saturation behavior.
- **Batch sweep:** B ∈ {1, 4, 8} at D=1536, L=1024.
- torch fairness controls: record `torch.get_num_threads()`, run one
  configuration with `torch.set_num_threads(1)` vs default to bound
  intra-op parallelism effects; kernel gets the same treatment via the
  threading knob.

**Tier 2 — end to end** (exists; extend `bench_e2e.py`):
- Prompt-length sweep: 128 / 512 / 2048 tokens × 32 new tokens,
  mamba-130m; optionally mamba-370m for a model-size point.
- Report prefill ms, decode tok/s, total; tokens asserted identical.

### 4.3 Measurement discipline

- Median of ≥10 reps after ≥3 warmups (CI-quick mode: 5/1, labeled).
- Windows noise control: High Performance power plan, plugged in, no
  foreground apps; each suite run twice in separate processes — if medians
  differ >5%, the run is discarded and repeated.
- Shared-CI noise: numbers labeled provisional; each headline claim
  reproduced on Ampere (dedicated) before use in submission materials.
- Every JSON result embeds host, core count, torch version, thread counts,
  git SHA. Results land in `bench/results/` with a generator script that
  renders RESULTS.md tables from the JSONs — no hand-copied numbers.

### 4.4 Automation

- New CI workflow `bench.yml` with `workflow_dispatch` (manual trigger) +
  weekly cron: full (non-quick) Tier-1 sweep on ubuntu-24.04-arm, results
  posted as commit comment + artifact JSON. Keeps the per-push CI fast
  while making deep numbers one click away.
- Ampere runbook already in bench/README.md; extended with the sweep
  commands and a `--json` naming convention.

## 5. Execution order & effort

| Step | Surface | Effort | Blocks |
|---|---|---|---|
| 1. QEMU arm64 local suite (script + first run) | Q | ~1h (Docker install user-side) | nothing |
| 2. Fuzz soak + long-L stress + Miri | W/Q | ~1h | nothing |
| 3. bench_op sweeps + thread-scaling bench + results renderer | W | ~2h | nothing |
| 4. Run full W baseline (scalar + scaling on 32 cores) | W | ~30min wall | 3 |
| 5. bench.yml dispatch workflow; full A sweep on CI | A | ~1h | 3 |
| 6. Ampere dedicated run (user provisions instance) | A | ~30min once instance exists | 3 |
| 7. RESULTS.md baseline document w/ W/Q/A-tagged tables | — | ~30min | 4,5 |

Steps 1–5 need nothing from the user except Docker Desktop (step 1 only).
Step 6 is the only one requiring a cloud instance.

## 6. Risks / validity notes

- **QEMU timing is never a number.** Emulated runs validate instruction
  semantics only; any timing from Surface Q is excluded by construction.
- **CI runners are shared**: expect ±10-20% run-to-run; medians + the
  Ampere reproduction bound this.
- **torch.compile on Windows** needs MSVC `cl` on PATH (VS2019 exists on
  this box; a devshell wrapper may enable it — attempted once, else the
  compile baseline stays A-only, as it already works there).
- **32-core W scaling ≠ Arm scaling** — cache topology differs; the W curve
  demonstrates the parallelization design scales, the A curve (4 cores CI,
  4 OCPU Ampere) demonstrates it on target silicon.
