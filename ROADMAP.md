# ROADMAP — How this gets built and tested (mostly for free)

Deadline: **Aug 14, 2026, 4:00 PM PDT** (~6.5 weeks from July 2).
Principle: **develop and test on free Arm hardware; rent Graviton4 only for a few hours of final headline numbers.**

---

## 1. Compute strategy (free-first)

| Tier | Hardware | Cost | Used for |
|---|---|---|---|
| **Daily dev** | Oracle Cloud **Always-Free Ampere A1** (Arm Neoverse N1, up to 4 OCPUs / 24 GB RAM, permanently free tier) | $0 | Writing/debugging the kernel, correctness tests, iterating on the PyTorch bridge, running the model on CPU. Neoverse N1 has full NEON — same intrinsics as Graviton. |
| **CI** | **GitHub Actions arm64 runners** (free for public repos, e.g. `ubuntu-24.04-arm`) | $0 | Correctness suite + build on every push; the green-badge-on-real-Arm DX signal. |
| **Local (if available)** | Apple Silicon Mac / Raspberry Pi 5 | $0 | Instant edit-compile-test loop; NEON is NEON. Also validates the "runs on a judge's MacBook" path. |
| **Headline numbers** | AWS **Graviton4 `c8g`** on-demand (e.g. `c8g.4xlarge`/`c8g.16xlarge`) for the benchmark sessions + demo video | ~$5–20 total | Final op benchmarks, core-scaling curve, Performix profiles, end-to-end recon timing, cost table, video recording. Two or three sessions of 1–3 hours each; terminate the instance between sessions (script the setup so it rebuilds in minutes). Check whether the hackathon or AWS's current free-tier credits cover this — they may. |
| **GPU** | none | $0 | Deliberately not needed: we use a **published checkpoint**, never train. The GPU column in the cost table cites public on-demand pricing; if we want a measured GPU number, one ~$1 spot hour suffices — optional. |

Everything else is free: fastMRI data (free with registration; we never redistribute it), Arm Performance Studio / Performix (free download), Rust toolchain, PyTorch CPU wheels, PyPI publishing, Gradio, YouTube hosting.

---

## 2. How it's built (architecture of the work)

**Layer 0 — Ground truth.** Extract `selective_scan_ref` (the pure-PyTorch reference in `mamba-ssm`) into our test harness. Every implementation we ever write is compared against it. This is the single most important artifact in the project: it makes correctness a mechanical check instead of a judgment call.

**Layer 1 — Scalar Rust kernel.** A plain, readable Rust implementation of the recurrence (discretize → scan → output projection), no intrinsics. Purpose: nail the math, the memory layout (B, D, L, N dims; strides; contiguity), and the FFI signature while everything is still easy to debug. Kept forever as the in-crate reference.

**Layer 2 — Optimizations, one at a time, each gated by the test suite:**
1. **Fused discretization** — compute `exp(Δ·A)` and `Δ·B·x` inside the scan loop instead of materializing intermediates (memory-traffic win, often bigger than the ALU win).
2. **NEON vectorization** — `core::arch::aarch64` intrinsics across `d_state` (16 → four `float32x4` lanes) and channel blocks. Stable Rust; no nightly.
3. **Chunked/associative scan** — exploit the linearity of the recurrence (the Mamba-2/SSD insight) to process time in chunks: compute per-chunk transition products in parallel, then a short sequential pass over chunk boundaries. Converts the time axis from latency-bound to throughput-bound.
4. **Threading** — rayon across batch × channel groups. This is where the Cloud-track story lives: publish the 1→N-core scaling curve.
5. **2D/bidirectional variant** — the VMamba-style multi-directional scan the MRI model actually uses (forward/backward × row/column). Reuses the 1D core with different traversal orders; the traversals are independent → more thread-level parallelism.
6. *(Stretch, in priority order)* PyPI aarch64 wheels via `maturin` → mamba-130m tokens/sec generalization benchmark → BF16-storage/fp32-accumulate experiment (Graviton4 only) → SVE2 path (nightly Rust).

**Layer 3 — PyTorch bridge.** Rust `cdylib` with `extern "C"` entry points → minimal C++/`pybind11` glue registering a PyTorch custom op → a Python monkey-patch/shim so the published model code calls our op instead of the fallback, **without modifying the checkpoint or the model source**. A contiguity-normalizing wrapper on the Python side keeps the FFI surface simple (contiguous fp32 in, contiguous fp32 out).

**Layer 4 — Application.** The chosen MRI model (Week-1 gate: MambaRecon vs. DH-Mamba, U-Mamba fallback) running end-to-end on CPU with our op swapped in; the Gradio side-by-side demo; the benchmark + quality scripts.

---

## 3. How it's tested (thoroughly, all free)

Correctness has four independent nets, all runnable on the free tier and in CI:

1. **Unit tests vs. `selective_scan_ref`** (pytest). Random tensors across a grid of shapes (batch, d_model, d_state, seq length incl. edge cases L=1, odd L, L≫chunk size), asserting `max|Δ|` within fp32 tolerance (rtol/atol tuned once, then frozen). Runs in seconds; zero data.
2. **Property-based tests** (Rust `proptest`): random shapes/strides/values, invariants like "chunked result == sequential scalar result," "threaded == single-threaded," "NEON == scalar." These catch the FFI/stride/edge-case bugs that unit grids miss.
3. **Determinism & sanitizer passes**: run the suite under `cargo miri` (UB detection) where feasible and with `RUSTFLAGS` debug assertions; assert bit-identical output across repeat runs at fixed thread count (catches data races).
4. **End-to-end quality gate**: reconstruct fastMRI validation slices with (a) reference fallback and (b) our op; assert PSNR/SSIM/NMSE deltas ≤ tolerance. This is the "the whole model still works" test. A **synthetic Shepp–Logan phantom** variant of the same script ships in-repo so anyone (including judges) can run an end-to-end check with no dataset.

Performance testing, kept honest:

- **Microbenchmarks** with `criterion` (Rust) + a Python harness timing the op through the full PyTorch bridge (so FFI overhead is included, not hidden).
- **Two baselines**: stock fallback *and* `torch.compile` on the reference — pre-empting the "you beat a strawman" critique.
- **Methodology hygiene**: pinned CPU frequency awareness, warmup iterations, median-of-N reporting, fixed thread counts per row, versions pinned in a lockfile. All benchmark scripts checked in; `RESULTS.md` states the exact instance type, AMI, and commands.
- **Performix profiles** before/after on Graviton (cycles, cache behavior) — both evidence and debugging tool.

CI (free, arm64): build + unit + property tests on every push; a nightly job additionally runs the synthetic end-to-end check. Badges in the README.

---

## 4. Week-by-week plan

**Week 1 (Jul 2–8) — De-risk everything. Gate week.**
- Spin up Oracle Ampere A1; set up Rust + PyTorch-CPU toolchain; script the environment (reused later for Graviton).
- **Trap check #1:** does `mamba-ssm` pip-install without CUDA on arm64? (Historically no — it builds CUDA extensions.) Determine integration shape: patch vs. standalone module.
- **Trap check #2:** which recon checkpoint (MambaRecon / DH-Mamba) downloads and runs on CPU, however slowly, and what is its exact scan structure (directions, d_state, d_model)? Verify U-Mamba fallback in parallel.
- Register for fastMRI access (approval can take days — do this day 1).
- Extract `selective_scan_ref` into the test harness; measure the real CPU-vs-GPU-folklore gap ourselves.
- **Gate:** model + integration shape locked by Jul 8.

**Weeks 2–3 (Jul 9–22) — MVP kernel.**
- Scalar Rust kernel + full correctness suite green (Layer 1, tests 1–3).
- Fused discretization → NEON → chunked scan → rayon, each landed behind green tests.
- C-ABI shim + PyTorch custom op; first end-to-end reconstruction on CPU with our op.
- arm64 CI live.
- **Gate:** end-to-end recon at quality parity, measurably faster than fallback, by Jul 22.

**Week 4 (Jul 23–29) — 2D variant + benchmarks.**
- Bidirectional/2D scan variant matching the model's blocks.
- Full benchmark harness: seq-length sweep, core-scaling curve, `torch.compile` baseline, peak memory.
- First Graviton `c8g` session: real numbers + Performix profiles (this is when the ~$5–20 spend starts).
- Synthetic-phantom demo path; quality-gate script finalized on real fastMRI slices.

**Week 5 (Jul 30–Aug 5) — Stretch + WOW, only if MVP is green.**
- PyPI wheels (`maturin`) → mamba-130m tokens/sec table → Gradio side-by-side demo → BF16 experiment → SVE2 (in that order; stop wherever time runs out).
- Second Graviton session for any re-benchmarks + demo rehearsal.

**Week 6 (Aug 6–13) — Freeze + package.**
- **Code freeze Aug 8.** From here: docs, video, writeup only.
- Dry-run the README setup on a *fresh* instance from scratch — judges test exactly this.
- Record the <3-min video on the Graviton instance (side-by-side demo centerpiece; no copyrighted music).
- Devpost writeup; MIT license visible in the GitHub About sidebar; RESULTS.md final.
- **Submit Aug 12–13, not 3:50 PM on the 14th.**

---

## 5. Standing risk register

| Risk | Mitigation |
|---|---|
| No recon checkpoint runs on CPU | U-Mamba segmentation fallback (pre-verified Week 1); kernel work unchanged |
| `mamba-ssm` won't install CPU-only | Standalone package + shim import path; reframed as a feature ("makes Mamba runnable on CPU") |
| FFI stride/dtype bugs | Contiguity-normalizing Python wrapper; property tests over strides; scalar reference kept in-crate |
| Chunked-scan numerical drift | fp32 accumulation; tolerance gate vs. `selective_scan_ref`; chunk size as tunable |
| Speedup underwhelms vs. `torch.compile` | Threading + chunking are levers `torch.compile` can't reach for a sequential scan; if a table row is close, publish it anyway — honesty over cherry-picking |
| Graviton spend creep | All dev on free tier; Graviton sessions scripted, time-boxed, instance terminated after each |
| fastMRI approval delayed | Registered day 1; synthetic phantom path means development never blocks on it |
| Timeline slip | Stretch list is strictly ordered and droppable; MVP alone (kernel + benchmarks + parity + docs) is a complete submission |
