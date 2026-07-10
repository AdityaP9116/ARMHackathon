# Integration Plan — Arm-Optimized Selective Scan Kernel (Rust + NEON)

**Goal:** a drop-in, pip-installable replacement for the slow CPU selective-scan path in the
PyTorch/Mamba ecosystem, hand-optimized for Arm CPUs. One kernel; every Mamba model that
routes through it gets faster on Arm.

---

## 0. The operation being replaced (mathematical spec)

The selective scan (`selective_scan_fn` in `mamba-ssm`, `slow_forward` in HF `transformers`):

Inputs (shapes use B=batch, D=channels, L=seq len, N=state size, typically N=16):

| Tensor | Shape | Role |
|---|---|---|
| `u` | (B, D, L) | input activations |
| `delta` | (B, D, L) | timestep (post-softplus) |
| `A` | (D, N) | state matrix (negative, real) |
| `B` | (B, N, L) | input projection (input-dependent) |
| `C` | (B, N, L) | output projection (input-dependent) |
| `D_skip` | (D,) | skip connection |
| `z` | (B, D, L) | optional gate (SiLU) |

Per (batch b, channel d), with state `h ∈ R^N` initialized to 0, for t = 0..L-1:

```
Ābar_t = exp(delta[t] * A[d, :])          # ZOH discretization, elementwise over N
Bbar_t = delta[t] * B[:, t] * u[t]        # Euler discretization (matches mamba-ssm)
h      = Ābar_t ⊙ h + Bbar_t              # the sequential recurrence
y[t]   = dot(C[:, t], h) + D_skip[d] * u[t]
out[t] = y[t] * silu(z[t])                # if gated
```

Key structural facts that drive the whole design:
- The recurrence is a **first-order linear recurrence** `h_t = a_t h_{t-1} + b_t`, which is
  **associative** under `(a, b) ∘ (a', b') = (a·a', a'·b + b')` — this is what makes the
  chunked/parallel scan possible and is the thing `torch.compile` cannot do.
- Cost per timestep per channel ≈ 16 `exp` + ~48 FLOPs. **`exp` dominates.** A fast NEON
  polynomial `exp` is the single biggest performance lever.
- (b, d) pairs are fully independent → embarrassingly parallel across B×D (D is 1536+ in
  real models), which is where rayon goes.

---

## 1. Repository & crate architecture

```
ARMHackathon/
├── kernel/                     # Rust workspace
│   ├── Cargo.toml              # [workspace]
│   ├── arm-scan-core/          # pure-Rust kernel library (unit-testable, no FFI)
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── scalar.rs       # reference scalar implementation (ground truth in Rust)
│   │       ├── neon/
│   │       │   ├── mod.rs      # runtime feature detection + dispatch
│   │       │   ├── exp.rs      # fast NEON exp (polynomial)
│   │       │   └── scan.rs     # vectorized sequential + chunked scan
│   │       ├── chunked.rs      # associative chunked scan (arch-independent logic)
│   │       └── parallel.rs     # rayon work partitioning over (B × D)
│   └── arm-scan-ffi/           # cdylib, C-ABI surface only (thin, all unsafe here)
│       └── src/lib.rs
├── python/
│   └── arm_scan/
│       ├── __init__.py         # load lib, expose selective_scan()
│       ├── op.py               # torch.library custom-op registration
│       └── patch.py            # monkeypatch entry points for HF transformers / mamba-ssm
├── tests/
│   ├── golden/                 # .npy test vectors generated from PyTorch reference
│   └── gen_golden.py           # generates them (runs the pure-PyTorch ref scan)
├── bench/
│   ├── microbench.rs           # criterion, op-level
│   └── e2e.py                  # tokens/sec end-to-end vs torch.compile baseline
└── .github/workflows/ci.yml    # arm64 runners: build, test, wheel
```

Two-crate split is deliberate: `arm-scan-core` is safe Rust with isolated `unsafe` NEON
blocks and full unit tests; `arm-scan-ffi` is the only place raw pointers cross a boundary.

---

## 2. Phase-by-phase plan

### Phase 0 — Ground truth & de-risk (days 1–3)

The kernel is worthless without an unimpeachable correctness oracle. Do this first.

1. On any machine (x86 is fine), `pip install torch` and vendor the **pure-PyTorch reference
   scan** (`selective_scan_ref` from mamba-ssm's repo — it's a single self-contained
   function; copy it, don't install mamba-ssm, which needs CUDA to build).
2. Write `tests/gen_golden.py`: runs the reference at **float64**, saves inputs + outputs as
   `.npy` for a grid of shapes — `(B, D, L, N)` ∈ {(1,4,8,16), (2,64,128,16), (1,1536,512,16),
   (4,768,1024,16)} plus edge cases L=1, D=1, and with/without `z` gating.
3. Also golden-check the target model path: load the chosen application checkpoint
   (audio/ECG/RF — whichever wins the pivot) on CPU, run one forward pass, confirm it routes
   through the reference scan, and record layer input/output pairs. This doubles as the
   Week-1 checkpoint-viability flag from PROJECT_CONCEPT.md.

**Exit criteria:** golden vectors committed; target model runs on CPU through the reference
scan end to end.

### Phase 1 — Rust scaffold + scalar kernel (days 3–7)

1. `cargo new` the workspace as laid out above. Core API (all-contiguous v1):

```rust
pub struct ScanDims { pub batch: usize, pub dim: usize, pub len: usize, pub state: usize }

/// u:(B,D,L) delta:(B,D,L) a:(D,N) b:(B,N,L) c:(B,N,L) d_skip:(D) z:opt(B,D,L) out:(B,D,L)
pub fn selective_scan_f32(dims: ScanDims, u: &[f32], delta: &[f32], a: &[f32],
                          b: &[f32], c: &[f32], d_skip: &[f32],
                          z: Option<&[f32]>, out: &mut [f32]);
```

2. Implement `scalar.rs` as a direct transcription of the math — clarity over speed. Use
   `f32` state (matching what PyTorch does) but validate against the f64 goldens with
   tolerance `max_abs_err < 1e-4`, `mean_rel_err < 1e-5` (loose enough for f32 exp, tight
   enough to catch real bugs).
3. Test harness: `ndarray-npy` (or a 60-line hand-rolled .npy reader) loads goldens;
   `proptest` fuzzes shapes/values against a second naive implementation.
4. Wire criterion microbenchmarks now, so every later optimization has a baseline number
   from day one.

**Exit criteria:** scalar kernel passes all goldens + property tests on Apple Silicon and on
a GitHub Actions `ubuntu-24.04-arm` runner.

### Phase 2 — NEON vectorization (week 2)

The core register-level design, per (b, d) pair, N=16:

1. **State in registers:** `h` is 16 floats = four `float32x4_t`. Load once, keep resident
   across the entire sequence loop — zero state memory traffic.
2. **Hoist the A row:** `A[d, :]` is loop-invariant → four registers loaded once per channel.
3. **Fast NEON exp** (`neon/exp.rs`): range-reduce `x = k·ln2 + r`, degree-5 polynomial for
   `2^r`, reassemble exponent with integer ops. ~10 instructions per vector vs. a libm call
   per lane. Unit-test exhaustively against `f64::exp` over the domain delta·A actually
   occupies (A < 0, delta ∈ (0, ~10) → argument ∈ (−∞, 0]; clamp underflow to 0).
4. **Inner loop (fused discretization):** per timestep, entirely in registers:
   `vmulq(delta_t, a_row)` → NEON exp → `vfmaq` for `h = ābar·h + (delta_t·u_t)·B_t` →
   `vfmaq` dot-product accumulation with `C_t` → horizontal `vaddvq` for `y_t`.
   No intermediate tensors ever touch memory — this *is* the fusion lever.
5. **Dispatch:** `std::arch::is_aarch64_feature_detected!("neon")` at init; scalar fallback
   retained (also keeps x86 CI able to run correctness tests via the scalar path).
6. Handle N not divisible by 4 with a masked scalar tail (N=16 is the fast path; keep the
   general path correct, not fast).

**Exit criteria:** bit-tolerance parity with scalar path on all goldens; criterion shows the
expected ~3–4× over scalar on one core.

### Phase 3 — Chunked scan + rayon (weeks 2–3)

1. **Chunking (`chunked.rs`):** split L into chunks of ~64–256 (tune to L1/L2). Two-pass
   structure per chunk using the associative composition `(a, b) ∘ (a', b') = (a·a', a'·b + b')`:
   - Pass A (parallel-friendly, vectorized **across time**): precompute all `ābar_t` and
     `bbar_t` for the chunk — batches the exps for maximum throughput.
   - Pass B (sequential within chunk, vectorized **across state**): run the recurrence and
     the `C` dot products.
   Chunk boundaries carry `h` forward sequentially; since chunks also expose each chunk's
   composed `(A_prod, B_comb)`, single-sequence latency can later be parallelized across L
   too (stretch — only needed when B×D is small, e.g. B=1 streaming audio).
2. **Threading (`parallel.rs`):** rayon `par_chunks` over the B×D pairs, contiguous channel
   blocks per thread (cache-friendly, zero synchronization — pairs are independent). Respect
   `RAYON_NUM_THREADS`; also expose a thread-count arg through the FFI so PyTorch's own
   thread pool doesn't oversubscribe against rayon.
3. Add a heuristic: below a work threshold (B·D·L small), skip rayon — thread spawn overhead
   dominates tiny problems and would poison the microbenchmarks.

**Exit criteria:** near-linear scaling to 4 cores on Oracle Ampere A1 for D≥256; goldens
still pass under all thread counts (run tests with `RAYON_NUM_THREADS ∈ {1, 2, 8}`).

### Phase 4 — PyTorch integration (week 3)

1. **FFI (`arm-scan-ffi`):** one `#[no_mangle] extern "C"` function taking raw `*const f32`
   pointers + the dims struct; returns an error code. Contiguous-only in v1 — the Python
   wrapper calls `.contiguous()` (cheap, and honest to benchmark). All pointer validation at
   the boundary; `catch_unwind` so a Rust panic never unwinds into Python.
2. **Python op (`op.py`):** register via `torch.library.custom_op("arm_scan::selective_scan", ...)`
   with a `register_fake` shape function — this makes the op **compose with `torch.compile`**
   instead of graph-breaking, which matters because the fair baseline is torch.compile.
   Load the cdylib via `ctypes` (no pybind/pyo3 build complexity; the ABI is 1 function).
3. **Patch layer (`patch.py`)** — the "drop-in" story, two targets in priority order:
   - **HF `transformers`**: patch `MambaMixer.slow_forward`'s inner loop (this is the path
     *every* HF Mamba model hits on CPU — highest leverage, covers language models for the
     "it generalizes" benchmark).
   - **The application model** (MambaRecon-style or the audio/ECG/RF pick): patch its local
     scan function by module path.
   Both are `arm_scan.patch()` → returns list of what got patched; `arm_scan.unpatch()` for
   A/B benchmarking in the same process.
4. **Numerics disclosure:** document that results match the reference to fp32 tolerance, not
   bit-exactly (exp approximation + FMA reassociation); show output-level model metrics
   (e.g. SNR/accuracy on the demo task) are unchanged.

**Exit criteria:** `pip install -e .; python -c "import arm_scan; arm_scan.patch()"` →
HF Mamba model produces tolerance-identical logits with the kernel engaged (assert via a
call counter), under both eager and `torch.compile`.

### Phase 5 — Packaging & CI (week 3–4, overlaps)

1. **Wheels via maturin** (`maturin build --release`), `manylinux_2_28_aarch64` +
   `macosx_arm64`. GitHub Actions matrix:
   - `ubuntu-24.04-arm`: cargo test, clippy, golden tests, wheel build.
   - `macos-14` (Apple Silicon): cargo test + wheel.
   - `ubuntu-latest` (x86): scalar-path correctness only (keeps the safety net cheap).
2. CI runs the *Python-level* golden test against the built wheel — catches ABI/stride bugs
   that Rust unit tests can't see.
3. Version-pin nothing torch-side except `torch>=2.1` (custom_op API floor).

**Exit criteria:** green CI producing installable aarch64 wheels on every push.

### Phase 6 — Benchmarks (week 4)

Three tiers, all reproducible from `bench/` with pinned seeds and reported hardware:

1. **Op-level (criterion + Python timeit):** kernel vs. (a) naive PyTorch ref, (b)
   `torch.compile`'d ref, at shapes (1,768,{128,512,2048},16) and (8,1536,1024,16).
   Report the ladder: scalar → +NEON → +chunking → +rayon, so each lever's contribution
   is visible. Target: 2–5× over torch.compile at op level.
2. **End-to-end (tokens/sec or samples/sec):** HF Mamba-130M/370M generate() on CPU,
   patched vs. unpatched-compiled. Target: 1.5–2.5×.
3. **Application demo:** the chosen signal task (audio/ECG/RF) — real-time factor
   (audio-seconds processed per wall-second) before/after, on Oracle Ampere (free) with
   headline numbers reproduced once on Graviton (~$5–20 budget per ROADMAP.md).

Report medians of ≥10 runs after warmup, single-tenant instances, fixed thread counts.

### Phase 7 — Application demo & submission wrap (week 4–5)

1. Wire the chosen model through `arm_scan.patch()`; build the minimal demo (e.g. audio:
   noisy WAV in → denoised WAV out → listen; plus RTF number on screen).
2. Update README/PROJECT_CONCEPT/ROADMAP to the final track + application (currently they
   still say MRI/Cloud); add the results table and a 3-minute demo video script.

---

## 3. Test strategy (running throughout, not a phase)

| Net | What it catches |
|---|---|
| Golden vectors (PyTorch f64 → Rust) | math transcription errors |
| Scalar ↔ NEON ↔ chunked ↔ threaded parity tests | each optimization layer independently |
| proptest fuzzing (shapes, extreme delta/A values) | edge cases, over/underflow in exp |
| Python-level golden test on the wheel | FFI/stride/ownership bugs |
| Model-level metric check (logits / task metric) | "fast but subtly wrong" |

Rule: **every optimization layer must reproduce the previous layer's outputs within
tolerance before it can be benchmarked.** Speed claims only from correct kernels.

---

## 4. Risks & mitigations

| Risk | Mitigation |
|---|---|
| NEON exp accuracy tail | exhaustive domain testing vs f64; clamp underflow; f64 accumulation available behind a feature flag |
| rayon vs PyTorch thread oversubscription | explicit thread-count knob through FFI; document `OMP_NUM_THREADS` interplay |
| HF transformers internals shift between versions | patch() probes and reports; pin tested version range; local-model patch path as fallback |
| torch.compile baseline being surprisingly good at large L | chunked scan is the moat — compile can't parallelize the recurrence; benchmark honestly and show the ladder |
| Only x86 dev machine locally | Apple Silicon / GH Actions arm64 / Oracle A1 per ROADMAP; scalar path keeps x86 dev viable |
| SVE2 stretch eats schedule | strictly after Phase 6 numbers are locked; NEON is the product |

---

## 5. Dependency graph (what blocks what)

```
Phase 0 (goldens) ──► Phase 1 (scalar) ──► Phase 2 (NEON) ──► Phase 3 (chunk+rayon)
                            │                                        │
                            └──────────► Phase 4 (FFI+PyTorch) ◄─────┘  (FFI can start
                                              │                          against scalar)
                             Phase 5 (CI/wheels, parallel from wk3)
                                              │
                                        Phase 6 (bench) ──► Phase 7 (demo + docs)
```

Phase 4's FFI/patch work can begin against the **scalar** kernel as soon as Phase 1 lands —
integration risk retired early, optimizations drop in behind a stable ABI.
