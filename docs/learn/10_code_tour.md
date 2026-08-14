# 10. Code tour — which file does what

A map of the repository, in the order data actually flows through it.

## The path a call takes

```
your PyTorch code
      │
      ▼
python/arm_scan/op.py          torch.library custom op + fake kernel
      │
      ▼
python/arm_scan/_ffi.py        ctypes: load the .so/.dylib/.dll
      │  C ABI — plain pointers and integers
      ▼
kernel/arm-scan-ffi/src/lib.rs the ONLY place raw pointers exist
      │  safe Rust types
      ▼
kernel/arm-scan-core/src/      the actual maths
```

## The Rust kernel — `kernel/`

### `arm-scan-core/src/`

| File | What it does |
|---|---|
| `lib.rs` | Public types: `ScanDims`, `ScanInput`, `Threading`. Backend dispatch |
| `scalar.rs` | **The reference.** Direct transcription — clarity over speed. Correctness baseline *and* non-Arm fallback. **Not dead code** |
| `float.rs` | The `Float` trait abstracting f32/f64, so one implementation serves both |
| `parallel.rs` | `for_each_channel` — rayon over `batch × dim`. Used by *every* backend, so even scalar scales |
| `neon/mod.rs` | The chunked two-pass NEON kernel. `CHUNK = 128` |
| `neon/exp.rs` | Polynomial `exp` for NEON — the hot spot, ~half of runtime |
| `neon/math.rs` | `softplus`, `silu`, other vectorized helpers |
| `neon/profile.rs` | Per-phase timing hooks |
| `mamba3/scalar.rs` | Mamba-3 reference kernel |
| `mamba3/tiled.rs` | Cache-blocked but **portable** — the NEON kernel's structural twin, runnable on x86. `TILE = 32` lives here |
| `mamba3/mimo.rs` | Rank-`r` MIMO. **Scalar only** — no NEON path |
| `neon/mamba3.rs` | The NEON Mamba-3 kernel |

Everything under `neon/` is `#[cfg(target_arch = "aarch64")]` — **it does not
compile on x86 at all.** That is why `make check-cross` exists.

### `arm-scan-core/tests/`

| File | What it checks |
|---|---|
| `golden.rs`, `golden_mamba3.rs` | recorded input/output pairs replay correctly |
| `property.rs`, `property_mamba3.rs` | invariants over randomly generated inputs |

### `arm-scan-ffi/src/lib.rs`

The C ABI boundary. Null checks, overflow-checked size arithmetic, and
`catch_unwind` so a Rust panic never unwinds into Python. Currently **ABI v7**.

## The Python layer — `python/arm_scan/`

| File | What it does |
|---|---|
| `_ffi.py` | ctypes loader; finds and validates the shared library |
| `op.py` | Mamba-1 `torch.library` custom op + registered fake kernel |
| `mamba3.py` | Mamba-3 op. **Owns shape validation** — the FFI cannot |
| `mamba3_noncausal.py` | `forward + backward − diagonal`. Composes over the public op |
| `bidirectional.py` | Fused both-directions call |
| `ss2d.py` | Mamba-1 four-direction cross-scan as two traversal pairs |
| `ss2d_mamba3.py` | The same layout logic on the Mamba-3 primitive — **no new kernel** |
| `numpy_api.py` | Torch-free path, so correctness can be checked without PyTorch |
| `patch.py` | `arm_scan.patch()` — monkey-patches HF Mamba to use our kernel |

The split between `ss2d.py` and `ss2d_mamba3.py` is the generality argument made
concrete: the topology layer is **recurrence-agnostic**, so a new recurrence gets
2D for free.

## Tests — `tests/`

| File | What it does |
|---|---|
| `golden/` | The recorded ground truth. 53 MB, committed — **no GPU ever needed** |
| `gen_golden.py`, `gen_golden_2d.py` | Regenerate Mamba-1 goldens |
| `verify_golden*.py` | **Independent** NumPy re-derivation, replayed through the real C ABI |
| `reference/selective_scan_ref.py` | Vendored Mamba-1 ground truth |
| `reference/mamba3_ref.py` | Our Mamba-3 CPU reference — the one Stage 1 validated |
| `check_ffi.py` | Goldens through the C ABI |
| `check_mamba3_*.py` | The seven Mamba-3 gates |
| `check_mamba3_model.py` | Real 187M logits vs the official model |

## Applications — `apps/`

| Directory | What it is |
|---|---|
| `mamba3_lm/` | The published 187M Mamba-3 running on CPU through our kernel. `load.py` fetches from HF |
| `mri_diffusion/` | SS2D diffusion MRI reconstruction. **Demoted** — off the critical path, but still CI-gated because it is the only end-to-end exercise of the SS2D kernel |

## Benchmarks — `bench/`

| File | What it measures |
|---|---|
| `bench_op.py` | Mamba-1 kernel vs eager vs `torch.compile` |
| `bench_mamba3.py` | Mamba-3 kernel |
| `bench_mamba3_lm.py` | the real 187M model end to end |
| `bench_ss2d_mamba3.py` | 2D Mamba-3 at vision grid sizes |
| `bench_mamba3_noncausal.py` | causal vs non-causal |
| `bench_longctx.py` | the constant-memory claim — the 128k row |
| `run_baseline.sh` | the full Mamba-1 session incl. core scaling. **No Mamba-3 — see §3b of the run sheet** |
| `GRAVITON_SESSION.md` | the run sheet you execute on the instance |
| `AWS_FIRST_TIME.md` | account setup through termination, for a first-time AWS user |

## Tooling — `tools/`

| File | What it does |
|---|---|
| `capture_mamba3_goldens.py` | **Stage 0.** Records ground truth from the official GPU kernels |
| `check_mamba3_recurrence.py` | Demonstrates the paper-vs-community disagreement |
| `setup_linux.sh` | Provisions a Linux box for Rust gates + golden capture |
| `setup_cuda_toolchain.sh` | Checksum-verified CUDA toolchain for re-capturing goldens |

## Docs — where things live

**Root is judge-facing; current design documents live in `docs/project/`.**

| File | Purpose |
|---|---|
| `README.md` | the pitch and the precise claims |
| `RUNNING_THE_KERNEL.md` | complete setup and execution runbook |
| `CONTRIBUTING.md` | correctness, safety, and benchmarking rules |
| `docs/project/STATUS.md` | where the work stands right now |
| `docs/project/MAMBA3_IMPLEMENTATION_PLAN.md` | stages, prior art, claims policy |
| `docs/project/THREE_PATHS_INTEGRATION.md` | the three demonstrations, scoped |
| `docs/project/MAMBA3_KERNEL_WORKPLAN.md` | file-by-file execution |
| `docs/learn/` | **this folder** — teaching material |
| `docs/archive/` | superseded plans and measurement history — real evidence of how decisions were made |

## Commands worth knowing

```bash
make validate        # the judge path: ~5 min, no data, no AWS account
make test            # kernel gates + goldens through the C ABI
make test-mamba3     # all 7 Mamba-3 gates
make check-cross     # typecheck the aarch64 path from an x86 box, ~1 s
```

```bash
cd kernel && cargo test --release
cd kernel && cargo clippy --all-targets -- -D warnings && cargo fmt --check
```

Run correctness under several thread counts — `RAYON_NUM_THREADS ∈ {1,2,8}` —
because parallel output must be **bit-identical** to sequential.

---

## Where to go next

- The claims and the pitch: [`README.md`](../../README.md)
- Current state and open gaps: [`docs/project/STATUS.md`](../project/STATUS.md)
- The Graviton run: [`bench/GRAVITON_SESSION.md`](../../bench/GRAVITON_SESSION.md)
