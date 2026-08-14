# MAMBA3_KERNEL_WORKPLAN — modifying the kernel for Mamba-3, paths 1–3(i)

**Written Aug 6, 2026.** The file-level execution plan for
[`MAMBA3_IMPLEMENTATION_PLAN.md`](MAMBA3_IMPLEMENTATION_PLAN.md) Stages 2–5,
grounded in the code audit in that document's §14.

**Scope: one kernel, three paths.** 1D unidirectional, 1D bidirectional, and 2D
causal cross-scan all run the *same* Mamba-3 primitive — they differ only in
traversal order, which lives in Python above the kernel. **Non-causal / VNCT is
explicitly out of scope here**: dropping causality removes the recurrence
entirely and yields a GEMM-shaped kernel that shares packaging and tests but
almost no compute code. It gets its own plan once this lands.

**The Mamba-1 path is not touched.** Every item below is additive.

---

## 1. Design decisions — made now, not left open

| Decision | Choice | Why |
|---|---|---|
| Module layout | New `src/mamba3/{mod,scalar}.rs` + `src/neon/mamba3.rs` | Mirrors the existing `scalar.rs` / `neon/mod.rs` split; groups Mamba-3 without touching Mamba-1 |
| Pass B form | **Sequential recurrence**, not the chunked dual form | It is the direct transcription of the verified Stage-1 reference, so correctness is mechanical — and it *preserves the sequential-recurrence moat*, the thing `torch.compile` provably cannot restructure. The dual form is a later optimisation, gated on measurement |
| Vectorisation axis in Pass B | Across the **state matrix**, not across time | Same structure as Mamba-1's Pass B; the `t` dependency stays serial by necessity |
| RoPE angle pre-pass | **Stays in Python** (`cumsum(tanh(angle)·π·dt)`) | This is exactly how upstream splits it — `angle_dt_fwd` is a separate kernel from `mamba3_siso_fwd`. Mirroring it keeps our goldens directly comparable |
| `sin`/`cos` | **Kernel accepts precomputed `cos`/`sin` planes** in v1 | Avoids writing and accuracy-testing new NEON transcendentals on the critical path. Correctness first, fusion second — the repo's standing rule. Revisit only if profiling says the extra memory traffic matters |
| Precision | **f32 in, f32 out**, internally f32 | Callers pass bf16-rounded values when bit-matching the GPU. Do NOT implement bf16 storage yet — it is a separate, measurable decision |
| Threading | New `for_each_head` in `parallel.rs` | `for_each_channel` assumes one output scalar per (channel, t); see §14.3 of the plan |
| MIMO / multi-group | **Rejected at validation** | SISO only. The reference already raises rather than silently truncating |

---

## 2. Layout contract

Chosen to match what the model and our goldens already produce, so **no
transposes are needed anywhere**. All row-major, fully contiguous.

| Tensor | Shape | Notes |
|---|---|---|
| `q`, `k` | `(batch, len, 1, dqk)` | Groups axis is 1 (SISO). Contiguous in `dqk` at each `t` — directly loadable |
| `v`, `z` | `(batch, len, heads, dv)` | Contiguous in `dv` per (t, head) — the rank-1 update's `v` vector |
| `adt`, `dt`, `trap` | `(batch, heads, len)` | Contiguous in `t` — ideal for Pass A |
| `cos`, `sin` | `(batch, len, heads, dqk/2)` | Precomputed by the Python pre-pass |
| `q_bias`, `k_bias` | `(heads, dqk)` | fp32 parameters |
| `d_skip` | `(heads,)` | |
| `out` | `(batch, len, heads, dv)` | Matches the goldens exactly |
| `last_state` | `(batch, heads, dv, dqk)` | Per head, a `dv × dqk` matrix |
| `last_bx` | `(batch, heads, dqk)` | `scale_T * k_T`. **Not a resume carry** — see §6 |

**Note the difference from Mamba-1**, which is `(batch, dim, len)` — channel-major.
Mamba-3 is time-major. That is why `for_each_channel` does not fit.

---

## 3. Work items, in dependency order

### M0 — Types and validation ✅ **DONE**
**New:** `src/mamba3/mod.rs`

```rust
pub struct Mamba3Dims { pub batch, heads, dv, dqk, len: usize }

pub struct Mamba3Input<'a, T> {
    pub q, k: &'a [T],            // (b, l, 1, dqk)
    pub v: &'a [T],               // (b, l, h, dv)
    pub adt, dt, trap: &'a [T],   // (b, h, l)
    pub q_bias, k_bias: &'a [T],  // (h, dqk)
    pub cos, sin: &'a [T],        // (b, l, h, dqk/2)
    pub d_skip: Option<&'a [T]>,  // (h,)
    pub z: Option<&'a [T]>,       // (b, l, h, dv)
    pub reverse: bool,            // the 1D half of bidirectional + 2D
}
```

`Mamba3Variant { Siso = 0 }` is born here with `Mimo = 1` reserved, so MIMO is
an addition rather than a refactor.

Validation mirrors `validate()`: zero dims, exact length checks per tensor,
`dqk % 2 == 0`, and an explicit rejection of anything implying groups > 1.

**Gate:** unit tests for every rejection path.

### M1 — `Float` trait additions ✅ **DONE**
**Modify:** `src/float.rs`

Add `sigmoid()` (trapezoid gate) and `tanh()` — both expressible on the existing
`exp` core, both with a default method body so the impls stay one-liners.
**No `sin`/`cos`** — §1 defers those by taking precomputed planes.

**Gate:** accuracy sweep against `f64` libm, same shape as the existing
exp/softplus/silu tests.

### M2 — `for_each_head` ✅ **DONE**
**Modify:** `src/parallel.rs`

Sibling to `for_each_channel`, chunking by the Mamba-3 shapes:

```rust
pub(crate) fn for_each_head<T, S, I, F>(
    len: usize, dv: usize, dqk: usize,
    out: &mut [T],                    // chunks of len*dv
    last_state: Option<&mut [T]>,     // chunks of dv*dqk
    last_bx: Option<&mut [T]>,        // chunks of dqk
    threading: Threading, init: I, f: F,
)
```

Work heuristic recalibrated: the per-step cost is `~3·dv·dqk` FMAs, not
`state` lane-steps, so `should_parallelize` needs a Mamba-3 variant.

**Gate:** a property test that output is bit-identical at
`RAYON_NUM_THREADS ∈ {1,2,8}`. Holds by construction — heads are disjoint, no
cross-thread reduction — but assert it, don't assume it.

### M3 — Scalar reference kernel ✅ **DONE**
**New:** `src/mamba3/scalar.rs`

Direct transcription of `tests/reference/mamba3_ref.py`. Clarity over speed;
this is the in-crate oracle and the non-Arm fallback.

Per (batch, head), with `S: [dv][dqk]`:

```
for i in 0..len:
    t = if reverse { len-1-i } else { i }
    λ  = sigmoid(trap[t]);  γ = dt[t]·λ
    λ' = sigmoid(trap[t±1]); scale = γ + dt[t±1]·(1-λ')   // ± follows traversal
    α  = exp(adt[t])
    q_t = rope(q[t] + q_bias, cos[t], sin[t])
    k_t = rope(k[t] + k_bias, cos[t], sin[t])
    y   = α · (q_t · Sᵀ)                                   // (dqk)·(dv,dqk) -> dv
    y  += (d_skip + γ·(q_t·k_t)) · v[t]                    // diagonal term
    S   = α·S + scale·(v[t] ⊗ k_t)                         // rank-1 update
    out[t] = y ⊙ silu(z[t])
```

**Two details the reference proved and that are easy to get wrong:**
- RoPE is **interleaved** — pairs `(2i, 2i+1)`, not split-halves.
- Under `reverse`, the 2-tap looks the *other* way: `scale` reads `t+1` forward
  and `t-1` backward. This is the one place `reverse` is more than an index flip.

**Gate:** matches all 10 Stage-0 goldens within the bf16 bound (≤ 8 ULP at
tensor scale), replayed by a new `tests/golden_mamba3.rs`.

### M4 — NEON kernel ✅ **DONE** (split M4a portable-blocked / M4b intrinsics)
**New:** `src/neon/mamba3.rs`

Chunked two-pass, same skeleton as `neon/mod.rs`, reusing `chunks_in_scan_order`
and `vexpq_f32_nonpos` directly.

**Pass A** (per `CHUNK` timesteps, vectorised across time — all pointwise):
`λ = sigmoid(trap)`, `γ`, `scale`, `α = exp(adt)` via the existing non-positive
exp, and the rotated `q_t`/`k_t` (even/odd lane pairs with `vfma`/`vfms`).

**Pass B** (vectorised across the state matrix, serial in `t`) — **the new
engineering.** `S` is `dv × dqk` = 8 KB at the sweep shapes, **32 KB at the
187M shape**, against 64 KB L1d/core on the Neoverse N2 we measured. It cannot
be register-resident like Mamba-1's 16-float state.

Structure: **block over `dqk` in tiles**, and for each tile do the update and
its contribution to `y` in one pass, so `S` is touched once per timestep:

```
for tile in dqk.chunks(TILE):          # TILE ~ 32 f32 = 8 NEON registers
    for row in 0..dv:
        S[row][tile] = α·S[row][tile] + (scale·v[row])·k_t[tile]   # FMA
        y[row]      += q_t[tile] · S[row][tile]                    # dot, accum
```

`TILE` is a **tunable to be swept, not derived** — start at 32, sweep
{16, 32, 64} at the real shapes. The plan's §14.2 warning applies: choose it by
measurement.

**Epilogue:** `out = (y + d_skip·v) ⊙ silu(z)`, vectorised across `dv`.

**Chunk carry:** `S` **and** `bx_last` (`k_t·scale` from the boundary step).

**Gates:** NEON↔scalar parity ≤ 3e-7; goldens at the bf16 bound; rayon
bit-identity at 1/2/8 threads.

### M5 — Dispatch and public API ✅ **DONE**
**Modify:** `src/lib.rs`

`pub fn mamba3_scan(...)` / `..._with_options(...)` / `..._with_state(...)`
mirroring the Mamba-1 family, with the same `Backend`/`Threading` enums reused
verbatim. `mod mamba3;` alongside the existing modules.

### M6 — FFI ✅ **DONE** (ABI 6; bumped to **7** by B2 for MIMO)
**Modify:** `arm-scan-ffi/src/lib.rs`

Third entry point `arm_scan_mamba3_scan_f32` with `#[repr(C)] Mamba3DimsC`.
Same discipline: null checks, overflow-checked size arithmetic before any slice
is formed, `catch_unwind` at the boundary. **Bump `arm_scan_abi_version()`** and
update the Python loader's check.

**Gate:** goldens replayed through the **real C ABI**, extending
`tests/check_ffi.py`.

### M7 — Python op + topologies ✅ **DONE**
**New:** `python/arm_scan/mamba3.py`

- `_ffi.mamba3_raw` ctypes binding.
- `torch.library.custom_op` **with a registered fake kernel** so it composes
  with `torch.compile` — non-negotiable, since `torch.compile` is the baseline
  every number is measured against.
- The angle pre-pass: `theta = cumsum(tanh(angle)·π·dt)`, then `cos`/`sin`.
- `mamba3_scan_pair(...) -> (fwd, bwd)` for the two topologies below.

**Path 2 (bidirectional):** `bidirectional_scan` already takes an injected
primitive — point it at `mamba3_scan_pair`. **Zero new orchestration.**

**Path 3(i) (2D causal cross-scan):** `ss2d.py`'s `grid_to_views` /
`views_to_grid` / merge are pure layout ops with no recurrence knowledge. The
only change is widening the `scan_pair` seam, whose current signature is
Mamba-1's parameter list — either a tensor-bundle form or a parallel
`ss2d_scan_mamba3` sharing the same helpers.

**Gate:** per-direction 2D goldens (non-square, non-multiple-of-4 grids) plus
pair-vs-oracle parity at 1/2/8 threads — the same shape as the existing SS2D
gate.

### M8 — Bench + CI ✅ **DONE**
`bench/bench_mamba3.py` on the ladder (scalar → NEON → NEON+rayon) against
eager and `torch.compile`. Extend the `mri-app` arm64 CI job (which already has
torch) with the new gates. `make test-mamba3` grows to cover the Rust path.

---

## 3a. Status — all of M0–M8 complete (Aug 7, 2026)

Verified on x86: 28 Rust tests, `make test`, `make test-mamba3`
(reference + torch op), fmt and clippy clean on **both** targets, aarch64
cross-compiles including all test targets. Verified on Arm by CI: the NEON
gates (`mamba3_neon_matches_naive`, `mamba3_neon_parallel_bit_identical`) pass
on `linux-arm64` and `macos-arm64`.

Worst deviation from the captured official-kernel ground truth is **4.47 bf16
ULP** (bound 8) at every layer — Python reference, Rust scalar, Rust blocked,
and the torch op — which is the floor bf16 output quantisation allows.

**Dedicated-hardware numbers: DONE Aug 11, 2026** on `c8g.16xlarge` (Graviton4,
Neoverse-V2, 64 vCPU) — see `README.md` and `bench/results/`. The SS2D pair
rewrite **regressed** there (0.96× vs 1.80× on x86) and the P1-7 verdict
**flipped to justified** at 46.1% overhead, which makes a fully fused
`selective_scan_2d` the highest-value remaining kernel work.

## 4. Sequencing

| Step | Work | Est. |
|---|---|---|
| M0–M2 | Types, `Float` additions, `for_each_head` | ~0.5 day |
| M3 | Scalar kernel + goldens | ~1 day |
| **M4** | **NEON kernel** | **2–3 days** |
| M5–M6 | Dispatch + FFI | ~0.75 day |
| M7 | Python op + both topologies | ~0.5 day |
| M8 | Bench + CI | ~0.25 day |

**~5–6 days.** M4 dominates and carries all the uncertainty; everything else is
pattern-following. Note M3 alone already yields a correct, portable, threaded
Mamba-3 kernel — slow, but shippable, and it unblocks M7's topology work in
parallel with M4.

## 5. Risks

| Risk | Mitigation |
|---|---|
| `S` tiling underperforms | `TILE` is a swept parameter; the scalar path is always a correct fallback. Profile before optimising — the phase profiler already exists |
| 32 KB state thrashes L1 at the 187M shape | Measure at both sweep and production shapes; blocking over `dv` as well as `dqk` is the escape hatch |
| `reverse` 2-tap direction bug | Highest-risk correctness item, since it is the one place `reverse` is more than an index flip. Covered by a dedicated golden and by flip-forward-flip equivalence |
| Arithmetic intensity claim doesn't hold | It is a hypothesis, not a result. Measure at M4; publish either way |
| Scope creep into MIMO / non-causal | Both rejected at validation; non-causal has its own plan |

## 6. Found in review: Mamba-3 is not resumable, and cannot be made so by a carry

`last_bx` is plumbed through the core, the C ABI and the dims struct, and it is
**written but never read**. More importantly it does not do what its name
suggests: `scale_t = dt_t*lam_t + dt_{t+1}*(1 - lam_{t+1})` depends on the
*next* timestep, so the trapezoid looks forward and a segment's last step cannot
be completed without the first step of the segment after it.

Mamba-1's `h0` / `last_state` contract therefore **does not transfer**. A
resumable Mamba-3 needs lookahead into the following chunk — an extra *input* —
not a carry out of the preceding one. That is a design item for any future
decode or chunked path, not a bug to patch.

Documented at all three layers rather than removed, since the plumbing is
already through the ABI and a chunked path will want the slot. The Python layer
deliberately does not expose it.

## 7. Explicitly not doing

Chunked dual form (Pass B stays sequential until measured), bf16 storage, fused
`sin`/`cos`, MIMO, non-causal/VNCT, and any decode-step specialisation. Each is
a named next lever, not a gap.
