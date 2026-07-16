# TOPOLOGY_IMPLEMENTATION_PLAN — 1D bidirectional & 2D cross-scan (SS2D)

**Status:** the plan. Written Jul 14, 2026. Companion to [`APPLICATIONS.md`](APPLICATIONS.md) (which topology feeds which showcase app) and [`INTEGRATION_PLAN.md`](INTEGRATION_PLAN.md) (Phases 0–6, already landed).

> **Execution is tracked in per-topology logs** — what has actually been built against this plan, what it is verified against, and what broke on the way. This file stays the plan; the logs are the record.
> - §2 (1D bidirectional) → [`BIDIRECTIONAL_LOG.md`](BIDIRECTIONAL_LOG.md). **§2.1 is landed and verified on Arm. §2.2 was measured and REJECTED** — the flip copies cost ~2%, not enough to fuse. Read that before touching §2.2.
> - §3 (2D cross-scan / SS2D) → not started; gets its own log when it does. Note §3's copy overhead is a *different* question — it materializes four grid views, not one flipped sequence — so §2's rejection does **not** transfer to it. Measure it separately.

**Scope:** how to take 1D bidirectional and 2D cross-scan from "correct via Python rearrangement" to "fast AND correct in Rust," using the already-shipped 1D unidirectional kernel as the architectural template. Section 1 documents that template as the reference; Sections 2 and 3 are the plans for the two new topologies; Section 4 is cross-cutting methodology and sequencing.

---

## 1. Reference: how 1D unidirectional was built (fast AND correct)

Every later topology should be held to this same bar and reuse this same skeleton. Five layers, each independently testable:

### 1.1 Ground truth
- `tests/reference/selective_scan_ref.py` — vendored, unmodified upstream reference (matches HF `transformers` `MambaMixer.slow_forward` bit-for-bit).
- `tests/gen_golden.py` generates 16 fixed-seed cases into `tests/golden/*.npz`, covering shape edges (`edge_L1`, `edge_D1`), NEON-specific edges (`state13_neon_tail` — state not a multiple of 4), optional-input combinations (`no_z`, `no_D`, `no_bias`, `no_softplus`), grouped B/C, and one *real* case (`hf_mixer_layer0` — an actual mamba-130m forward pass, not synthetic). Each case's `manifest.json` entry records the recorded `f32_max_abs_err` — the tolerance floor a correct f32 kernel must land near, not just under.
- `tests/verify_golden.py` — an independent re-derivation in numpy, so the goldens aren't trusted on the strength of the reference implementation alone.

### 1.2 Scalar reference kernel (`kernel/arm-scan-core/src/scalar.rs`)
Direct transcription of the recurrence, clarity over speed — this is simultaneously the correctness oracle inside the crate, the non-Arm fallback, and what keeps x86 CI meaningful. Channel iteration is already routed through `parallel::for_each_channel`, so even the fallback scales across cores. **This file is the one every optimized backend is diffed against**, and it must never be deleted or let drift.

### 1.3 NEON fast kernel (`kernel/arm-scan-core/src/neon/mod.rs`)
The actual "fast" engineering, structured as a **chunked two-pass pipeline** per channel (`CHUNK = 128` timesteps, sized to keep scratch in L1):
- **Pass A1** (`discretize_chunk`) — `dt = softplus(delta + bias)`, `dt·u`, vectorized across time, 4 timesteps/iteration.
- **Pass A2** — `ābar = exp(dt·A)` and `b̄ = (dt·u)·B` for the whole chunk, with no cross-timestep dependency, so the NEON `exp` polynomial (`neon/exp.rs`) issues at full pipeline throughput instead of being serialized inside the recurrence.
- **Pass B** — the actual serial recurrence `h = ābar⊙h + b̄` and the `C·h` dot product, vectorized *across state* instead of across time — pure loads/FMAs, `h` resident in registers for the `state == 16` fast path (`channel_n16`), a zero-padded general path for any other state size (`channel_general`, exercised by the `state13_neon_tail` golden).
- **Epilogue** — `out = (y + D·u)·silu(z)`, vectorized across the row.
- A one-time (batch, group) transpose of B/C from `(state, len)` to `(len, state_padded)` up front, amortized across every channel that streams it, because NEON can't do the strided load the native layout would otherwise require.

### 1.4 Parallelism (`kernel/arm-scan-core/src/parallel.rs`)
`for_each_channel` dispatches disjoint `(batch, channel)` row chunks to rayon with per-worker scratch (`for_each_init`, zero per-channel allocation). Channels are fully independent — no shared mutable state, no reductions — so parallel output is **bit-identical** to sequential regardless of thread count, which is a hard property enforced by `tests/property.rs`, not just an aspiration.

### 1.5 FFI, torch op, HF patch
- `arm-scan-ffi/src/lib.rs` — the only crate with raw pointers. One entry point (`arm_scan_selective_scan_f32`), `#[repr(C)]` dims struct, overflow-checked size arithmetic before any slice is formed, `catch_unwind` at the boundary so a Rust panic becomes an error code instead of unwinding into the FFI caller. An `arm_scan_abi_version()` the Python loader checks before calling anything else.
- `python/arm_scan/op.py` — a `torch.library.custom_op` with a registered fake/meta kernel (so it composes with `torch.compile` instead of graph-breaking — necessary because `torch.compile` is the fair baseline, not eager).
- `python/arm_scan/patch.py` — monkeypatches `MambaMixer.slow_forward`, transcribing upstream's non-scan code exactly and routing only the recurrence through the kernel. Adapts at patch time to the two `transformers` API generations by inspecting `orig`'s signature; falls back to the original implementation for decode steps, training-mode `mambapy`, and any unrecognized future signature — the fallback always forwards the original arguments verbatim, so it can never see a shape mismatch.

### 1.6 Why this is trusted
Golden-vs-f64 tolerance (`< 1e-4`, and near the recorded `f32_max_abs_err` floor, not orders of magnitude above it) + NEON-vs-scalar parity (`≤ 3e-7`) + rayon-vs-sequential bit-identity, all re-checked at `RAYON_NUM_THREADS ∈ {1,2,8}`, plus `tests/check_ffi.py` (goldens through the *real* C ABI, not just the Rust unit tests) and `tests/check_hf_patch.py` / `check_hf_slow_path.py` (real HF model, token-identical output). **This five-layer + four-gate shape is the template for both topologies below.**

---

## 2. 1D bidirectional

Candidate target apps, per `APPLICATIONS.md` (**not yet decided** — see its open questions): genomics (Caduceus-class, 131k-token contexts), audio enhancement, ECG/EEG. Everything below is written to be app-agnostic — swap in whichever is chosen without changing the plan; only the merge-operator verification in §2.1 and the HF mixer class in §2.4 are app-specific details to fill in once that decision lands.

### 2.1 Correctness-only path (ship first, ~a day including test wiring)
No Rust changes. In Python: call `selective_scan` (§1.5) twice — once as-is, once with `u`/`delta`/`b`/`c` flipped via `torch.flip(-1)`, flipping the second call's output back before merging. **Before writing this, confirm the merge operator against the actual target checkpoint's code** (sum vs. concat-then-project vs. gated combine differ across BiMamba/Caduceus-style implementations — do not assume sum).

Deliverable: `python/arm_scan/bidirectional.py` (new) exposing a `bidirectional_scan(...)` function with the same tensor-shape contract as `op.selective_scan` plus a merge strategy parameter, callable from whichever app-specific integration (genomics/audio/ECG model class) needs it. This unblocks app-level demos and benchmark numbers immediately, independent of anything in §2.2.

### 2.2 Fast path: fused `reverse` flag in Rust

> ### ✅ BUILT — but read why, because it is NOT a speedup.
>
> `bench/bench_bidirectional.py` measured what the flip copies cost: a **ceiling**
> of 1.085× at L=128, falling to 1.025× at L=512 — i.e. **~2%**, shrinking as L
> grows. As a bidirectional optimization this does not pay, and that finding
> stands.
>
> It was built anyway because **SS2D needs a backward traversal regardless** — §3.2's
> row-backward and column-backward directions reuse the 1D scan with this exact
> flag. `reverse` is the *substrate* for the 2D cross-scan; removing bidirectional's
> flip copies is a side effect, not the reason.
>
> **Ship it as:** *"a fused backward traversal, the substrate for the 2D cross-scan."*
> **Never as:** *"we made bidirectional faster."*
>
> **One correction to the design below:** it claims the NEON work needs SIMD
> lane-reversal (`vrev64q_f32`/`vextq_f32`). **It does not.** Pass A is pointwise in
> time, and Pass B vectorizes across *state* while `t` is a scalar index — so
> reversing is one subtraction, no shuffles. The real change was a chunk-order
> iterator plus a flipped index. Full write-up:
> [`BIDIRECTIONAL_LOG.md`](BIDIRECTIONAL_LOG.md) Step 3.

**Original estimate: half a day**, because the recurrence math doesn't change — only traversal order does.

| Layer | Change |
|---|---|
| `arm-scan-core/src/lib.rs` | Add `reverse: bool` to `ScanInput` (a per-call semantic, not a `Backend`/`Threading` choice — it changes what answer is computed, not which code path computes an unchanged answer). Default `false` via any existing constructors/tests. |
| `arm-scan-core/src/scalar.rs` | In the per-channel closure (`scalar.rs:47`), iterate a step index `i` and derive `t = if reverse { len - 1 - i } else { i }` for every indexed access (`u_row[t]`, `delta_row[t]`, `bc_idx` from `t`, `out_row[t]`). `h` still evolves once per step in scan order — this is purely an indexing change, no new math. |
| `arm-scan-core/src/neon/mod.rs` | The real work. The chunked two-pass scan walks chunks low-to-high address with a forward inter-chunk carry (`h0..h3` persisting across the `while start < len` loop in `channel_n16`/`channel_general`). Reversing means: (a) iterate `start` from `len` down to `0` in `CHUNK`-sized steps, (b) within each chunk's Pass A1/A2, reverse the *lane order* before Pass B's sequential FMA chain (a NEON lane-reverse via `vrev64q_f32` + `vextq_f32`, or by writing `scratch.dt`/`scratch.dtu`/`abar`/`bbar` back-to-front during Pass A2 instead of adding a separate shuffle step), (c) un-reverse before storing `out_row`. The B/C transpose step (`mod.rs:63-78`) needs no change — it's direction-agnostic; only the per-chunk *consumption* order flips. |
| `arm-scan-core/src/parallel.rs` | **No change.** Channel independence doesn't depend on scan direction. |
| `arm-scan-ffi/src/lib.rs` | Add a `reverse: c_int` parameter to `arm_scan_selective_scan_f32`; bump `arm_scan_abi_version()` to 3 (this is a signature change per the existing versioning contract in the module doc). |
| `python/arm_scan/op.py` | Add `reverse: bool = False` to `_selective_scan_op` and the public `selective_scan()` wrapper; thread it through `_ffi.scan_raw`. Registered-fake kernel needs no change (output shape is identical). |
| `python/arm_scan/bidirectional.py` | Once the flag exists, swap the §2.1 flip-based implementation's *internals* to call `selective_scan(..., reverse=True)` for the backward pass instead of flipping four tensors — same call site and merge logic, strictly less copying. This is a drop-in internal swap, not a new public API. |

**Note — overlaps with [`IMPROVEMENT_IDEAS.md`](IMPROVEMENT_IDEAS.md):** the NEON chunk-reversal work above and that document's §4.2 (cache-blocking over L, to protect a long-sequence demo at whatever length one is chosen) both restructure the same `while start < len` chunk loop in `neon/mod.rs`. Don't implement them independently — whichever lands first should leave the loop nest in a shape the other can build on, or the chunk-loop surgery gets redone twice. Also relevant: `IMPROVEMENT_IDEAS.md` §2.4/§7.6 propose a streaming API (`h0` input + `last_state` output, already half-present as `last_state`) for constant-memory processing of arbitrarily long sequences — that's architecturally the same "carry state across a chunk boundary" idea as `reverse`, so it's worth deciding once whether `h0` support is a shared prerequisite for both bidirectional and any long-sequence streaming demo, rather than building the carry-state plumbing twice.

### 2.3 Correctness gates (new)
- Extend `tests/gen_golden.py` with `reverse: true` cases (reuse existing shapes — tiny/small/medium/state13_neon_tail — with the reference computed by flip-then-reference, matching §2.1's Python semantics exactly, since that's the ground truth this topology is defined against).
- New Rust test: NEON-reverse vs. scalar-reverse parity (mirrors the existing forward parity test).
- New Rust property test: rayon-reverse bit-identical to sequential-reverse at `RAYON_NUM_THREADS ∈ {1,2,8}`.
- `tests/check_ffi.py` extended to pass `reverse=1` goldens through the real C ABI.
- End-to-end: whichever bidirectional HF model class is chosen (§2.4) gets a `check_hf_patch.py`-style token-identical (or tolerance-bound, if the checkpoint isn't deterministic) parity check against its unpatched forward.

### 2.4 HF/ecosystem integration
Per the API-design decision already made: `arm_scan.patch(model)` should introspect the model's mixer class and dispatch automatically rather than requiring the caller to name a topology. This phase adds: identifying the actual bidirectional mixer class for whichever checkpoint is chosen (Caduceus's `BiMambaWrapper` or equivalent), and a `patch.py` dispatch branch for it, following the same "transcribe upstream, replace only the scan calls" discipline as `_mixer_scan_forward` does today.

### 2.5 Exit criteria
- All new goldens pass at the same tolerance discipline as §1.6.
- NEON-reverse output bit-parity with scalar-reverse within existing tolerance.
- Rayon-reverse bit-identical to sequential-reverse.
- Measured: fused `reverse` path is faster than the §2.1 flip-based path at the target app's realistic shape (this is the actual justification for having built it — verify it, don't assume it).
- One showcase app (genomics, audio, or ECG per whichever is chosen) runs end-to-end through the fused path with a quality-parity check.

---

## 3. 2D cross-scan (SS2D)

Candidate target apps, per `APPLICATIONS.md` (**not yet decided**): MRI reconstruction (MambaRecon) or VMamba-style vision classification. As with §2, this plan doesn't depend on which is chosen — only §3.4's risk register differs slightly (MRI carries more checkpoint/CUDA-coupling risk than VMamba, per `APPLICATIONS.md`). This is new surface, not a flag on the existing op — the shape contract genuinely changes from flat `len` to a `(height, width)` grid.

### 3.1 Correctness-only path (ship first, ~a day)
No Rust changes. In Python, given a `(B, D, H, W)` patch grid: build the four traversal views —
1. row-major forward,
2. row-major backward (`flip(W)`),
3. column-major forward (transpose `H`↔`W`, then treat `H` as the scan axis),
4. column-major backward (transpose + `flip`),

reshape each to `(B, D, H*W)`, **stack along the batch dimension** (`4B` effective batch — this needs zero new Rust or FFI, it's exactly the existing `ScanDims.batch` knob), make **one call** to `selective_scan`, split the output back into 4, invert each transform, and merge per whatever the target checkpoint actually does (again: verify against the real model code before assuming sum).

Deliverable: `python/arm_scan/ss2d.py` (new), `ss2d_scan(...)` with a `(B, D, H, W)`-shaped contract. This is the one that unblocks the MRI/vision app end-to-end and produces real numbers before any new kernel work is committed to.

### 3.2 Fast path: fused four-direction traversal in Rust
**Estimated effort: ~a week — the largest single item in this plan, and, per `APPLICATIONS.md`, the one piece of genuine white space** (no CPU SS2D implementation exists anywhere).

**Design decision, made now rather than left open:** implement column-direction traversal via an in-kernel **transpose-then-reuse-the-row-scan** strategy, not a second from-scratch strided-load NEON scan variant. Rationale: row-direction traversal is *already free* — a grid row is contiguous in row-major memory, so it's literally the existing `neon::scan`/`scalar::scan` called once per row (with `reverse` from §2 for the backward direction), zero new SIMD code. Column traversal is strided (stride = `W`) and NEON has no efficient strided load; transposing a tile into scratch and reusing the verified, already-fast row kernel confines all new risk to a transpose micro-kernel instead of a second scan implementation with its own correctness surface to prove out. This directly answers `APPLICATIONS.md`'s open question about whether the fused path is worth building: it is, *because* it can be built almost entirely out of already-trusted code.

| Layer | Change |
|---|---|
| `arm-scan-core/src/lib.rs` (or new `arm-scan-core/src/scan2d.rs`, referenced from `lib.rs`) | New `ScanDims2D { batch, dim, height, width, state, groups }` and a new public `selective_scan_2d(dims2d, input, out4, last_state4, opts)` entry point. `out4` is `(4, batch, dim, height, width)` — one plane per direction — or four separate output slices; pick whichever the merge step in §3.1 finds more natural to consume, since Python is doing the merge either way. |
| new `arm-scan-core/src/transpose.rs` | A cache-tiled `(H, W) → (W, H)` transpose micro-kernel for f32, NEON-accelerated (4x4 tile transpose via `vtrn`/`vzip`-style shuffles is standard and low-risk) with a scalar fallback identical in spirit to `scalar.rs`. This is the one genuinely new piece of SIMD code in this whole plan. |
| `arm-scan-core/src/scan2d.rs` | Orchestration: for each of the 4 directions, either (a) call the existing row-scan `H` times directly (row directions), or (b) transpose the `(H, W)` tile, call the existing row-scan `W` times, transpose the output back (column directions). Threading: parallelize over `(batch × channel × direction)` — more parallelism than the 1D case, which is a good benchmark story on Arm's core counts. |
| `arm-scan-ffi/src/lib.rs` | New entry point `arm_scan_selective_scan_2d_f32` (separate from the 1D ABI — different dims struct, not worth overloading the existing signature). Same discipline: null checks, overflow-checked size arithmetic, `catch_unwind`. Bump `arm_scan_abi_version()` again. |
| `python/arm_scan/_ffi.py`, `op.py` | New `_ffi.scan2d_raw` ctypes binding; new `torch.ops.arm_scan.selective_scan_2d` custom op with a registered fake kernel (same `torch.compile`-composability requirement as the 1D op). |
| `python/arm_scan/ss2d.py` | Once the fused op exists, swap internals from "stack 4B batch, one 1D call" to "one call to `selective_scan_2d`" — same public function signature, strictly less memory traffic (per-direction copies go from ~4x grid size to ~1-2x). |

**Note — overlaps with [`IMPROVEMENT_IDEAS.md`](IMPROVEMENT_IDEAS.md):** its §3.6 already flags the existing B/C plane transpose in `neon/mod.rs` as a scalar strided loop that should become a vectorized 4×4 NEON tile transpose — that's the same primitive proposed above as `transpose.rs`. If that general kernel-perf item lands first (independent of any topology work), SS2D can reuse it directly instead of building it from scratch, which would shrink this section's effort estimate below the week quoted here. Also relevant: its §4.1 (stop reallocating/rezeroing the `bt`/`ct` workspace every call) matters more here than in the 1D case, since four-direction orchestration multiplies the number of transpose calls per forward pass — worth sequencing §4.1 before committing to this section's full effort estimate.

### 3.3 Correctness gates (new)
- New golden generator (extend `tests/gen_golden.py` or add `tests/gen_golden_2d.py`) producing `(H, W)` grid cases at a few sizes (including a non-square grid and an `H`/`W` not a multiple of 4, to exercise the transpose tail path the way `state13_neon_tail` exercises the 1D general path). Ground truth: feed the vendored reference the four permuted views exactly as §3.1 does, and compare against `selective_scan_2d`'s four raw output planes *before* merge — this isolates kernel bugs from merge-strategy bugs.
- Transpose micro-kernel gets its own unit tests (round-trip identity, non-multiple-of-4 tiles) independent of the scan tests.
- NEON-vs-scalar parity and rayon bit-identity tests, same structure as §1.6/§2.3, run per-direction.
- `tests/check_ffi.py` extended for the new 2D C ABI entry point.
- End-to-end: whichever app (MRI or VMamba) gets a quality-parity gate (PSNR/SSIM/NMSE for MRI per `PROJECT_CONCEPT.md`'s existing framing, or top-1 accuracy for VMamba) between the fused path and the §3.1 Python-only path, to confirm the fusion didn't silently change the answer.

### 3.4 Risks specific to this topology
- **Merge-strategy mismatch** — verify against actual checkpoint code before generating goldens; do not assume sum-merge.
- **Copy overhead may not dominate at small grids** — `APPLICATIONS.md`'s open question #2. Measure the §3.1 unfused path's overhead *before* committing the week to §3.2; if flip/permute traffic is already a small fraction of scan time at the target app's realistic resolution, the fused kernel is a nice-to-have for the writeup, not a blocker for the demo.
- **Research checkpoints are CUDA-coupled** — per `APPLICATIONS.md`, MRI/vision SS2D reference repos may bundle CUDA-only ops that need to be forced onto a CPU reference path before any golden can even be generated. Budget time for this *before* the week estimate above, not inside it.

### 3.5 Exit criteria
- All new 2D goldens pass at the same tolerance discipline as §1.6, checked per-direction before merge.
- Transpose micro-kernel round-trips exactly (bit-identical, it's a permutation).
- NEON-vs-scalar and rayon bit-identity hold per direction.
- Measured: fused path faster than the §3.1 stacked-batch path at the target app's realistic grid size — if not, ship §3.1 for the demo and say so honestly in `RESULTS.md`, per the "benchmark honestly" rule in `CLAUDE.md`.
- Chosen showcase app (MRI or VMamba) runs end-to-end through the fused path with the appropriate quality-parity gate.

---

## 4. Cross-cutting methodology & sequencing

### 4.1 Testing discipline (applies to both topologies, no exceptions)
Same four gates as §1.6: golden-vs-f64 tolerance near the recorded floor, NEON-vs-scalar parity, rayon-vs-sequential bit-identity at `RAYON_NUM_THREADS ∈ {1,2,8}`, and real-C-ABI golden replay via `tests/check_ffi.py`. Never loosen a tolerance to make a new topology's test pass — if a chunk-reversal or transpose implementation doesn't hit the same bar as the forward 1D path, the bug is in the implementation, not the test.

### 4.2 Benchmarking discipline
Use the same three-surface framework as `BASELINE_TEST_PLAN.md` (Surface W for correctness/scaling shape, Surface Q for NEON correctness without a timing claim, Surface A for the numbers that go in `RESULTS.md`). Every new topology's headline number needs the same rigor already applied to 1D: `torch.compile` as the real baseline, medians after warmup, fixed thread counts, host/git-SHA-tagged JSON feeding the results generator.

### 4.3 Recommended sequencing
1. **Now:** §2.1 (bidirectional, Python-only) and §3.1 (SS2D, Python-only) — both are low-risk, unblock app-level demos and real measurements, and directly answer the open "does copy overhead matter" question for each topology.
2. **Once an app is chosen per `APPLICATIONS.md`:** measure the Python-only path at the app's actual shapes. This is the gate that decides whether §2.2 and §3.2 are worth spending calendar time on before Aug 14, or whether the correctness-only paths are good enough for the submission with the fusion work framed as future work in the writeup.
3. **If pursued:** §2.2 (half a day) before §3.2 (a week) — bidirectional is strictly cheaper, de-risks the `reverse`-flag plumbing (FFI version bump, op.py threading) that §3.2's column-direction reuse depends on anyway (column scans use `reverse` for their backward direction too), and unblocks whichever of genomics/audio/ECG is chosen independent of whether SS2D ever lands.
4. **§3.2 only if time and the §3.1 measurement justify it** — per the risk register in §3.4, this is the one item on the whole roadmap that could genuinely eat a week it doesn't have.

### 4.4 What NOT to build
No unified "topology" enum on the core `selective_scan` function (see the API-design discussion this plan follows from) — 1D forward/backward stays one function family with a `reverse` flag, 2D stays a separate function family with its own dims struct, and merge/direction-selection logic stays in Python where it's model-specific and cheap to iterate on without touching the Rust ABI.
