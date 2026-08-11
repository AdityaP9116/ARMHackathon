# MAMBA3_IMPLEMENTATION_PLAN — an Arm/NEON Mamba-3 kernel for all three scan topologies

**Rewritten Aug 6, 2026**, after a prior-art sweep that materially changed the
plan. Supersedes the previous revision and the Mamba-3 sections of
[`SUBMISSION_ENDGAME_PLAN.md`](docs/archive/SUBMISSION_ENDGAME_PLAN.md). Every external claim
below was verified the same day against the actual repositories and APIs, not
against summaries — three claims from a research digest failed that check, and
two pieces of prior art nobody had spotted turned up.

---

> **Execution plan for the three paths:
> [`THREE_PATHS_INTEGRATION.md`](THREE_PATHS_INTEGRATION.md)** (Aug 7) — written after the
> kernel itself was finished, and grounded in the real checkpoint's parameter table rather
> than in estimates.

## 0. The shape of the project, in one paragraph

An **Arm-optimised Mamba-3 kernel in Rust** (NEON + chunked scan + rayon),
exposed as a PyTorch custom op, covering all three scan topologies. **2D is the
headline** — no CPU implementation of any 2D Mamba-3 exists, and we ship two
formulations plus the comparison between them. **1D is the evidence track** —
it is the only topology with published weights and an authoritative oracle, so
it is how we prove the kernel is correct and fast. **Bidirectional comes along
for free** because the topology layer is recurrence-agnostic; we claim no
novelty there. **We train nothing and alter no published model.**

### The three paths, honestly

| | 1D unidirectional | 1D bidirectional | **2D / vision** |
|---|---|---|---|
| Architecture published | ✅ Mamba-3 | ❌ none for M3 | ✅ **VNCT** (ECCV 2026) |
| Weights available | ✅ 187M–1.5B + MIMO | ❌ | ❌ none anywhere |
| Reference code | ✅ official (GPU-only) | ❌ | ❌ **repo is 404** |
| Authoritative oracle | ✅ captured (Stage 0) | n/a | ❌ **none — see §7** |
| Prior CPU work | mamba-rs, burn-mamba, mamba.c | burn-mamba | **none** |
| Accuracy claim possible | ✅ | ❌ | ❌ |
| **Role** | **earns credibility** | freebie | **earns novelty** |

---

## 1. Prior art — verified Aug 6, 2026

Checked by fetching the repos and papers, not by search summary.

| Work | What it is | What it does NOT do |
|---|---|---|
| [`state-spaces/mamba`](https://github.com/state-spaces/mamba) | The official Mamba-3 ([arXiv:2603.15569](https://arxiv.org/abs/2603.15569), ICLR 2026). Triton SISO prefill, TileLang MIMO, CuTe decode | **No CPU path at all** — `modules/mamba3.py` asserts if the kernels are missing. Will not install without `nvcc` |
| [`silvermpx/mamba-rs`](https://github.com/silvermpx/mamba-rs) | Mamba-3 SISO in Rust, CPU (rayon/BLAS) + CUDA, bit-deterministic | Standalone runtime, no PyTorch interop; x86/CUDA focus, **no Arm/NEON**; 1D only |
| [`swfsql/burn-mamba`](https://github.com/swfsql/burn-mamba) | Mamba-1/2/3 for the Burn framework, **incl. bidirectional wrappers**; parallel + recurrent modes. *Actively developed* | **"No custom CUDA/Triton kernels — portable Burn tensor ops"** — a clarity-first reference, not a hand-optimised kernel; no Arm tuning; 1D only |
| [`kroggen/mamba.c`](https://github.com/kroggen/mamba.c) | Mamba/Mamba-2/**Mamba-3** inference in pure C, 203★ | Portable C, no NEON intrinsics; standalone, not PyTorch-callable; 1D only |
| [VNCT](https://arxiv.org/abs/2607.03589) | **The** 2D Mamba-3 architecture. Non-causal lift of the trapezoidal dynamics, 2D RoPE, optional low-rank MIMO. ECCV 2026 | **Code unreleased** (`github.com/anvitha305/VNCT` → 404, checked twice). Trained on 8× RTX 6000 Ada. No CPU/Arm anywhere |
| [VMamba](https://github.com/MzeroMiko/VMamba) / [2DMamba](https://arxiv.org/abs/2412.00678) | The SS2D cross-scan reference; GPU-side 2D tiling precedent | **CUDA-only**; Mamba-1/2 lineage, not Mamba-3 |
| [MFil-Mamba](https://github.com/puskal-khadka/MFil-Mamba) | Multi-filter vision SSM, code released | Its own README: *"based on Mamba, VMamba"* — **VMamba lineage, not Mamba-3** |

### Claims policy — what we may and may not say

**May claim, to the best of our knowledge:**
1. **First CPU implementation of a 2D Mamba-3** (any architecture, any hardware).
2. **First causal-vs-non-causal 2D SSM comparison on CPU** — the novel *result*.
3. **First PyTorch-callable, NEON-optimised Mamba-3 scan** — none of mamba-rs,
   burn-mamba or mamba.c is a PyTorch drop-in, and none targets Arm.

**Must never claim:**
- "First Mamba-3 on CPU" — three projects have it.
- "First Mamba-3 in Rust" — mamba-rs and burn-mamba.
- **Anything about bidirectional Mamba-3** — burn-mamba ships it.
- Any accuracy/quality result for 2D or bidirectional — there are no weights.

---

## 2. Leverage — what the repo already provides

> **Corrected after a code review of the existing kernel (same day).** Two
> reuse claims that appeared in earlier revisions of this plan and in
> `HANDOFF.md` do not survive contact with the code: `parallel.rs` is **not**
> reusable as-is, and Pass B is **not** structurally unchanged. Details in §14.
> The *guarantees* still hold (heads are independent; the recurrence equation is
> the same shape); the concrete functions do not.

| Asset | Reuse for Mamba-3 |
|---|---|
| `parallel.rs` (rayon over independent rows) | **Pattern reusable, function is not** — its signature assumes one output scalar per (channel, timestep) and a flat per-channel state. Needs a sibling driver; see §14.3. Bit-identity still holds by construction |
| `vexpq_f32_nonpos` and the NEON math | Direct — `exp` is ~54% of kernel time, already tuned |
| Chunked two-pass pipeline shape | The template; Mamba-3's Pass B is the same `h = a⊙h + b̄` |
| FFI discipline (`catch_unwind`, overflow-checked sizing, ABI versioning) | New entry points beside the old |
| `torch.library` custom op + fake kernel | Same pattern, so Mamba-3 composes with `torch.compile` |
| Golden methodology (capture → independent re-derivation → C-ABI replay) | Already applied; Stage 0/1 done |
| **SS2D orchestration** (`ss2d_scan`, traversal pairs; 1.80× on x86, 0.96× on 64-core Graviton4) | **Recurrence-agnostic** — retargets to Mamba-3 by swapping the primitive |
| `bidirectional_scan` + fused reverse | Same seam |
| Wheels, arm64/macOS/x86 CI, bench harness | Unchanged |

**The Mamba-1 kernel is not modified.** Mamba-3 is additive: new files beside
the existing ones, new C-ABI entry points, one shared crate.

---

## 3. Stage 0 — Ground truth ✅ **DONE (Aug 6)**

10 goldens / 7 output shapes (L ∈ {1, 63, 64, 128, 255, 256, 2048}, batch 1–2,
d_state 64/128) in `tests/golden/mamba3/`, plus `model_forward.npz` and
`model_shape.json`. 19 MB. Replay verified under a Python with **no torch, no
mamba_ssm, no CUDA** — the GPU is never needed again.

Three findings that shape everything downstream:

1. **Install from git, not PyPI.** The release rejects every published Mamba-3
   checkpoint *and* predates PR #997, which fixes **silent forward-pass
   corruption on Blackwell**. The capture script now refuses to run without the
   fix, detected by source inspection (patched and unpatched both report
   `2.3.2.post1`).
2. **The kernel is mixed precision with no flag.** `Q/K/V/Trap/Angles/Z` → bf16
   on entry; `ADT/DT` stay fp32 "for stability"; `Q_bias/K_bias/D` stay fp32 as
   parameters; output bf16. **Stages 4–5 must mirror this split.** Goldens
   record inputs *post-cast*, so a reference is fed exactly what the kernel was.
3. **`model_shape.json` reveals the naming**: the mixer's params are `B_bias` /
   `C_bias` while the kernel signature calls them `Q_bias` / `K_bias`. Stage 6's
   weight mapping is mechanical because of that file.

## 4. Stage 1 — CPU reference ✅ **DONE (Aug 6)**

`tests/reference/mamba3_ref.py` reproduces the official kernel to **4.47 bf16
ULP** at tensor scale across all 10 goldens. Gated by
`tests/verify_golden_mamba3.py` / `make test-mamba3`, wired into arm64 CI.

**The recurrence is settled — the paper's form, not the community's.** Proven by
diffing the kernel's own `Scale_store` buffer: `scale = dt·σ(trap) +
dt₊₁·(1−σ(trap₊₁))` matched to **3.4e-8**. Independently corroborated by the
Mamba-3 paper and by `mamba.c`'s README.

Two details no published description contains, both found by diffing internal
buffers rather than final outputs:
- RoPE is **interleaved** (pairs `2i, 2i+1`), not split-halves.
- The angle is accumulated by a **separate pre-pass kernel**:
  `theta = cumsum(tanh(angle)·π·dt)`. The `tanh(·)·π` squashing is the piece
  that no end-to-end sweep would recover — omitting it leaves ~7e-2 error,
  close enough to read as a rounding problem.

**Tolerance, corrected:** the original "< 1e-4 at f64" is unsatisfiable against
a bf16 output. The gate is now: round the f64 reference to bf16, require ≤ 8
ULP at tensor scale. Evidence-based — golden `_04` is L=1 with *no*
accumulation and still sits at 0.45 ULP; a structurally wrong reference is off
by thousands.

**Known limits:** SISO only (multi-group/MIMO now raise rather than silently
truncate); no initial-state support yet — Stage 5 and any decode path need it.

---

## 5. Stage 2 — Scalar Rust ✅ **DONE (Aug 7)**

**File:** `kernel/arm-scan-core/src/mamba3.rs`

Direct transcription of `mamba3_ref.py`, clarity over speed. Becomes the
in-crate oracle and the non-Arm fallback, exactly as `scalar.rs` is today.

- New per-step inputs: `lambda` (trapezoid gate, pre-sigmoid) and the `Bx`
  2-tap. **The state shape changes** from Mamba-1's per-channel vector
  `(d_state)` to a per-head matrix `(headdim_v × headdim_qk)` — this is the
  largest structural difference and it drives the loop nest.
- The block-variant enum is born here (`Mamba3Siso = 0`, `Mamba3Mimo = 1`
  reserved) so later work is addition, not refactor.
- `parallel.rs` untouched.

**Gate:** matches the Stage-1 reference within the bf16 bound on all 10
goldens, replayed via a Rust-side loader.

## 6. Stage 3 — NEON ✅ **DONE (Aug 7)** — split portable-blocked / intrinsics

**File:** `kernel/arm-scan-core/src/neon/mamba3.rs`

| Phase | Change |
|---|---|
| Pass A1 | `dt = softplus(...)` — unchanged |
| Pass A2 | `α = exp(ADT)` — unchanged, reuses `vexpq_f32_nonpos`. **New:** the 2-tap blend `scale_t = dt·λ + dt₊₁·(1−λ₊₁)`, pointwise in time so it vectorises across 4 timesteps |
| **Pass B** | **A NEW inner kernel — this is the hard part, not a port.** The equation `h = α⊙h + b̄` is the same shape, but Mamba-1's state is 16 f32 = **64 bytes, held in 4 NEON registers** (`channel_n16`), which is the entire reason it is fast. Mamba-3's per-head state is `headdim_v × headdim_qk` = **8–32 KB**. It cannot be register-resident, so this becomes a cache-blocked rank-1 update over a tiled matrix. See §14.2 |
| RoPE | Pointwise even/odd lane rotations. Kept in Python initially; fuse only if measured |
| Epilogue | `out = y + D·x`, then `silu(Z)` gate |
| Chunk carry | carries `h` **and** `Bx_last` |

**Gate:** NEON↔scalar parity; rayon bit-identical at `RAYON_NUM_THREADS ∈ {1,2,8}`.

## 7. Stage 4 — FFI + torch op ✅ **DONE (Aug 7)** — ABI 6, now **7** (MIMO)

`arm_scan_mamba3_scan_f32`, same discipline as the existing entry point. Bump
`arm_scan_abi_version()`. `python/arm_scan/mamba3.py` with a registered fake
kernel so it composes with `torch.compile`. **Mirror the mixed-precision
contract from §3.2.**

**Gate:** goldens replayed through the real C ABI.

---

## 8. Stage 5 — The two extra topologies ✅ **1D bidirectional DONE; 2D causal wiring pending**

### 5a. Bidirectional *(~half a day, no novelty claimed)*

Add `reverse` to the Mamba-3 entry point; the 2-tap reverses with the traversal
(`Bx_prev` becomes `Bx_next`). `bidirectional_scan` then works unchanged.
Ship it as capability, cite burn-mamba, claim nothing.

### 5b. 2D — **the headline, two formulations** *(~1 week)*

Both retarget the existing `ss2d_scan` machinery by swapping the scan primitive.

**(i) Causal cross-scan.** VMamba-style four directions as two traversal pairs,
exactly as our Mamba-1 SS2D already runs. Keeps the **sequential-recurrence
moat** — the thing `torch.compile` provably cannot restructure.

**(ii) Non-causal, VNCT-style.** Drop the causal mask → the intra-chunk term
becomes two dense GEMMs, plus 2D RoPE. **Be honest about the moat here:**
GEMMs are exactly what BLAS and compilers are good at, so expect a much thinner
margin than the scan enjoys. That is a finding, not a failure.

**The oracle problem, stated plainly.** For 1D we captured ground truth from
official kernels. **For VNCT no authoritative source exists** — the code is
unreleased. We validate our Rust against *our own PyTorch reading of the
paper*, which proves the kernel implements the reference, **not** that the
reference implements VNCT as intended. Say so in the writeup. Mitigations:
per-direction 2D goldens (non-square, non-multiple-of-4), independent numpy
re-derivation, and cross-checking block semantics against `burn-mamba` and
`mamba.c` where they overlap.

**Gate:** 2D goldens per direction; pair-vs-oracle parity at 1/2/8 threads.

## 9. Stage 6 — CPU model path, 1D ✅ *(DONE Aug 7 — the credibility anchor)*

**Directory:** `apps/mamba3_lm/` — `block.py`, `model.py`, `load.py`.

`mamba_ssm`'s `Mamba3` cannot run on CPU, so the block is reimplemented in plain
PyTorch — projections, RoPE angles (**including the `tanh(·)·π` pre-pass**),
BCNorm, gated MLP, embedding and tied head — with the scan routed to our kernel.
The published 187M weights load with `strict=True` and no key remapping.

**Gates, both passing:**

- `tests/check_mamba3_block.py` — the mixer against the real block, driven by
  the published layer-0/1 weights: **1.36 bf16 ULP** (bound 16). In CI; needs no
  checkpoint download because the weights ride inside the golden.
- `tests/check_mamba3_model.py` — logits against `model_forward.npz`: **98.05%**
  argmax, drift 4.5e-3 of logit range, **0 unexplained flips**. Not in CI (357 MB).

**Three things this stage established that were not known going in:**

1. **Mamba-3 has no `conv1d`.** The mixer has exactly 8 parameters.
2. **`A` is data-dependent** — out of `in_proj` through `heavy_tail`, not a
   learned `A_log`. There is no `A` tensor to load.
3. **The official kernel is not reproducible across processes.** Two runs
   disagree on up to 5/256 argmax positions (~2.9e-3 relative logit drift), while
   two forwards inside one process are bit-identical — `triton.autotune` picks
   its config by timing. The gate therefore cites this measured floor instead of
   comparing against a 100% nothing can reach.

Fixed here too: the golden sweep seeded itself with `abs(hash(name))`, and
Python salts `str.__hash__` per process, so the sweep goldens were **not
regenerable**. Now seeded from `golden_inputs.case_seed` (sha256) and verified
bit-identical across two independent captures.

## 10. Stage 7 — The comparison study *(the novel result)*

Not a demo — a **measurement nobody has published**:

- **Causal cross-scan vs non-causal aggregation**, same kernel family, across
  grid size × channel count × core count, on Arm.
- **Scan-form vs dual-form** at matched shapes: our Mamba-1 kernel vs Mamba-3.
- Baselines always named: `torch.compile` first, eager for colour, plus our own
  scalar rung for the ablation ladder.
- Long-context 1D on the real 187M checkpoint: constant memory where the
  reference needs GBs of intermediates.

Publish unflattering rows — especially the non-causal margin.

## 11. Stage 8 — Graviton *(~3 hours, and it is still unbooked)*

`bench/GRAVITON_SESSION.md`, extended with Mamba-3 rows. **Every number in this
repo is still x86 or a shared 4-core CI runner.** This is the only remaining
item that cannot be done from a laptop, and no amount of kernel work
substitutes for it.

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| VNCT code drops and differs from our reading | Watch the repo; our 2D goldens are per-direction so a correction is localised. Being first to implement is still first |
| Non-causal margin vs BLAS is thin | Expected and pre-declared; the comparison **is** the result |
| No oracle for 2D | Two independent implementations of our own + cross-check against burn-mamba/mamba.c semantics |
| Prasanna's group publishes a CPU/FPGA VNCT follow-up | Their lab does hardware acceleration — plausible. Move early, claim precisely |
| Matrix-shaped state hurts NEON | It raises arithmetic intensity, which is where CPU does *better*; measure at Stage 3 before optimising |
| Mamba-3 spec still settling | Goldens are the contract, pinned to a tag — not repo HEAD |

## 13. Descope ladder — cut in this order

*(§14 follows the ladder; it is the code-level audit this plan rests on.)*

1. **Non-causal 2D variant** — keeps causal SS2D, loses the comparison
2. **Bidirectional** — no novelty anyway
3. **Stage 6 model path** — demo against goldens instead; weakens "runs the
   model" to "implements the operator, verified against the official kernel"
4. **All of Mamba-3** — the Mamba-1 submission is complete, measured and
   CI-green. It remains the floor.

---

## 14. Kernel audit — what fits the three paths, and what does not

Read against the actual code on Aug 6, not from memory. File:line references
are to `kernel/arm-scan-core/src/`.

### 14.1 Reusable unchanged

| Asset | Why it carries over |
|---|---|
| NEON `exp` (`neon/exp.rs`), incl. `vexpq_f32_nonpos` | `exp(ADT)` has the identical non-positive precondition (`A<0`, `dt>=0`). `exp` is ~54% of kernel time — this is the single most valuable piece of reuse |
| Chunked two-pass shape, `CHUNK = 128` (`neon/mod.rs:44`) | Pass A stays pointwise-in-time and fully vectorisable; the structure holds |
| `chunks_in_scan_order(len, reverse)` (`neon/mod.rs:195`) | Reverse traversal is already a chunk-order iterator plus a flipped index — directly reusable for bidirectional Mamba-3 |
| FFI discipline (`arm-scan-ffi`) | `catch_unwind`, overflow-checked sizing, ABI versioning, null checks — pattern applies verbatim to a new entry point |
| `torch.library` op + fake kernel | Same pattern gives `torch.compile` composability |
| Golden / parity / bit-identity test harness | Already proven on Mamba-3 in Stages 0–1 |
| **`ss2d.py` orchestration** | The best-designed extension point in the repo: `grid_to_views` / `views_to_grid` / merge are **pure layout ops with no recurrence knowledge**, and `scan_pair` is an injected primitive. Retargeting to Mamba-3 means passing a different function |

### 14.2 The big one: Pass B is a new kernel, not a port

`channel_n16` (`neon/mod.rs:284`) keeps the **entire recurrent state in four
NEON registers** — `h0..h3`, 16 f32, 64 bytes (`neon/mod.rs:300`, `365-382`).
The time loop then streams `abar`/`bbar` past registers that never spill. That
is why it is fast, and it is why `state == 16` gets a dedicated path at
`neon/mod.rs:137`.

Mamba-3's state is a **matrix per head**, `headdim_v × headdim_qk`:

| Config | State per head |
|---|---|
| Mamba-1 (`state=16`) | 64 B — **4 NEON registers** |
| Mamba-3 sweep shapes (32×64) | 8 KB |
| Mamba-3 187M (64×128) | **32 KB** |

At 32 KB it cannot be register-resident and barely fits L1d (64 KB/core on the
Neoverse N2 we measured). So Pass B becomes a **cache-blocked rank-1 update**
over a tiled matrix — a different loop nest, different blocking, different
spill behaviour. The `channel_n16` / `channel_general` split does not carry
over; both are `state`-vector-shaped.

**Consequence for planning:** Stage 3's estimate of 1–1.5 days assumed a port.
Budget more, and expect the tile/blocking choice to need measurement rather
than derivation. The compensating upside is real: a matrix state has far higher
arithmetic intensity than a 16-element vector, which is the regime where CPU
competes best — but that has to be demonstrated, not assumed.

### 14.3 `parallel.rs` needs a sibling driver

`for_each_channel` (`parallel.rs:24`) hard-codes Mamba-1's output shape:

- `n_channels = out.len() / len` (`parallel.rs:38`) assumes **one output scalar
  per (channel, timestep)**. Mamba-3 emits a `headdim_v` vector per (head, t).
- `out.chunks_exact_mut(len)` (`:58`, `:71`) — would need `len * headdim_v`.
- `last_state.chunks_exact_mut(state)` (`:51`) — Mamba-3's carry is
  `headdim_v * headdim_qk`, plus `Bx_last` and the RoPE angle.
- `should_parallelize(n_channels, len, state)` (`:18`) is a work heuristic
  calibrated on lane-steps that no longer mean the same thing.

Precedent: the bidirectional topology already required a separate
`for_each_channel_bidir` (`parallel.rs:96`) rather than a generalisation. A
`for_each_head` sibling is consistent with that, and cheap.

**The bit-identity guarantee is unaffected** — heads remain independent with no
cross-thread reduction, so schedule-independence holds by construction. That
part of the earlier claim was right; "unchanged code" was not.

### 14.4 New types required (no reuse possible)

- `ScanDims { batch, dim, len, state, groups }` (`lib.rs:80`) → Mamba-3 needs
  `{ batch, heads, headdim_v, headdim_qk, len, chunk }`.
- `ScanInput` (`lib.rs:89`) carries `u, delta, a, b, c, d_skip, z, delta_bias`
  → Mamba-3 needs `Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z`.
  Disjoint. This is why a flag on the existing op was rejected.
- Third FFI entry point beside `arm_scan_selective_scan_f32` and
  `..._bidirectional_f32`.

### 14.5 `Float` trait gaps

`float.rs` provides `exp`, `ln_1p`, `softplus`, `silu` only. Mamba-3 additionally
needs **`sigmoid`** (the trapezoid gate), **`tanh`** (angle squashing), and
**`sin`/`cos`** (RoPE) — scalar and NEON. `sigmoid` and `tanh` can be built on
the existing exp core; `sin`/`cos` are genuinely new vector math and are the
one place a NEON polynomial has to be written and accuracy-tested from scratch.

### 14.6 Something that gets *simpler*

The one-time B/C transpose (`neon/mod.rs`, `(state,len) → (len,state_padded)`)
exists because NEON cannot do the strided load the native layout needs. Mamba-3
hands us `Q`/`K` as `(b, l, groups, dqk)` — **already time-major**. The
transpose disappears. It was measured at ~0.1% of runtime so this is not a
speed win, but it is one less correctness surface, and it retires the
tile-transpose item (P1-6) for this track.

### 14.7 Path-by-path fit

| | Reuses | Needs new |
|---|---|---|
| **1D unidirectional** | exp, chunk pipeline, FFI/torch patterns, test harness | Pass B inner kernel, dims/input types, `for_each_head`, sigmoid/tanh/sin/cos |
| **1D bidirectional** | *Everything* from 1D plus `chunks_in_scan_order`, `for_each_channel_bidir` pattern, `bidirectional_scan` | Only the 2-tap's reversal (`Bx_prev` → `Bx_next`) |
| **2D causal cross-scan** | *Everything* from 1D plus the whole `ss2d.py` layer — `grid_to_views`, merge, pair machinery | Widen the `scan_pair` seam: its signature is Mamba-1's parameter list, so it needs a tensor-bundle form or a parallel `ss2d_scan_mamba3` |
| **2D non-causal (VNCT)** | **Almost nothing from the kernel** — infrastructure only | Dropping causality removes the recurrence entirely: two dense GEMMs + 2D RoPE. This is a *different compute pattern*, not an extension. Note `matrixmultiply` is currently only a transitive **dev**-dependency via `ndarray`; using it in the core lib means adding a real dependency, or hand-rolling a NEON microkernel |

**The honest summary:** paths 1, 2 and 3(i) share one kernel with a new Pass B.
Path 3(ii) is a second, GEMM-shaped kernel that shares the packaging and the
test harness but little else — which is also exactly why its `torch.compile`
margin will be thinner, and why measuring both is the interesting result.
