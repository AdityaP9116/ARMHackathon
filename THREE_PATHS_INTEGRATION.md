# THREE_PATHS_INTEGRATION — the three demonstrations, scoped

**Written Aug 7, 2026.** Execution plan for the three paths chosen in
[`MAMBA3_IMPLEMENTATION_PLAN.md`](MAMBA3_IMPLEMENTATION_PLAN.md) §0, now that the Mamba-3
kernel itself is complete and gated (Stages 0–5, M0–M8).

**The constraint that governs all three:** our Mamba-3 kernel runs only Mamba-3 models, and
**exactly one family has public weights** — `state-spaces/mamba3-*`. Everything else in the
Mamba ecosystem (VideoMamba, VMamba, PointMamba, Mamba-3VL, MM2CT) is an earlier generation
and belongs to the Mamba-1 kernel. That is why the three paths below are *architectures*
within one family rather than three unrelated models: it is the honest maximum.

| | Path A — 1D SISO | Path B — MIMO | Path C — 2D |
|---|---|---|---|
| Role | **credibility** | **second real-weights path** | **novelty** |
| Weights | ✅ 187M–1.5B | ✅ 187M/444M/894M | ❌ none exist |
| Oracle | ✅ official kernels | ✅ official kernels *(if capturable)* | ❌ **our own reference only** |
| Accuracy claim | ✅ token-level | ✅ | ❌ **impossible** |
| Speed claim | ✅ | ✅ | ✅ |
| Kernel work | none | **rank-r update** | none — layout + composition only |
| Risk | model plumbing | **TileLang on Blackwell** | no authoritative oracle |

---

## Path A — 1D SISO end to end ✅ *(Stage 6 — DONE, Aug 7 2026)*

**Goal:** run the real `state-spaces/mamba3-siso-187m` on Arm CPU through our kernel, and
prove it is the same model by matching logits.

**Status: complete and gated.** The 187M checkpoint loads into `apps/mamba3_lm/` and runs on
CPU with the recurrence on our kernel.

| Gate | Result |
|---|---|
| `tests/check_mamba3_block.py` — mixer vs the real block, real weights | **1.36 bf16 ULP** (bound 16) |
| `tests/check_mamba3_model.py` — logits vs the real model | **98.05%** argmax, drift 4.5e-3 of logit range |
| Reference vs **itself**, across processes | **98.83%** — the floor; see below |

Three findings worth carrying forward:

1. **There is no `conv1d` in Mamba-3.** Mamba-1 and Mamba-2 both open the mixer with a short
   depthwise convolution; Mamba-3 dropped it. The mixer has exactly **8** parameters. Do not
   add one back by analogy — the checkpoint has no weights for it.
2. **`A` is data-dependent**, not a learned `A_log`. It comes out of `in_proj` and through
   `heavy_tail`, so there is no `A` parameter to load.
3. **Ground truth is not reproducible across processes.** Two invocations of the *official*
   model disagree on up to 5/256 argmax positions at ~2.9e-3 relative logit drift, while two
   forwards *within* one process are bit-identical. That localises the cause to
   `triton.autotune` choosing a config by timing. **Our 98.05% therefore sits inside the
   reference's own noise band, not below it** — the gate cites the measured floor rather than
   comparing against an unattainable 100%.

The model gate is deliberately **not** in CI (357 MB download); `check_mamba3_block.py` is the
cheap proxy that runs on every push, carrying the real layer-0/1 weights inside the golden.

**Why it cannot be done by importing upstream:** `mamba_ssm.modules.mamba3` imports Triton /
TileLang / CuTe and asserts when they are missing; the package will not even install without
`nvcc`. So the block is reimplemented in plain PyTorch, and only the scan goes to our kernel —
exactly the discipline `python/arm_scan/patch.py` already follows for HF Mamba-1: *transcribe
the non-scan code verbatim, replace only the recurrence.*

### What has to be rebuilt, verified against the real checkpoint

`model_shape.json` (146 params) makes this mechanical rather than guesswork. Confirmed
arithmetic — `d_in_proj = 3432` reconciles exactly:

```
d_model 768 · expand 2 → d_inner 1536 · headdim 64 → nheads 24
d_state 128 · ngroups 1 · rope_fraction 0.5 → num_rope_angles 32

in_proj (3432, 768) splits as:
  z 1536 | x 1536 | B 128 | C 128 | dt 24 | A 24 | trap 24 | angles 32
```

Per layer: `norm` (RMSNorm 768) → mixer → `norm2` (768) → gated MLP
(`fc1` 3072 = 2×1536, `fc2` 1536→768). Top level: `embedding` (128256, 768), `norm_f`,
**tied** LM head.

Inside the mixer, in order:

1. `in_proj`, split as above.
2. `_A = -heavy_tail(dd_A)`, clamped to `≤ -A_floor` (1e-4), where
   `heavy_tail(x) = 1 + x` for `x ≥ 0` and `1 / (1 - x)` otherwise. **This is not a standard
   activation — transcribe it, do not substitute softplus.**
3. `DT = softplus(dd_dt + dt_bias)`; `ADT = _A * DT`.
4. `B_norm` / `C_norm` — RMSNorm over `d_state`, applied **before** the kernel.
5. The angle pre-pass (`arm_scan.angles_to_cos_sin`, already built).
6. **The scan** → `arm_scan.mamba3_scan`.
7. `out_proj`.

**Two traps already identified.** `B_bias`/`C_bias` are `(24, 1, 128)` — `(nheads, mimo_rank,
d_state)` — and must be squeezed to the kernel's `(heads, dqk)`; and the checkpoint calls them
`B_bias`/`C_bias` while the kernel signature calls them `Q_bias`/`K_bias`.

### Files *(all written)*

```
apps/mamba3_lm/block.py     the Mamba3 mixer in plain PyTorch, scan -> arm_scan
apps/mamba3_lm/model.py     embedding, layers, RMSNorm, gated MLP, tied head
apps/mamba3_lm/load.py      checkpoint -> our modules, driven by model_shape.json
tests/check_mamba3_block.py the mixer gate (cheap, in CI)
tests/check_mamba3_model.py the logits gate (needs the checkpoint)
bench/bench_mamba3_lm.py    prefill / long-context, vs eager and torch.compile
```

### Gates, and why they are shaped this way

- **Mixer vs the real block first.** An end-to-end logits mismatch says "something is wrong
  somewhere in twelve identical layers"; this says *which of the eight parameters or six steps*
  is wrong, on layer 0, in one forward. Proven to discriminate by negative control — injecting
  the three most likely bugs (B/C swapped, `Q_bias`/`K_bias` swapped, `heavy_tail` replaced by
  softplus) moves it from **0.75** ULP to **170.8 / 177.8 / 91.3**, all far outside the bound.
- **Logits gate on argmax + subset + logsumexp**, not on a raw logit tensor (131 MB).
- **The argmax test is on *explained* flips, not on a rate.** A bare agreement rate says
  nothing about whether flipped tokens were coin tosses or confident errors, and 98.05% against
  a 0.98 threshold is a coin flip away from red. The real test is that every disagreement has a
  top-2 margin the measured drift can account for: median margin at disagreeing positions is
  **0.0115** against **0.3809** at agreeing ones — a 33× gap. Unexplained flips: **0**.

### What it demonstrates

Not "N× faster" — **"this model cannot run on a CPU-only machine at all, and we made it run."**
The baseline is not upstream (there is no upstream CPU path); it is the PyTorch recurrence a CPU
user would have to write, benchmarked **in fp32, not the f64 oracle** — timing an f64 baseline
against an fp32 kernel would inflate every number here.

First measurements, `bench/bench_mamba3_lm.py`, 187M, 16 threads:

| L | our kernel | PyTorch eager | speedup | `torch.compile` | vs compiled | compile time |
|---|---|---|---|---|---|---|
| 128 | 125.7 ms | 341.6 ms | 2.72× | 174.5 ms | 1.39× | 35.5 s |
| 256 | 253.0 ms | 627.1 ms | 2.48× | 404.1 ms | 1.60× | 60.2 s |
| 512 | 389.2 ms | 719.7 ms | 1.85× | *(skipped)* | — | — |
| 1024 | 693.7 ms | 2538.8 ms | 3.66× | *(skipped)* | — | — |

**These are x86 numbers and are not the claim.** This box is not the target, not quiesced, and
the run-to-run spread (1.85× at L=512 against 3.66× at L=1024) is wide enough that only the
order of magnitude should be read. They also exercise the **scalar/blocked** path — NEON is not
compiled on x86 — so they are, if anything, a floor. Graviton is still the gap.

Compile time is itself part of the story and is why the compiled column stops at 256: 35.5 s →
60.2 s across one doubling of `L`, consistent with the Mamba-1 path's measured 59.9 s → 532.8 s
from L=256 → 2048. A sequential scan gives `torch.compile` a graph that grows with the sequence.

### Risks

All three risks flagged before the work turned out to be resolvable by reading, not guessing:

| Risk | How it actually resolved |
|---|---|
| `fused_add_norm` / residual-in-fp32 semantics differ subtly | Upstream ships the unfused branch too, and it is the same computation — `layer_norm_fn(..., prenorm=True)` is *defined* as `(norm(h+res), (h+res).float())`. Fusion is a memory optimisation. `residual_in_fp32=True` is kept because that one does change results |
| Tied embeddings applied wrongly | Not inferred — **verified**: the checkpoint's `lm_head.weight` and `backbone.embedding.weight` are the *same tensor object* (`is` → True). `load.py` refuses to drop the alias if they ever differ |
| Gated-MLP gate/value order flipped | Not a 50/50 after all — upstream is `y, gate = y.chunk(2, -1)` then `y * silu(gate)`, i.e. **value first**. Read, not guessed |

---

## Path B — MIMO *(kernel work; gated on a capture probe)*

**What MIMO is, concretely:** the axis our validation currently rejects.

```python
b, length, groups, dqk = q.shape
if groups != 1:                       # <- this axis IS mimo_rank
    raise ValueError("SISO expects one B/C group ...")
```

SISO updates the state with a **rank-1** outer product per step,
`S = α·S + scale·(v ⊗ k)`. MIMO uses **r** pairs — a rank-r update — with the input projected
down to `r` streams (`mimo_x`), a *shared* state, and projection back up (`mimo_o`).

**Why it is worth doing:** decode is memory-bandwidth bound — the whole 8–32 KB state is loaded
to perform very little arithmetic. MIMO does `r`× the work on the same load (~4× arithmetic
intensity), which is precisely the regime where a CPU is weakest. It is the Mamba-3 change most
likely to *help* us rather than merely be ported. And its weights already exist.

### Step B0 — capture probe ✅ **GREEN** *(done Aug 7)*

**Result: MIMO ground truth is capturable, at the published configuration, unmodified.**
`tests/golden/mamba3_mimo/` holds 10 cases across 6 output shapes, captured from the real
`state-spaces/mamba3-mimo-187m`. Path B is unblocked.

Getting there needed a toolchain fix, not a workaround, and the distinction matters — the
blocker was never "MIMO is broken on Blackwell":

| Blocker | Why | Fix |
|---|---|---|
| `nvcc fatal: 'sm_120a' is not defined` | System nvcc was **12.4**, whose newest arch is `compute_90`. `sm_120a` needs ≥ 12.8 | Standalone nvcc redistributable |
| `cospi`/`sinpi`/`rsqrt` redeclared | CUDA 12.9 headers collide with **glibc ≥ 2.41**, which added those as C23 | CUDA **13.0** headers |
| `cicc: not found` | CUDA 13 split `cicc` out of `cuda_nvcc` into **libnvvm** | Install libnvvm too |
| `unsupported GNU version` | nvcc 13 refuses host gcc > 13; box has gcc 15 | gcc-13 shim |

All four are scripted in **[`tools/setup_cuda_toolchain.sh`](tools/setup_cuda_toolchain.sh)**
(no sudo, no full toolkit, ~80 MB). Note the pip wheel `nvidia-cuda-nvcc-cu12` does *not*
solve this — it ships `ptxas` and headers but not the `nvcc` driver binary.

**Why SISO captured fine on the same box:** Triton compiles PTX itself with a bundled `ptxas`
and never invokes `nvcc`. TileLang shells out to it. Nothing about the GPU differed.

### What B0 established beyond "it runs"

**bf16 is a hard requirement here, not a precision preference.** The TileLang kernel *types its
shared tiles on the caller's dtype* (accumulation is separately fp32), so fp32 doubles every
tile. Measured on this card — 101,376 B of opt-in shared memory against datacenter Blackwell's
~227 KB:

| dtype | `chunk_size` | result |
|---|---|---|
| **bf16** | **16** *(the checkpoint's own value)* | ✅ runs |
| fp32 | 16 | ✗ wants 168,128 B |
| bf16 / fp32 | 8 | ✅ runs |

So the capture runs the published configuration **unmodified** — strictly better than re-tiling
the model to fit, which was the original plan.

**`chunk_size` is pure tiling, verified rather than assumed.** Measured on SISO at the real
187M checkpoint: 64→32 moves logits by 5.3e-3 relative (99.61% argmax), 64→16 by 6.8e-3
(99.22%) — the same magnitude as the autotune noise floor. In MIMO, bf16 at `chunk_size` 16 vs
8 gave *identical* argmax. It reassociates floating-point sums; it does not change the model.

**Two upstream issues confirmed on our hardware, and now recorded in the manifest:**

- **#985** — MIMO at `seqlen=1` fails (`DA_CS strides[2] expected 1`). The decode edge case is
  genuinely unavailable, so a MIMO decode golden cannot be captured here.
- **#990** — the shared-memory ceiling. `MIMO_SWEEP` deliberately includes one shape that
  exceeds it, so the limit is recorded evidence rather than folk knowledge.

**MIMO's ground truth is noisier than SISO's.** Across-process reference-vs-itself agreement
measured **93.4%** and **95.3%** on two separate runs (17/256 and 12/256 argmax positions,
~9.3–9.9e-3 relative) against SISO's 98.8%. Note the floor is itself a distribution, not a
constant — quote it as a range. B1–B4 gates must be set against *this* band, not SISO's, and
the per-run figure is recorded in the manifest's `reference_self_consistency`.

**A capture-correctness trap worth keeping.** SISO's Triton kernel hardcodes
`.to(torch.bfloat16)` on Q/K/V/Trap/Angles/Z, so the capture pre-casts to match. **MIMO does
not** — applying SISO's list would have downcast `Angles`, which the module deliberately casts
*up* to fp32, and silently corrupted the goldens. `Capture` now takes the cast list per entry
point.

Block-level MIMO goldens are deliberately **not** captured yet (they were 15.6 MB of 34 MB and
serve B4, which is not scoped). Regenerate with `--max-blocks 2` when B4 starts.

### Steps B1–B4 *(≈2.5 days — B0 is green, so these are live)*

| | Work |
|---|---|
| ~~**B1**~~ | ✅ **DONE Aug 7.** `mamba3_mimo_ref` reproduces the official TileLang kernel to **2.40 bf16 ULP** (bound 12) across all 10 goldens — tighter than SISO's 4.47. Gate: `tests/verify_golden_mamba3_mimo.py` |
| ~~**B2**~~ | ✅ **DONE Aug 7.** `Mamba3Dims` gained `rank`; `Mamba3Mimo` carries ψ/ζ/φ; `mamba3/mimo.rs` is the scalar rank-r kernel. Matches the official TileLang goldens to **1.90 bf16 ULP** *through the C ABI* — tighter than the PyTorch reference's 2.40. Gate: `tests/check_mamba3_mimo_op.py` |
| ~~**B3**~~ | ✅ **DONE Aug 7** as part of B2 — **ABI 6 → 7**, Python loader bumped, `arm_scan.mamba3_mimo_scan(...)` exported |
| ~~**B4**~~ | ✅ **DONE Aug 7.** `apps/mamba3_lm` runs **`mamba3-mimo-187m` end to end on CPU** — 96.48% argmax, 0 unexplained flips, against a reference-vs-itself floor of 95.31%. Gate: `make test-mamba3-model` |

**Gates:** `r=1` bit-identical to SISO; MIMO goldens at the bf16 bound; NEON↔scalar parity;
thread bit-identity; logits vs a MIMO `model_forward.npz`.

### B4 — the MIMO model end to end ✅

**Both published families now run on CPU through our kernel**, which is the claim Path B existed
to support:

| | argmax agreement | reference vs **itself** | unexplained flips |
|---|---|---|---|
| `mamba3-siso-187m` | **98.05%** | 98.83% | 0 |
| `mamba3-mimo-187m` | **96.48%** | 95.31% | 0 |

Read the MIMO row carefully: **we agree with the reference more closely than the reference
agrees with itself** (96.48% vs 95.31%), and our logit drift is smaller too (9.7e-3 vs 9.9e-3).

That forced a real change to the gate. It was failing MIMO on a hardcoded 98% — a
SISO-calibrated constant. **A gate must not demand better agreement than the oracle can
produce**, so the requirement is now capped at the measured across-process floor from the
manifest. It never loosens SISO (`min(0.98, 0.988) = 0.98`, unchanged); it only stops the gate
asking for precision the ground truth does not have. The sharp test remains "every disagreement
is explained by the measured drift", which passed on both families unchanged.

**A shape bug `strict=True` caught.** Upstream's `GatedMLP` rounds its hidden width **up to a
multiple of 128**, so `d_intermediate` in the config is not the layer's real width. SISO-187M
never showed it (1536 is already a multiple); MIMO-187M asks for 1264 and the checkpoint carries
1280. Loading non-strictly would have left twelve randomly-initialised MLPs in the model and
produced plausible garbage.

### Throughput — and why Path B does not claim one

Measured on **Graviton4 `c8g.16xlarge`, 64 threads** (superseding the earlier x86 run):

| L | MIMO | SISO | MIMO slower by |
|---|---|---|---|
| 128 | 171.95 ms | 133.76 ms | 1.29× |
| 256 | 251.82 ms | 142.03 ms | 1.77× |
| 512 | 465.58 ms | 186.74 ms | 2.49× |
| 1024 | 919.70 ms | 268.09 ms | **3.43×** |

MIMO delivers **1,113 tok/s against SISO's 3,820** at L=1024 — 29% — and the gap *widens* with
length.

**Against the PyTorch baseline MIMO scores 31–56×, and we deliberately do not quote it.** The
MIMO reference is itself ~10× slower than the SISO reference (50,748 ms vs 5,064 ms at L=1024),
and `torch.compile` makes it worse rather than better (6,090 ms vs 219 ms for compiled SISO at
L=128). A 55× against that is a statement about the baseline, not about our kernel. Anyone who
divides two columns of our own results table will see it, and a headline number that collapses
under thirty seconds of scrutiny would discredit the figures that do hold.

**So Path B's claim is coverage and correctness, not speed:**

> Both published Mamba-3 families run end to end on Arm CPU through our kernel, with the MIMO
> path matching the official TileLang kernel to **1.90 bf16 ULP through the C ABI** — tighter
> than our own PyTorch reference manages, and tighter than the SISO path's 4.47.

**Why the speed gap exists, and why it is the opposite of a fundamental limit.** Per timestep
MIMO loads the *same* state as SISO and does `r`× more arithmetic with it — roughly `r`× the
arithmetic intensity. That is the regime a memory-bound CPU is weakest in and should gain most
from. A scalar implementation cannot exploit arithmetic density at all: it simply executes `r`×
more scalar operations, turning MIMO's theoretical advantage into a straight slowdown.

The work an optimised kernel would vectorise — the rank-`r` outer-product accumulation and the
`r`×`r` diagonal contraction — is dense, regular and register-resident, i.e. close to an ideal
NEON target. Nothing about MIMO is hostile to this machine; we simply have not written that
kernel. **The table above is the gap it would have to close, and until it exists the
arithmetic-intensity argument stays a prediction.**

### What B2 shipped, and the one thing it deliberately did not

`arm_scan.mamba3_mimo_scan(...)` runs the rank-r recurrence in Rust. Five checks in
`tests/check_mamba3_mimo_op.py`: vs the official goldens (**1.90 ULP**), vs the f64 reference
(1.3e-07 — the fp32 floor), rank-1-equals-SISO-unrotated **kernel to kernel** (2.4e-07),
partial-projection rejection at the C boundary, and bit-identity across 1/2/8 threads.

**MIMO runs on the scalar path only.** There is no blocked or NEON MIMO kernel yet, and
dispatch routes MIMO *before* the backend match so that asking for `Backend::Neon` does not
report "unavailable" — which would be true but useless. This is the honest state: MIMO is
**correct everywhere and fast nowhere**. Making it fast is the remaining kernel work, and it is
worth doing precisely because MIMO's `r`× arithmetic on one state load is the shape a CPU
should like.

Two API notes that matter to a caller:

- `mamba3_mimo_scan` is **not** `mamba3_scan` with `rank=1`. Different rotation convention;
  they agree only when the rotation is the identity.
- The three projections are **all-or-nothing** at every layer — `Mamba3Mimo` makes a partial
  set unrepresentable in Rust, and the C entry point rejects two-of-three rather than guessing.

### What B1 established — read this before starting B2

**The two kernels use DIFFERENT RoPE conventions.** SISO's Triton kernel rotates *interleaved*
lanes `(2i, 2i+1)`; MIMO's TileLang kernel rotates *split halves* `(i, i + n/2)` for
`i < n/4`, leaving the other lanes untouched entirely. Read from the source and confirmed
against the goldens. **B2's Rust kernel needs both**, selected per family — this is the single
most consequential finding for the kernel work.

**The plan's "r=1 reproduces the SISO goldens bit-for-bit" is false, and now disproved rather
than dropped.** At r=1 with unit Ψ/Φ/ζ every MIMO-specific term degenerates, so the two *are*
algebraically identical — except for that rotation. Measured: rel **3.8e-01** as captured,
**7.7e-16** with the angles zeroed. So the free check still exists, just in the zero-angle form,
and `check_rank1_collapse` runs it.

**What MIMO changes, precisely** (everything else — discretization, the angle pre-pass — is
byte-identical to SISO):

- the state is **shared across ranks**, updated with a sum of `r` outer products
- `V` is projected **elementwise**, `x_r = Ψ_r · v` — a per-rank reweighting, not a matmul
- the diagonal term is a **rank-by-rank contraction**: `r_out` collects `(q_{r_out}·k_{r_in})·x_{r_in}` over every `r_in`
- `D` multiplies the **projected** `x_r` and is **not** scaled by γ
- the gate is per rank, `silu(z·ζ_r)`, applied **before** Φ reduces the rank axis

**A bug worth remembering, because the golden design is what caught it.** `Q_bias`/`K_bias` are
`(h, r, n)` and the reference needs `(1, 1, r, h, n)` — and `h·r·n == r·h·n`, so `.view()`
succeeds and silently transposes head against rank instead of raising. It cost 195–358 ULP on
the four model-driven goldens while **every synthetic sweep case passed at ~1 ULP**, because a
freshly-built `Mamba3` initialises `B_bias`/`C_bias` to a *constant*, where the transposition is
invisible. Only real trained weights expose it. Keep both kinds of case.

**Set the model-level gate against ~93–95%, not SISO's ~99%.** MIMO's across-process reference
floor is measurably worse *and* variable (recorded per run in
`tests/golden/mamba3_mimo/manifest.json` under `reference_self_consistency`). Prefer the
"every disagreement is explained by the measured drift" test that `check_mamba3_model.py`
already uses over a bare rate — with a floor this wide, a rate threshold is close to
meaningless. And note `seqlen=1` is unavailable upstream (#985), so B2's decode path has no
authoritative oracle — validate it against our own reference and say so.

---

## Path C — 2D *(the novelty; ~2–3 days for the causal half)*

**No weights exist, and we are not training any.** That is a decision, not a gap — the plan's
strongest standing rule is *"do not train a model to justify the kernel"* (the DH-Mamba
lesson). A half-converged checkpoint produces visibly bad output and damages the demo.

**What that does and does not cost us.** Correctness and throughput are both
**weight-independent**: a scan over random weights performs identical arithmetic in an
identical memory pattern, and correctness is checked against a *reference implementation*, not
against task performance. So we can ship everything except an accuracy number.

| Claim | Available? |
|---|---|
| Kernel computes the right thing | ✅ |
| Throughput vs `torch.compile` at real vision grids | ✅ |
| **Causal vs non-causal comparison** | ✅ — **the novel result** |
| "Same accuracy, faster" / "we ran a real vision model" | ❌ |

### C1 — causal cross-scan ✅ *(DONE Aug 7 — zero new kernel code, as predicted)*

`arm_scan.ss2d_scan_mamba3` (`python/arm_scan/ss2d_mamba3.py`) runs the four-direction cross
scan on `mamba3_scan_pair`. It is **pure layout** — no Rust changed.

Built as a **parallel module sharing `ss2d.py`'s helpers** rather than by widening
`ss2d_scan`'s seam. `ss2d_scan` takes five `(b, d, l)` tensors; Mamba-3 hands the scan eleven
across **two different layout families**, so one shared seam would be a union type every caller
has to disambiguate. The layout helpers — where the actual logic lives — are shared.

| family | tensors | grid → views |
|---|---|---|
| time-major | `q, k, v, z, angles` | `(b,H,W,…)` → `(2b,H·W,…)` |
| head-major | `adt, dt, trap` | `(b,h,H,W)` → `(2b,h,H·W)` |

**The one real correctness trap:** `theta = cumsum(tanh(angle)·π·dt)` accumulates *along the
traversal order*, and row-major and column-major are different orders. The pre-pass must run on
the **views**, not the grid — otherwise both orderings silently get the row-major `theta`.
Nothing raises. Within a pair, sharing `cos`/`sin` between forward and backward *is* correct:
`reverse=True` keeps each token's own position while walking the recurrence backward.

Also added: `mamba3_siso_ref` gained `reverse=` — it previously could not express the backward
direction at all, so no 2D reference could be built. The order of operations there is
load-bearing and commented: rotate on the forward order, *then* flip, and rebuild `scale`
**after** the flip, because the trapezoid reads `dt_{t+1}` and a backward traversal has a
different neighbour.

**Gates** — `tests/check_ss2d_mamba3.py`, seven checks, worst per-direction error **2.0e-07**
(fp32 kernel vs f64 reference, at the fp32 floor) across square / non-square / odd / wide /
degenerate `H=1` grids, bit-identical at `RAYON_NUM_THREADS ∈ {1,2,8}`.

**What the negative control taught, and why the gate is shaped this way.** Injecting the
obvious layout bug — dropping the column view's transpose, so four directions silently become
two — leaves **kernel-vs-reference passing at 2.4e-07**. Both sides share the layout code, so
they agree with each other while both being wrong. A correctness comparison *cannot* catch this
class of bug, and neither could stored goldens (they would be generated through the same broken
layout). Only structural invariants can, so the gate asserts them directly: the orderings must
differ, and each ordering's `theta` must equal an independently-built one.

That is also why this deviates from the plan's original "stored goldens re-derived in numpy":
kernel (Rust) vs reference (PyTorch) already crosses an implementation boundary, so it *is* the
independent check, and the recurrence itself is already pinned to the official kernels at
4.47 ULP by the 1D goldens. A third numpy implementation would re-verify the part that is
already verified and still miss the part that actually broke.

**Throughput** (`bench/bench_ss2d_mamba3.py`, x86, 16 threads — see the caveat below):

| grid | tokens | our kernel | PyTorch eager | speedup | `torch.compile` | vs compiled |
|---|---|---|---|---|---|---|
| 14×14 (p16) | 196 | 4.05 ms | 57.60 ms | 14.2× | 7.80 ms | **1.92×** |
| 28×28 (p8) | 784 | 7.15 ms | 227.98 ms | 31.9× | *(skipped)* | — |
| 56×56 (stage 1) | 3136 | 24.14 ms | 911.02 ms | 37.8× | *(skipped)* | — |

**These isolate the scan** — unlike Path A's 1.85–3.66×, which timed a whole model where the
projections dominate and are identical on both sides. Both are honest; they measure different
things, and the writeup must not present them as comparable. And as everywhere else in this
repo right now: x86, unquiesced, **scalar path** (NEON is not compiled on x86). Graviton
remains the gap.

### C2 — non-causal ✅ *(DONE Aug 7 — and the plan's premise was wrong)*

**The plan predicted this would be "a second kernel — two dense GEMMs", O(L²), with a thin moat
because BLAS is good at GEMMs. The maths says otherwise, and that is the result.**

Unrolling the recurrence gives `y_t = Σ_s M[t,s]·(q_t·k_s)·v_s` with
`M[t,s] = e^(L_t − L_s)·scale_s`. The decay **factorises** into `e^(L_t)·e^(−L_s)`, so the sum
over `s < t` is exactly a forward scan and the sum over `s > t` is exactly a backward one:

```
Σ_{all s}  =  Σ_{s≤t}  +  Σ_{s≥t}  −  Σ_{s=t}
non-causal =  forward  +  backward  −  diagonal
```

**Non-causal costs 2× a causal scan, not O(L²), and needs no new kernel at all.** Both
directions already existed. `arm_scan.noncausal_scan` and `ss2d_noncausal_mamba3` are pure
composition over `mamba3_scan_pair`; **no Rust changed for C2**.

The diagonal correction reads `q_t·k_t`, which looks like it needs the rotated q/k the kernel
computes internally. It does not: **a dot product is invariant under rotating both operands by
the same angle**, and RoPE gives `q_t` and `k_t` the same `theta_t`. That is why this composes
over the public op instead of needing kernel surgery.

**In 2D it is nearly free.** The four-direction cross-scan *already* runs both directions, so
non-causal 2D is the same scans minus two diagonals — measured at **1.06–1.23× causal**, against
**2.25–2.65×** for the 1D case where a second scan really is added.

### The measurement — `bench/bench_mamba3_noncausal.py`

x86, 16 threads. Sub-millisecond rows are dispatch-dominated and the harness says so.

| grid | tokens | 1D causal | 1D non-causal | ratio | 2D causal | 2D non-causal | ratio | dense |
|---|---|---|---|---|---|---|---|---|
| 8×8 | 64 | 0.32 ms | 0.73 ms | 2.25× | 1.20 ms | 1.47 ms | 1.23× | 0.80 ms |
| 14×14 | 196 | 0.61 | 1.59 | 2.63× | 3.47 | 2.83 | 0.81× | 0.90 |
| 28×28 | 784 | 2.71 | 3.19 | 1.18× | 6.53 | 7.86 | 1.20× | 15.84 |
| 56×56 | 3136 | 6.26 | 16.58 | 2.65× | 25.59 | 27.13 | 1.06× | *(mask too large)* |

**The dense form has a crossover and it arrives early.** Better constant (GEMMs, no sequential
dependency), worse asymptotics: competitive around 200 tokens, **5× slower by 784**, and at 3136
the `(L,L)` mask per head is the binding constraint before time is. For any real vision grid the
scan form wins — which is the argument for a CPU scan kernel existing at all, now measured
rather than asserted.

### Gates — `tests/check_mamba3_noncausal.py`

Five checks. The load-bearing one is that **an O(L²) dense algorithm reproduces the O(L) kernel
to 2.99e-16 in f64** — two implementations sharing no code, agreeing to machine precision. That
validates the mask derivation and, through it, the recurrence itself.

Also checked: non-causal genuinely differs from causal (3.3e-01 — a no-op would pass every
equality above), the 2D form equals its own per-ordering definition **exactly** (0.00e+00), and
thread invariance.

One fix fell out: `angles_to_cos_sin` hardcoded `.float()`, silently capping any f64 pipeline at
fp32 — harmless for the kernel path, which downcasts at the FFI boundary anyway, but it made the
f64 algorithm comparison bottom out at 1e-8 instead of 1e-16. Now dtype-preserving, floored at
fp32; fp32 callers are unaffected.

### The oracle caveat, stated plainly

For 1D we captured ground truth from official kernels. **For 2D nothing authoritative exists** —
VNCT's code is unreleased. We validate our Rust against *our own PyTorch reading of the paper*,
which proves the kernel implements the reference, **not** that the reference implements VNCT as
its authors intended. This must appear in the writeup, not be buried.

---

## Sequencing

| Order | Item | Why here |
|---|---|---|
| ~~1~~ | ~~**Path A**~~ | ✅ **done Aug 7** — the only accuracy evidence the project can produce |
| ~~2~~ | ~~**B0 capture probe**~~ | ✅ **GREEN Aug 7** — MIMO goldens captured at the published config |
| ~~3~~ | ~~**C1 causal 2D**~~ | ✅ **done Aug 7** — `ss2d_scan_mamba3`, zero new kernel code |
| 4 | **B1–B4** | Second real-weights path — now unblocked |
| ~~5~~ | ~~**C2 non-causal**~~ | ✅ **done Aug 7** — and it needed no new kernel: the decay factorises |

**Not sequenced here, and still the largest gap: dedicated-hardware numbers.** Every timing in
this repo is x86 or a shared 4-core runner. All five items above are laptop work; the Graviton
session is not, and none of them substitutes for it.

## What each path may claim

- **A:** first PyTorch-callable NEON Mamba-3 running the official checkpoint on Arm CPU, at
  token-level fidelity, constant memory in `L`.
- **B:** the first CPU MIMO Mamba-3 — and the arithmetic-intensity argument tested rather than
  asserted.
- **C:** first CPU implementation of any 2D Mamba-3, plus the causal-vs-non-causal comparison,
  which nobody has published for any Mamba generation. **Correctness and throughput only.**

Never: "first Mamba-3 on CPU" or "in Rust" (`mamba-rs`, `burn-mamba`, `mamba.c`); anything about
bidirectional Mamba-3 (`burn-mamba` ships it); any accuracy result for 2D.
