# SUBMISSION_ENDGAME_PLAN — the last 10 days (Aug 4 → Aug 14, 2026)

**Written Aug 4, 2026.** Supersedes the week tables in [`ROADMAP.md`](ROADMAP.md) §4 and
[`SS2D_REPOSITIONING_PLAN.md`](SS2D_REPOSITIONING_PLAN.md) §7, both of which assumed the Jul 20 and
Jul 27 weeks happened. They did not — the last commit is `eb707f6`, **Jul 18**. This doc is the
single source of truth for what ships, what gets cut, and in what order. Where it disagrees with an
older plan doc, this one wins.

**The situation in one line:** the 1D kernel half is finished and excellent; the SS2D + diffusion
half that the project repositioned around on Jul 17 is ~⅓ built; there are **zero numbers from Arm
hardware anywhere in the repo**; and there are 10 days left, of which the last 2 are buffer.

---

## STATUS — updated Aug 4, 2026 (end of the hole-closing pass)

**Closed in code today.** All verified locally on x86 against the real cdylib through the C
ABI — so the *scalar* backend is proven and the **NEON path still needs re-verification on
Arm**, which the `mri-app` CI job now does on every push.

| Item | State |
|---|---|
| K1 SS2D → fused bidirectional pairs | ✅ **1.77–1.82× (geomean 1.80×)** block total at production grids, 1.81–1.90× on scan alone; overhead **21–25% → 7.2–13.8%** |
| K2 parity gate vs the legacy oracle | ✅ three gates (refactor / kernel / cross-backend), green at `RAYON_NUM_THREADS` ∈ {1,2,8} |
| K3 2D goldens | ✅ 6 cases; independent numpy re-derivation agrees to ~5e-16; kernel replay at **1.0× the recorded f32 floor** |
| H7 `ss2d_scan` public contract | ✅ real `(B,D,H,W)` API in `arm_scan.ss2d`, exported from the package |
| T2 de-hardcode + `edm_min.py` | ✅ app tests run on a clean checkout; no CC-BY-NC-SA dependency on the judge path |
| T3 CI + Makefile | ✅ new `mri-app` job; `make validate` now covers SS2D + diffusion, not just 1D |
| D1 `demo.py` | ✅ end-to-end, emits PNG + metrics JSON, `--save-prior` for repeat takes |
| T1 doc truth-pass | ✅ stale sequencing tables redirected here; Mamba-3 docs bannered post-submission |

**Two defects found while verifying** — neither visible from reading the code:

1. **Phase C's parity gate was vacuous.** `out_proj`/`head` are zero-init
   (identity-at-init residual), so an untrained net returns `c_skip * x` no matter what the
   scan computed — the check reported `max_abs = 0.000e+00` because every scan output was
   multiplied by zero. It now activates those layers first and measures 8.3e-7.
2. **The reference scan's backward was ~23× its forward** (147 s vs 6.3 s at a 32×32 grid).
   Indexing `deltaA[:, :, t]` inside the time loop registers one `select` backward per
   timestep, each scattering into a full `(b,d,l,n)` gradient buffer. Hoisting one `unbind`
   outside the loop: **backward 147 s → 6.7 s**, training step 153 s → 12.9 s. That is what
   makes small-scale prior training practical at all, so it directly de-risks A1.

**A third finding, and it is a real one: Phase D has never passed.** Its gate asserts the
R=4 reconstruction beats zero-filled by >1 dB; measured, it is **2.75 dB worse**
(13.69 → 10.94 dB), and it fails before reaching its own kernel-vs-reference parity check.
This is *not* a regression from the work above, and the evidence rules the alternatives out:

- Phase B bring-up **passes** — the same backbone under the same precond/loss trains
  (loss 0.59 → 0.22) and denoises hard: **11.4× MSE reduction at σ=0.3, 17.0× at σ=1.0**,
  with persistence round-tripping. The model is fine.
- Pair-vs-legacy cross-scan parity is 4e-6 and the 2D goldens replay at 1.0× the f32 floor.
  The scan is fine.
- Phase C parity (now non-vacuous) passes at 8.3e-7. The kernel-in-the-loop is fine.
- `CLAUDE.md` itself only ever claimed *"A (GO), B, C are gated green; D's gate test
  **exists**"* — it never claimed D passed.

**→ Now fully diagnosed, with measurements, in
[`PHASE_D_DIAGNOSIS.md`](PHASE_D_DIAGNOSIS.md).** Read that before touching anything.

The hypothesis originally recorded here — that hard DC at high σ was splicing clean k-space
into a noise-dominated iterate — **was wrong, and the measurement says so.** DC is applied to
`net(x, σ)`, the *denoised estimate* x̂₀, which is the standard, correct scheme. Replacing the
denoiser with an oracle reconstructs to **151 dB**, so the sampler, FFT, mask and DC are
exact and there is no sampler bug. The real causes, ranked: the evaluation data (smooth
Gaussian bumps) has almost no energy outside the sampled low frequencies, so zero-filled is
already near-optimal and a prior can only lose; the mask delivers effective R=2.67 when it
says R=4; the sampler runs to σ=80 while the prior is trained at `ln σ ~ N(−1.2, 1.2)` (4.6 SD
out); and the gate itself blocks the kernel-parity check behind a model-quality assertion.
**Do not fix this by relaxing the assertion** — the >1 dB bar is the app's entire quality
claim.

**Consequence for the plan below:** the Aug 4 and Aug 5 rows are done, and **P1-5, P1-6 and
P1-7 are cut** — the pair rewrite pulled non-scan overhead to 7.2–13.8%, under this repo's
own 15% bar for justifying a fused `selective_scan_2d`, and that verdict is now backed by
`bench/bench_ss2d.py` output rather than judgement. Caveat worth carrying: the worst shape
lands at 13.8%, close enough to the bar that Arm's different cache behaviour could push it
back over. Re-run `bench_ss2d.py` in Graviton session 1 before treating the cut as final.

**Measurement hygiene note, learned the hard way:** the first pass at these numbers ran at
`--reps 2` while two other jobs were competing for the same 16 cores, and produced a
*0.50× "regression"* on one shape that does not exist — the quiesced re-run puts that same
shape at 1.82×. Every number above is from the quiet re-run. Treat any benchmark taken
while something else is running as void, on Graviton especially.

**Still open, and still the two things that decide the outcome:** no Arm hardware numbers
(H1) and no trained prior (H3). The demo currently reconstructs *worse* than zero-filled on
a short in-process prior; it prints an explicit warning instead of presenting a broken image
as a result. That is a training-budget problem, not a kernel one — the kernel-vs-reference
cross-check in the same run agrees to 4.9e-6.

---

## Part 1 — The holes

Ordered by what actually loses the competition, not by size.

### H1 — No Arm hardware evidence exists. *(Existential.)*

This is an **Arm** optimization challenge, Cloud AI track, and every number in the repo is from
either a Windows x86 i9 box or a shared 4-core GitHub Actions runner:

- [`bench/results/RESULTS.md`](bench/results/RESULTS.md) — entirely `windows-i9`, dated Jul 13.
  The scalar backend, on x86. It exercises the fallback path, not NEON.
- `RESULTS_ci-arm64.md` — real aarch64, but a shared 4-core runner, and the repo's own
  [`BASELINE_TEST_PLAN.md`](BASELINE_TEST_PLAN.md) classifies that as **provisional, not
  headline-grade**.
- The **core-scaling curve** — the single most important chart for a *Cloud* track submission, the
  "CPUs scale where the sequential scan hinders the GPU" argument — exists only as `windows-i9_t1`
  through `t32`. It has never been run on Arm.
- [`bench/ARM_BASELINE.md`](bench/ARM_BASELINE.md) opens by admitting it: *"the kernel's headline
  claim is currently unproven: the only recorded numbers are x86 + scalar fallback."* That was
  written weeks ago and is still true.

Consequence: the README's headline table (24.1× / 3.7×) is provisional by the repo's own rules, and
the four "still to land" rows are all unstarted. A judge who is an Arm engineer will look for the
Graviton row first.

**The runbook to fix this already exists and is good** (`ARM_BASELINE.md` is a complete
measure/profile/interpret procedure, plus `bench/setup_ampere.sh`, `bench/run_baseline.sh`,
`sync.ps1`). Nobody has executed it. This is a scheduling failure, not an engineering one.

### H2 — The SS2D headline is unproven, and the cheapest big kernel win is sitting unused. *(Critical.)*

Claim (2) in the README — *first fast CPU SS2D cross-scan* — is currently backed by one Python
batching change and one x86 measurement. Specifically:

- No SS2D numbers on Arm at all (`ss2d_windows-i9.json` is the only file).
- **No `torch.compile` baseline for SS2D at real shapes.** `bench_ss2d.py` measures
  arm-total-vs-arm-scan *overhead split*, and only compares against the torch reference at the
  `mini_96x80` grid. `CLAUDE.md`'s "benchmark honestly" rule requires `torch.compile` as the
  baseline that matters; the SS2D story currently has no such row at the shapes it claims.
- **No 2D goldens.** `TOPOLOGY_IMPLEMENTATION_PLAN.md` §3.3 requires per-direction goldens
  (including non-square and non-multiple-of-4 grids) to land *before* the Rust they gate. There is
  no `gen_golden_2d.py` and no 2D case in `tests/golden/`.
- P1-5 (cache-block over L), P1-6 (tile transpose), P1-7 (fused `selective_scan_2d`) are all
  unstarted — no `transpose.rs`, no 2D entry point in `lib.rs`, no 2D FFI symbol.

**And the thing that makes this fixable:** the repo already built, gated, and shipped
`selective_scan_bidirectional` (ABI v5, `neon::scan_bidirectional`, `for_each_channel_bidir`) —
one call that emits both time directions while computing the direction-independent **Pass A
(discretize + exp, ~85% of runtime) once instead of twice**. `BIDIRECTIONAL_LOG.md` measures the
exp-sharing at **1.58–1.75×, geomean ~1.67×, reproduced across four CI runs**.
`python/arm_scan/bidirectional.py`'s docstring even states that the pattern it accelerates — same
projected tensors traversed both ways — *"is also the 1D case of the SS2D cross-scan."*

`SS2DBlock.forward` does not use it. It builds `cat([rows, rows.flip(-1), cols, cols.flip(-1)])`
and makes one plain **forward** call on a 4B batch — so Pass A is computed **four times** when there
are only **two** distinct Pass A inputs (`rows` and `cols`). The two halves of each pair differ only
in traversal order, which is exactly what the fused kernel exists to exploit.

This is the highest-leverage change available in the entire project, it is **pure Python**, and it
runs on an already-verified kernel path. See K1 below.

### H3 — There is no trained prior, so there is no application. *(Critical, and the biggest time sink if handled wrong.)*

- The Route A/B prior decision and the GPU budget — flagged in `CLAUDE.md` as *"the only open
  decision that can starve everything downstream"* — are **still open** after 16 days.
- No training or distillation script exists anywhere in the repo (`grep` for `distill`/`train_prior`
  returns only the test files).
- Everything app-side runs at **32×32** on a toy prior trained inline for 200 steps inside
  `test_phase_d_partial.py`. `MambaSS2DNet` has **never been run at the real 384×320 resolution** —
  not once, not even a single forward pass. Phase A measured the *U-Net teacher* at 8.8 s/NFE on
  x86; our backbone at L=122,880 × 6 blocks × 4 directions is unmeasured at full size.
- The README promises "PSNR/SSIM/NMSE parity at R=2–8" and "$/reconstruction". None of that is
  reachable without weights.
- fastMRI requires NYU registration, which Phase A flagged as "start early." It wasn't started.

Route A (distill CSI's 65.5M-param U-Net into `MambaSS2DNet` at 384×320) is days of GPU work plus
tuning, and a half-converged prior produces a *visibly bad* reconstruction — the WOW criterion
backfires. With 10 days and H1/H2 outstanding, **Route A is not affordable.** See A1.

### H4 — The app is not reproducible by anyone but you, and may have a license problem. *(High — this is the DX score, 15 points.)*

- All three files in `apps/mri_diffusion/tests/` hardcode
  `REF = Path(r"C:\Users\Adity\Claude\Projects\reference\ambient-diffusion-mri")`. Nobody else can
  run them. Note it isn't even the current user's home directory — it's a stale path.
- **None of the app tests run in CI.** `.github/workflows/` has no reference to `mri_diffusion` or
  `ss2d`; CI gates the kernel only. So Phases B/C/D are "gated green" by a manual run on one
  Windows box, weeks ago.
- `make validate` covers the kernel only — the README's "judge runs this in 5 minutes" promise does
  not touch the SS2D or diffusion path that the whole pitch now rests on.
- **License exposure to verify before shipping:** the app tests import `dnnlib`, `training.networks`
  and `training.loss` from the CSI fork of NVlabs/edm. NVlabs/edm is distributed under
  **CC BY-NC-SA 4.0** (non-commercial, share-alike) — which does not compose with this repo's MIT
  license, and "non-commercial" is a poor fit for a corporate-sponsored contest. Confirm the actual
  license terms, and make the judge-facing path depend on **our own implementation of the EDM
  equations** (they are ~20 lines of preconditioning arithmetic from the paper) rather than on their
  source. Re-implementing published equations is fine; vendoring NC-licensed code into an MIT repo
  is not.

### H5 — No demo, no video, no writeup, and no demo *script*. *(High — items 5 and 6 of `CLAUDE.md`'s own definition of done.)*

There is no `demo.py`, no Gradio app, no image-producing script anywhere. The video needs a visual
artifact that does not exist yet: the side-by-side (zero-filled vs. reconstruction vs. ground truth)
with a live timer. That has to be built before it can be filmed, and filming happens on rented
hardware, so it must exist *before* the Graviton session — not during it.

### H6 — The repo's own documentation no longer tells the truth about the repo. *(Medium, but it is the first thing a judge reads.)*

- `CLAUDE.md` "Where things stand" is headed **"as of Jul 13"** with a Jul 17 parenthetical, and
  lists P1-5/P1-6/P1-7 as "next" — 16 days stale.
- `ROADMAP.md` §4 and `SS2D_REPOSITIONING_PLAN.md` §7 both show Jul 20 / Jul 27 work as scheduled;
  it never happened, and this week's row says "Graviton session 1," which has not been booked.
- 22 markdown files, ~5,700 lines, against a `CLAUDE.md` rule that says *"keep them
  non-duplicative."* Four separate documents now contain a sequencing table.
- The two most recent commits added `RESEARCH_TRIAGE_MAMBA2_2D.md` and the 246-line
  `MAMBA3_KERNEL_PLAN.md` — both explicitly **post-submission**, on `feature/mamba3`, by their own
  scope guard. The last activity before the repo went quiet went into work that, by its own terms,
  cannot help the Aug 14 entry. (The Mamba-3 plan is good and should survive as "future work" in the
  writeup — one paragraph, not a headline.)

### H7 — Two smaller correctness/claims gaps worth 20 minutes each.

- `ss2d.py` (31 lines) is a `scan_fn` seam plus a monkeypatcher; it does **not** implement the
  `ss2d_scan((B, D, H, W))` contract that `TOPOLOGY_IMPLEMENTATION_PLAN.md` §3.1 specifies and the
  README implies ("`arm_scan.ss2d` routes VMamba-style 4-direction cross-scans"). The actual
  cross-scan logic lives in the *app's* `SS2DBlock`, so the reusable-artifact claim is weaker than
  stated. Either move the direction logic into `ss2d.py` (preferred — it is also what makes the
  "reusable kernel" claim true) or soften the README wording.
- `python/README.md`, `bench/README.md`, `tests/README.md` were supposed to gain `ss2d.py` /
  `bench_ss2d.py` / 2D-golden mentions "as each lands" (SS2D plan §3). Check they don't promise
  things that still don't exist.

---

## Part 2 — Strategy: the three calls that make 10 days feasible

**Call 1 — Cut Route A distillation. Train small instead.**
Target a **128×128** (or 160×160) single-coil SS2D-Mamba EDM prior on synthetic/phantom + IXI-style
data, ~2–4 GPU-hours on one rented spot instance (~$3). This keeps *"diffusion-prior MRI
reconstruction running on Arm CPU through our kernel"* literally true end-to-end, at a resolution
where the latency numbers are honest, and it kills the single largest schedule risk. State the scope
plainly in the README: **we claim a working CPU pipeline with kernel-vs-reference quality parity, not
SOTA reconstruction quality.** The 384×320 grid still appears — as *kernel* benchmark rows via
`bench_ss2d.py`, which needs no weights.

**Call 2 — Cut P1-5, P1-6, and P1-7. Ship the fused-bidirectional SS2D instead.**
P1-7 is a week; there isn't one. But cutting it does **not** cost the "first fast CPU SS2D" claim,
because K1 makes SS2D run on a genuine *kernel-level* optimization — Pass A shared across
directions inside NEON, threaded over channels — not on a Python rearrangement. That is a real,
defensible, measurable kernel contribution, it reuses an already-gated code path, and it costs a day
instead of a week. The measured 21–25% overhead finding gets published as-is, with the fused 2D
kernel named as the identified next lever.

**Call 3 — Hardware first, polish second.**
The Graviton session is the only irreplaceable item: everything else can be written on a laptop, and
no amount of code quality substitutes for a missing headline number. Book session 1 for **Aug 6**,
after K1 lands (so the SS2D rows measure the good path) and before anything else competes for time.

---

## Part 3 — The plan, day by day

Two hard gates. Everything else is droppable in priority order.

| Day | Track | Deliverable |
|---|---|---|
| **Tue Aug 4** ✅ | Truth + kernel + correctness + demo | **DONE — three days' worth.** T1 doc truth-pass; T2 de-hardcode + `edm_min`; **K1** SS2D → fused bidirectional pairs; K2 parity gate; K3 2D goldens; T3 CI + Makefile; **D1 `demo.py`**; Phase D diagnosed; CI unbroken (red since Jul 17); 8 commits pushed |
| **Wed Aug 5** | Unblock CI + Phase D | Open the PR so `mri-app` runs (**first NEON verification**); Phase D **D-1…D-5.1** (gate split, mask fix, phantom eval data, σ_max, `prior_report.py`) — all no-GPU; **G1: prior route decided** |
| **Thu Aug 6** | **GRAVITON SESSION 1** | Full `ARM_BASELINE.md` runbook + `bench_ss2d.py` + per-NFE. ~3 h, terminate after |
| **Fri Aug 7** | Results | R1 `RESULTS.md` from real JSON; R2 README truth-alignment; **G2: fused-2D go/no-go** |
| **Sat Aug 8** | App | **D-5** train the prior (GPU, bounded, stop when `prior_report` clears) |
| **Sun Aug 9** | App / slack | **D-6** re-run the quality gate; quality table; absorb slippage |
| **Mon Aug 10** | **CODE FREEZE** + **GRAVITON SESSION 2** | Final numbers, demo capture, video footage. ~2 h |
| **Tue Aug 11** | Submission | V1 video cut (<3 min); W1 Devpost writeup |
| **Wed Aug 12** | **SUBMIT** | Repo tagged, About sidebar license checked, links verified |
| Thu–Fri Aug 13–14 | Buffer | Untouched unless something broke |

**Position as of Aug 4:** ~2 days ahead on the code track, 0 days of progress on the two
items that actually decide the outcome (Arm numbers, trained prior). The slack exists
precisely to spend on those — do not spend it on more code.

**G1 (Aug 5):** prior route. Recommendation: small-scale per Call 1. *If training has not
started by end of Aug 6, drop to phantom-only and reframe — do not let this eat the Graviton or
video days.*

**G2 (Aug 7, after real numbers exist):** fused `selective_scan_2d`. Default is **no**. Only
reconsider if K1's measured Arm overhead is still >15% *and* Aug 8–9 came in clean, which is
unlikely.

---

## Part 4 — Integration plan, item by item

Every item states files, the change, its gate, and its exit criterion. Standing rules from
`CLAUDE.md` apply throughout: correctness gates speed; never loosen a tolerance; `torch.compile` is
the baseline that matters; publish unflattering rows.

---

### K1 — Route SS2D's four directions through the fused bidirectional kernel *(the centerpiece)*

**Why:** four directions = two traversal-order pairs over two distinct input sets (`rows`, `cols`).
The fused kernel computes each pair's shared Pass A once. Expected: ~1.67× on the scan portion
(75–79% of block time, per `ss2d_windows-i9.json`) plus deletion of all four `torch.flip` copies
from the remaining 21–25%. Uses `arm_scan_selective_scan_bidirectional_f32`, already shipped, already
golden-gated, ABI v5. **No new Rust.**

**Files:** `apps/mri_diffusion/backbone/mamba_ss2d.py`, `python/arm_scan/ss2d.py`,
`apps/mri_diffusion/backbone/torch_scan.py`.

**1. Introduce a pair-shaped seam.** `SS2DBlock` currently has `scan_fn(seq,…) -> out`. Add
`scan_pair_fn(seq, …) -> (fwd, bwd)`, both outputs indexed at `t` (the kernel's `reverse` writes
output at index `t` — traversal order changes, layout never does; see the `ScanInput::reverse`
doc comment in `lib.rs`).

- Reference implementation in `torch_scan.py`, defined as **exactly today's math** so parity is
  meaningful: `fwd = selective_scan_torch(seq, …)`;
  `bwd = selective_scan_torch(flip(seq), flip(delta), flip(B), flip(C), …).flip(-1)`.
- Arm implementation in `ss2d.py`: `bidirectional_scan(…, merge="none")`, which dispatches to the
  fused op and falls back to two calls for untied weights or when `last_state` is requested
  (neither applies here).

**2. Rewrite `SS2DBlock.forward`'s scan section:**

```python
rows = s.flatten(2)                      # (b, inner, H*W)  row-major
cols = s.transpose(2, 3).flatten(2)      # (b, inner, W*H)  col-major
seqs = torch.cat([rows, cols], dim=0)    # (2b, inner, L) — ONE projection pass
fwd, bwd = self.scan_pair_fn(seqs, ...)  # Pass A computed once per pair
r_f, c_f = fwd.chunk(2, dim=0)
r_b, c_b = bwd.chunk(2, dim=0)
merged = ((r_f + r_b).view(b, self.inner, h, w)
          + (c_f + c_b).view(b, self.inner, w, h).transpose(2, 3))
```

`x_proj`/`dt_proj` stay shared across directions (they are time-pointwise, so flipping commutes with
them — this is why the fusion is exact, and it is the same argument P0-1 already relied on).

**3. Three things that will bite, all of them checkable:**

- **`D` is applied per direction and summed.** Today each of the 4 outputs carries its own `D·u`, so
  a merged pair carries `2·D·u`. `bidirectional_scan` with a shared `D` does the same — see the
  "GOTCHA" section in `bidirectional.py`'s docstring. So the swap is numerically equivalent **only
  if** `D` keeps being applied in both directions. Do not "fix" this; it changes trained semantics.
  K2 catches it if you get it wrong.
- **Rayon rows per call drop from 4B to 2B** (384→192 at inner=96, b=1). Still ≫ core count on any
  Graviton, and each call now does twice the work per row. Confirm on hardware, don't assume.
- **NEON reverse is not bit-identical to flip-forward-flip** (~1e-7, because the 4-timestep vector
  body and the scalar tail evaluate softplus/SiLU differently and flipping moves timesteps across
  that boundary — documented in `lib.rs`). K2 uses a tolerance, not equality.

**Gate:** K2 below. **Exit:** `bench_ss2d.py` shows lower total block time at all four real shapes,
and Phase-C sampling parity still passes.

---

### K2 — Parity gate for the K1 refactor *(must land with K1, same commit)*

**File:** `apps/mri_diffusion/tests/test_ss2d_pair_parity.py` (new).

Assert the new forward equals the old 4-way-flip-stack forward, at ≥2 grid shapes including one
non-square and one odd-sized (e.g. 32×32, 24×40, 17×23), on **both** paths:

1. torch-reference pair vs. the old torch 4-stack — should be tight (~1e-6); this validates the
   *refactor*.
2. kernel pair vs. torch-reference pair — the standing fp32 tolerance; this validates the *kernel*.

Pin `torch.manual_seed`, run at `RAYON_NUM_THREADS ∈ {1,2,8}` for the kernel path. Keep the old
4-stack code as a `_forward_legacy` method used only by this test — it is the oracle, exactly like
`scalar.rs` is for the kernel. **Exit:** green, and wired into CI by T3.

---

### K3 — 2D goldens *(the gate `TOPOLOGY_IMPLEMENTATION_PLAN.md` §3.3 requires)*

**Files:** `tests/gen_golden_2d.py`, `tests/golden/grid_*.npz`, `tests/verify_golden_2d.py`.

Ground truth = the vendored f64 reference fed the four permuted views exactly as §3.1 defines,
compared **per direction before merge** so kernel bugs are isolated from merge bugs. Cases: square,
non-square, `H`/`W` not multiples of 4, and one grid at a real aspect ratio (384×320 shape family,
small channel count to keep the file size sane). Record each case's `f32_max_abs_err` floor in the
manifest, same format as the 1D goldens. Add the independent numpy re-derivation.

**Why it still matters even though P1-7 is cut:** it makes the SS2D correctness claim mechanical
rather than "the app test passed on my laptop," and it is the artifact that lets a future fused
kernel land safely. **Exit:** both implementations agree in f64; floors recorded; replayed through
the real C ABI in `check_ffi.py`.

---

### T1 — Documentation truth-pass *(2 hours, do it first — everything else references it)*

- `CLAUDE.md` "Where things stand" → rewrite to today, and point to this doc for sequencing.
- `ROADMAP.md` §4 and `SS2D_REPOSITIONING_PLAN.md` §7 → replace both tables with a one-line pointer
  here. Do not maintain three sequencing tables.
- `MAMBA3_KERNEL_PLAN.md` / `RESEARCH_TRIAGE_MAMBA2_2D.md` → add a banner: **post-submission,
  `feature/mamba3`, not part of the Aug 14 entry.** (Keep them. They are strong writeup material as
  "where this goes next" — just not on the critical path.)
- README `## Status` → rewrite once, honestly, after G1.

### T2 — De-hardcode the app tests *(1–2 hours; blocks T3)*

Replace `REF = Path(r"C:\Users\Adity\...")` in all three test files with
`os.environ.get("ADM_REF")` → skip-with-message when unset. Then go further: implement the EDM
preconditioning and loss **from the paper's equations** in a new `apps/mri_diffusion/edm_min.py`
(`c_skip`/`c_out`/`c_in`/`c_noise` and the EDM loss weighting — the exact formulas are already
transcribed in `MRI_DIFFUSION_IMPLEMENTATION_PLAN.md` §2), so the phantom track has **no dependency
on the CSI/EDM tree at all**. This simultaneously fixes reproducibility and the H4 license exposure.
Keep the CSI path available behind `ADM_REF` for the real-data work.

**Exit:** `python apps/mri_diffusion/tests/test_phase_d_partial.py` runs on a clean checkout with no
env vars and no external repo.

### T3 — App tests into CI *(1 hour)*

Add a `mri-app` job to `.github/workflows/ci.yml` on `ubuntu-24.04-arm`: build the FFI, install
torch, run backbone bring-up + K2 parity + Phase-D phantom (small sizes, time-boxed). Extend the
`Makefile`: `validate` gains the phantom end-to-end so the README's 5-minute judge path exercises the
SS2D and diffusion surface the pitch is about — not just the 1D kernel.

---

### D1 — The demo script *(half a day; build it BEFORE the Graviton session)*

**File:** `apps/mri_diffusion/demo.py`.

Produces a single PNG/GIF: **ground truth | zero-filled (R=4) | our reconstruction**, with PSNR/SSIM
annotated and a wall-clock/per-NFE timer, plus a printed line naming instance type, thread count, and
torch version. Runs on the phantom track with no credentials, and on real data when `--data` is
given. This is the video's only visual, and it must exist before you are paying for hardware.

**Exit:** produces the image in one command on a laptop.

---

### Graviton session 1 *(Thu Aug 6, ~3 hours, scripted, terminate after)*

`c8g.16xlarge` for the core-scaling curve (you cannot show scaling on 4 cores). Follow
`bench/ARM_BASELINE.md` exactly — it is already written; do not improvise on the clock. Capture, in
order, stopping if correctness fails:

1. `cargo test --release -- --nocapture` — confirm `backend Auto resolves to NEON`. **If this is not
   green, stop.**
2. `cargo bench` ladder: `scalar_seq → neon_seq → neon_par` (the ablation the README promises).
3. `bench_op.py` vs eager **and** `torch.compile`, JSON-tagged.
4. `bench_e2e.py` mamba-130m, tokens-identical confirmed (generality row).
5. **Core-scaling sweep** 1→64 threads → efficiency curve. *The Cloud-track chart.*
6. **`bench_ss2d.py` at the real 384×320 / 192×160 shapes** — post-K1, plus a `torch.compile`
   baseline row at the mini grid (H2's missing baseline).
7. **Per-NFE latency** for `MambaSS2DNet` at the demo resolution → $/reconstruction using published
   on-demand pricing.
8. `perf record` / `perf stat` → IPC + stall classification for the strengths/weaknesses section.

Commit every JSON (they are gitignored by default — add deliberately, per `ARM_BASELINE.md` §9).

### R1 / R2 — Results and README *(Fri Aug 7)*

`render_results.py` regenerates `RESULTS.md`; then rewrite the README's "Results so far" table from
**real Graviton rows**, naming instance type, core count, and torch version on every one. Delete or
demote the provisional CI rows. Write the strengths/weaknesses paragraph from the profile —
`ARM_BASELINE.md` §8 is a checklist for this, and stating the weaknesses with the profile that proves
them is worth more to Arm-engineer judges than hiding them.

---

### A1 — The small prior *(Sat Aug 8, bounded to one GPU session)*

Per Call 1: `train_prior.py` (new, ~150 lines) — EDM loss from `edm_min.py`, `MambaSS2DNet` at
128×128, torch reference scan (the kernel registers no autograd; **say so in the README**), fixed
step budget, checkpoint every N steps so a partial run is still usable. Data: synthetic phantoms +
IXI if it downloads cleanly. **Do not block on fastMRI.**

**Hard stop:** if it is not training by end of Aug 6, cut to phantom-only and reframe. Budget the
spend explicitly (this is the `PROJECT_CONCEPT.md` amendment that has been open since Jul 17).

### A2 — Quality table *(Sun Aug 9)*

R ∈ {2,4,8}: PSNR/SSIM/NMSE for zero-filled vs. reconstruction, **and** kernel-path vs.
reference-path parity at identical output quality. The parity column is the one that matters for the
kernel claim; the absolute quality column is the honest, scoped demo result.

### V1 / W1 — Video and writeup *(Aug 10–11)*

<3 min, shot on Graviton, no copyrighted music: the problem (CUDA-only op, CPU falls back), `make
validate` running green on Arm, the core-scaling curve, the side-by-side recon with the timer, the
prior-art table and the three precise claims. Devpost writeup mirrors the README. **Submit Aug 12.**

---

## Part 5 — Explicitly cut, and how to say so

Cutting is a result when it is measured and stated. Each of these gets a sentence in `RESULTS.md`:

| Cut | Honest framing |
|---|---|
| Fused `selective_scan_2d` (P1-7) | "We measured flip/permute overhead at 21–25% of SS2D block time at production grids, which clears our own 15% bar; we instead captured the larger share by routing the four directions through the fused bidirectional kernel (Pass A shared, ~1.67× measured), and name the fully fused 2D kernel as the identified next lever." |
| Cache-block over L (P1-5), tile transpose (P1-6) | Named next levers with the roofline reasoning already recorded in `IMPROVEMENT_IDEAS.md`. |
| Route A distillation | "No public Mamba-backbone EDM MRI checkpoint exists; we trained a small prior rather than claim a quality result we could not support in the time available." |
| SVE2 FEXPA (P2-9) | Already scoped as a stretch. One line. |
| Mamba-3 / SSD substrate | One paragraph of "where this goes next," pointing at the existing plan doc. It is genuinely good work — it is just not this deadline's work. |
| fastMRI real-data recon | Phantom + small-scale track, credential-free by design — which is also the DX argument. |

---

## Part 6 — Risks specific to this plan

| Risk | Trigger | Mitigation |
|---|---|---|
| K1 is slower on Arm than the 4-way stack | Fewer rayon rows per call (2B vs 4B) starves a 64-core box | Measured in Graviton session 1 step 6; keep `_forward_legacy` behind a flag and publish whichever wins, with the reason |
| Graviton session overruns / instance unavailable | Capacity or a red correctness gate | Session is scripted; Ampere A1 or a 4-core arm64 runner is the fallback surface, labelled provisional per `BASELINE_TEST_PLAN.md` |
| Training produces a visibly bad reconstruction | Too few steps at 128×128 | Quality is explicitly scoped as "pipeline works + kernel parity," not SOTA; the parity column carries the kernel claim regardless |
| K1 breaks trained-semantics via the `D` gotcha | `D` applied once instead of twice after the swap | K2 parity test against `_forward_legacy` catches it before anything is benchmarked |
| Doc churn eats a day | 22 files, 4 sequencing tables | T1 is time-boxed to 2 hours: replace tables with pointers, don't rewrite prose |
| Everything slips | It always does | Aug 13–14 is buffer, and the MVP (kernel + Graviton numbers + parity + docs + video) is a complete submission without A1/A2 |

**The one-sentence fallback:** if Aug 8 arrives and only the kernel track is done, ship the 1D + SS2D
kernel with real Graviton numbers, the phantom demo, and an honest "the trained prior is future work"
— that is still a coherent, defensible entry. What is *not* recoverable is arriving at Aug 12 with no
Arm numbers.
