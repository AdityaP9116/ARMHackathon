# PHASE_D_DIAGNOSIS — why the reconstruction gate fails, and the plan to fix it

**Written Aug 4, 2026.** Companion to [`SUBMISSION_ENDGAME_PLAN.md`](SUBMISSION_ENDGAME_PLAN.md).
Phase D (`apps/mri_diffusion/tests/test_phase_d_partial.py`) asserts the R=4 reconstruction
beats zero-filled by >1 dB. It does not — it is **2.75 dB worse** — and it has never passed.
This document is the diagnosis, measured rather than argued, and the plan that follows from it.

---

## 1. What is actually still open

Three things, and only three. Everything else in
[`SUBMISSION_ENDGAME_PLAN.md`](SUBMISSION_ENDGAME_PLAN.md) Part 1 is closed.

| # | Issue | Severity | Fixable from a laptop? |
|---|---|---|---|
| **A** | **No Arm hardware numbers.** Every figure in the repo is x86 or a shared 4-core CI runner. | Existential — it is an Arm contest | No. Needs a Graviton/Ampere session. |
| **B** | **No trained prior**, therefore no defensible reconstruction-quality result. | High | Partly — needs a GPU session, but the target is now quantified (§4). |
| **C** | **Phase D's quality gate fails**, and its test design conflates two independent claims. | High | Yes — §5 D-1…D-4. |

**B and C are the same root cause wearing two hats.** The kernel is not implicated in either.

---

## 2. The evidence

Three experiments, run today. Each is cheap and repeatable; §5 D-1 turns the first into a
permanent gate.

### 2.1 The sampler is provably correct

Replace the learned denoiser with an **oracle** that returns the true image. Then
`DC(x_true) = x_true` identically, so Heun must return the truth. Measured:

| Check | Result |
|---|---|
| `ifft(fft(x)) == x` | max_abs 2.4e-07 |
| `zero_filled(full mask) == x` | max_abs 2.4e-07 |
| `DC(truth) == truth` | max_abs 2.4e-07 |
| **Oracle denoiser → reconstruction** | **151.31 dB** (max_abs 3.6e-07), at both 10 and 18 steps |

So the FFT convention, the mask, the data-consistency projection and the Heun integrator are
all exact. **No sampler bug exists.** This also retires the hypothesis recorded earlier in
`SUBMISSION_ENDGAME_PLAN.md` — that hard DC at high σ was splicing clean k-space into a
noise-dominated iterate. It is not: DC is applied to `net(x, σ)`, the *denoised estimate*
x̂₀, which is the standard and correct scheme.

### 2.2 How accurate must the denoiser be?

Same oracle, with controlled error `ε` added to its output (nominal R=4, 10 steps,
zero-filled baseline 19.23 dB, bar 20.23 dB):

| ε (denoiser RMSE) | 0.00 | 0.01 | 0.02 | 0.05 | 0.10 | 0.20 | 0.40 |
|---|---|---|---|---|---|---|---|
| reconstruction | 151.3 dB | 48.99 | 42.97 | 35.01 | 28.99 | 22.97 | 16.95 |
| clears bar? | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |

Clean −20·log₁₀(ε) behaviour; the bar is crossed at **ε ≈ 0.28**. That is the accuracy
target a prior must hit, and it is the number §4 turns into an acceptance test.

### 2.3 Where the real prior actually fails

A 300-step, 147K-parameter prior (the size Phase D trains inline), measured against the exact
σ ladder `heun_posterior` walks:

| σ | 80.0 | 42.4 | 21.1 | 9.72 | 4.07 | 1.50 | 0.47 | 0.117 | 0.020 | 0.002 |
|---|---|---|---|---|---|---|---|---|---|---|
| RMSE | 0.525 | 0.515 | 0.508 | 0.476 | 0.394 | **0.289** | 0.143 | 0.034 | 0.009 | 0.002 |
| vs 0.28 bar | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ok | ok | ok | ok |

**Six of the ten steps are above the bar**, and they are the high-σ steps that fix global
structure. At σ=80 the RMSE (0.525) is roughly the data's own scale — the denoiser is
returning essentially nothing informative. The mechanism is plain: EDM trains with
`ln σ ~ N(−1.2, 1.2)`, which puts σ=21 at **3.4 SD** and σ=80 at **4.6 SD** out. The sampler
is being run far outside the support the prior was ever trained on.

### 2.4 …but shrinking σ_max does not rescue it

If §2.3 were the whole story, lowering `sigma_max` into the trained region would fix it. It
does not. Same prior, sweeping σ_max and step count:

| σ_max | 80 | 20 | 10 | 5 | 2 | 1 | *(zero-filled)* |
|---|---|---|---|---|---|---|---|
| R=2 recon | 12.85 | 13.45 | 13.99 | 14.92 | 16.86 | **17.99** | **18.76** |
| R=4 recon | 11.19 | 11.83 | 12.42 | 13.46 | 15.68 | **16.77** | **16.90** |

Monotone improvement as σ_max falls — consistent with §2.3 — but it **asymptotes to
zero-filled from below and never crosses it**. As σ_max→0 the sampler degenerates to
`DC(init)` ≈ zero-filled, so the limit is the baseline. Every increment of prior influence
costs PSNR: the prior is not merely weak, it is **anti-informative at every σ**.

Two levers ruled out cheaply along the way: **step count is not the issue** (10 vs 18 steps
differs by ≤0.5 dB everywhere), and **measurement-based initialisation is not the issue**
(`x = zero_filled + σ_max·n` vs `σ_max·n` differs by ≤0.12 dB — at σ_max=80 the init is
noise-dominated either way).

---

## 3. What is actually wrong — four distinct defects

Ranked by how much of the gap each explains.

### D-α — The evaluation task is close to unwinnable as configured *(the big one)*

`toy_batch` generates **smooth superposed Gaussian bumps**. Their energy is almost entirely
in low spatial frequencies — exactly the lines a centre-ACS Cartesian mask *keeps*. So
zero-filling is already near-optimal (18.76 dB at R=2), there is almost no high-frequency
content left for a prior to restore, and any hallucination in the unmeasured region is a
**net loss** under PSNR. A generative prior cannot demonstrate value on data whose missing
components carry almost no energy.

### D-β — The mask does not deliver the acceleration it claims

```python
m = (torch.rand(w) < 1.0 / R).float()
m[w//2 - acs//2 : w//2 + acs//2] = 1.0     # ACS added ON TOP
```

ACS lines are forced *after* the 1/R draw, so the true sampling fraction is
`1/R + (acs/w)·(1 − 1/R)`. Measured at 32×32, acs=6: **nominal R=4 → fraction 0.375
(effective R = 2.67)**; nominal R=2 → 0.688 (effective 1.45). Every R in the repo is
overstated, which both flatters zero-filled and misreports the headline.

### D-γ — Sampler σ range exceeds the prior's trained support

§2.3. σ_max=80 against `ln σ ~ N(−1.2, 1.2)` means the first ~6 of 10 steps run 2–4.6 SD
outside training. Worth ~5 dB (§2.4) — real, but not sufficient alone.

### D-δ — The gate conflates the kernel claim with the model claim

Phase D asserts kernel-vs-reference parity **after** the quality assertion, so a weak prior
prevents the parity check from ever running. The submission's claim is about the *kernel*;
the >1 dB bar measures the *model*. One failing should not mask the other.

---

## 4. The accuracy target, made concrete

**Superseded and corrected (Aug 4).** §2.2's figure of 0.28 was an *absolute* RMSE measured
on the smooth-bump data with the old, mislabelled mask. It does not transfer: a threshold in
absolute units is meaningless across image families with different amplitudes.

Re-derived on the corrected setup (phantom evaluation data, mask delivering true R), the
oracle sweep crosses the ">1 dB better than zero-filled" bar at a **constant multiple of the
data's own RMS**:

| setup | crossing point |
|---|---|
| phantom 32px, R=4 | 0.96 × RMS |
| phantom 32px, R=8 | 0.96 × RMS |
| phantom 64px, R=4 | 0.95 × RMS |
| phantom 64px, R=8 | 0.93 × RMS |
| bumps 64px, R=4 (old setup) | 1.00 × RMS |

So the target is scale-invariant: **NRMSE = RMSE / RMS(data) < 0.95 at every σ on the
ladder.** The old 0.28 was that same threshold for one dataset (bump RMS 0.262 →
0.28/0.262 ≈ 1.07).

Better still, the sweep is a clean −20·log₁₀ line, so denoiser accuracy **predicts**
reconstruction gain before any sampling is run:

> expected PSNR gain over zero-filled ≈ **−20 · log₁₀(NRMSE)**

NRMSE 0.5 → ~+6 dB, 0.25 → ~+12 dB, 0.125 → ~+18 dB. Clearing 0.95 buys only the *minimum*
passing result, so `tools/prior_report.py` also flags rungs above a **0.5 target**.

Validation: run on the failing 300-step prior, the tool reports a limiting NRMSE of 1.60 and
predicts "no better than −4.1 dB". The measured Phase D result was **−2.75 dB** — the right
magnitude and sign, obtained without sampling at all. That is what makes it usable as a
training stop condition (§5 D-5) rather than a post-hoc explanation.

---

## 5. Implementation plan

Ordered so that **everything that does not need a GPU lands first**, and so the submission's
kernel claim stops depending on model quality within the first hour.

### D-1 — Split the gate: pipeline vs. quality *(≈1 h, no GPU, do this first)*

The single highest-value change: it makes the kernel claim permanently testable.

**New `apps/mri_diffusion/tests/test_phase_d_pipeline.py`** — prior-independent, no training,
runs in seconds, goes in CI:

1. FFT/mask/DC identities (§2.1 rung 0) — each < 1e-5.
2. **Oracle-denoiser exactness**: reconstruction > 100 dB. This is the real regression gate on
   the sampler, and it would catch any future breakage in DC, the mask, or Heun.
3. **Sampler-level kernel parity**: run `heun_posterior` with a *fixed randomly-initialised*
   net (weights activated per `test_phase_c_parity.activate_output_layers`) on the arm_scan
   path and the torch-reference path; assert agreement at the standing tolerance and
   `kernel_calls > 0`. Quality-independent, so it always runs.
4. The ε-sweep of §2.2 at two points, asserting monotonicity — cheap insurance that the
   quality metric itself is wired up correctly.

**Rename the existing test to `test_phase_d_quality.py`**, and have it *require* a real prior:
`--checkpoint` (or `PRIOR_CKPT`), skipping with a clear message when absent instead of
training a doomed 200-step prior inline. Keep the >1 dB assertion **exactly as it is** — it is
the app's quality claim and must not be softened.

*Exit:* `test_phase_d_pipeline.py` green in CI on arm64; `test_phase_d_quality.py` skips
cleanly with no checkpoint.

### D-2 — Fix the mask's acceleration *(≈30 min, no GPU)*

`cartesian_mask` should target a **total** fraction of `1/R`, ACS included:

```python
n_total = max(acs, round(w / R))
n_random = n_total - acs                     # ACS is part of the budget
# draw n_random columns uniformly from the non-ACS set, without replacement
```

Add an assertion to D-1 that `|mask.mean() − 1/R| < 1/w` for R ∈ {2,4,8}. Then **re-measure
every R-labelled number in the repo** — zero-filled baselines will drop, which makes the bar
*easier*, and makes the headline honest. Note in `RESULTS.md` that earlier R labels were
optimistic; do not quietly restate them.

### D-3 — Evaluate on data with high-frequency content *(≈1 h, no GPU)*

Move `shepp_logan` / `phantom_batch` out of `demo.py` into
`apps/mri_diffusion/data.py`, and use it for Phase D and the demo. Sharp ellipse
boundaries put real energy in high frequencies, so undersampling actually destroys
information a prior can restore — the regime where a diffusion prior is *supposed* to win.
Keep `toy_batch` for the bring-up test, where smoothness is fine (it only checks that
denoising works).

Report the zero-filled baseline for both datasets side by side; the gap between them *is* the
argument for why the phantom is the right evaluation set.

### D-4 — Make σ_max a property of the prior, not a constant *(≈1 h, no GPU)*

Store the training σ distribution on the model at construction
(`net.sigma_max_trained = exp(P_mean + 3·P_std) ≈ 11` for the current defaults) and have
`heun_posterior` default `sigma_max` to it rather than to 80. Two coupled knobs, one source
of truth:

- either lower the sampler to the prior's support (worth ~5 dB today, §2.4), or
- raise the training distribution to cover the sampler (`P_mean=-0.4, P_std=1.6` puts σ=80 at
  ~2.9 SD instead of 4.6) — preferable once there is GPU budget, because it keeps EDM's
  standard σ_max=80 and its well-tested schedule.

Pick **one** and make the sampler and the loss agree; the present mismatch is the bug.

### D-5 — Train a real prior, with a measured acceptance test *(GPU, ≈2–6 h)*

This is the actual fix for D-α and item **B**, and it is the only step needing hardware.

1. **`tools/prior_report.py`** (no GPU, ≈30 min): loads a checkpoint, prints the §2.3 σ-vs-RMSE
   table with a pass/fail against the 0.28 bar, plus the ε-curve position. Run it on every
   candidate; **stop training when it passes**, not at a fixed step count.
2. **`apps/mri_diffusion/train_prior.py`**: EDM loss from `edm_min`, phantom data (D-3),
   checkpoint every N steps, resumable. Scale that actually has a chance:
   `model_channels=64–128`, `num_blocks_per_level=2`, 128×128, **5k–20k steps**. The current
   147K-parameter/300-step configuration is ~2 orders of magnitude short.
3. Train on **one rented GPU** (the ~$3 spot instance in the endgame plan's Call 1). Note
   training must run on the torch reference scan — the kernel has no autograd — so the
   `unbind` fix (backward 147 s → 6.7 s) is what makes this affordable at all.
4. Acceptance: `prior_report.py` green at every σ, *then* run `test_phase_d_quality.py`.

### D-6 — Re-run and report *(≈30 min)*

With D-2 + D-3 + D-4 + D-5: re-run the quality gate at R ∈ {2,4,8}, regenerate the demo image
at 128×128 with `--checkpoint`, and fill the PSNR/SSIM/NMSE table. If it clears the bar, that
is the app result. If it does not, §6.

---

## 6. If D-5 cannot happen before Aug 14

Ship the honest version — it is still a coherent submission, because **the kernel claim never
depended on the prior**:

- `test_phase_d_pipeline.py` (D-1) proves the sampler is exact and the kernel is in the loop
  with verified parity. That is the contest-relevant claim.
- Report the quality result as **negative, with this diagnosis attached**: the σ-vs-RMSE
  table, the oracle bound, and the ε-curve showing exactly how accurate a prior must be. A
  measured negative result with a quantified target is a far better look than a silent skip,
  and it is exactly the "benchmark honestly" rule in `CLAUDE.md`.
- Frame the app as *"the pipeline runs end-to-end on Arm CPU through our kernel at N s/NFE,
  with kernel-vs-reference parity at 5e-6; reconstruction quality is bounded by prior training
  budget, which we quantify."*

**Do not** close the gap by relaxing the >1 dB assertion, by reporting the effective-R numbers
as if they were nominal, or by picking the σ_max that flatters the result. All three are
available and all three are disqualifying.

---

## 7. Suggested order

| Order | Item | GPU? | Effort | Unblocks |
|---|---|---|---|---|
| 1 | D-1 split the gate | no | 1 h | kernel claim stops depending on the prior; CI-able |
| 2 | D-2 mask acceleration | no | 30 m | honest R labels everywhere |
| 3 | D-3 phantom evaluation data | no | 1 h | makes the task winnable at all |
| 4 | D-4 σ_max ↔ training support | no | 1 h | ~5 dB, and removes a real inconsistency |
| 5 | D-5.1 `prior_report.py` | no | 30 m | turns "train more" into a stop condition |
| 6 | D-5.2–4 train the prior | **yes** | 2–6 h | the actual quality result |
| 7 | D-6 re-run + report | no | 30 m | the app table |

Items 1–5 are ~4 hours on a laptop and are worth doing **regardless of whether D-5 happens** —
they convert an unexplained red gate into a green pipeline gate plus a quantified, honestly
reported open question.
