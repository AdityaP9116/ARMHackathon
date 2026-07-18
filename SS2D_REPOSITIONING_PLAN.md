# SS2D_REPOSITIONING_PLAN — double down on vision/diffusion Mamba

**Written Jul 17, 2026.** Follows the prior-art verification recorded in
[`PROJECT_CONCEPT.md`](PROJECT_CONCEPT.md) ("Prior-art verification"). This doc is two things:
(1) the repo-wide list of what to change now that the framing is SS2D/diffusion-first, and
(2) the kernel optimization plan **specialized to the actual workload** the diffusion app runs —
grounded in the real shapes from `apps/mri_diffusion/backbone/mamba_ss2d.py` and
`PHASE_A_FINDINGS.md`, not generic shapes. It prioritizes items already scoped in
[`TOPOLOGY_IMPLEMENTATION_PLAN.md`](TOPOLOGY_IMPLEMENTATION_PLAN.md) and
[`IMPROVEMENT_IDEAS.md`](IMPROVEMENT_IDEAS.md) rather than re-deriving them — read those for the
*how*, this for the *which and in what order*.

---

## 1. Why the reframe (one paragraph, for the record)

The novelty check (Jul 17) found real prior art for 1D language Mamba on CPU/Arm — llama.cpp's
`ssm_scan`, BitMamba-2 on NEON, mamba.rs/Candle in Rust — so "fast Mamba on Arm" is contested
ground. What has **zero** prior art: a SIMD `selective_scan` callable from PyTorch as a drop-in,
a fast CPU SS2D cross-scan (VMamba is CUDA-only; DiM/ZigMa/DiffuSSM have no CPU path), and
diffusion-prior MRI recon on CPU. The repo already bet on exactly this surface (topology plan §3,
the MRI diffusion app, Phase A = GO). The reframe just makes the bet explicit everywhere: **SS2D
inside a diffusion sampling loop is the headline; 1D language rows are generality evidence.**

---

## 2. Done in this change

- **`README.md`** — rewritten: SS2D/diffusion-first pitch, a prior-art table (judges see we did
  the search), three precise "to the best of our knowledge" claims, provisional CI numbers with
  their caveats, MRI-diffusion component rows, numerics disclosure.
- **`PROJECT_CONCEPT.md`** — amended, dated: primary application = SS2D-Mamba diffusion MRI
  (MambaRecon demoted to fallback), Training reopened (Route A distillation recommended),
  prior-art section added with the claims policy ("never claim first-Mamba-on-Arm").

## 3. Still to change, doc by doc (each ≤ 30 min, do opportunistically)

| File | Change |
|---|---|
| `CLAUDE.md` | "Where things stand" is frozen at Jul 13 — now wrong twice over (application is decided; Phase A–B code exists). Rewrite that section; add the claims policy to "Rules of engagement" so future sessions can't reintroduce an over-claim. |
| `ROADMAP.md` | Week-by-week plan predates the diffusion pivot. Replace remaining weeks with §7's sequencing; keep the compute strategy section as-is. |
| `APPLICATIONS.md` | Header says "nothing locked yet." Add a dated banner: decided — MRI-diffusion took the SS2D slot; pointer to the decision log. Don't rewrite the brainstorm (it's honest history). |
| `INTEGRATION_PLAN.md` | Phase 7 floats an audio/ECG/RF pivot — the disagreement `CLAUDE.md` calls a bug. Mark Phase 7 resolved-by `MRI_DIFFUSION_IMPLEMENTATION_PLAN.md`. |
| `IMPROVEMENT_IDEAS.md` | §10 priority shortlist is ordered for the 1D language workload. Add a note re-pointing to §5 below for the SS2D-first ordering (don't maintain two orderings). |
| `python/README.md`, `bench/README.md`, `tests/README.md` | Mention `ss2d.py`, `bench_ss2d.py`, and the 2D goldens as each lands (not before — READMEs promising things that don't exist is an open wound the main README just had treated). |
| **`Makefile`** | **Doesn't exist; README promises `make validate`.** Create: `validate` = cargo tests + `check_ffi.py` + `verify_golden.py` + `bench_op.py --quick`. This is a judge-facing deliverable, not a doc fix. |

## 4. The workload, characterized (what the kernel must now be fast at)

From `mamba_ss2d.py` (locked recipe) and Phase A: grid **384×320** at level 1 (row-flatten →
**L = 122,880** per direction), **192×160** at level 2 (L = 30,720); channels inner = 96 (enc/dec)
and 192 (mid); `d_state=16` (the NEON fast path); 6 SS2D blocks per denoiser call; 4 directions
per block; Heun sampling at 18–35 steps ≈ 35–69 NFE. So one reconstruction ≈
**6 × 4 × 69 ≈ 1,650 kernel invocations**, each `B=1, D≈96–192, L≈30k–123k`.

What that profile changes vs. the 1D language workload the kernel was tuned on:

1. **L is ~60× longer** than the L=2048 the ladder was measured at. The one-time B/C transpose
   buffer is `L × state_padded × 4B ≈ 7.9 MB × 2` at level 1 — this **blows past L2** on every
   Arm core we target. Cache behavior, not exp throughput, may now be the binding constraint.
2. **Allocation count explodes.** The `bt`/`ct` workspace is reallocated+zeroed per call ×
   1,650 calls per reconstruction (IMPROVEMENT_IDEAS §4.1's worst case, realized).
3. **Parallelism is thin per call** (B=1 → only 96–192 rayon rows) but **wide across directions**
   (×4) and across sampler seeds (Tamir's code batches seeds). Getting the 4 directions and the
   seed batch into one call is worth more than any micro-optimization.
4. **The torch side flips and transposes constantly.** `SS2DBlock.forward` does 2 `flip`s of a
   `(1, 96, 122880)` tensor + 2 output flips + `transpose` per block per call — copies the fused
   kernel exists to delete.
5. The exp phase (85% of single-thread time at L=2048) still matters, but the ladder already took
   −15%; the remaining swing there (SVE2 FEXPA) is Graviton-specific and stays a stretch.

## 5. Kernel optimization plan, ordered for this workload

Everything obeys the standing gates: golden `< 1e-4` near the recorded floor, NEON↔scalar parity,
rayon bit-identity at `RAYON_NUM_THREADS ∈ {1,2,8}`, C-ABI replay. New 2D goldens per topology
plan §3.3 (including non-square and non-multiple-of-4 grids) land **before** the Rust they gate.

### P0 — this week, Python only, no new Rust  ✅ DONE (Jul 17)

> **Measured:** P0-1 batched 4-direction call landed (kernel calls/forward
> 12→3; per-NFE 542→23 ms on x86; Phase-C parity re-gated PASS). P0-2
> `bench_ss2d.py` at real shapes: overhead 21–25% on every real shape →
> **fused `selective_scan_2d` (P1-7) JUSTIFIED** by the 15% rule
> (`bench/results/ss2d_windows-i9.json`).

1. **Stack the 4 directions into one batched scan call** (`SS2DBlock.forward` + `ss2d.py`).
   Topology plan §3.1 already prescribes it; the backbone currently makes 4 separate
   `_scan_dir` calls. Build the 4 views, stack to `4B` batch, one `x_proj`/`dt_proj` (they're
   shared across directions), one kernel call → 4× the rayon rows (384–768) and ¼ the FFI
   crossings. This is the single cheapest big win and also what the fused kernel's contract wants.
2. **`bench_ss2d.py` — measure the unfused path at the real shapes** (384×320/96ch,
   192×160/192ch, seed-batch 1 and 4) vs. torch reference and `torch.compile`. This answers the
   topology plan's gating question — how much of SS2D time is flip/permute/copy traffic vs. scan —
   with the workload's own numbers, and decides how much of P1 is justified. Also gives the
   per-NFE number the honest-framing rules require.

> **External-research triage (Jul 18):** a literature sweep (VSSD, 2DMamba, EfficientViM,
> COREY, FairyFuse — verified) converged on this same ordering; see
> [`RESEARCH_TRIAGE_MAMBA2_2D.md`](RESEARCH_TRIAGE_MAMBA2_2D.md). Deltas folded in: P1-5 gains a
> static `CHUNK` sweep at SS2D shapes + a preceding roofline run; P1-6 uses a `vld4q_f32`
> de-interleave 4×4 transpose (fewer shuffles than vtrn/vzip); P1-7 cites 2DMamba's SRAM tiling
> as GPU-side precedent. VSSD and SSD-on-CPU are rejected-with-reasons there (they'd dissolve
> the sequential-recurrence moat) — writeup material, not work items.

### P1 — Rust, in dependency order

> **Status (Jul 17):** P1-3 ✅ (thread-local B/C plane cache, `3177ded`);
> P1-4 ✅ (reverse flag landed via bidirectional PR#8, ABI v5 — `ss2d.py`
> still to adopt it to delete the per-block flips). Next: P1-5, P1-6.

3. **Workspace reuse** (IMPROVEMENT_IDEAS §4.1): thread-local scratch arena, keyed by size,
   no per-call alloc/zero. Was a "nice-to-have" at 1 call per forward; at 1,650 calls per
   reconstruction it's first. Small, self-contained, no numerics impact (gates must still pass).
4. **`reverse` flag** (topology plan §2.2, ~half a day): deletes the 4 per-block `torch.flip`
   copies, and is a prerequisite the fused 2D path needs for its backward directions anyway.
   ABI bump per the existing versioning contract.
5. **Cache-block over L** (IMPROVEMENT_IDEAS §4.2): stream B/C transposition per chunk instead of
   materializing the full 7.9 MB plane. At L=123k this is the difference between L2-resident and
   memory-bound. Coordinate with the `reverse` chunk-loop surgery (topology plan §2.2's overlap
   note) — restructure the loop once, not twice.
6. **Vectorized 4×4 tile transpose** (IMPROVEMENT_IDEAS §3.6 = topology plan's `transpose.rs`):
   shared primitive — speeds the existing B/C transpose now, becomes the column-direction engine
   of the fused kernel next.
7. **Fused `selective_scan_2d`** (topology plan §3.2, the week-sized item): transpose-then-reuse-
   row-scan, threading over `batch × channel × direction`, new C-ABI entry point. **Go/no-go on
   P0-2's measurement**, per the plan's own rule: if flips+permutes are <15% of SS2D time at the
   real grid, ship the stacked-batch path for the demo and publish that finding honestly in
   `RESULTS.md`; the transpose kernel (item 6) is justified either way.

### P2 — Graviton-specific, on the rented `c8g` only

8. **Seed-batch core-scaling curve** (1→64 cores, batch of sampler seeds × 4 directions):
   this is the "CPUs scale where the GPU is hindered by the sequential scan" chart for the video.
9. **SVE2 FEXPA exp** (IMPROVEMENT_IDEAS §3.2, nightly Rust): the remaining transcendental swing;
   stretch, only if the schedule holds after §7's week 3.

### Explicitly not doing (measure-and-reject material for the writeup)

L-dimension Blelloch scan (§4.3 — the chunked two-pass already recovers most ILP without the
3-phase pass), Mamba-2 SSD duality on CPU (§7.1 — documented rejection), fp16 plane storage
(§5.1 — not until fp32 numbers are locked), any new scan pattern beyond plain 4-direction SS2D
(ZigMa zigzag etc. — the kernel contract shouldn't chase per-paper scan orders before Aug 14).

## 6. App-side items the kernel plan depends on

- **Route A/B decision + GPU budget** (`MRI_DIFFUSION_IMPLEMENTATION_PLAN.md` §8/§14) — the only
  open decision that can starve everything downstream. Decide this week; distillation wall-clock
  is the critical path to a quality-parity table.
- The kernel registers **no autograd** — training/distillation runs the torch reference path on
  GPU; the kernel is inference-only. Say so in the README when the training section lands.
- **Phantom track** stays mandatory: judge-runnable end-to-end (sampler + backbone + kernel) with
  no credentials, mirroring `make validate`.

## 7. Sequencing to Aug 14 (submit Aug 12–13)

| Week | Kernel | App / results |
|---|---|---|
| Jul 20 | P0-1, P0-2, P1-3 | Route A/B decision; start distillation/training; Makefile `validate` |
| Jul 27 | P1-4, P1-5, P1-6 | Prior trained/distilled; Phase C/D parity on arm64 CI; 2D goldens |
| Aug 3 | P1-7 (if measurement justifies) | **Graviton session 1** (`c8g`, scripted, terminated after): headline ladder, per-NFE, $/recon, core-scaling |
| Aug 10 | freeze; P2-9 only if green | **Graviton session 2**: demo video; `RESULTS.md` final; Devpost writeup; **submit Aug 12–13** |

Standing risk: if distillation slips past Jul 31, fall back to MambaRecon (decision log's fallback
row) — the kernel work above is identical either way, which is the point of having ordered it
SS2D-first rather than app-first.
