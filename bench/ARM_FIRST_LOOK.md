# ARM_FIRST_LOOK — the first Arm measurements this project has had

**Aug 4, 2026.** Source: the `Profile kernel` workflow, run
[30915946873](https://github.com/AdityaP9116/ARMHackathon/actions/runs/30915946873),
commit `e13e03f`, artifact `kernel-profile`.

> **Provisional, not headline.** This is a **4-core shared GitHub Actions runner**, which
> `BASELINE_TEST_PLAN.md` classifies as provisional by rule. The dedicated-instance numbers
> come from [`GRAVITON_SESSION.md`](GRAVITON_SESSION.md) and supersede everything here.
> Phase *proportions* and scaling *ratios* are far more robust to a noisy shared host than
> absolute timings, so those are what this document leans on.

Recorded because until now **every** number in the repo was x86 or the scalar fallback, and
these already settle two open decisions.

---

## The host

**Neoverse-N2**, 4 cores, 256 KiB L1d, 4 MiB L2, 128 MiB L3.

```
fp asimd ... sve asimdfhm ... sve2 svebf16 svei8mm i8mm bf16
```

**SVE2 is present**, along with `bf16` and `i8mm`. Worth noting because "no SVE2 yet" is
listed as a known weakness and SVE2 `FEXPA` (P2-9 / `IMPROVEMENT_IDEAS.md` §3.2) has been a
stretch item — the free CI hardware can actually exercise it.

---

## The ablation ladder (`cargo bench`, criterion)

| Shape | scalar_seq | neon_seq | neon_par | NEON gain | threading gain | total |
|---|---|---|---|---|---|---|
| `small_d64_l128` | 1.6955 ms | 415.08 µs | 129.61 µs | **4.08×** | 3.20× | **13.1×** |
| `mamba130m_layer_l512` | 82.178 ms | 20.407 ms | 5.1194 ms | **4.03×** | **3.99×** | **16.1×** |

Read against `ARM_BASELINE.md` §3:

- **NEON gain 4.03–4.08×** sits at the top of the expected 3–4× band. The SIMD path and the
  hand-vectorized `exp` are doing their job on real Arm — not merely on paper.
- **Threading 3.99× on 4 cores = 99.7% scaling efficiency** at the mamba shape. The bar for
  "genuine strength" is 0.7. At 4 cores there is no bandwidth wall at all.
  **The open question is where that curve bends on 64 cores** — which is precisely what the
  Graviton session's 1→64 sweep exists to answer, and now looks worth the time.

---

## Where single-core time actually goes

`profile_phases`, single-threaded, stable across every shape measured:

| Phase | L=128 | L=512 | L=2048 | B=8, L=1024 |
|---|---|---|---|---|
| **exp** | 53.6% | 52.7% | **53.7%** | **53.7%** |
| discretize | 20.3% | 20.5% | 20.3% | 20.3% |
| recurrence | 12.5% | 12.6% | 12.8% | 12.8% |
| epilogue | 8.4% | 8.2% | 6.8% | 6.8% |
| projection | 5.2% | 6.0% | 6.3% | 6.3% |
| **transpose** | **0.0%** | **0.0%** | **0.1%** | **0.1%** |

### Three things this settles

**1. `exp` is the hot spot, at ~54%, and it does not move.** That makes SVE2 `FEXPA`
(§3.2) the highest-value remaining kernel optimization, not a speculative one — and the CI
hardware above supports it. Everything else is a rounding error by comparison.

**2. The transpose costs ~0.1%.** This retires two items:
- `IMPROVEMENT_IDEAS.md` §3.6 — vectorizing the scalar B/C plane transpose into a 4×4 NEON
  tile transpose. It is optimizing 0.1% of runtime.
- It independently supports cutting **P1-6** (the tile-transpose micro-kernel) and weakens
  the case for **P1-7**, whose column-direction strategy was built around that same
  transpose primitive. The SS2D overhead measurement already put P1-7 under its 15% bar;
  this says the primitive it depends on would not have bought much either.

**3. Phase shares are flat from L=128 to L=2048.** No cache cliff appears in that range, so
§4.2 cache-blocking over L (**P1-5**) is not urgent at the sequence lengths measured. Worth
re-checking at the diffusion workload's L=122,880, which is 60× longer than anything here.

---

## Tooling defect found

`bench/profile/dump_asm.sh` produced **nothing** — `cargo asm` exited with
"Multiple targets found", because `arm-scan-core` also builds an example, two test targets
and a bench. Fixed by passing `--lib`. The asm audit (`IMPROVEMENT_IDEAS.md` §3.7 — checking
for exp-constant re-materialization and `h0..h3` register spills) has therefore never
actually run; it will on the next profile.

---

## What this does *not* tell us

- Nothing about **SS2D on Arm** — this workflow is Rust-only and predates the pair path
  meeting NEON. The `mri-app` CI job covers that.
- Nothing **vs `torch.compile`** — this is the internal ablation ladder only.
- Nothing about **scaling past 4 cores**, which is the actual Cloud-track argument.
- No hardware counters (IPC, stalls, cache misses) — `perf_ampere.sh` needs root.
