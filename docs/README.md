# docs/ — the working record

The repository root holds only what a reader needs to evaluate the project. Everything else
lives here, kept rather than deleted: how a decision was reached is itself evidence, and
several of these documents record measurements that superseded an earlier belief.

## `archive/` — history, superseded plans, and measurement logs

| Document | What it is | Status |
|---|---|---|
| `INTEGRATION_PLAN.md` | The original build plan, Phases 0–6 | **Landed.** Goldens → scalar → NEON → rayon → C ABI → torch op → wheels → CI |
| `TOPOLOGY_IMPLEMENTATION_PLAN.md` | How 1D bidirectional and SS2D were to be built | §2 landed (fused bidirectional). §3.1 landed (SS2D via pairs). §3.2 (fused `selective_scan_2d`) **cut by measurement** |
| `BIDIRECTIONAL_LOG.md` | Build log for the fused bidirectional kernel | The exp-sharing result (1.58–1.75×) that SS2D is now built on |
| `BIDIRECTIONAL_SPEEDUP_IDEAS.md` | Design options considered for that kernel | Superseded by the log |
| `OPTIMIZATION_LOG.md` | Per-optimization measured attribution | Historical; `bench/results/` is now the live record |
| `BASELINE_TEST_PLAN.md` | Benchmark methodology — the three surfaces | Still the governing methodology; `bench/README.md` is the practical version |
| `BASELINE_REPORT.md` | First baseline write-up | Superseded by `bench/results/RESULTS.md` |
| `PROFILING.md`, `PROFILING_EXPLAINED.md` | How to profile, and how to read it | Still accurate; `bench/ARM_BASELINE.md` §7 is the current entry point |
| `IMPROVEMENT_IDEAS.md` | Kernel optimization backlog | Partly **retired by measurement** — §3.6 (vectorised transpose) targets 0.1% of runtime |
| `SS2D_REPOSITIONING_PLAN.md` | The Jul 17 pivot to SS2D | **Executed** |
| `MAMBA_DIFFUSION_MRI_PLAN.md` | MRI strategy | Superseded by `MRI_DIFFUSION_IMPLEMENTATION_PLAN.md` |
| `PHASE_D_DIAGNOSIS.md` | Why the reconstruction quality gate fails | **Still live** — read before touching the sampler |

## `roadmap/` — after the submission

| Document | What it is |
|---|---|
| `MAMBA3_KERNEL_PLAN.md` | Staged program for a Mamba-3 kernel: SSD substrate → M3 core → 2D |
| `MAMBA2_SSD_PLAN.md` | The SSD substrate that is ~90% of the above |
| `RESEARCH_TRIAGE_MAMBA2_2D.md` | External research survey, verified and triaged |

**One correction worth carrying:** these documents state that no public Mamba-3 checkpoints
exist. That was true when written (Jul 18) and is **no longer true** — `state-spaces` has
published SISO and MIMO checkpoints (arXiv 2603.15569). What remains true is that there is
**no CPU implementation of Mamba-3 anywhere**, which is what makes the plan interesting
rather than obsolete.
