# Documentation

This directory separates current project guidance from teaching material, future roadmaps, and
the historical record. The distinction matters: archived plans explain how decisions were made,
but they are not current instructions.

## Start here

| Document | Purpose |
|---|---|
| [`../README.md`](../README.md) | Project overview, claims, measured results, and architecture |
| [`../RUNNING_THE_KERNEL.md`](../RUNNING_THE_KERNEL.md) | Fresh-machine setup, build, validation, usage, benchmarking, and troubleshooting |
| [`project/STATUS.md`](project/STATUS.md) | Concise current capabilities, limitations, and next work |
| [`project/README.md`](project/README.md) | Index of the active design and implementation documents |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | Correctness, safety, benchmarking, and pull-request expectations |

## Directory map

| Directory | Audience | Contents |
|---|---|---|
| [`project/`](project/) | Contributors and reviewers | Current decisions, the three-path design, implementation sequence, and kernel workplan |
| [`learn/`](learn/) | Students and new contributors | A professional, first-principles path from SSMs through Mamba, kernels, NEON, correctness, and benchmarking |
| [`presentation/`](presentation/README.md) | Submission reviewers | Browser-based presentation deck, fact-checked narration, and recording notes |
| [`roadmap/`](roadmap/) | Future contributors | Work deliberately deferred beyond the current submission |
| [`archive/`](archive/) | Maintainers and auditors | Superseded plans, investigation logs, diagnoses, and historical session notes |

Operational benchmark instructions live beside the code in [`../bench/`](../bench/), while raw
dedicated-machine bundles are preserved in [`../bench/artifacts/`](../bench/artifacts/).

## Archive policy

Documents enter `archive/` when their execution plan is no longer current. They are retained
because measurements, failed approaches, and decision history are useful evidence. A document
in `archive/` may still contain accurate technical analysis, but it must not override the root
README, `project/STATUS.md`, or the current project documents.

Notable records include:

| Document | Why it remains useful |
|---|---|
| [`archive/HANDOFF_2026-08-11.md`](archive/HANDOFF_2026-08-11.md) | Full pre-submission session state before this repository cleanup |
| [`archive/INTEGRATION_PLAN.md`](archive/INTEGRATION_PLAN.md) | Original Phase 0–6 kernel integration plan; the implementation landed |
| [`archive/TOPOLOGY_IMPLEMENTATION_PLAN.md`](archive/TOPOLOGY_IMPLEMENTATION_PLAN.md) | Design history for fused bidirectional and SS2D topologies |
| [`archive/PHASE_D_DIAGNOSIS.md`](archive/PHASE_D_DIAGNOSIS.md) | Live technical diagnosis to read before modifying the MRI sampler |
| [`archive/SPIKE_FINDINGS.md`](archive/SPIKE_FINDINGS.md) | Feasibility investigation that found the relevant Mamba-3 prior art |
| [`archive/BASELINE_TEST_PLAN.md`](archive/BASELINE_TEST_PLAN.md) | Benchmark methodology behind the practical guidance in `bench/README.md` |

One historical correction should be carried when reading older files: public Mamba-3 SISO and
MIMO checkpoints now exist under `state-spaces`. The project does not claim to be the first
Mamba-3 CPU or Rust implementation; its claims are narrower and listed precisely in the root
README.
