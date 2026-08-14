# Project documentation

This directory contains the current design record for Arm Scan. These documents explain
what the project is building, why its scope was chosen, how the three Mamba-3 paths fit
together, and what implementation work remains.

For a first visit, use this order:

1. [`STATUS.md`](STATUS.md) — current capabilities, measured results, limitations, and next work.
2. [`PROJECT_CONCEPT.md`](PROJECT_CONCEPT.md) — decisions, rejected alternatives, and rationale.
3. [`THREE_PATHS_INTEGRATION.md`](THREE_PATHS_INTEGRATION.md) — the 1D SISO, MIMO, and 2D demonstrations.
4. [`MAMBA3_IMPLEMENTATION_PLAN.md`](MAMBA3_IMPLEMENTATION_PLAN.md) — staged implementation and claims policy.
5. [`MAMBA3_KERNEL_WORKPLAN.md`](MAMBA3_KERNEL_WORKPLAN.md) — file-level kernel engineering plan.

The root [`README.md`](../../README.md) is the judge-facing project overview. The root
[`RUNNING_THE_KERNEL.md`](../../RUNNING_THE_KERNEL.md) is the reproducible setup and execution
runbook. Superseded plans and session records are preserved in [`../archive/`](../archive/).
