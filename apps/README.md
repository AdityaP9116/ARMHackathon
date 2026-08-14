# Applications and integration surfaces

The kernel is the product; these applications prove that its interfaces work in realistic
PyTorch code.

| Directory | Role | Status |
|---|---|---|
| [`mamba3_lm/`](mamba3_lm/) | Loads the published Mamba-3 SISO and MIMO 187M checkpoints and routes their recurrent mixer through Arm Scan | Active submission path; used for real-model correctness and throughput |
| [`mri_diffusion/`](mri_diffusion/) | Minimal SS2D diffusion MRI pipeline and fastMRI data tooling | Preserved and CI-gated, but demoted from the submission critical path because its quality gate has not passed |

These directories do not contain model weights or datasets. See
[`../RUNNING_THE_KERNEL.md`](../RUNNING_THE_KERNEL.md) for execution commands and
[`../docs/project/THREE_PATHS_INTEGRATION.md`](../docs/project/THREE_PATHS_INTEGRATION.md) for
the current demonstration scope.
