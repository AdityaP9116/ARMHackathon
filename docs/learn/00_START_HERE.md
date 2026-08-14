# Start here — understanding this project from first principles

This folder explains what this project is, from "what is a state space model"
through to "what does line 300 of the NEON kernel do." It assumes you can
program, and assumes nothing else.

**On jargon:** the terms are kept, not removed. You need to say *selective scan*,
*discretization*, *SIMD*, *ULP* and *arithmetic intensity* to talk to anyone in
this field, and a judge will expect them. Every term is defined the first time
it appears, and the definitions are honest rather than simplified into
uselessness.

---

## The project in one paragraph

Mamba is a neural network architecture that competes with the transformer. At
its core is an operation called the **selective scan**. In PyTorch, that
operation is **CUDA-only** — it runs fast on NVIDIA GPUs and falls back to a slow
Python-level loop on CPUs. This project writes that operation by hand in Rust,
optimized for **Arm processors** (the CPUs in AWS Graviton servers, Apple
Silicon, and phones), and exposes it to PyTorch so existing models can use it
with no code changes. It does this for two generations of the architecture —
Mamba-1 and Mamba-3 — and for three different *shapes* of the scan.

## Reading order

Each file assumes the ones before it. Read in order the first time.

| # | File | What you will understand after it |
|---|---|---|
| 1 | [State space models](01_state_space_models.md) | What an SSM is, and why "linear time, constant memory" matters |
| 2 | [Mamba and the selective scan](02_mamba_and_selective_scan.md) | What Mamba added, and why the scan is hard to make fast |
| 3 | [Mamba-3: what changed](03_mamba3_whats_new.md) | The newest generation and why we targeted it |
| 4 | [What a kernel is](04_what_a_kernel_is.md) | Kernels, SIMD, NEON — why anyone writes assembly-adjacent code in 2026 |
| 5 | [Why CPU, why Arm](05_why_cpu_on_arm.md) | The argument the whole project rests on |
| 6 | [Inside our kernel](06_inside_our_kernel.md) | The actual Rust: two passes, chunking, threading |
| 7 | [The three topologies](07_the_three_topologies.md) | 1D, bidirectional, 2D cross-scan |
| 8 | [Proving correctness](08_how_we_prove_correctness.md) | Goldens, oracles, ULPs — the part that makes the speed claims believable |
| 9 | [Benchmarking honestly](09_how_we_benchmark.md) | Baselines, `torch.compile`, and how to not fool yourself |
| 10 | [Code tour](10_code_tour.md) | Which file does what |

## If you only have ten minutes

Read [§5, Why CPU and why Arm](05_why_cpu_on_arm.md). That file contains the
argument. Everything else is either the background needed to state it or the
engineering needed to support it.

## How this relates to the rest of the repo

These files are **teaching material**. They are not the source of truth for what
is done or planned — that lives in:

- [`README.md`](../../README.md) — the pitch, and the precise claims
- [`docs/project/STATUS.md`](../project/STATUS.md) — where the work currently stands
- [`CLAUDE.md`](../../CLAUDE.md) — standing engineering and claims rules

If this folder ever disagrees with those, they win.
