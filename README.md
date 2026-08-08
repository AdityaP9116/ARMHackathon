# ARMHackathon — Arm-Optimized Selective Scan for the PyTorch Mamba Ecosystem

**Arm Create: AI Optimization Challenge 2026 — Cloud AI track**

State-space models run in **linear time with constant memory** — exactly what a CPU is good
at, and exactly what a transformer's growing KV cache is not. But in PyTorch, where these
models actually live, the selective scan at their core is **CUDA-only**. On CPU it falls back
to an unoptimized sequential loop, and `torch.compile` cannot rescue it: it cannot restructure
a sequential recurrence, and its compile time *grows with sequence length*.

This project ships **hand-written Arm/NEON selective-scan kernels in Rust**, exposed as
PyTorch custom ops — for both the deployed **Mamba-1** ecosystem and the newest
**[Mamba-3](https://arxiv.org/abs/2603.15569)** (ICLR 2026), whose official kernels are
Triton, TileLang and CuTe and have **no CPU path at all**.

## Two kernels, one library

They are separate code, deliberately: Mamba-3's state is a **matrix per head** where Mamba-1's
is a **vector per channel**, and their tensor sets are disjoint. One entry point serving both
would be half-ignored on every call. They share threading, the C ABI, packaging, CI, and the
entire correctness harness.

| | Mamba-1 kernel | Mamba-3 kernel |
|---|---|---|
| Status | shipped, measured | shipped, gated |
| Topologies | 1D · fused bidirectional · SS2D cross-scan | 1D · bidirectional · 2D *(wiring in progress)* |
| Drop-in | `arm_scan.patch()` for HF `transformers` Mamba | `arm_scan.mamba3_scan(...)` |
| Ground truth | vendored f64 reference | **captured from the official GPU kernels** |

```python
import arm_scan
arm_scan.patch()                    # HF Mamba on CPU now runs the kernel
out = arm_scan.mamba3_scan(q, k, v, adt, dt, trap, q_bias, k_bias, angles=angles)
```

## What we claim, precisely

Prior art is real, and we checked it repo by repo rather than by search summary — the table
is published so a judge can see the search was done.

| Prior work | What it is | What it does not do |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | CPU `ssm_scan` for GGUF Mamba/Mamba-2 | Not PyTorch-callable; needs model conversion; 1D only |
| [`silvermpx/mamba-rs`](https://github.com/silvermpx/mamba-rs) | Mamba-3 SISO in Rust, CPU + CUDA | Standalone runtime, no PyTorch interop; x86/CUDA focus, **no Arm/NEON** |
| [`swfsql/burn-mamba`](https://github.com/swfsql/burn-mamba) | Mamba-1/2/3 for Burn, incl. bidirectional wrappers | Portable tensor ops by design — **no custom kernels**, no Arm tuning |
| [`kroggen/mamba.c`](https://github.com/kroggen/mamba.c) | Mamba/Mamba-2/Mamba-3 inference in pure C | Portable C, no NEON intrinsics; not PyTorch-callable |
| [VMamba](https://github.com/MzeroMiko/VMamba) / [2DMamba](https://arxiv.org/abs/2412.00678) | The SS2D cross-scan reference | **CUDA-only**; Mamba-1/2 lineage |
| [VNCT](https://arxiv.org/abs/2607.03589) | The 2D Mamba-3 architecture (ECCV 2026) | **Code unreleased**; GPU-only; no CPU or Arm path |

**To the best of our knowledge** this is: (1) the **first Arm/NEON selective scan exposed as a
PyTorch custom op** — a drop-in for existing checkpoints, no model conversion; (2) the **first
fast CPU SS2D cross-scan** on any architecture; (3) the **first PyTorch-callable,
NEON-optimised Mamba-3 scan**.

**We never claim** "first Mamba on CPU", "first Mamba-3 on CPU", or "first Mamba-3 in Rust" —
three projects above have those. Nor anything about bidirectional Mamba-3, which `burn-mamba`
ships.

## Correctness, and why it is unusually strong here

Mamba-3 has **no CPU reference anywhere** — upstream's module imports GPU kernels and asserts
if they are missing — so there was nothing to diff a CPU implementation against. We captured
ground truth directly from the official Triton kernels, driven by the real
`state-spaces/mamba3-siso-187m` checkpoint: 10 cases across 7 shapes, committed, replayable
with **numpy alone**. The GPU is never needed again.

Our reference reproduces those to **4.47 bf16 ULP** at tensor scale — and so do the Rust scalar
kernel, the cache-blocked kernel, and the PyTorch op, digit for digit. That is the floor bf16
output quantisation allows.

Four independent nets, all enforced in CI on arm64, macOS-arm64 and x86:

- golden vectors vs an f64 reference **and** an independent numpy re-derivation
- NEON ↔ scalar parity, and **bit-identical** output at any thread count
- goldens replayed through the **real C ABI**, not just the Rust tests
- `reverse` defined as an equivalence (flip → scan → flip) and checked as one

Two findings we document rather than bury: upstream's released `mamba-ssm` wheel predates a fix
for **silent forward-pass corruption on Blackwell**, so capturing through it would have
validated everything downstream against garbage; and the golden input draws had to move off
`torch.Generator` *and then* off float32 transcendentals, because neither reproduces across
versions or across architectures.

## Numerics, disclosed

NEON `exp` polynomials and FMA reassociation mean results match the reference to fp32
tolerance, not bit-exactly. Every golden records its own error floor and every change is gated
against it. Patched HF mamba-130m produces **token-identical** greedy output.

## Try it

```bash
git clone https://github.com/AdityaP9116/ARMHackathon && cd ARMHackathon
make validate      # kernel + SS2D + diffusion gates, ~5 min, no data, no AWS account
make test-mamba3   # Mamba-3 reference + torch op + Path A mixer vs official-kernel truth
```

Running the real 187M model on your CPU (downloads ~357 MB):

```bash
make test-mamba3-model    # logits vs the official GPU model, SISO and MIMO
python bench/bench_mamba3_lm.py --quick
python bench/bench_mamba3_lm.py --model state-spaces/mamba3-mimo-187m --quick
```

Runs on AWS Graviton, Oracle Ampere, Raspberry Pi 5 and Apple Silicon. Correctness validation
never needs credentials.

## Status — honestly

**Done:** both kernels, every correctness gate, the PyTorch integration, wheels, CI across three
platforms — and **both published Mamba-3 families running end to end on CPU**
(`apps/mamba3_lm/`): `mamba3-siso-187m` at **98.05%** argmax agreement and the rank-4
`mamba3-mimo-187m` at **96.48%**. The SISO mixer matches the official block to 1.36 bf16 ULP.

That floor is worth a sentence, because it surprised us: the official kernel is
`triton.autotune`d, so it picks its config by timing candidates at first call. Two runs of the
**official model** on the same GPU with the same seed disagree on up to 5/256 tokens, while two
forward passes inside one process are bit-identical. Our agreement sits inside that band, so
the gate is written against the measured floor rather than an unreachable 100%.

**Not done, and it is the gap that matters: there are no dedicated-hardware numbers.** Every
timing in this repository comes from an x86 box or a shared 4-core CI runner, which this
project's own rules classify as provisional. A Graviton session is the outstanding work, and no
amount of kernel engineering substitutes for it.

Also here, and as far as we can tell the first CPU implementation of any 2D Mamba-3:
`arm_scan.ss2d_scan_mamba3` runs the four-direction cross-scan on the Mamba-3 recurrence as pure
layout over `mamba3_scan_pair` — **no new kernel code**. Measured 14–38× over the PyTorch
recurrence and 1.9× over `torch.compile` at vision grid sizes. **Correctness and throughput
only**: no 2D Mamba-3 weights have ever been published, so no accuracy claim is available, and
we do not make one.

The **causal-vs-non-causal comparison** — which nobody has published for any Mamba generation —
is now measured: dropping causality costs **2×** in 1D and **~1×** in 2D, because the
four-direction cross-scan already runs both directions. It needed no new kernel, because the
decay factorises. The O(L²) dense formulation is implemented as an independent check and
reproduces the O(L) kernel to **2.99e-16** — then loses to it by 784 tokens.

Still outstanding: **making MIMO fast** — it is correct everywhere but runs on the portable
scalar path only, so its arithmetic-intensity advantage on CPU is still a prediction rather
than a measurement.

## Where to look

| | |
|---|---|
| [`MAMBA3_IMPLEMENTATION_PLAN.md`](./MAMBA3_IMPLEMENTATION_PLAN.md) | Current plan: stages, prior art, claims policy |
| [`THREE_PATHS_INTEGRATION.md`](./THREE_PATHS_INTEGRATION.md) | The three demonstrations, scoped: what each can and cannot claim |
| [`MAMBA3_KERNEL_WORKPLAN.md`](./MAMBA3_KERNEL_WORKPLAN.md) | File-by-file execution, and the audit of the existing kernel |
| [`PROJECT_CONCEPT.md`](./PROJECT_CONCEPT.md) | Decision log — what was chosen, what was rejected, why |
| [`docs/`](./docs/README.md) | The working record: superseded plans, measurement logs, diagnoses |
| [`apps/mri_diffusion/`](./apps/mri_diffusion/) | SS2D-Mamba diffusion MRI — **demoted, still CI-gated** ([plan](./docs/archive/MRI_DIFFUSION_IMPLEMENTATION_PLAN.md)) |

## License

MIT — see [`LICENSE`](./LICENSE) (also set in the repository About sidebar, per contest rules).
