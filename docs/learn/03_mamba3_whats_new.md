# 3. Mamba-3: what changed, and why we targeted it

Mamba-3 ([arXiv 2603.15569](https://arxiv.org/abs/2603.15569), ICLR 2026) is the
current generation. This file covers what is different and why those differences
mattered for our kernel.

## The four changes that matter

### 1. A better discretization: exponential-trapezoidal

Mamba-1 discretizes with a **first-order** approximation: assume the input is
constant across the step. That is the crudest possible choice.

Mamba-3 uses a **trapezoidal** rule: approximate the input over the step as a
*blend* of the value at the start and the value at the end. Geometrically, you
are approximating the area under a curve with a trapezoid instead of a
rectangle — more accurate for the same step size.

The recurrence becomes **three terms** instead of two:

```
h_t = α_t·h_{t-1} + β_t·(B_{t-1}·x_{t-1}) + γ_t·(B_t·x_t)
      ^^^^ decay    ^^^^ previous input     ^^^^ current input
```

That extra middle term — the *previous* step's input — is the trapezoid.

**Why this was good news for us.** Group the last two terms together and call
their sum `b̄_t`:

```
h_t = α_t·h_{t-1} + b̄_t
```

That is *exactly* the shape our existing kernel already computes. `b̄_t` is a
**2-tap convolution** over the input stream — a weighted sum of the current and
previous values — and computing it is **pointwise in time**, meaning step `t`
does not depend on step `t-1`. So it goes in the parallelizable pass, and the
sequential part of the kernel is **unchanged**.

This is why targeting Mamba-3 was feasible in days rather than weeks.

### 2. Data-dependent rotations (RoPE) on B and C

Mamba-3 applies **RoPE** — rotary position embedding — to `B` and `C`. RoPE
encodes position by *rotating* a vector by an angle proportional to its position.
It is standard in modern transformers; Mamba-3 brings it into the SSM, with the
rotation angle computed from the data.

The practical effect for a kernel author: extra pointwise trigonometry before the
scan. Cheap relative to the scan itself, which is why our implementation keeps
the angle computation in Python and only the scan in Rust.

There is a subtlety here worth knowing, because it produced a real result in this
project: **a dot product is invariant under rotating both operands by the same
angle.** If `q` and `k` are both rotated by `θ_t`, then `q·k` is unchanged. That
fact is what let us implement non-causal Mamba-3 *on top of the public kernel
API* rather than inside the kernel — see [§7](07_the_three_topologies.md).

### 3. MIMO — multi-input, multi-output

Mamba-1 is **SISO**: single-input, single-output. Each head processes one stream.

Mamba-3 adds **MIMO** with a rank parameter `R`: each head processes `R` streams
at once. The state stops being a vector per channel and becomes a **matrix per
head**.

The argument for MIMO is **arithmetic intensity** — the ratio of arithmetic
operations to bytes moved from memory. Higher is better, because memory is
usually the bottleneck, not the arithmetic. MIMO does more maths per byte loaded.

**On a CPU that argument should matter even more than on a GPU**, because CPUs
have proportionally less memory bandwidth relative to their compute. That makes
MIMO-on-CPU an interesting question.

**We have not answered it.** Our MIMO kernel is correct but runs on the portable
scalar path only — there is no NEON MIMO kernel. So the arithmetic-intensity
argument remains a **prediction**, not a result. Saying so plainly is the honest
position, and it is stated in the README.

### 4. No Conv1D

Mamba-1 has a short causal convolution before the scan. Mamba-3 removes it —
the trapezoidal discretization already mixes adjacent timesteps, so the
convolution is redundant.

For us this is pure simplification: one less operation, and no need for the
`causal-conv1d` dependency.

## Why we targeted Mamba-3

**Novelty.** Arm-optimized CPU kernels for Mamba-1 and Mamba-2 already exist —
llama.cpp has one, `mamba.rs` has one. Mamba-3 is new enough that the ecosystem
has not caught up.

But **prior art is real and we checked it repo by repo.** Three projects already
do Mamba-3 on CPU:

| Project | What it does | What it does not |
|---|---|---|
| `silvermpx/mamba-rs` | Mamba-3 SISO in Rust, CPU + CUDA | standalone runtime, no PyTorch interop; x86/CUDA focus, **no Arm/NEON** |
| `swfsql/burn-mamba` | Mamba-1/2/3 for the Burn framework | portable tensor ops by design — **no custom kernels** |
| `kroggen/mamba.c` | Mamba/2/3 inference in pure C | portable C, no NEON intrinsics; not PyTorch-callable |

So we **never claim** "first Mamba-3 on CPU" or "first Mamba-3 in Rust." Those
are taken. What survives is narrower and defensible: **first PyTorch-callable,
NEON-optimized Mamba-3 scan**, and **first CPU implementation of a 2D Mamba-3**.

This matters more than it might seem. A judge who finds `mamba-rs` in a table we
omitted would discount everything else we say. Publishing the table is how you
demonstrate the search was actually done.

## The problem that dominated the project: no oracle

To claim a fast implementation is *correct*, you diff it against a trusted
reference. For Mamba-1 that reference ships with upstream as `selective_scan_ref`
— plain PyTorch, runs anywhere.

**Mamba-3 has nothing of the sort.** `mamba_ssm/modules/mamba3.py` imports Triton,
TileLang and CuTe kernels and asserts if they are missing. There is **no CPU path
anywhere in the file**. The package will not even install without `nvcc`.

The obvious substitute — a community PyTorch reimplementation — turned out to
compute a **different recurrence from the paper**. Measured on inputs where the
states are O(1), so this is not floating-point rounding:

```
community vs paper, same gate ........................ max_abs 1.06
best gate remapping (1 - gate/2) ..................... max_abs 0.145
structural difference alone (decay on the prev term) . max_abs 0.363
```

The paper carries the decay on the *previous* input term; the community version
does not. **No remapping of the gate reconciles them.**

We cannot tell from outside which is right — and it does not matter, because the
only authoritative answer is **what the official kernels compute**, since that is
what the published weights were trained against.

So the project's very first step was: rent a GPU, run the official kernels,
**record their inputs and outputs to disk**, commit those files, and never need a
GPU again. Those recordings are the *goldens*, and they are the foundation of
everything in [§8](08_how_we_prove_correctness.md).

Reproduce the disagreement yourself:

```bash
python tools/check_mamba3_recurrence.py
```

---

**Next:** [What a kernel is](04_what_a_kernel_is.md)
