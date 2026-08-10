# 5. Why CPU, and why Arm

**This is the argument the project rests on.** If you read one file, read this
one.

## The obvious objection

> Neural networks run on GPUs. Why write a CPU kernel at all?

It is the right question, and it has a real answer.

## Reason 1 — inference is not training

Training happens once, on a GPU cluster. **Inference happens forever**, every
time a user makes a request. Most of the total compute spent on a deployed model
is inference, and the economics there are different: latency, cost per request,
and *availability of hardware* matter more than peak throughput.

## Reason 2 — CPUs are what you actually have

GPU instances are expensive, capacity-constrained, and frequently unavailable in
the region you want. CPU instances are cheap, abundant, and already running your
web tier.

For a model small enough to serve on CPU, "no GPU required" removes a
procurement problem, a cost line, and a scaling constraint at once.

## Reason 3 — the architecture genuinely fits

This is the part that is specific to SSMs rather than generic CPU advocacy.

| | Transformer | SSM (Mamba) |
|---|---|---|
| Time cost | **O(L²)** | **O(L)** |
| Memory during generation | KV cache, **grows with L** | fixed state, **constant** |
| Parallelism available | very high (big matmuls) | limited (sequential scan) |

A GPU's advantage is *massive parallelism*. A transformer's attention is a huge
matrix multiply — perfect for that. The sequential scan is not: it is a chain of
dependent steps, which wastes most of a GPU's width.

So the scan is the operation where a GPU's advantage is **smallest**, and where a
CPU's strengths — large cheap memory, strong single-thread performance, deep
cache hierarchy — count for **most**.

And the constant-memory property is a straightforward win on a machine with
hundreds of gigabytes of RAM and no transfer cost to reach it.

**Our measured evidence**, at a mamba-130m layer shape:

| L | reference's intermediate | our kernel | our RSS rise | torch reference | torch RSS rise |
|---|---|---|---|---|---|
| 2,048 | 0.20 GB | **0.07 s** | 0 MB | 0.41 s | 315 MB |
| 8,192 | 0.81 GB | **0.28 s** | 25 MB | 2.02 s | 1,179 MB |
| 32,768 | 3.22 GB | **1.11 s** | 101 MB | 8.20 s | 4,753 MB |
| **131,072** | **12.88 GB** | **4.60 s** | ~0 MB | **not attempted** | — |

Two findings, and the second is the one that matters:

- **Speed:** 5.9× → 7.2× → 7.4× as L grows. The advantage *widens* with length.
- **Memory:** the reference climbs to 4.7 GB and would need **12.88 GB of
  intermediates** at 128k. Ours stays flat.

At 128k the honest claim is not "faster." It is **"runs at all."** That is a
capability difference, not a performance difference, and it is the strongest
single measurement in the project.

*(Reproduce with `python bench/bench_longctx.py`; needs `psutil`.)*

## Why Arm specifically

### Arm is where server CPUs are going

**AWS Graviton** is Arm. Graviton4 (Neoverse-V2) is the current generation, and
AWS reports the majority of new CPU capacity going in is Graviton. Azure Cobalt
and Google Axion are Arm. Apple Silicon is Arm. Every phone is Arm.

Optimizing for Arm is optimizing for where the deployment is heading, not where
it has been.

### The price argument

Graviton instances are typically **~20% cheaper** than comparable x86 for similar
performance. For an inference workload running continuously, that is a direct and
permanent cost reduction.

### The gap in the ecosystem

This is the actual opening. PyTorch's Mamba path is **CUDA-only**. On any CPU you
get the slow fallback; on an **Arm** CPU there was no optimized `selective_scan`
callable from PyTorch at all.

So: an architecture that suits CPUs, deployed on the hardware the industry is
moving to, with a missing piece exactly where the two meet.

## What we claim — precisely

Over-claiming is the fastest way to lose credibility with an engineer judge, so
the claims are narrow and the prior art is published:

| Prior work | What it is | What it does not do |
|---|---|---|
| llama.cpp | CPU `ssm_scan` for GGUF Mamba/Mamba-2 | not PyTorch-callable; needs model conversion; 1D only |
| `silvermpx/mamba-rs` | Mamba-3 SISO in Rust, CPU + CUDA | standalone runtime, no PyTorch interop; **no Arm/NEON** |
| `swfsql/burn-mamba` | Mamba-1/2/3 for Burn, incl. bidirectional | portable tensor ops — **no custom kernels** |
| `kroggen/mamba.c` | Mamba/2/3 inference in pure C | portable C, no NEON; not PyTorch-callable |
| VMamba / 2DMamba | the SS2D cross-scan reference | **CUDA-only**; Mamba-1/2 lineage |
| VNCT | the 2D Mamba-3 architecture | **code unreleased**; GPU-only |

**To the best of our knowledge**, this is:

1. the **first Arm/NEON selective scan exposed as a PyTorch custom op** — a
   drop-in for existing checkpoints, no model conversion
2. the **first fast CPU SS2D cross-scan** on any architecture
3. the **first PyTorch-callable, NEON-optimized Mamba-3 scan**
4. the **first CPU implementation of a 2D Mamba-3**, causal and non-causal, plus
   the causal-vs-non-causal comparison — which nobody has published for any
   Mamba generation

**We never claim** "first Mamba on Arm/CPU", "first Mamba-3 on CPU", or "first
Mamba-3 in Rust." Those belong to the projects above.

Claim (4) carries a caveat stated inline rather than in a footnote: **there is no
authoritative 2D oracle.** VNCT's code is unreleased and no 2D Mamba-3 weights
exist, so our 2D work is validated against our own reading of the operator — two
independent algorithms agreeing to 2.99e-16. That proves the kernel implements
the reference; it does **not** prove the reference implements VNCT as intended.
**No accuracy claim is available for 2D and none is made.**

## The generality argument

One kernel, three scan **topologies** ([§7](07_the_three_topologies.md)), each
proven on a different workload.

That structure is deliberate. A single optimization on a single model invites
"you tuned for your benchmark." Three unrelated shapes of the same operation,
each with correctness gates and measured throughput, is evidence that **the
kernel itself is general** — which is a much harder thing to dismiss.

---

**Next:** [Inside our kernel](06_inside_our_kernel.md)
