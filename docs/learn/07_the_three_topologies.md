# 7. The three topologies

A **topology** here means the *pattern in which the scan traverses the data* —
not a different kernel, but a different route through it. Our argument for
generality rests on one kernel serving all three.

## 1. 1D unidirectional — left to right

The basic case from [§2](02_mamba_and_selective_scan.md). Start at token 0, walk
forward, each step depending on the last.

**Where it is used:** language models. Text generation is inherently causal —
predicting token `t` may only use tokens before it.

**Its demonstration: long context.** Not "we generate text N% faster" — the
capability claim. Our memory is flat in `L` while the PyTorch reference's grows
linearly, so past some length the reference simply cannot run and we can. See the
table in [§5](05_why_cpu_on_arm.md); at L=131,072 the reference would need 12.88 GB
of intermediates.

**Measured:** 3.71× vs `torch.compile`.

## 2. 1D bidirectional — both directions

Run the scan forward *and* backward, then combine. Position `t` now sees context
from both sides.

**Where it is used:** anything where the whole sequence is available up front —
classification, speech enhancement, encoders. Not generation.

### The fused optimization, and why it works

The naive approach runs the kernel twice. Ours runs **one fused call** that does
both directions.

The win comes from the two-pass structure in [§6](06_inside_our_kernel.md):

> **Pass A is direction-independent.**

`dt`, `ābar = exp(dt·a)` and `b̄` depend only on the input and the parameters —
**not** on which way you walk. Only Pass B's sequential recurrence has a
direction.

Since Pass A is ~85% of the work, sharing it between the two directions is close
to a free second scan.

**Measured:** 6.39–8.99× vs `torch.compile` — our strongest-measured topology.

### An honest gap

This topology has **no application demonstration**. We investigated speech
enhancement (SEMamba) and found it uses **outer** bidirectionality: separate
weights per direction, with the whole mixer — including its causal convolution —
re-run on the flipped input. A causal convolution over flipped input is *not* the
flip of the convolution over the input, so the two directions' scan inputs are
genuinely different tensors. **A kernel `reverse` flag buys nothing there.**

So we present the bidirectional kernel on its measured merits rather than
attached to a model. And we make **no novelty claim** about bidirectional
Mamba-3, because `burn-mamba` already ships it.

Stating this is better than quietly omitting it.

## 3. 2D cross-scan (SS2D) — images

An image is 2D; a scan is 1D. To apply an SSM to an image you must choose a
traversal order — and any single order is biased. Scanning left-to-right,
top-to-bottom means a pixel "sees" everything above and to the left, and nothing
below or to the right.

**SS2D** (from VMamba) fixes this with **four** scans:

```
1. →  left to right, top to bottom
2. ←  right to left, bottom to top
3. ↓  top to bottom, column-major
4. ↑  bottom to top, column-major
```

Each pixel ends up with context from all four directions; the results are summed.

### Our optimization: two pairs, not four scans

The four directions form **two traversal pairs**: (1,2) are the same row-major
path in opposite directions, and (3,4) are the same column-major path in opposite
directions.

A pair in opposite directions is *exactly* what the fused bidirectional kernel
does. So instead of four independent forward scans, we run **two fused pair
calls** — computing Pass A **twice** instead of four times, and eliminating four
`torch.flip` tensor copies.

**Measured:** **1.77–1.82× (geomean 1.80×)** over the four-scan formulation.
Non-scan overhead fell from 21–25% to 7.2–13.8%.

This is the clearest example of the layering paying off: `ss2d_scan` →
`scan_pair` → `bidirectional_scan` → *a scan primitive*. **That layer is
recurrence-agnostic**, which is why pointing it at the Mamba-3 primitive required
**no new kernel code at all**.

### A trap worth knowing

The RoPE angle pre-pass must run on the traversal **views**, not on the grid.
Otherwise both orderings silently share the row-major `θ`, and the column-major
scan gets the wrong positional encoding — a bug that produces plausible-looking
numbers rather than an error.

## The non-causal result

This one is worth understanding in detail, because it is the project's most
interesting *finding* rather than its most impressive number.

**Causal** = position `t` sees only `s ≤ t`. **Non-causal** = `t` sees all
positions. Vision wants non-causal — there is no reason a pixel should not see
the pixel to its right.

Our plan predicted this needed a **second kernel**: two dense matrix multiplies,
O(L²), with a weak moat because BLAS libraries are excellent at GEMMs.

**The plan was wrong, and the maths says so.** Unrolling the recurrence gives

```
y_t = Σ_s M[t,s]·(q_t·k_s)·v_s      with    M[t,s] = e^(L_t − L_s)·scale_s
```

The decay **factorises**: `e^(L_t − L_s) = e^(L_t)·e^(−L_s)`. So the sum over
`s < t` is exactly a forward scan and the sum over `s > t` is exactly a backward
one:

```
Σ_all s  =  Σ_{s≤t}  +  Σ_{s≥t}  −  Σ_{s=t}
non-causal =  forward  +  backward  −  diagonal
```

**Non-causal costs 2× a causal scan, not O(L²) — and needs no new kernel.** Both
directions already existed. **No Rust changed.**

The diagonal correction reads `q_t·k_t`, which looks like it needs the rotated
`q`/`k` the kernel computes internally. It does not: **a dot product is invariant
under rotating both operands by the same angle**, and RoPE gives `q_t` and `k_t`
the same `θ_t`. That is why this composes over the *public* op.

In 2D it is nearly free: the four-direction cross-scan **already** runs both
directions, so non-causal 2D is the same scans minus two diagonals. **Measured
1.06–1.23× causal**, versus 2.25–2.65× in 1D where a second scan really is added.

The dense O(L²) form is implemented anyway — as an independent oracle and as the
other half of the comparison. It is competitive around 200 tokens, 5× slower by
784, and by 3,136 the `(L,L)` mask per head binds before time does. **For any
real vision grid the scan form wins** — which is the argument for a CPU scan
kernel existing at all, now measured rather than asserted.

### Why this is good evidence

The load-bearing gate is that an **O(L²) dense algorithm reproduces the O(L)
kernel to 2.99e-16 in f64**. Two implementations sharing no code, agreeing to
machine precision. That validates the mask derivation and, through it, the
recurrence itself.

---

**Next:** [Proving correctness](08_how_we_prove_correctness.md)
