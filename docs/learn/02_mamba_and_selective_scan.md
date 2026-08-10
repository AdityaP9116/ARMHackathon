# 2. Mamba and the selective scan

## What Mamba changed

Earlier SSMs had **fixed** `A`, `B`, `C`, `Δ` — the same values for every token
in the sequence. The model's memory behaviour was decided at training time and
never varied. It could not decide that one word mattered more than another.

Mamba's contribution: **make the parameters depend on the input.** At each
position, the model computes its own `B_t`, `C_t` and `Δ_t` from the token at
that position.

```
Δ_t, B_t, C_t = f(x_t)          # computed per token, not fixed
h_t = exp(Δ_t·A)·h_{t-1} + Δ_t·B_t·x_t
y_t = C_t·h_t
```

This is called **selectivity**, and the resulting layer is **S6** — Structured
State Space Sequence model with a Selective scan. The architecture built around
it is **Mamba**.

### What selectivity buys

Because `Δ_t` is now per-token, the model gains an explicit forgetting control:

- **Large `Δ_t`** → `exp(Δ_t·A)` is small (`A` is negative) → the old state is
  mostly erased. *"Something important happened, reset and focus on this."*
- **Small `Δ_t`** → `exp(Δ_t·A)` is near 1 → the state passes through unchanged.
  *"This token is filler, keep what I had."*

The model **learns** to produce those values. That is the whole idea: content-
dependent memory in a linear-time model.

## Why this makes the operation hard to compute

Selectivity is exactly what makes Mamba good, and exactly what makes it slow.

With **fixed** `A`, the whole sequence can be computed as a **convolution** —
which is a Fast Fourier Transform, which is highly parallel and which GPUs
execute beautifully.

With **input-dependent** `A_t`, `B_t`, `C_t`, that trick dies. Every step's
transition is different, so you are back to walking the sequence one step at a
time.

That walk is the **selective scan** — and it is the operation this entire project
exists to make fast.

## What the scan looks like in code

Stripped to its essentials, from our own reference implementation
([`kernel/arm-scan-core/src/scalar.rs`](../../kernel/arm-scan-core/src/scalar.rs)):

```rust
for i in 0..len {
    let t = if input.reverse { len - 1 - i } else { i };

    let mut dt = delta_row[t] + bias;
    if input.delta_softplus { dt = dt.softplus(); }
    let dt_u = dt * u_row[t];

    let mut y = T::ZERO;
    for (n, h_n) in h.iter_mut().enumerate() {
        let bc_idx = bc_base + n * len + t;
        let new = (dt * a_row[n]).exp() * *h_n + dt_u * input.b[bc_idx];
        *h_n = new;                              // update the state
        y = y + input.c[bc_idx] * new;           // read an output out of it
    }
    out_row[t] = y;
}
```

Line by line, this is the maths from [§1](01_state_space_models.md):

| Code | Maths |
|---|---|
| `dt = delta_row[t] + bias`, `softplus` | compute the step size `Δ_t` (softplus forces it positive) |
| `(dt * a_row[n]).exp()` | `Ā = exp(Δ·A)` — the discretization |
| `* *h_n` | decay the old state |
| `+ dt_u * input.b[bc_idx]` | add the new input, `Δ·B·x` |
| `y += input.c[bc_idx] * new` | read the output, `C·h` |

**Two loops.** The outer walks time and is **strictly sequential** — `h_n` on
iteration `t` depends on its value at `t-1`. The inner walks the **state
dimension** `n` and is **fully independent** — state element 0 never reads state
element 1.

That asymmetry is the single most important fact about optimizing this operation,
and [§6](06_inside_our_kernel.md) is built entirely around exploiting it.

## The shapes involved

Real tensors, with the names used throughout the code:

| Name | Shape | Meaning |
|---|---|---|
| `u` | `(batch, dim, len)` | the input sequence |
| `delta` | `(batch, dim, len)` | per-token step size `Δ` |
| `A` | `(dim, state)` | the decay matrix — **not** input-dependent, learned |
| `B`, `C` | `(batch, groups, state, len)` | input-dependent, per token |
| `h` | `(batch, dim, state)` | the hidden state |

`dim` is the channel count (768 for a 130M model), `state` is the state size per
channel (typically 16). **`dim` channels are completely independent of each
other** — another axis of free parallelism, and the one our threading uses.

Note `A` is **not** input-dependent, only `B`, `C` and `Δ` are. That is a
deliberate design choice in Mamba, and it is why `A` has no `len` axis.

## Why `torch.compile` cannot rescue this

`torch.compile` is PyTorch's optimizing compiler. It fuses operations, removes
overhead, and generates efficient code. It is genuinely good.

It cannot fix this, for a structural reason: **it cannot restructure a sequential
recurrence.** The loop above has a real data dependency at every step. No
compiler can parallelize across `t` without changing the algorithm — and
changing the algorithm is a human decision about numerics, not something a
compiler may do silently.

Worse, when it unrolls the loop, **compile time grows with sequence length**. Our
own measurements: 59.9 s at L=256, rising to 532.8 s at L=2048. You pay that
before running a single token.

This is why `torch.compile` is the honest baseline for our benchmarks
([§9](09_how_we_benchmark.md)) and also why beating it is possible at all. If a
compiler could solve this, there would be no project here.

## Where the CPU story starts

In PyTorch, Mamba's fast path is a **CUDA kernel** — hand-written code that runs
on NVIDIA GPUs. On a CPU there is no such kernel, so you get a fallback: the
recurrence expressed as PyTorch operations, interpreted step by step.

That fallback is slow for two compounding reasons:

1. **Per-operation overhead.** Every tiny operation is a separate dispatched
   PyTorch call, with Python-level bookkeeping around it. At L=2048 that is
   thousands of dispatches.
2. **It materialises enormous intermediates.** The natural way to write it in
   PyTorch computes `exp(Δ·A)` for *every timestep at once*, producing a tensor
   of shape `(batch, dim, len, state)`. At L=131,072 that is **12.88 GB** — for
   a single layer, in temporary storage that is discarded immediately.

Our kernel streams through in fixed-size chunks and never allocates that array
at all. Measured: at L=131,072 ours runs in **4.60 s** with flat memory, while
the reference needs 12.88 GB of intermediates and was not attempted.

At that length the honest claim is not "faster" — it is **"runs at all."**

---

**Next:** [Mamba-3: what changed](03_mamba3_whats_new.md)
