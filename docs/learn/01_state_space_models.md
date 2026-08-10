# 1. State space models

## The problem every sequence model solves

You have a sequence — words in a sentence, samples in an audio clip, pixels in a
row. You want to produce an output at each position that depends on everything
that came before. The question is *how you remember the past*.

There are two classic answers, and they have opposite cost profiles.

### The transformer's answer: remember everything, look at it all

At position `t`, a transformer **attends** to all previous positions. Every token
can look directly at every other token.

- **Cost to produce one output:** proportional to `t` — it examines every earlier
  position.
- **Cost for a sequence of length L:** proportional to **L²**.
- **Memory during generation:** a **KV cache** (the stored keys and values for
  every previous token) that **grows linearly with L**.

Quadratic cost is why long context is expensive, and the growing KV cache is why
a long chat session eats more and more GPU memory.

### The RNN's answer: keep a summary

A recurrent neural network keeps a fixed-size **hidden state** `h` — a summary of
everything seen so far. At each step you update the summary and emit an output:

```
h_t = (something) · h_{t-1} + (something) · x_t     # update the summary
y_t = (something) · h_t                             # read the summary
```

- **Cost for a sequence of length L:** proportional to **L**. Linear.
- **Memory:** **constant.** `h` is the same size at position 10 and position
  10,000,000.

That is dramatically better. The catch is the arrow of causality: `h_t` needs
`h_{t-1}`, which needs `h_{t-2}`. The steps are **sequential** — you cannot
compute step 500 before step 499. Classic RNNs were also hard to train on long
sequences (vanishing gradients), which is why transformers displaced them.

## What a state space model is

A **state space model (SSM)** is an RNN with a particular, principled structure
borrowed from control theory. The continuous-time form is:

```
h'(t) = A·h(t) + B·x(t)      # how the state evolves
y(t)  = C·h(t)               # what you can observe
```

Read it in words:

- `x(t)` is the **input** at time `t`
- `h(t)` is the **hidden state** — the memory
- `A` says **how the state decays and mixes on its own**, with no input
- `B` says **how new input enters** the state
- `C` says **how you read an output** out of the state

This describes an enormous number of physical systems: a mass on a spring, the
charge on a capacitor, the temperature of a room. That is the origin of the name
— it is the *state* of the system, in a *space* of possible states.

### Discretization — from continuous to steps

The equations above are continuous — they use a derivative `h'(t)`. Real data
arrives in discrete steps: token 1, token 2, token 3. Converting continuous
dynamics into per-step updates is called **discretization**, and it introduces a
**step size**, conventionally `Δ` (delta) — how much time each step represents.

The result is a recurrence you can actually run:

```
h_t = Ā·h_{t-1} + B̄·x_t
y_t = C·h_t
```

where `Ā` and `B̄` (read "A-bar", "B-bar") are the discretized versions of `A`
and `B`. The most common discretization gives:

```
Ā = exp(Δ·A)
```

**That `exp` is not decoration — it is the single most expensive arithmetic
operation in the whole kernel.** Remember it; it comes back in
[§6](06_inside_our_kernel.md), where it accounts for roughly half the runtime.

Why an exponential? Because a continuous system with decay rate `A`, left alone
for time `Δ`, decays by exactly `exp(Δ·A)`. `A` is negative in practice — so
`exp(Δ·A)` is between 0 and 1, and the state fades rather than exploding. You can
see this constraint in our own test generator, which draws `a` from
`-16.0..-0.01`: negative by construction.

## Why this is attractive for hardware

The SSM keeps the RNN's cost profile — **linear time, constant memory** — while
having enough mathematical structure to be trained effectively on long
sequences.

Now connect that to a CPU. A CPU has:

- modest parallelism compared to a GPU (tens of cores, not thousands)
- **a lot of memory**, cheaply (hundreds of GB on a server; a GPU has 32–80 GB)
- no memory transfer cost to "get the data onto the device"

A transformer's growing KV cache fights a CPU's weakness (raw parallel
throughput) and ignores its strength. An SSM's constant memory footprint is
almost tailor-made for it.

**That observation is the seed of this entire project**, and it is stated
directly in our own README: *"State-space models run in linear time with
constant memory — exactly what a CPU is good at, and exactly what a
transformer's growing KV cache is not."*

## The catch, stated plainly

Linear and constant sound strictly better than quadratic and growing. So why did
transformers win?

1. **Sequential dependency.** `h_t` needs `h_{t-1}`. A transformer's attention
   over a known sequence is a big matrix multiply — embarrassingly parallel, and
   GPUs eat those. The scan is a chain.
2. **Early SSMs had fixed dynamics.** `A`, `B`, `C` were the same for every
   token. The model could not decide "this word matters, remember it" — its
   memory behaviour was fixed in advance. That limitation is what **Mamba**
   fixed, and it is the subject of the next file.

---

**Next:** [Mamba and the selective scan](02_mamba_and_selective_scan.md)
