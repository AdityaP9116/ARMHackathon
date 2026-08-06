# SPIKE_FINDINGS — feasibility of the three demo slots

**Aug 6, 2026.** A timeboxed investigation, run *before* committing days to any of
the three planned demonstrations. Everything below was executed or read
first-hand; nothing is inferred from documentation alone.

| Slot | Verdict | Cost to build |
|---|---|---|
| **1D unidirectional — long context** | 🟢 **GREEN.** Measured, works today | ~1 day |
| **1D bidirectional — speech enhancement** | 🟡 **AMBER.** Checkpoint exists, but our *fused* kernel does not apply and the dependency needs surgery | 2–3 days |
| **Mamba-3 kernel** | 🔴 **RED for this deadline** | Weeks, and blocked on an oracle problem |

---

## 1. Long context — GREEN, and the numbers are good

Measured directly on the scan at a mamba-130m layer shape (B=1, D=768, N=16),
x86, 16 threads. The claim was never "we are N% faster" — it is that our memory
is flat in sequence length while the PyTorch reference materialises a
`(B, D, L, N)` intermediate that grows linearly.

| L | reference's intermediate | our kernel | our RSS rise | torch reference | torch RSS rise |
|---|---|---|---|---|---|
| 2,048 | 0.20 GB | **0.07 s** | 0 MB | 0.41 s | 315 MB |
| 8,192 | 0.81 GB | **0.28 s** | 25 MB | 2.02 s | 1,179 MB |
| 32,768 | 3.22 GB | **1.11 s** | 101 MB | 8.20 s | 4,753 MB |
| **131,072** | **12.88 GB** | **4.60 s** | ~0 MB | **not attempted** | — |

Two results, and the second is the demo:

- **Speed**: 5.9× → 7.2× → 7.4× as L grows. The advantage *widens* with length.
- **Memory**: the reference climbs 315 MB → 4.7 GB and would need **12.9 GB of
  intermediates** at 128k. Ours stays flat — the rises shown are output-tensor
  allocation and allocator reuse, not scan state.

So at 128k the honest framing is not "faster" but **"runs at all."** That is a
capability gap, and it is the strongest single measurement in the project.

*(Kernel memory reads as ~0 at some rows because the allocator reuses the
previous iteration's blocks. The flatness is real; the exact MB is noisy.)*

---

## 2. Mamba-3 — RED, and for a reason I did not anticipate

The recurrence *shape* is still close to ours — `h_t = α_t h_{t-1} + (2-tap input)`
is our Pass B with one extra term. That part of the earlier optimism holds. Two
things kill it for a 10-day window:

**`mamba-ssm` cannot be installed without CUDA.** Not "runs slowly on CPU" —
`pip install` fails during metadata generation:

```
UserWarning: mamba_ssm was requested, but nvcc was not found.
torch.__version__ = 2.13.0+cpu
error: metadata-generation-failed
```

**There is no pure-PyTorch reference to check a kernel against.** Reading
`mamba_ssm/modules/mamba3.py` directly: it imports `mamba3_siso_combined`
(Triton), `mamba3_mimo_combined` (TileLang) and `mamba3_step_fn` (CuTe). Missing
imports raise assertions — **there is no CPU fallback and no torch-only path
anywhere in the file.**

That is the blocker. Our entire correctness methodology is *diff against a
trusted reference*: goldens from a vendored oracle, then scalar↔NEON parity.
For Mamba-3 the oracle does not exist, so we would have to **write it from the
paper first**, and validating *that* against the official implementation
requires a GPU. The kernel is the easy part; the correctness infrastructure is
the expensive part.

### Correction — two of the claims above were wrong

A follow-up prior-art search (prompted by the user, not by me) turned up two
repositories that change this section materially. Both verified:

**[`rishikksh20/mamba3-pytorch`](https://github.com/rishikksh20/mamba3-pytorch)** —
a pure-PyTorch Mamba-3 with trapezoidal discretization, RoPE on B/C, and both
SISO and MIMO. Runs on CPU, no CUDA. So **the oracle problem is solvable
without a GPU** — this is a usable reference, the same role
`tests/reference/selective_scan_ref.py` plays for Mamba-1. (It is a community
re-implementation, so it would itself need validating against the official
kernels before being trusted as ground truth.)

**[`silvermpx/mamba-rs`](https://github.com/silvermpx/mamba-rs)** — **Mamba-3
SISO, in Rust, on CPU, with rayon parallelization** and optional BLAS. This is
direct prior art, and it is the important finding: the argument for pivoting to
Mamba-3 was "nobody has it on CPU." That is **no longer true**.

What survives is narrow. `mamba-rs` is a standalone runtime — "no framework
dependency," a library and CLI, no PyTorch interop — and its optimization story
is x86/CUDA with no Arm/NEON. So a pivot would leave us claiming *"first
PyTorch-callable, NEON-optimized Mamba-3 scan"*: defensible, but a much thinner
claim that invites "isn't that mamba-rs with a Python binding?"

Both are now in the README prior-art table. We claim to have checked; a judge
finding `mamba-rs` in a table we omitted it from would be far more damaging
than the row itself.

**What this does NOT touch:** `mamba-rs` is 1D only. VMamba is CUDA-only.
2DMamba is GPU-only. **"First fast CPU SS2D cross-scan on any architecture"
stands unchallenged** — and it is the claim we already have working, gated, and
measured.

**Net effect on the recommendation: strengthens the case for not pivoting.**
The novelty argument was the entire reason to spend ~5 of 8 remaining days on
Mamba-3, and it just got substantially weaker.

---

## 3. Speech enhancement — AMBER, and it invalidates an assumption

**SEMamba** ([RoyChao19477/SEMamba](https://github.com/RoyChao19477/SEMamba),
IEEE SLT 2024) is a real, pretrained, non-causal speech-enhancement Mamba —
PESQ 3.55 on VoiceBank-DEMAND, 3.69 with PCS, weights published, and there is a
working HF Space. As a demo target it is exactly what we wanted.

**But it is the wrong *kind* of bidirectional for our fused kernel.** From
`models/mamba_block.py`:

```python
x_backward = torch.flip(x, [1])          # flip the INPUT
y_backward = self.backward_blocks(...)   # SEPARATE weights
torch.cat([y_forward, y_backward], -1)
```

Separate `forward_blocks` and `backward_blocks` means this is the **"outer"**
bidirectional pattern — the whole mixer, including the causal convolution and
the projections, is re-run on the flipped input. `python/arm_scan/bidirectional.py`
already documents why that matters:

> a causal conv over flipped input is not the flip of the conv over input, so
> the two directions' scan inputs are genuinely different tensors… A kernel
> `reverse` flag buys it nothing.

**Consequence:** SEMamba would exercise our *unidirectional* kernel twice
(3.71× vs `torch.compile`), **not** the fused bidirectional path (6.39–8.99×).
The reason we picked audio — to light up our strongest-measured topology —
does not survive contact with the actual model.

**Second obstacle:** SEMamba imports `from mamba_ssm.modules.mamba_simple import
Mamba`, so it inherits the CUDA install blocker above. Its `requirements.txt`
does not list `mamba-ssm`, which is misleading — the import is there in the code.

Getting it running on CPU means vendoring or stubbing `mamba_ssm` and routing
its scan to ours. That is real work, and it is also the strongest framing
available: **the model cannot run on a CPU-only machine today at all.** "We
made a CUDA-only model run on an Arm CPU in real time" beats "we made it 3.7×
faster" — but it is 2–3 days, not one.

---

## What I would do with this

**Build the long-context demo.** It is green, the numbers are already measured,
and the 128k row is the strongest evidence in the project.

**Treat audio as the stretch**, with eyes open: it costs 2–3 days, needs
dependency surgery, and pays out on the unidirectional kernel rather than the
bidirectional one. Worth it for an audible demo and a real-time-factor metric —
but not worth displacing the Graviton session.

**Keep Mamba-3 as roadmap**, now with a first-hand justification: no CPU
implementation exists, the reference will not install without a GPU, and there
is no torch-only oracle to build against.

**The fused bidirectional kernel still has no application.** That gap is real
and this spike did not close it. The honest options are to find an "inner"
bidirectional model (weights shared, flip after the projections), or to present
the bidirectional kernel on its own measured merits — 6.39–8.99× vs
`torch.compile` on Arm, which is a legitimate kernel result even without an
application attached.
