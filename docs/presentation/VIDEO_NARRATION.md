# Arm Scan video narration

**Target:** approximately 2 minutes 25–40 seconds, including natural pauses

**Recording deck:** [`arm_hackathon_deck_manim.html`](arm_hackathon_deck_manim.html)

**Format:** voiceover only; slide titles and cues below are not spoken

The deck uses manual navigation; its `data-duration` values do not automatically advance the
slides. Slide 2 automatically replays after a 1.8-second pause, so let at least one full cycle
finish before advancing. Hold the final slide for roughly 16 seconds while reading the three
claims.

## Slide 1 — Title

Arm Scan is a PyTorch-callable selective-scan library optimized for Arm CPUs, covering three
Mamba-3 paths.

## Slide 2 — Attention versus state

During standard attention prefill, each token compares with prior context, so total work grows
quadratically. Mamba instead updates one fixed recurrent state per token: linear work with a
bounded inference state.

## Slide 3 — The missing CPU path

Mamba-3's official fast scans use GPU-specific Triton, TileLang, and CuTe kernels. On CPU, users
fall back to many small PyTorch operations. `torch.compile` helps at short lengths, but cannot
turn the recurrence into one fused, Arm-aware scan.

## Slide 4 — Kernel design

I built that scan in Rust as a PyTorch custom operation: one native call, fixed 128-token chunks,
four FP32 values per NEON instruction, and Rayon across independent rows. Correctness against
captured ground truth gates every benchmark.

## Slide 5 — Path 1A: Mamba-3 SISO speed

Using the published 187-million-parameter SISO checkpoint at 1,024 tokens, the full forward
pass fell from 5,064 milliseconds in eager PyTorch to 268 with Arm Scan: 18.89 times faster,
or 3,820 tokens per second, on Graviton4. Argmax agreement was 98.05 percent with zero
unexplained flips.

## Slide 6 — Path 1B: long-context memory

Separately, at the original scan shape, 128,000 tokens finished in 4.60 seconds using about 17
kilobytes of fixed scratch per thread. PyTorch was not run because two intermediates alone
require 12.88 gigabytes. The 6.1-million-token operator projection is untested—not a full-model
Mamba-3 claim.

## Slide 7 — Path 2: Mamba-3 MIMO

Rank-four MIMO accumulates four updates into the same state. It matches the GPU oracle to 1.90
BF16 ULPs, and rank one collapses to SISO. This is correctness, not speed: the scalar path is
3.43 times slower and still needs a NEON microkernel.

## Slide 8 — Path 3: Mamba-3 2D cross-scan

For 2D data, Arm Scan traverses each grid in four directions, giving pixels context from both
axes. At 56 by 56, it is 92.5 times faster than eager PyTorch on Graviton4 and matches
independent dense math. Without a public 2D checkpoint or authoritative oracle, this is
operator throughput and correctness—not model accuracy.

## Slide 9 — Close

To the best of our knowledge: the first Arm NEON selective scan exposed as a PyTorch custom
operation, the first fast CPU SS2D cross-scan, and the first PyTorch-callable NEON Mamba-3
scan. Code and results are on GitHub.

---

## Recording notes — do not read aloud

- Say **SISO** as “sigh-soh” and **MIMO** as “my-moh.”
- Say **ULPs** as “ulps,” or spell it out as “units in the last place” if that feels more natural.
- Pause briefly after the three headline numbers: **18.89×**, **12.88 GB**, and **92.5×**.
- The 128K reference memory figure is analytically determined from the required tensor shape;
  the reference itself was deliberately not run at that size.
- The 6.1-million-token value is only an operator-level memory projection for the measured scan
  shape. Do not call it a demonstrated Mamba-3 context window.
- MIMO's 31–56× ratios against the PyTorch recurrence are intentionally omitted because the
  reference baseline is pathological. The honest measured result is that the current scalar
  MIMO implementation is slower than SISO.
- Use “to the best of our knowledge” before the novelty claims. Never say “first Mamba-3 on CPU”
  or “first Mamba-3 in Rust”; prior standalone CPU implementations exist.
- Before recording, update the deck's visible repository address from
  `github.com/AdityaP9116/ARMHackathon` to `github.com/AdityaP9116/Arm-Scan`.

## Verification sources

- [`../../README.md`](../../README.md) — public claims, Graviton measurements, and limitations
- [`../project/STATUS.md`](../project/STATUS.md) — current three-path status
- [`../../bench/results/m3-lm-siso-graviton.json`](../../bench/results/m3-lm-siso-graviton.json) — SISO model timings
- [`../../bench/results/m3-lm-mimo-graviton.json`](../../bench/results/m3-lm-mimo-graviton.json) — MIMO timings
- [`../../bench/results/m3-2d-graviton.json`](../../bench/results/m3-2d-graviton.json) — 2D timings
- [`../archive/SPIKE_FINDINGS.md`](../archive/SPIKE_FINDINGS.md) — measured 128K long-context run
- [Official `state-spaces/mamba` repository](https://github.com/state-spaces/mamba) — current Mamba-3 GPU implementation and installation requirements
- [VNCT paper](https://arxiv.org/abs/2607.03589) — published non-causal 2D Mamba-3 architecture used to bound the 2D claims
