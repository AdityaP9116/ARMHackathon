# Current project status

**Last consolidated:** August 13, 2026

**Submission target:** Arm Create: AI Optimization Challenge 2026, Cloud AI track

Arm Scan is a Rust and Arm NEON implementation of the selective-scan operations used by
Mamba-1 and Mamba-3, exposed to Python as PyTorch custom operations. The implementation is
correctness-gated first, then benchmarked against both eager PyTorch and `torch.compile`.

## Shipped and verified

- Mamba-1 scalar, NEON, threaded, C ABI, PyTorch integration, fused bidirectional scan, and
  SS2D cross-scan.
- Mamba-3 SISO scalar, cache-blocked, NEON, threaded, C ABI, and PyTorch integration.
- Mamba-3 MIMO correctness path, including official TileLang-derived goldens and the published
  `mamba3-mimo-187m` model running end to end on CPU.
- Causal and non-causal 2D Mamba-3 cross-scan, with an independent dense oracle for the
  operator-level comparison.
- Reproducible correctness suites and CI across Linux arm64, macOS arm64, and x86.
- Dedicated AWS Graviton4 measurements on a 64-vCPU Neoverse-V2 instance. Raw results are in
  [`../../bench/results/`](../../bench/results/), and the captured session bundle is preserved
  in [`../../bench/artifacts/`](../../bench/artifacts/).

## The three submission paths

| Path | What it demonstrates | Current result |
|---|---|---|
| 1D SISO | Long-context, unidirectional Mamba-3 and real checkpoint integration | Up to **18.89×** over the PyTorch recurrence at L=1024; chunk scratch stays roughly **17 KB per thread** instead of materializing the reference's sequence-sized intermediate |
| MIMO | Correct support for Mamba-3's rank-*r* state update | Correct through the real C ABI and real model; **not yet a tuned speed result** because this path remains scalar |
| 2D | Four-direction Mamba-3 cross-scan for image-like grids | Up to **92.5×** over the PyTorch recurrence at 56×56; operator correctness only, because no authoritative 2D Mamba-3 implementation or published weights are available |

## Important limitations

- MIMO does not yet have its own cache-blocked NEON kernel. Its current value is architectural
  coverage and correctness, not a headline speedup.
- The 2D path has no published Mamba-3 weights or authoritative upstream 2D oracle. It makes no
  downstream accuracy, segmentation, or classification claim.
- The estimated multi-million-token context ceiling is a memory-accounting projection, not a
  measured model-quality or end-to-end inference result. The measured long-context result is
  L=131,072.
- On Graviton4, the older Mamba-1 SS2D traversal-pair rewrite regressed at 64 cores. That result
  is disclosed; it justifies a future fully fused 2D kernel rather than being hidden.

## Highest-value remaining engineering

1. Build and tune the dedicated MIMO NEON/cache-blocked kernel.
2. Implement the fully fused Mamba-1 2D operator justified by the Graviton overhead result.
3. Extend long-context measurements beyond 128K while recording total resident memory and
   separating kernel scratch from unavoidable input/output storage.
4. Keep submission claims synchronized with the measured evidence in the root README.

For setup and exact commands, use [`RUNNING_THE_KERNEL.md`](../../RUNNING_THE_KERNEL.md).
