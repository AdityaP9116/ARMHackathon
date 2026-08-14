# Contributing to Arm Scan

Arm Scan is a correctness-first performance project. A faster result is accepted only after it
reproduces the established reference within the existing numerical gate.

## Development setup

Follow [`RUNNING_THE_KERNEL.md`](RUNNING_THE_KERNEL.md) for platform prerequisites, build steps,
and troubleshooting. The shortest local verification path is:

```bash
make validate
make test-mamba3
```

The first command validates the Mamba-1 kernel and application-level gates. The second runs the
Mamba-3 SISO, MIMO, 2D, and non-causal correctness suites without requiring a GPU.

## Change rules

1. Preserve the scalar implementation as the portable correctness reference.
2. Keep raw pointers inside `kernel/arm-scan-ffi`; isolated NEON `unsafe` blocks require a
   `SAFETY` explanation.
3. Never loosen a tolerance to make a test pass. Diagnose the numerical or indexing error.
4. Gate correctness before collecting performance numbers.
5. Benchmark against eager PyTorch and `torch.compile`, with warm-up, fixed thread counts,
   named hardware, and medians.
6. Do not commit model weights, fastMRI data, signed dataset links, credentials, or local output.
7. Update [`docs/project/STATUS.md`](docs/project/STATUS.md) when a capability or limitation
   materially changes; move superseded plans to `docs/archive/` instead of deleting them.

## Before opening a pull request

```bash
cd kernel
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --release
cd ..
make validate
make test-mamba3
git diff --check
```

If the change affects Arm-only code, also run `make check-cross` on x86 and rely on the native
arm64 CI jobs for final compilation and execution. Performance changes should include the raw
result file, hardware description, command, thread count, and a short interpretation.
