# arm-scan

Arm-optimized Mamba selective scan for the PyTorch ecosystem: a Rust + NEON
kernel (chunked scan, rayon-threaded) behind a drop-in patch for Hugging
Face `transformers` Mamba models on CPU.

```python
import arm_scan
arm_scan.patch()      # HF Mamba's CPU path now runs the kernel
# ... use any Mamba model as usual ...
print(arm_scan.stats())  # confirm engagement
```

Direct op (torch): `arm_scan.selective_scan(u, delta, A, B, C, D=..., z=...)`
NumPy-only (no torch): `arm_scan.selective_scan_numpy(...)`

See the repository root for the kernel, benchmarks, and correctness
methodology: https://github.com/AdityaP9116/ARMHackathon
