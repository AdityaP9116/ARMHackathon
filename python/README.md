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

Two-direction (bidirectional / non-causal models), one call that computes the
shared discretize+`exp` pass once instead of twice:
`arm_scan.bidirectional.bidirectional_scan(..., merge="sum"|"none")`

**Mamba-3** (SISO, the `state-spaces/mamba3-siso-*` checkpoints), whose official
kernels are Triton/TileLang/CuTe and have no CPU path at all:

```python
out = arm_scan.mamba3_scan(q, k, v, adt, dt, trap, q_bias, k_bias,
                           angles=angles, D=D, z=z)      # (b, l, h, dv)
fwd, bwd = arm_scan.mamba3_scan_pair(...)                # both traversals
```

`trap` is **pre-sigmoid** (the kernel applies it, matching upstream) and
`angles` are **raw** — `arm_scan.angles_to_cos_sin` runs the accumulation
pre-pass, mirroring upstream's split between `angle_dt_fwd` and
`mamba3_siso_fwd`. `mamba3_scan_pair` is the seam the bidirectional and 2D
cross-scan topologies build on: both are traversal orders over one primitive.

2D cross-scan (VMamba-style SS2D) over a `(B, D, H, W)` token grid — the four
directions run as two traversal-order pairs on that same fused kernel:

```python
from arm_scan.ss2d import ss2d_scan, use_arm_scan
out = ss2d_scan(u, delta, A, B, C, D=D, merge="sum")   # (b, d, h, w)
use_arm_scan(model)   # swap every SS2D block in a model onto the kernel
```

`merge="none"` returns the four direction planes unmerged, for models with a
learned combine — and is the form the 2D goldens check, per direction, before
any merge.

See the repository root for the kernel, benchmarks, and correctness
methodology: https://github.com/AdityaP9116/Arm-Scan
