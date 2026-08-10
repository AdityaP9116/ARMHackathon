# 4. What a kernel is, and why anyone hand-writes one

## The word "kernel"

Overloaded term. Three unrelated meanings:

1. **OS kernel** — the core of Linux. *Not this.*
2. **Convolution kernel** — the small filter in a CNN. *Not this.*
3. **Compute kernel** — a routine that performs one operation, hand-optimized for
   particular hardware. **This one.**

When someone says "the CUDA kernel for attention," they mean a function, usually
written in a low-level language, that does one job (attention) as fast as a
specific chip allows.

Our project ships a **selective scan kernel**: one function that computes the
recurrence from [§2](02_mamba_and_selective_scan.md), written in Rust, optimized
for Arm CPUs.

## Why not just write it in PyTorch?

You can, and PyTorch will run it. It will be slow, for reasons that are about
*plumbing*, not arithmetic.

### Dispatch overhead

Every PyTorch operation — a multiply, an add — is a full function call through
several layers: Python, the dispatcher (which decides CPU vs GPU, which dtype),
then the actual computation. That overhead is fine when the operation is a
1024×1024 matrix multiply that takes milliseconds. It is ruinous when the
operation is multiplying two small numbers and you do it 100,000 times.

The scan is the ruinous case: a long sequence of tiny dependent operations.

### Materialising intermediates

To use PyTorch efficiently you batch operations across the whole tensor. So you
compute `exp(Δ·A)` for *every timestep at once* — allocating a `(batch, dim, len,
state)` tensor. As noted in [§2](02_mamba_and_selective_scan.md), at L=131,072
that is **12.88 GB**, written to memory and then read back once.

A hand-written kernel computes a chunk, uses it, and discards it — the values
stay in registers and cache and never touch main memory.

### No control over memory layout

Performance on modern hardware is dominated by **cache behaviour**. A CPU reads
memory in 64-byte **cache lines**; if the value you need is not in cache, you
wait ~200 cycles for main memory — enough time for ~800 arithmetic operations.

A kernel author controls the traversal order so that data is used while it is
still in cache. From PyTorch you have almost no such control.

## SIMD: the reason hand-written kernels win

**SIMD** = Single Instruction, Multiple Data. One instruction operating on
several values at once.

A modern CPU has wide registers — on Arm, **NEON** provides 128-bit registers.
A 32-bit float is 4 bytes, so **one NEON register holds 4 floats**, and one NEON
instruction does 4 multiplications simultaneously.

```
Scalar:  a[0]*b[0]      → 1 result  per instruction
NEON:    a[0..4]*b[0..4] → 4 results per instruction
```

Up to **4× throughput**, for free, if your data is laid out so that four
consecutive values can be loaded and processed together.

That last condition is the entire craft. Getting 4× requires the values you want
to process in parallel to be **adjacent in memory** and **independent of each
other**. Recall from [§2](02_mamba_and_selective_scan.md): time is sequential but
the **state dimension is independent**. That is precisely the axis we vectorize
along — see [§6](06_inside_our_kernel.md).

### The instruction set landscape

| Architecture | SIMD |
|---|---|
| x86 (Intel/AMD) | SSE, AVX2 (256-bit), AVX-512 (512-bit) |
| **Arm (Graviton, Apple Silicon, phones)** | **NEON (128-bit)**, SVE/SVE2 (variable width) |

The instructions are **not portable**. NEON code does not run on x86 and vice
versa. This is why our Rust has `#[cfg(target_arch = "aarch64")]` around the NEON
modules, and why **an x86 machine does not compile or execute that code at all** —
a fact that matters a great deal for our benchmarking, since it means every
Mamba-3 measurement taken on x86 is measuring the *portable* path, not the NEON
one.

Our profiling found SVE2 present on the Neoverse-N2 CI runner. We do not use it
yet; NEON is the shipped path.

## Why Rust rather than C

C is traditional for kernels. Rust gives:

- **Memory safety by default** — bounds checks, no use-after-free
- **The same generated code** — both compile through LLVM; a Rust loop and a C
  loop typically produce identical assembly
- **`unsafe` blocks where you need them** — SIMD intrinsics require `unsafe`, but
  it is *localized and visible* rather than ambient
- **A real build and test system** — `cargo test` runs our whole correctness suite

Our rule, from `CLAUDE.md`: all raw-pointer handling lives in the FFI crate
(`arm-scan-ffi`); `unsafe` in the core crate is confined to isolated NEON blocks
with a `SAFETY` comment explaining why it is sound.

## How a Rust kernel gets called from Python

Four layers, each with a job:

```
   Python / PyTorch
        │  torch.library custom op  ── python/arm_scan/op.py
        ▼
   ctypes loader                    ── python/arm_scan/_ffi.py
        │  C ABI (plain pointers and integers)
        ▼
   arm-scan-ffi (cdylib)            ── the only place raw pointers live
        │  safe Rust types
        ▼
   arm-scan-core                    ── scalar / tiled / NEON kernels
```

1. **`arm-scan-core`** — the actual maths. Safe Rust, no pointers, testable.
2. **`arm-scan-ffi`** — compiles to a **cdylib** (a `.so`/`.dylib`/`.dll`) exposing
   a **C ABI**: the calling convention nearly every language can speak. It does
   null checks, overflow-checked size arithmetic, and `catch_unwind` so a Rust
   panic never unwinds into Python (which would be undefined behaviour).
3. **`_ffi.py`** — loads the shared library with `ctypes`.
4. **`op.py`** — registers the function as a **`torch.library` custom op**, so
   PyTorch treats it as a first-class operation. A **fake kernel** (a shape-only
   stub) is registered alongside it so `torch.compile` can reason about shapes
   without executing.

The payoff of that last layer is `arm_scan.patch()`: it monkey-patches Hugging
Face's Mamba implementation to call our kernel. An existing model gets faster
**with no code changes** — which is the developer-experience argument, worth 15%
of the judging rubric.

## The FFI's sharpest edge

Worth stating explicitly because it has bitten this project:

**The C ABI cannot validate buffer lengths.** It receives a pointer and a count.
If the count is wrong, the kernel reads memory it does not own — not an error, a
silent out-of-bounds read that surfaces as `NaN` much later, far from the cause.

So **the Python layer owns shape correctness** (`_check_shapes` in
`python/arm_scan/mamba3.py`). This is a deliberate division of responsibility,
not an oversight.

---

**Next:** [Why CPU, why Arm](05_why_cpu_on_arm.md)
