# Running the Arm selective-scan kernels

This is the operational runbook for building, validating, using, and
benchmarking the kernels in this repository. It is written for a fresh machine
and uses commands that exist in the current tree.

The shortest correct path on an Arm64 Ubuntu machine is:

```bash
git clone https://github.com/AdityaP9116/Arm-Scan
cd Arm-Scan
bash bench/setup_ampere.sh
source .venv/bin/activate
source "$HOME/.cargo/env"
python -m pip install -e ./python
make validate
make test-mamba3
```

Do not benchmark until both validation commands pass.

---

## 1. What this repository builds

The project contains two native Rust kernels behind one C ABI and one Python
package:

| Kernel | Public Python entry point | Intended workload |
|---|---|---|
| Mamba-1 selective scan | `arm_scan.selective_scan(...)` | Existing Hugging Face Mamba checkpoints |
| Mamba-3 SISO/MIMO scan | `arm_scan.mamba3_scan(...)` / `arm_scan.mamba3_mimo_scan(...)` | Published Mamba-3 checkpoints and new topologies |

The native library is built as:

```text
kernel/target/release/libarm_scan_ffi.so       # Linux
kernel/target/release/libarm_scan_ffi.dylib    # macOS
kernel/target/release/arm_scan_ffi.dll          # Windows
```

On `aarch64`, `backend="auto"` selects the Arm implementation. On a non-Arm
host, the code can still be built and checked through the scalar backend, but
that is not evidence of NEON performance.

### Important runtime constraints

- The PyTorch custom operations are **inference-only**. No autograd formula is
  registered.
- Direct PyTorch inputs **must already be on CPU**. The Python boundary makes
  them contiguous and converts their dtype to `float32`; it does not move GPU
  tensors back to the CPU.
- The Mamba-3 checkpoint wrapper can load BF16 weights, but the selective scan
  itself executes through the CPU FP32 kernel.
- `RAYON_NUM_THREADS` controls native-kernel parallelism. Set it **before the
  Python process starts**.
- Mamba-3 MIMO is implemented and correctness-gated, but its current compute
  path is scalar. Do not present its timing as a tuned NEON result.

---

## 2. Choose the right machine

### Recommended: Arm64 Linux

Use Ubuntu 22.04 or 24.04 on one of:

- AWS Graviton (`c8g`, `c7g`)
- Oracle Ampere A1
- an Arm64 Linux workstation
- Raspberry Pi 5 with a 64-bit operating system

Confirm the architecture before building:

```bash
uname -m
nproc
lscpu | grep -Ei 'model name|architecture'
```

For an Arm run, `uname -m` must print `aarch64` or `arm64`. If it prints
`x86_64`, the native library will not exercise NEON.

### Apple Silicon

Apple Silicon is supported by the macOS Arm CI job. Install Xcode command-line
tools, Rust, and Python, then follow the manual build in section 4:

```bash
xcode-select --install
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### x86 Linux, Windows, or WSL2

These are useful for scalar correctness and documentation work, not for Arm
performance claims. Ordinary WSL2 on an Intel/AMD Windows machine is still
`x86_64`. Use the arm64 CI jobs or a real Arm machine for NEON validation.

For first-time AWS setup, follow
[`bench/AWS_FIRST_TIME.md`](bench/AWS_FIRST_TIME.md). For the complete measured
benchmark session, use
[`bench/GRAVITON_SESSION.md`](bench/GRAVITON_SESSION.md).

---

## 3. Automated setup on Arm Ubuntu

The repository provides a one-time provisioning script for Arm Linux:

```bash
git clone https://github.com/AdityaP9116/Arm-Scan
cd Arm-Scan
bash bench/setup_ampere.sh
```

That script:

1. installs the compiler, Git, Python, and optional `perf` packages;
2. installs Rust with `rustup` when necessary;
3. creates `.venv` in the repository;
4. installs NumPy, PyTorch, and Transformers;
5. builds `arm-scan-ffi` in release mode.

Activate both environments after setup and whenever you reconnect:

```bash
cd ~/Arm-Scan
source .venv/bin/activate
source "$HOME/.cargo/env"
```

Install the Python package in editable mode for use from your own scripts. The
repository's tests add `python/` to their import path themselves, but external
applications do not:

```bash
python -m pip install -e ./python
```

Verify the environment:

```bash
python --version
rustc --version
cargo --version
python -c "import torch; print('torch', torch.__version__, 'threads', torch.get_num_threads())"
```

---

## 4. Manual installation

Use this section when you do not want to run `bench/setup_ampere.sh`.

### 4.1 Install system packages

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y build-essential curl git pkg-config libssl-dev \
    python3-pip python3-venv
```

### 4.2 Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal
source "$HOME/.cargo/env"
rustup component add rustfmt clippy
```

### 4.3 Clone and create the Python environment

```bash
git clone https://github.com/AdityaP9116/Arm-Scan
cd Arm-Scan
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-dev.txt
python -m pip install -e ./python
```

`requirements-dev.txt` installs NumPy, PyTorch, Transformers, and Matplotlib.
If only the NumPy C-ABI surface is needed, use the smaller environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -e ./python
```

### 4.4 Build the native library

From the repository root:

```bash
make build
```

Equivalent direct Cargo command:

```bash
cd kernel
cargo build --release -p arm-scan-ffi
cd ..
```

Confirm that Python finds the exact artifact just built:

```bash
python -c "import arm_scan; print(arm_scan.lib_path())"
```

On Linux, the printed path should end in
`kernel/target/release/libarm_scan_ffi.so` unless a wheel or `ARM_SCAN_LIB`
override is intentionally being used.

### 4.5 Optional explicit library override

The loader searches in this order:

1. `ARM_SCAN_LIB`;
2. next to the installed `arm_scan` package;
3. `kernel/target/release` and `kernel/target/debug` in a source checkout.

To force one library:

```bash
export ARM_SCAN_LIB="$PWD/kernel/target/release/libarm_scan_ffi.so"
python -c "import arm_scan; print(arm_scan.lib_path())"
```

Use a full path. If Python reports an ABI-version mismatch, rebuild the native
library; do not bypass the ABI check.

---

## 5. Validate before running benchmarks

Correctness gates performance in this project. A benchmark from a failing
build is invalid.

### 5.1 Main kernel and application validation

```bash
make validate
```

This performs:

- release build of the C ABI library;
- Rust unit/property tests;
- one-dimensional goldens through the real C ABI;
- independent one- and two-dimensional golden re-derivations;
- SS2D traversal-pair parity;
- MRI application integration gates;
- a quick op-level benchmark after correctness passes.

Success means the command exits with status `0`; no test may print `FAIL`.

### 5.2 All Mamba-3 gates

If `make validate` was not run first, build explicitly:

```bash
make build
make test-mamba3
```

The Mamba-3 target covers:

- official GPU-captured SISO goldens;
- the PyTorch custom operation;
- real captured block inputs and weights;
- 2D cross-scan;
- official GPU-captured MIMO goldens;
- MIMO C-ABI and rank-collapse checks;
- causal and non-causal formulations.

Expected final markers include `MAMBA-3 OP CHECK: PASS` and
`SS2D MAMBA-3 CHECK: PASS`, followed by a zero exit status.

### 5.3 Real 187M model gate

This downloads the published SISO and MIMO checkpoints, approximately 357 MB
each on the first run:

```bash
make test-mamba3-model
```

The test compares CPU logits against committed outputs captured from the
official GPU implementation. Model agreement is judged against the observed
BF16/autotuning floor, not against an artificial requirement of bit identity.

### 5.4 Confirm NEON directly

The PyTorch operation selects `auto`. To prove the NEON backend itself is
available, use the NumPy API and force `backend="neon"`:

```bash
python - <<'PY'
import numpy as np
import arm_scan

rng = np.random.default_rng(0)
B, D, L, N = 1, 8, 32, 16
u = rng.standard_normal((B, D, L), dtype=np.float32)
delta = rng.random((B, D, L), dtype=np.float32) * np.float32(0.1)
A = -(rng.random((D, N), dtype=np.float32) + np.float32(0.5))
Bm = rng.standard_normal((B, N, L), dtype=np.float32)
Cm = rng.standard_normal((B, N, L), dtype=np.float32)

y = arm_scan.selective_scan_numpy(
    u, delta, A, Bm, Cm,
    backend="neon", threading="rayon",
)
print("library:", arm_scan.lib_path())
print("output:", y.shape, y.dtype, "finite:", np.isfinite(y).all())
PY
```

On Arm64 this should complete and print an output shape of `(1, 8, 32)`. On a
non-Arm host, forcing NEON should report that the backend is unavailable; that
is expected and prevents an x86 run from being mislabeled as Arm.

---

## 6. Run the three Mamba-3 paths

Set the native and PyTorch thread counts before launching a benchmark process:

```bash
export RAYON_NUM_THREADS="$(nproc)"
export OMP_NUM_THREADS="$(nproc)"
export MKL_NUM_THREADS="$(nproc)"
T="$(nproc)"
mkdir -p bench/results
```

The `--threads` option sets PyTorch's CPU thread count. The
`RAYON_NUM_THREADS` environment variable controls the Rust pool.

### Path 1A: Mamba-3 SISO

First run a short synthetic-shape benchmark:

```bash
python bench/bench_mamba3.py --quick --threads "$T"
```

Run the full kernel-shape sweep and save machine-readable results:

```bash
python bench/bench_mamba3.py \
    --threads "$T" \
    --tag "$(hostname)" \
    --json bench/results/m3-kernel-"$(hostname)".json
```

Run the published 187M SISO model end to end:

```bash
python bench/bench_mamba3_lm.py \
    --model state-spaces/mamba3-siso-187m \
    --quick \
    --threads "$T"
```

For the complete length sweep:

```bash
python bench/bench_mamba3_lm.py \
    --model state-spaces/mamba3-siso-187m \
    --threads "$T" \
    --json bench/results/m3-lm-siso-"$(hostname)".json
```

Every shape is checked against the FP32 PyTorch recurrence before it is timed.
`torch.compile` is deliberately skipped at longer lengths because compiling an
unrolled recurrence becomes the dominant cost.

### Path 1B: long-context memory behavior

Install the one optional dependency used for peak-RSS sampling:

```bash
python -m pip install psutil
```

Then run:

```bash
python bench/bench_longctx.py
```

This is an operator-level Mamba-1-shaped selective-scan experiment. It tests
lengths through 131,072 and reports:

- native-kernel wall time;
- measured process RSS rise;
- PyTorch-reference time where the allocation is safe;
- the theoretical size of the reference's `dA + dBu` intermediates.

Do not relabel this as a measured full-model Mamba-3 context window. It proves
the bounded-scratch scan design and supports a memory-capacity projection at
that operator shape.

### Path 2: Mamba-3 MIMO rank 4

Run the MIMO correctness gates first:

```bash
python tests/verify_golden_mamba3_mimo.py
python tests/check_mamba3_mimo_op.py
python tests/check_mamba3_model.py --dir tests/golden/mamba3_mimo
```

Then run the real published MIMO model:

```bash
python bench/bench_mamba3_lm.py \
    --model state-spaces/mamba3-mimo-187m \
    --quick \
    --threads "$T"
```

Full run with JSON output:

```bash
python bench/bench_mamba3_lm.py \
    --model state-spaces/mamba3-mimo-187m \
    --threads "$T" \
    --json bench/results/m3-lm-mimo-"$(hostname)".json
```

Interpret this correctly: the MIMO path proves architecture coverage and
correctness. Its current scalar implementation is not a tuned MIMO NEON
kernel, so its runtime is a floor and a description of remaining work—not a
MIMO speedup claim.

### Path 3: Mamba-3 2D cross-scan

Quick correctness-gated benchmark:

```bash
python bench/bench_ss2d_mamba3.py --quick --threads "$T"
```

Full benchmark at 14x14, 28x28, and 56x56 grids:

```bash
python bench/bench_ss2d_mamba3.py \
    --threads "$T" \
    --json bench/results/m3-2d-"$(hostname)".json
```

Run the causal/non-causal comparison:

```bash
python bench/bench_mamba3_noncausal.py \
    --threads "$T" \
    --json bench/results/m3-noncausal-"$(hostname)".json
```

The 2D benchmark uses random, shape-correct operator inputs because no
published 2D Mamba-3 checkpoint exists. It proves numerical correctness and
throughput; it does not make an image-task accuracy claim.

### Generate the explanatory figures

```bash
make viz
```

Outputs:

```text
bench/results/scan_1d.png
bench/results/mimo_rank.png
bench/results/ss2d_scans.png
```

---

## 7. Call the kernels from Python

### 7.1 Mamba-1 direct PyTorch operation

```python
import torch
import arm_scan

B, D, L, N = 1, 8, 32, 16
g = torch.Generator().manual_seed(0)
u = torch.randn(B, D, L, generator=g)
delta = torch.rand(B, D, L, generator=g) * 0.1
A = -(torch.rand(D, N, generator=g) + 0.5)
Bm = torch.randn(B, N, L, generator=g)
Cm = torch.randn(B, N, L, generator=g)

out, final_state = arm_scan.selective_scan(
    u, delta, A, Bm, Cm,
    return_last_state=True,
)
print(out.shape)          # (1, 8, 32)
print(final_state.shape)  # (1, 8, 16)
```

Tensor contract:

| Tensor | Shape |
|---|---|
| `u`, `delta`, optional `z` | `(batch, dim, length)` |
| `A` | `(dim, state)` |
| `B`, `C` | `(batch, state, length)` or `(batch, groups, state, length)` |
| optional `D`, `delta_bias` | `(dim,)` |
| optional `initial_state` | `(batch, dim, state)` |

For streaming, pass the previous `final_state` back as `initial_state`. Preserve
the original sequence order and use `reverse=False` for a resumable state.

### 7.2 Patch a Hugging Face Mamba-1 model

```python
import torch
import arm_scan
from transformers import MambaForCausalLM

patched = arm_scan.patch()
print("patched:", patched)

model = MambaForCausalLM.from_pretrained(
    "state-spaces/mamba-130m-hf",
    torch_dtype=torch.float32,
).eval().cpu()

input_ids = torch.randint(0, model.config.vocab_size, (1, 128))
with torch.no_grad():
    output_ids = model.generate(input_ids, max_new_tokens=8)

print(output_ids.shape)
print(arm_scan.stats())
```

`stats()` must show `patched: True`, `fast_calls > 0`, and
`kernel_calls > 0`. Decode steps may use the upstream single-token update;
the optimization target here is prefill.

### 7.3 Mamba-3 SISO direct operation

```python
import torch
import arm_scan

B, L, H, DV, DQK = 1, 64, 4, 16, 32
g = torch.Generator().manual_seed(0)
randn = lambda *s: torch.randn(*s, generator=g)

q = randn(B, L, 1, DQK)
k = randn(B, L, 1, DQK)
v = randn(B, L, H, DV)
dt = torch.nn.functional.softplus(randn(B, H, L) - 4.5)
adt = -torch.exp(randn(B, H, L) * 0.2) * dt
trap = randn(B, H, L)          # pre-sigmoid gate
q_bias = randn(H, DQK)
k_bias = randn(H, DQK)
angles = randn(B, L, H, DQK // 4)
D = randn(H)
z = randn(B, L, H, DV)

with torch.no_grad():
    out = arm_scan.mamba3_scan(
        q, k, v, adt, dt, trap, q_bias, k_bias,
        angles=angles, D=D, z=z,
    )

print(out.shape)  # (1, 64, 4, 16)
```

Mamba-3 uses time-major values `(batch, length, heads, dv)` and a matrix state
per head. Do not pass Mamba-1 tensors to `mamba3_scan`; the two ABIs are
separate by design.

For authoritative examples of MIMO and grid-shaped 2D inputs, use:

- `tests/check_mamba3_mimo_op.py`
- `tests/check_ss2d_mamba3.py`
- `bench/bench_ss2d_mamba3.py`

These examples are correctness-gated and stay synchronized with the API.

---

## 8. Produce a full Arm benchmark package

Run on an otherwise idle Arm machine. Check `uptime` first; a contended cloud
instance produces invalid comparisons.

```bash
uptime
mkdir -p bench/results
```

The reusable Mamba-1/application baseline is:

```bash
bash bench/run_baseline.sh "$(hostname)" 2>&1 | tee session.log
```

Mamba-3 is intentionally separate from that older baseline script. Run all
three Mamba-3 paths explicitly:

```bash
T="$(nproc)"
export RAYON_NUM_THREADS="$T"

python bench/bench_mamba3.py \
    --threads "$T" --json bench/results/m3-kernel-"$(hostname)".json
python bench/bench_ss2d_mamba3.py \
    --threads "$T" --json bench/results/m3-2d-"$(hostname)".json
python bench/bench_mamba3_noncausal.py \
    --threads "$T" --json bench/results/m3-noncausal-"$(hostname)".json
python bench/bench_mamba3_lm.py \
    --threads "$T" --json bench/results/m3-lm-siso-"$(hostname)".json
python bench/bench_mamba3_lm.py \
    --model state-spaces/mamba3-mimo-187m \
    --threads "$T" --json bench/results/m3-lm-mimo-"$(hostname)".json
```

For thread scaling, start a new process for every value so Rayon initializes a
fresh global pool:

```bash
for T in 1 2 4 8 16 32 64; do
  if [ "$T" -le "$(nproc)" ]; then
    RAYON_NUM_THREADS="$T" python bench/bench_op.py \
      --suite scaling-point --no-compile --reps 7 \
      --torch-threads "$T" \
      --tag "$(hostname)-t$T" \
      --json "bench/results/op_scaling_$(hostname)_t$T.json"
  fi
done
```

Benchmark rules:

1. validation must pass first;
2. record host, architecture, core count, PyTorch version, and commit SHA;
3. warm up before timing;
4. report medians over multiple repetitions;
5. save JSON rather than copying terminal numbers by hand;
6. do not run other compute-heavy work during measurement;
7. label synthetic operators separately from real-checkpoint runs.

---

## 9. Build and test a wheel

Build the native library first, then package it:

```bash
make build
python scripts/build_wheel.py
ls -lh python/dist/
```

Install the wheel in a clean environment:

```bash
python3 -m venv wheel-test-env
source wheel-test-env/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy python/dist/arm_scan-*.whl
python -c "import arm_scan; print(arm_scan.lib_path())"
deactivate
```

Wheels are platform-specific because they contain the native library, but one
`py3-none-<platform>` wheel covers supported Python 3 versions on that
platform.

---

## 10. Troubleshooting

### `arm_scan_ffi library not found`

Build the release library and inspect its path:

```bash
make build
find kernel/target/release -maxdepth 1 -type f -name '*arm_scan_ffi*' -print
python -c "import arm_scan; print(arm_scan.lib_path())"
```

If the Python package is outside the source checkout, install the wheel or set
`ARM_SCAN_LIB` to the full library path.

### ABI-version mismatch

The Python package and Rust library came from different commits. Rebuild:

```bash
cargo clean --manifest-path kernel/Cargo.toml
make build
```

Do not change the expected ABI number to suppress this error.

### `ModuleNotFoundError: arm_scan`

Install the package in editable mode:

```bash
source .venv/bin/activate
python -m pip install -e ./python
```

Repository tests and benchmarks insert `python/` into `sys.path`, but an
independent application should install the package.

### `backend unavailable on this platform`

You forced NEON on a non-Arm host, or the binary was built for the wrong
architecture:

```bash
uname -m
file kernel/target/release/libarm_scan_ffi.so
```

Use `backend="auto"` for portable execution. Use a real `aarch64` machine for
NEON measurements.

### The process uses the wrong number of cores

Set both pools before starting Python:

```bash
RAYON_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python bench/bench_mamba3.py --threads 8 --quick
```

Do not change `RAYON_NUM_THREADS` inside a long-lived process after the first
kernel call; Rayon's global pool may already be initialized.

### `torch.compile` takes a very long time

This is expected for the sequential reference because its graph grows with
sequence length. Use `--quick`, lower the compile limit, or skip compilation:

```bash
python bench/bench_mamba3.py --quick --no-compile
python bench/bench_mamba3_lm.py --quick --no-compile
python bench/bench_ss2d_mamba3.py --quick --no-compile
```

Do not silently remove the compile baseline from published comparisons; label
it as skipped and state why.

### MIMO is slower than SISO

That is the current expected result. MIMO routes through a scalar compute path
and needs a dedicated dense NEON microkernel. Correctness is the current MIMO
deliverable.

### A model download fails

Confirm network access and install the Hugging Face dependencies:

```bash
python -m pip install transformers huggingface_hub safetensors
```

Checkpoint files are cached under `~/.cache/huggingface/hub`. They are not
committed to this repository.

### Gradients or training fail

The native custom operations are inference-only. Use the PyTorch reference for
training and the native kernel for CPU inference/benchmarking. Wrap inference
in `torch.no_grad()` and call `.eval()` on models.

### Results vary between runs

Check for contention, thermal throttling, and mixed thread counts:

```bash
uptime
ps -eo pid,comm,%cpu --sort=-%cpu | head
```

Run warmups, use at least three measured repetitions, and publish medians. A
timing taken while another benchmark or install is running is void.

---

## 11. Safe stopping point

The installation is complete when all of the following are true:

```bash
uname -m                                      # aarch64/arm64 for NEON
python -c "import arm_scan; print(arm_scan.lib_path())"
make validate                                # exits 0
make test-mamba3                             # exits 0
python bench/bench_mamba3.py --quick         # correctness passes before timing
```

At that point the kernel is built, Python is loading the intended library, the
Mamba-1 and Mamba-3 correctness nets pass, and the machine is ready for the
three-path benchmark commands in section 6.
