#!/usr/bin/env bash
# Provision a Linux box for this repo: Rust correctness gates + Mamba-3 golden
# capture on an NVIDIA GPU. Works on native Linux and on WSL2; the one real
# difference between them is handled automatically (see the driver note below).
#
# WHY THIS EXISTS
# ---------------
# Two independent blockers on the Windows host, one fix:
#
#   1. `cargo test` has never run there. The active Rust toolchain is
#      x86_64-pc-windows-msvc but no MSVC linker is installed; the windows-gnu
#      toolchain is present but missing `dlltool`. Our core correctness gate —
#      scalar vs NEON vs threaded parity against the goldens — is unrunnable
#      locally, so we have been leaning entirely on CI for it.
#
#   2. Stage 0 needs `mamba-ssm`, which ships NO Windows wheels for any version
#      and builds from source only. Its SISO kernel is Triton, and official
#      Triton has no Windows wheels either.
#
# Linux solves both, and a third thing that matters more than either: it is what
# CI runs and what Graviton is. A bug reproduced on windows-gnu is not evidence
# about the deployment target.
#
# ON THE CUDA TOOLKIT: you probably do not need it, so this script does not
# install it. PyTorch's cu128 wheels bundle their own CUDA runtime, and Triton
# JIT-compiles through its own bundled LLVM rather than shelling out to nvcc.
# mamba-ssm wants nvcc only for CUDA extensions that the Triton SISO path does
# not use, and the fallback below skips them. If something genuinely demands
# nvcc, install it then — not preemptively.
#
# USAGE:
#   bash tools/setup_linux.sh
set -euo pipefail

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m!!! %s\033[0m\n' "$*"; exit 1; }

if grep -qi microsoft /proc/version 2>/dev/null; then
    ENV_KIND="WSL2"
else
    ENV_KIND="native Linux"
fi
say "Environment: $ENV_KIND"

say "System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential curl git pkg-config libssl-dev \
    python3-pip python3-venv

say "GPU visibility"
if ! command -v nvidia-smi >/dev/null; then
    if [ "$ENV_KIND" = "WSL2" ]; then
        # In WSL2 the GPU is passed through by the Windows driver. Installing a
        # driver inside the guest BREAKS that passthrough — never do it here.
        die "No nvidia-smi. Update the NVIDIA driver on the WINDOWS side (not in here)."
    fi
    die "No nvidia-smi. Install the NVIDIA driver, e.g.:
    sudo ubuntu-drivers install
  A Blackwell card (RTX 5090, sm_120) needs driver r570 or newer.
  Reboot after installing, then re-run this script."
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

say "Rust toolchain"
if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"
rustup component add rustfmt clippy
rustc --version

say "Python environment"
python3 -m venv ~/venv-arm
# shellcheck disable=SC1091
source ~/venv-arm/bin/activate
pip install -q --upgrade pip wheel setuptools ninja packaging
pip install -q numpy

say "PyTorch with CUDA 12.8 (Blackwell / sm_120 needs cu128 or newer)"
pip install -q torch --index-url https://download.pytorch.org/whl/cu128
python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch cannot see the GPU"
print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
      f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
PY

say "Triton (this is the piece that is simply impossible on Windows)"
pip install -q triton
python -c "import triton; print('triton', triton.__version__)"

# Optional for Mamba-3 — the architecture dropped Conv1D — but mamba-ssm's
# imports can still reach for it. Cheap insurance; failure is not fatal.
pip install -q causal-conv1d --no-build-isolation 2>/dev/null || \
    echo "  (causal-conv1d skipped — not required for Mamba-3 SISO)"

say "mamba-ssm (sdist only — expect a compile, 10-30 min)"
if ! pip install mamba-ssm --no-build-isolation; then
    echo "  full build failed; retrying without the CUDA extensions"
    echo "  (the SISO path we capture from is Triton, so these are not needed)"
    MAMBA_SKIP_CUDA_BUILD=TRUE pip install mamba-ssm --no-build-isolation
fi
python -c "import mamba_ssm; print('mamba_ssm', mamba_ssm.__version__)"

say "Rust correctness gates — these have never run on the Windows host"
cd "$(cd "$(dirname "$0")/.." && pwd)"
( cd kernel && cargo test --release 2>&1 | tail -25 )

say "Ready"
cat <<'EOF'
Capture the Mamba-3 ground truth next:

    source ~/venv-arm/bin/activate
    python tools/capture_mamba3_goldens.py --out tests/golden/mamba3

That writes .npz goldens + manifest.json + model_shape.json. Commit them; the
GPU is then never needed again — the Rust kernel is built and validated on CPU
against those files.
EOF
