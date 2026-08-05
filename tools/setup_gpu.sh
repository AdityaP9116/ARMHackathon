#!/usr/bin/env bash
# One-time setup for a rented GPU box, for TRAINING ONLY.
#
# Usage (on the instance):
#   git clone https://github.com/AdityaP9116/ARMHackathon
#   cd ARMHackathon && bash tools/setup_gpu.sh
#
# NOTE: this deliberately does NOT install Rust or build the kernel.
# Training runs on the pure-torch reference scan, because arm_scan registers
# no autograd — the kernel cannot be trained through. The kernel is an
# INFERENCE story, measured on Arm CPU (see bench/GRAVITON_SESSION.md). A GPU
# box needs none of it, which keeps this setup to a couple of minutes.
#
# Afterwards:  bash tools/run_training_session.sh --cache data/knee_128.pt
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== host =="
uname -a
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
    || echo "WARNING: nvidia-smi not found — is this actually a GPU box?"

echo
echo "== system packages =="
if command -v apt-get >/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq build-essential python3-venv python3-pip git curl
fi

echo
echo "== python env =="
python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
# CUDA wheels: the default PyPI torch bundles CUDA and is what we want here
# (unlike the Arm CPU side, which pins the cpu index).
.venv/bin/pip install --quiet torch numpy h5py

echo
echo "== verify =="
.venv/bin/python - <<'PY'
import torch
print(f"torch {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
else:
    print("!! No CUDA device. Training would fall back to CPU and take")
    print("!! orders of magnitude longer. Fix this before training.")
PY

echo
echo "== trainer smoke test (2 steps, phantoms, no data needed) =="
.venv/bin/python tools/train_prior.py --steps 2 --res 32 --batch 2 \
    --model-channels 16 --blocks 1 --log-every 1 --eval-every 2 \
    --save-every 2 --no-early-stop --out /tmp/_smoke.pt >/dev/null \
    && echo "trainer OK" || { echo "TRAINER FAILED — stop here"; exit 1; }
rm -f /tmp/_smoke.pt /tmp/_smoke.pt.resume

echo
echo "setup complete. next:"
echo "  source .venv/bin/activate"
echo "  # 1. prepare data (needs the fastMRI download; see tools/prepare_fastmri.py)"
echo "  # 2. bash tools/run_training_session.sh --cache data/knee_128.pt"
echo
echo "Remember to TERMINATE this instance when training finishes."
