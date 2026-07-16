#!/usr/bin/env bash
# One-time setup for a fresh Arm Linux instance (Oracle Ampere A1,
# Graviton, etc. — Ubuntu 22.04/24.04 assumed).
#
# Usage (on the instance):
#   git clone https://github.com/AdityaP9116/ARMHackathon
#   cd ARMHackathon && bash bench/setup_ampere.sh
#
# Afterwards run the baseline:   bash bench/run_baseline.sh ampere-a1
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== host =="
uname -a
nproc
grep -m1 -E "model name|Model" /proc/cpuinfo || true

echo "== system packages =="
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential python3-venv python3-pip git curl
# perf for the profiling tier (PROFILING.md / bench/profile/perf_ampere.sh).
# Oracle images run the -oracle kernel flavor, which linux-tools-generic
# does not cover — try the exact kernel package first, best-effort.
sudo apt-get install -y -qq linux-tools-common \
    "linux-tools-$(uname -r)" 2>/dev/null \
    || sudo apt-get install -y -qq linux-tools-common linux-tools-generic \
    || echo "WARN: perf not installed; profiling tier will be unavailable"
command -v perf >/dev/null && perf --version || true

echo "== rust toolchain =="
if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"
rustc --version

echo "== python env =="
python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet numpy torch transformers

echo "== build kernel =="
(cd kernel && cargo build --release -p arm-scan-ffi)

echo
echo "setup complete. next:"
echo "  source .venv/bin/activate && source ~/.cargo/env"
echo "  bash bench/run_baseline.sh ampere-a1"
