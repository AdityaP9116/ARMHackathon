# Benchmarks

Three reproducible tiers (INTEGRATION_PLAN.md Phase 6). All report medians
after warmup and print full host/environment info.

| Tier | Command | What it measures |
|---|---|---|
| Kernel ladder | `cargo bench` (in `kernel/`) | scalar_seq → neon_seq → neon_par, pure Rust, criterion |
| Op level | `python bench/bench_op.py` | kernel vs PyTorch eager vs **torch.compile** on the isolated scan, plan shapes |
| End to end | `python bench/bench_e2e.py` | HF mamba-130m `generate()`: prefill latency, decode tok/s, total — patched vs unpatched, token-identical output asserted |

The op-level `ref_compile` baseline is the fair fight: `torch.compile`
unrolls the sequential recurrence into an L-step graph (it cannot
restructure the scan), so its compile time explodes with L —
`--compile-max-len` (default 512) skips longer shapes and that limitation
is itself part of the result.

CI runs `bench_op.py --quick` on every push (ubuntu-24.04-arm, results in
the commit comment). Treat CI numbers as provisional — shared runners are
noisy; headline numbers come from dedicated instances:

## Headline-number runbook (Oracle Ampere / AWS Graviton)

```bash
# Ubuntu 22.04+ aarch64 instance, e.g. Ampere A1 4 OCPU / Graviton c7g.2xlarge
sudo apt-get update && sudo apt-get install -y build-essential python3-venv
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

git clone https://github.com/AdityaP9116/ARMHackathon && cd ARMHackathon
(cd kernel && cargo build --release -p arm-scan-ffi && cargo bench)

python3 -m venv env && source env/bin/activate
pip install numpy torch transformers

python bench/bench_op.py  --json bench/results/op_$(hostname).json
python bench/bench_e2e.py --prompt-tokens 512 --new-tokens 64 --reps 5 \
    --json bench/results/e2e_$(hostname).json
```

Report the instance type, core count, and torch version alongside every
number (the scripts capture them in the JSON).
