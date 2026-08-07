# Benchmarks

Three reproducible tiers (INTEGRATION_PLAN.md Phase 6). All report medians
after warmup and print full host/environment info.

| Tier | Command | What it measures |
|---|---|---|
| Kernel ladder | `cargo bench` (in `kernel/`) | scalar_seq → neon_seq → neon_par, pure Rust, criterion |
| Op level | `python bench/bench_op.py` | kernel vs PyTorch eager vs **torch.compile** on the isolated scan, plan shapes |
| End to end | `python bench/bench_e2e.py` | HF mamba-130m `generate()`: prefill latency, decode tok/s, total — patched vs unpatched, token-identical output asserted |
| SS2D (2D cross-scan) | `python bench/bench_ss2d.py` | The diffusion workload's own shapes (384×320 @ 96ch, 192×160 @ 192ch): traversal-pair vs the legacy four-forward-scans formulation (same kernel both sides), the scan-vs-overhead split that gates a fully fused `selective_scan_2d`, and eager / `torch.compile` baselines at reference-comparable grids. `--only` filters cases for time-boxed runs. |
| Diffusion app | `python bench/bench_diffusion.py` | The workload the pitch is about: **per-NFE denoiser latency** at production grids, peak RSS, projected reconstruction time at NFE ∈ {18, 69, 256}, and **$/reconstruction** given `--usd-per-hour`. Latency is prior-independent (an untrained net of the same shape costs the same to evaluate), so it runs before a prior exists; `--checkpoint` adds PSNR/SSIM/NMSE quality rows. |

Reconstruction time is **projected** as `per_nfe × NFE`, not measured — at 384×320 a single
denoiser call is seconds, so NFE=256 is most of an hour and would tell us nothing that the
multiplication does not. The sampler is a fixed number of denoiser calls plus negligible FFT
work, which is what makes that linear.

The op-level `ref_compile` baseline is the fair fight: `torch.compile`
unrolls the sequential recurrence into an L-step graph (it cannot
restructure the scan), so its compile time explodes with L —
`--compile-max-len` (default 512) skips longer shapes and that limitation
is itself part of the result.

CI runs `bench_op.py --quick` on every push (ubuntu-24.04-arm, results in
the commit comment). Treat CI numbers as provisional — shared runners are
noisy; headline numbers come from dedicated instances:

## Headline-number runbook (Oracle Ampere A1 / AWS Graviton)

Provision an aarch64 Ubuntu 22.04/24.04 instance (Oracle:
VM.Standard.A1.Flex, 4 OCPU / 24 GB — Always Free tier), then:

```bash
git clone https://github.com/AdityaP9116/ARMHackathon && cd ARMHackathon
bash bench/setup_ampere.sh          # one-time: apt deps, rustup, venv, build
bash bench/run_baseline.sh ampere-a1   # the full tagged baseline suite
```

`run_baseline.sh` gates on correctness first (cargo test + FFI golden
check), then runs: the criterion ladder, all four op suites (basic +
len/dim/batch sweeps), the RAYON_NUM_THREADS scaling loop, and the e2e
prompt-length sweep — writing tagged JSONs to `bench/results/` and
regenerating `bench/results/RESULTS.md`. `DRY_RUN=1` prints every command
first; `SKIP_*` toggles select subsets (see the script header). Copy the
results directory back (or commit it deliberately) when done.

The same script is the reusable harness for any future host: pass a
different tag (`graviton-c7g`, `ci-arm64`, …) and re-run
`render_results.py` — results group by tag automatically.

Mamba-3: `bench_mamba3.py` runs the Mamba-3 SISO scan against the pure-PyTorch
reference (eager and `torch.compile`) at the published checkpoints' shapes.
Correctness gates speed — every shape is diffed against the reference before its
timing is reported. The scalar -> blocked -> NEON ablation ladder is **not** in
it yet: that needs a backend selector plumbed through the torch op, and a flag
that silently benchmarked one backend three times would be worse than none.

SS2D: `bench_ss2d.py` measures the unfused 2D path at the real diffusion shapes and prints the fused-kernel go/no-go (overhead % vs the 15% rule); results land beside the other tagged JSONs.
