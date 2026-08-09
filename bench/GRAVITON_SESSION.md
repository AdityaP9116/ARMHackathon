# GRAVITON_SESSION — the run sheet

**Purpose:** turn ~3 rented hours into every Arm number the submission needs, with
zero improvisation on the clock. `ARM_BASELINE.md` explains *why* each measurement
matters and how to read it; **this file is what you actually execute.**

This session is the single highest-value action left in the project. Every performance
figure in the repo today is x86 or a shared 4-core CI runner. Nothing has run on
dedicated Arm hardware. It is an **Arm** contest.

---

## 0. Before you launch (do this on your laptop, not on the clock)

**Days ahead, not hours — this one can block the whole session:**

- [ ] **Check your EC2 vCPU quota.** Service Quotas → EC2 → *"Running On-Demand
      Standard (A, C, D, H, I, M, R, T, Z) instances"*. That number is in **vCPUs**,
      not instances. A `c8g.16xlarge` needs **64**. New accounts are frequently
      capped at 5–32, and an increase is a support request that can take **hours to
      days**. Request it now; it costs nothing if unused.
- [ ] **Know which student account you have.** They are not equivalent:
      *GitHub Student Pack credits* behave like a normal account (fine);
      *AWS Educate* often restricts instance types; **AWS Academy Learner Lab**
      blocks larger instance types outright *and* auto-terminates sessions
      (typically 4 h) — a 3-hour benchmark run is at real risk there.
- [ ] If 64 vCPU is a fight, take **`c8g.8xlarge` (32 vCPU)**. A 1→32 sweep is six
      points and shows the scaling story essentially as well. Do not lose days to a
      quota argument for the last doubling.

**Then:**

- [ ] `main` is pushed and CI is green. If CI is red, fix it first — you do not want
      to discover a build break at $3/hour.
- [ ] Decide the instance. **`c8g.16xlarge`** (Graviton4, Neoverse-V2, 64 vCPU) is the
      recommendation: the **core-scaling curve is the headline chart for a Cloud-track
      entry**, and you cannot draw a scaling curve on 4 cores. `c7g.16xlarge`
      (Graviton3) is a fine cheaper substitute. Check live pricing — on-demand for
      these is roughly $2.30–3.00/hour in us-east-1, so a 3-hour session is
      **≈$7–9**. Well inside the ~$5–20 budget in `CLAUDE.md`.
- [ ] AMI: **Ubuntu 24.04 LTS, arm64**. Storage: **40 GB** — torch, transformers, the
      mamba-130m download **and both 187M Mamba-3 checkpoints (~357 MB each)**.
- [ ] Have this file open, and `ARM_BASELINE.md` §8 for the interpretation table.

> **Set a calendar reminder to terminate the instance.** Stopping is not terminating.
> The most common way this budget gets blown is a forgotten running box.

---

## 1. Bring-up (~20 min, mostly unattended)

```bash
git clone https://github.com/AdityaP9116/ARMHackathon && cd ARMHackathon
bash bench/setup_ampere.sh
```

`setup_ampere.sh` installs build tools, perf, rustup, a `.venv` with
numpy/torch/transformers, and builds the cdylib. Despite the name it is correct for
Graviton — same Ubuntu, same aarch64.

Then confirm the machine is what you are paying for:

```bash
nproc && lscpu | grep -E "Model name|BogoMIPS|Flags" | head -3
```

- [ ] `nproc` reports the core count you paid for (64 on `c8g.16xlarge`)
- [ ] Nothing else is running on the box (`uptime` load ≈ 0)

---

## 2. Correctness gate — **abort point** (~10 min)

```bash
source .venv/bin/activate && source ~/.cargo/env
cd kernel && cargo test --release -- --nocapture 2>&1 | tee ../ct.log ; cd ..
make test-mamba3 2>&1 | tee mamba3.log
```

Three things to find in that output, in order of importance:

- [ ] **`backend Auto resolves to NEON on this host`** — the single most important
      line of the session. Without it you are benchmarking the scalar fallback and
      every number is worthless.
- [ ] every golden case prints `ok`, not `FAIL`
- [ ] `vexpq_f32` / `vsoftplusq_f32` worst error ≈ 1e-6…1e-7 (the vector math is
      accurate on *this* microarchitecture, which is not a given)
- [ ] **all 7 Mamba-3 gates green** (SISO, MIMO, 2D causal + non-causal). These have
      only ever run on CI's shared 4-core runner; this is their first dedicated-Arm
      execution.

> **If this is not green, stop and fix it. A fast wrong kernel is worth nothing, and
> the whole benchmark suite downstream would be measuring a lie.**

---

## 3. The full baseline (~90 min, one command, unattended)

```bash
USD_PER_HOUR=2.90 INSTANCE_TYPE=c8g.16xlarge THREADS_LIST="1 2 4 8 16 32 64" bash bench/run_baseline.sh graviton-c8g 2>&1 | tee session.log
```

Set `USD_PER_HOUR` to what you are **actually paying** (check the console — spot differs from
on-demand); it drives the $/reconstruction table. Add `PRIOR_CKPT=path.pt` if a trained prior
exists, to get the quality rows too.

That single command runs all seven stages. What it covers and roughly how long:

| Stage | What | ~Time |
|---|---|---|
| [1/7] | correctness gate + **2D goldens** + SS2D pair parity at 1/2/8 threads | 10 min |
| [2/7] | criterion ladder: `scalar_seq → neon_seq → neon_par` | 15 min |
| [3/7] | op-level vs eager **and `torch.compile`** (the baseline judges trust) | 20 min |
| [4/7] | **core-scaling sweep 1→64 threads** — the Cloud-track chart | 15 min |
| [5/7] | end-to-end mamba-130m generate (downloads ~300 MB once) | 15 min |
| [6/7] | **SS2D at production grids** + bidirectional | 15 min |
| [7/7] | **diffusion app**: per-NFE latency + $/reconstruction | 10 min |

Set `SKIP_E2E=1` if you fall behind — it is the slowest stage and the least
differentiating. Everything else is load-bearing.

**Live checks while it runs:**

- [ ] `neon_seq / scalar_seq` ≈ **3–4×** → NEON and the vector exp are working.
      Below 2× means the exp or memory layout is weak — note it, don't stop.
- [ ] `neon_par / neon_seq` should approach core count. On 64 cores, **≥ 8×** is
      healthy; **< 4×** is a scaling wall worth reporting honestly.
- [ ] SS2D prints a **P1-7 go/no-go**. On x86 the worst shape sat at 13.8% against a
      15% bar. **Whichever side it lands on here is the answer we publish** — this is
      the decision being re-taken, so record it either way.
- [ ] SS2D `pair-speedup` — x86 said 1.77–1.82×. If Arm shows a *regression*, that is
      the fewer-rayon-rows risk (2B vs 4B per call) materialising; keep
      `_forward_legacy` and publish whichever wins, with the reason.

---

## 3b. Mamba-3 — the headline work, and `run_baseline.sh` does NOT cover it (~35 min)

**Read this before you skip it.** `run_baseline.sh` was written before the Mamba-3
kernel existed and contains no reference to it. Running only §3 gets you Mamba-1,
SS2D and diffusion — and **zero Arm numbers for the work the submission now leads
with**. These four benchmarks are the gap.

All four take `--threads`, `--reps`, `--warmup` and `--json`, so they follow the same
discipline as everything else. Full core count; add `--quick` only if you are behind.

```bash
T=$(nproc)
python bench/bench_mamba3.py            --threads $T --json bench/results/m3-kernel-graviton.json
python bench/bench_ss2d_mamba3.py       --threads $T --json bench/results/m3-2d-graviton.json
python bench/bench_mamba3_noncausal.py  --threads $T --json bench/results/m3-noncausal-graviton.json
python bench/bench_mamba3_lm.py         --threads $T --json bench/results/m3-lm-siso-graviton.json
python bench/bench_mamba3_lm.py --model state-spaces/mamba3-mimo-187m \
                                        --threads $T --json bench/results/m3-lm-mimo-graviton.json
```

And the end-to-end model gate, which downloads both checkpoints (~357 MB each):

```bash
make test-mamba3-model 2>&1 | tee m3-model.log
```

What each one is for, and what to watch:

| Bench | Why it matters | Watch for |
|---|---|---|
| `bench_mamba3.py` | The Mamba-3 kernel ladder. x86 has no NEON path at all, so **this is the first real measurement of the NEON Mamba-3 kernel anywhere** | NEON vs scalar ≈ 3–4×, as Mamba-1 |
| `bench_ss2d_mamba3.py` | 2D Mamba-3 — claim (4), the novelty claim. x86 said 14–38× over the PyTorch recurrence, 1.9× over `torch.compile` | Whether the `torch.compile` margin holds on Arm |
| `bench_mamba3_noncausal.py` | The causal-vs-non-causal comparison — a result **nobody has published for any Mamba generation** | 1D ≈ 2×, 2D ≈ 1×. Rows under 1 ms are flagged dispatch-dominated — do not read those as findings |
| `bench_mamba3_lm.py` (SISO) | The real 187M model end to end. x86: 1.85–3.66× over the recurrence, 1.39–1.60× over `torch.compile` | The `torch.compile` margin is the number a judge trusts |
| `bench_mamba3_lm.py` (MIMO) | **MIMO is the scalar path only** — no blocked or NEON kernel exists | Report as a **floor, not a result.** It measures an unoptimised kernel, and saying so is the honest framing |

> **On MIMO:** do not present its timing as evidence about MIMO's arithmetic-intensity
> advantage on CPU. That argument is still a prediction. What this run gives you is the
> size of the gap an optimised MIMO kernel would have to close — which is a legitimate
> and interesting number, stated as such.

### Optional, if the session is running ahead: sweep `TILE`

`TILE = 32` in `kernel/arm-scan-core/src/mamba3/tiled.rs:50` is a **placeholder that
has never been swept**, and it can only be tuned here: the blocking exists to fit NEON
registers and Arm's L1, and **x86 does not execute that path at all**. It is a `const`,
so each value needs a rebuild (~1–2 min each).

```bash
for t in 16 24 32 48 64; do
  sed -i "s/pub(crate) const TILE: usize = .*/pub(crate) const TILE: usize = $t;/" \
      kernel/arm-scan-core/src/mamba3/tiled.rs
  (cd kernel && cargo build --release -p arm-scan-ffi -q)
  echo "== TILE=$t =="; python bench/bench_mamba3.py --quick --threads $(nproc)
done
git checkout kernel/arm-scan-core/src/mamba3/tiled.rs   # restore the default
```

Treat any win as **provisional until re-gated** — change the constant, then re-run
`make test-mamba3` before believing a number. Correctness gates speed, always.

---

## 4. Profiling — the strengths/weaknesses evidence (~20 min)

```bash
bash bench/profile/run_profile.sh graviton-c8g          # no root needed
sudo sysctl kernel.perf_event_paranoid=1
sudo bash bench/profile/perf_ampere.sh graviton-c8g     # hardware counters
```

This produces the phase breakdown (transpose / discretize / exp / recurrence /
epilogue) and the counter-based verdict. Read it with `ARM_BASELINE.md` §8:

| Signal | Verdict |
|---|---|
| IPC ≈ 2.5–3+ | compute-bound, near the ceiling — **ship it** |
| IPC < 1 + high backend stalls | memory-bound — name bf16 storage / better B/C blocking as the next lever |
| `vexpq_f32` dominates | exp is still the hot spot |
| transpose loop hot | B/C prep costs too much |

On shared cloud VMs the PMU is sometimes not exposed and counters read zero. If so,
`run_profile.sh`'s software sampling still shows *where* time goes — say so in the
writeup rather than quoting empty counters.

### Also run Arm's own profiler (~15 min, recommended)

**Arm Streamline CLI Tools** (part of Arm Performance Studio, free of charge) runs natively
on a Neoverse server and implements Arm's **top-down methodology** — it classifies cycles
rather than leaving us to infer compute-vs-memory bound from IPC by eye, attributes metrics
per function, and supports Rust. It is also *Arm's* tool on an *Arm* contest, which is worth
something to the judges by itself.

Install into a **separate venv** so its dependencies cannot perturb the benchmark
environment:

```bash
wget https://artifacts.tools.arm.com/arm-performance-studio/Streamline_CLI_Tools/get-streamline-cli.py
python3 -m venv sl-venv && source ./sl-venv/bin/activate
python3 get-streamline-cli.py install
python3 -m pip install -r ./streamline_cli_tools/bin/requirements.txt
export PATH=$PATH:$PWD/streamline_cli_tools/bin
```

Then find out what this instance actually exposes before profiling anything:

```bash
sysreport
```

- [ ] Record the counter count and whether **SPE** is present. Zero counters → hot-spot
      sampling only; **3** → top-down metrics at a reduced sample rate; **6+** → optimal.
      Without SPE you lose load-data-source metrics and some branch-mispredict accuracy.
      This degradation is why it is safe to run on a virtualized EC2 box at all.

Profile the op-level benchmark and read the top-down breakdown against the same question
`perf` answers: compute-bound (near the ceiling — ship it) or memory-bound (name bf16 storage
and better B/C blocking as the next lever). Command reference:
<https://developer.arm.com/documentation/109847/latest/>.

**Deactivate `sl-venv` and re-activate the project `.venv` before running any further
benchmarks.**

---

## 5. Retrieve and terminate (~10 min)

```bash
tar czf results-graviton.tgz bench/results session.log ct.log mamba3.log \
    m3-model.log bench/profile/out
```

Copy it off the box (from your laptop):

```bash
scp -i <key.pem> ubuntu@<instance-ip>:~/ARMHackathon/results-graviton.tgz .
```

- [ ] Archive is on your laptop and opens
- [ ] **Terminate the instance** (terminate, not stop)
- [ ] Confirm in the console that it is gone

---

## 6. If you only get ~1 hour

**Do §2 (correctness), then Mamba-3, then the scaling curve — in that order.** The
priority inverted when Mamba-3 landed: it is what the submission now leads with, and
it is the part with *no* Arm numbers of any kind. Mamba-1 at least has provisional CI
figures; the Mamba-3 NEON kernel has never run on dedicated Arm hardware at all.

```bash
T=$(nproc)
make test-mamba3
python bench/bench_mamba3.py      --threads $T --json bench/results/m3-kernel-graviton.json
python bench/bench_ss2d_mamba3.py --threads $T --json bench/results/m3-2d-graviton.json
python bench/bench_mamba3_lm.py   --threads $T --json bench/results/m3-lm-siso-graviton.json
SKIP_E2E=1 THREADS_LIST="1 4 16 64" bash bench/run_baseline.sh graviton-c8g 2>&1 | tee session.log
```

That is: the Mamba-3 gates, the Mamba-3 kernel, the 2D novelty claim, the real model
end to end, and a 4-point scaling curve. Everything the headline table and the
scaling story need.

**Cut first if you are still short:** the non-causal bench (the *comparison* is the
result, and it is already measured on x86 — Arm changes the constant, not the finding),
then MIMO (scalar-path floor, not a result), then diffusion.

---

## 7. Afterwards (laptop, ~2 h, no hardware)

1. `python bench/render_results.py` → regenerate `RESULTS.md` from the JSONs.
2. Rewrite the README table from **real Graviton rows**, tagged with instance type,
   core count and torch version. Demote the provisional CI numbers.
3. Write the strengths/weaknesses paragraph from the profile. State the weaknesses
   *with the evidence that proves them* — to an Arm-engineer judge that reads as
   competence, not as an admission.
4. Commit the headline JSONs deliberately (`bench/results/` is gitignored by default).

---

## Abort conditions — stop and think, do not push through

- `backend Auto` resolves to **scalar** → you are measuring the wrong kernel.
- Any golden case **FAILs** → correctness gates speed, always.
- SS2D pair parity fails on NEON but passed on x86 → a genuine NEON-specific bug in
  the fused traversal. Capture the failing case and stop; do not benchmark around it.
- Load average is not ≈0 → **a contended benchmark is void.** An earlier x86 run under
  load reported a 0.50× "regression" that the quiesced re-run put at 1.82×.
