# GRAVITON_SESSION — the run sheet

**Purpose:** turn ~3 rented hours into every Arm number the submission needs, with
zero improvisation on the clock. `ARM_BASELINE.md` explains *why* each measurement
matters and how to read it; **this file is what you actually execute.**

This session is the single highest-value action left in the project. Every performance
figure in the repo today is x86 or a shared 4-core CI runner. Nothing has run on
dedicated Arm hardware. It is an **Arm** contest.

---

## 0. Before you launch (do this on your laptop, not on the clock)

- [ ] The branch is pushed and CI is green (`mri-app` passing on `ubuntu-24.04-arm`).
      If CI is red, fix that first — you do not want to discover a build break at
      $3/hour.
- [ ] Decide the instance. **`c8g.16xlarge`** (Graviton4, 64 vCPU) is the
      recommendation: the **core-scaling curve is the headline chart for a Cloud-track
      entry**, and you cannot draw a scaling curve on 4 cores. `c7g.16xlarge`
      (Graviton3) is a fine cheaper substitute. Check live pricing — on-demand for
      these is roughly $2.30–3.00/hour in us-east-1, so a 3-hour session is
      **≈$7–9**. Well inside the ~$5–20 budget in `CLAUDE.md`.
- [ ] AMI: **Ubuntu 24.04 LTS, arm64**. Storage: **30 GB** (torch + transformers +
      the mamba-130m download need room).
- [ ] Have this file open, and `ARM_BASELINE.md` §8 for the interpretation table.

> **Set a calendar reminder to terminate the instance.** Stopping is not terminating.
> The most common way this budget gets blown is a forgotten running box.

---

## 1. Bring-up (~20 min, mostly unattended)

```bash
git clone https://github.com/AdityaP9116/ARMHackathon && cd ARMHackathon
git checkout feature/ss2d-fused-bidirectional     # until it is merged to main
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
```

Three things to find in that output, in order of importance:

- [ ] **`backend Auto resolves to NEON on this host`** — the single most important
      line of the session. Without it you are benchmarking the scalar fallback and
      every number is worthless.
- [ ] every golden case prints `ok`, not `FAIL`
- [ ] `vexpq_f32` / `vsoftplusq_f32` worst error ≈ 1e-6…1e-7 (the vector math is
      accurate on *this* microarchitecture, which is not a given)

> **If this is not green, stop and fix it. A fast wrong kernel is worth nothing, and
> the whole benchmark suite downstream would be measuring a lie.**

---

## 3. The full baseline (~90 min, one command, unattended)

```bash
THREADS_LIST="1 2 4 8 16 32 64" bash bench/run_baseline.sh graviton-c8g 2>&1 | tee session.log
```

That single command runs all six stages. What it covers and roughly how long:

| Stage | What | ~Time |
|---|---|---|
| [1/6] | correctness gate + **2D goldens** + SS2D pair parity at 1/2/8 threads | 10 min |
| [2/6] | criterion ladder: `scalar_seq → neon_seq → neon_par` | 15 min |
| [3/6] | op-level vs eager **and `torch.compile`** (the baseline judges trust) | 20 min |
| [4/6] | **core-scaling sweep 1→64 threads** — the Cloud-track chart | 15 min |
| [5/6] | end-to-end mamba-130m generate (downloads ~300 MB once) | 15 min |
| [6/6] | **SS2D at production grids** + bidirectional | 15 min |

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

---

## 5. Retrieve and terminate (~10 min)

```bash
tar czf results-graviton.tgz bench/results session.log ct.log bench/profile/out
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

Run this instead. It keeps every claim the submission actually rests on and drops
the rest:

```bash
SKIP_E2E=1 THREADS_LIST="1 4 16 64" bash bench/run_baseline.sh graviton-c8g 2>&1 | tee session.log
```

Correctness + ladder + op-level + a 4-point scaling curve + SS2D. That is enough for
the headline table, the scaling story, and the P1-7 decision.

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
