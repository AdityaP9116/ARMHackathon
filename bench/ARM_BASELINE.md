# Getting the ARM baseline — measure, profile, judge

The kernel's headline claim ("fast selective scan on Arm") is **currently unproven**:
the only recorded numbers are x86 + scalar fallback. This runbook produces the real
Arm numbers, a per-optimization ablation, and a profile that tells us whether the
kernel is near the hardware ceiling or leaving performance on the table.

Budget: ~1 afternoon, ~$0–10 (free Oracle Ampere, or a few Graviton hours).

---

## 0. The three baselines (say which one every number is against)

| Baseline | What it is | Why it matters | Expect |
|---|---|---|---|
| **eager** | PyTorch's generic scan (what the model runs today on CPU) | the "we fixed the un-optimized path" story | large (10–40×) — but a strawman |
| **torch.compile** | the reference, compiled | the **fair** fight; lead with this | honest 2–5× op / 1.5–2.5× e2e |
| **scalar (our own)** | our Rust scalar kernel, 1 thread | isolates what NEON+chunk+threads bought | the ablation ladder |

Rule: never quote a speedup without naming the baseline. The eager number is for color; the **torch.compile number is the one judges trust.**

---

## 1. Spin up an Arm box

Ubuntu 22.04+ aarch64 — Oracle **Ampere A1** (Always-Free, Neoverse N1) or AWS **Graviton** (`c7g`/`c8g`, Neoverse V1/V2). Prefer Graviton for the *headline* run (SVE2, newer), Ampere for free iteration.

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-venv git linux-tools-generic
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env
git clone https://github.com/AdityaP9116/ARMHackathon && cd ARMHackathon
```

---

## 2. Correctness FIRST (never benchmark an unverified kernel)

```bash
cd kernel && cargo test --release -- --nocapture
```

Look for, in the output:
- every golden case prints `ok` (not `FAIL`);
- `backend Auto resolves to NEON on this host` — confirms you're actually testing the NEON path, not the scalar fallback;
- `vexpq_f32 worst relative error` / `vsoftplusq_f32 worst error` lines are ~1e-6 to 1e-7 (the vector math is accurate on *this* microarchitecture).

If this isn't green, stop — a fast wrong kernel is worthless.

---

## 3. Tier 1 — the ablation ladder (`cargo bench`)

```bash
cargo bench            # from kernel/
```

Criterion prints a time for each rung × shape: `scalar_seq`, `neon_seq`, `neon_par`. Fill this in for the `mamba130m_layer_l512` shape:

| Rung | Time | vs previous rung | What it isolates |
|---|---|---|---|
| scalar_seq | ___ ms | — | baseline |
| neon_seq | ___ ms | ___× | **SIMD + vector-exp + fusion** (single core) |
| neon_par | ___ ms | ___× | **threading** across cores |

**Reading it:**
- `neon_seq / scalar_seq` ≈ **3–4×** → NEON and the vector exp are working. **< 2×** → the exp or memory layout is weak (profile it, step 7).
- `neon_par / neon_seq` should approach your core count. On 16 cores, **≥ 8×** is healthy; **< 4×** means a scaling wall (bandwidth or oversubscription — step 6).

---

## 4. Tier 2 — op-level vs torch.compile (the fair number)

```bash
cargo build --release -p arm-scan-ffi
cd .. && python3 -m venv env && source env/bin/activate
pip install numpy torch
python bench/bench_op.py --json bench/results/op_$(hostname).json
```

Record, per shape, **both** `speedup vs eager` and `speedup vs torch.compile`. The script also prints `kernel-vs-ref max_abs_err` per shape — confirm it stays < 1e-4 (correctness on real shapes, not just goldens). Note where `ref_compile` is *skipped* for long L — that skip is itself a result (torch.compile can't handle the long sequence; your kernel can).

---

## 5. Tier 3 — end-to-end on a real model

```bash
pip install transformers
python bench/bench_e2e.py --prompt-tokens 512 --new-tokens 64 --reps 5 \
       --json bench/results/e2e_$(hostname).json
```

Read three things: **prefill speedup** (where the kernel engages — should be the big one), **decode tok/s** (single-token steps fall back by design — should be ~unchanged), and **generate total** (the honest end-to-end, diluted by decode). Confirm `greedy tokens identical: True` and `fast_calls > 0` (the kernel actually ran).

---

## 6. Core-scaling sweep (threading strength/weakness)

```bash
cd kernel
for t in 1 2 4 8 16; do
  echo "=== $t threads ==="
  RAYON_NUM_THREADS=$t cargo bench --bench scan -- neon_par/mamba130m
done
```

Compute **scaling efficiency** = `time(1) / (time(N) × N)`.
- **> 0.7** → threading is a genuine strength; publish the scaling curve.
- **< 0.5** → weakness: you're memory-bandwidth-bound (all cores starving for the same B/C data) or fighting PyTorch's thread pool. This is the most likely weakness and worth finding.

---

## 7. Profile — where the single-core time actually goes

First confirm NEON was really emitted (cheap sanity check):

```bash
objdump -d kernel/target/release/libarm_scan_ffi.so | grep -c -E "fmla|fmul"   # should be > 0
```

Then sample the hot path:

```bash
sudo sysctl kernel.perf_event_paranoid=1
perf record -g python bench/bench_op.py --quick --no-compile
perf report            # look for time in channel_n16, vexpq_f32, the transpose loop
```

Hardware counters (bottleneck classification):

```bash
perf stat -e cycles,instructions,cache-references,cache-misses,\
stalled-cycles-backend,stalled-cycles-frontend \
  python bench/bench_op.py --quick --no-compile
```

> Caveat: on shared cloud VMs the PMU is often not exposed, so `perf stat` counters may read zero. On Graviton bare-metal-ish instances they usually work; otherwise use `perf record` software sampling (still shows *where* time goes) and Arm's **Performance Studio / Performix** (the contest's own tool) for counters.

**Interpretation — this is the strength/weakness verdict:**

| Signal | Meaning | Verdict |
|---|---|---|
| **IPC ≈ 2.5–3+** (instructions ÷ cycles) | executing near peak, few stalls | **compute-bound → near the ceiling; ship it** |
| **IPC < 1**, high `stalled-cycles-backend` | waiting on the memory system | **memory-bound → attack it** (bf16 storage, better B/C blocking) |
| **high `cache-misses` / LLC miss rate** | starving for data | bandwidth-bound → smaller working set, prefetch, bf16 |
| **`vexpq_f32` dominates `perf report`** | exp still the hot spot | sharpen exp (fewer poly terms if precision allows) |
| **the transpose loop is hot** | B/C prep costs too much | fold transpose into the chunk loop, or skip for groups=1 |

---

## 8. Turn it into a strengths/weaknesses statement

After the runs you can make *evidence-backed* claims like:

**Likely strengths (confirm with numbers):**
- Strong single-core NEON win (`neon_seq/scalar_seq`) from SIMD + hand-vectorized exp + fused loop.
- Constant memory (no `(B,D,L,N)` intermediate) — report peak RSS vs eager.
- Bit-exact correctness preserved end-to-end (tokens identical).
- Big prefill speedup — the regime that actually matters for throughput.

**Likely weaknesses (find and state honestly):**
- Threading scaling ceiling on many cores (bandwidth-bound) — the core-scaling curve will show it.
- Decode steps get no speedup (by design — they fall back).
- Non-N=16 shapes use the slower general path.
- No SVE2 yet — a named next lever, not a failure.
- The `torch.compile` gap is smaller than the eager gap — say so up front.

Stating the weaknesses *with the profile that proves them* is worth more to the judges than hiding them — it shows you understand the hardware.

---

## 9. Results to commit

Fill `RESULTS.md` and the README table from `bench/results/*.json`, always tagged with instance type, core count, and torch version. Commit the JSONs for the **headline** host (they're gitignored by default — add deliberately).

## 10. One-afternoon checklist

- [ ] Arm box up, repo cloned, toolchain installed
- [ ] `cargo test --release` green, Auto=NEON confirmed
- [ ] `cargo bench` ladder recorded (scalar → neon_seq → neon_par)
- [ ] `bench_op.py` recorded (vs eager **and** vs torch.compile)
- [ ] `bench_e2e.py` recorded (prefill / decode / total, tokens identical)
- [ ] core-scaling sweep → efficiency number
- [ ] `perf report` hot-symbol list + IPC/stall/cache classification
- [ ] strengths/weaknesses written from the evidence
- [ ] `RESULTS.md` + README table filled, JSONs committed for the headline host
