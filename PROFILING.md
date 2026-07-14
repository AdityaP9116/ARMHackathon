# PROFILING — finding the kernel bottleneck (all free)

**Goal:** measure how the NEON selective-scan spends its time — transpose vs
`exp` vs the recurrence chain vs memory stalls — so the
[`IMPROVEMENT_IDEAS.md`](./IMPROVEMENT_IDEAS.md) backlog gets worked in the
right order instead of by guess. Nothing here costs money.

The core tool is an **instrumented copy of the kernel** (`scan_profiled`, in
`kernel/arm-scan-core/src/neon/profile.rs`) that runs single-threaded and
reports per-phase timings. Read the **relative %**, not the absolute
nanoseconds. One caveat baked into the numbers: the transpose is a *serial
prologue* run once before the parallel channel loop, so at C cores its
wall-clock share grows ~×C (Amdahl) — a big transpose % is worse than it looks.

There is **no Rust toolchain on the Windows dev box**, so the primary path is
the GitHub Actions workflow (Tier 1), which runs on real Neoverse silicon for
free and needs nothing installed locally.

---

## Tier 1 (start here) — GitHub Actions, real Neoverse, one click

1. Push this branch to GitHub.
2. Go to the repo **Actions** tab → **Profile kernel** → **Run workflow**.
3. When it finishes (~10–20 min), read the **job summary** for the phase table,
   or download the **`kernel-profile`** artifact for the full output:
   - `profile-phases.txt` — the per-phase breakdown (the answer).
   - `bench-scan.txt` — the scalar → NEON → rayon criterion ladder.
   - `kernel-asm.txt` — disassembly of `vexpq_f32`/`channel_n16` (Tier 0 audit,
     done for you on native aarch64).

This same run also compiles the aarch64-only `profile.rs` and runs
`cargo fmt --check` + `clippy` on it — so it doubles as the lint gate for code
that can't be checked on the x86 box.

**How to read `profile-phases.txt`:**

| Largest phase | Bottleneck | Backlog item |
|---|---|---|
| `exp` | compute-bound on the polynomial | §3.1 cheaper exp, §3.2 SVE FEXPA |
| `recurrence` | serial FMA chain (esp. Graviton4) | §3.3 chain-breaking |
| `transpose` | redundant memory work | §2.1 layout flag, §4.1 workspace reuse |
| grows with `L` | falling out of cache | §4.2 cache-blocking |

That single table reorders the whole priority list in §10 of IMPROVEMENT_IDEAS.

---

## Tier 0 — local, no Arm hardware (optional cross-check)

If/when a Rust toolchain is installed (`rustup`), the assembly audit needs no
Arm machine — it cross-compiles and reads what LLVM emits:

```bash
bash bench/profile/dump_asm.sh        # writes bench/profile/out/asm/*.s
```

Look for (these are the §3.7 findings):
- the same exp constant re-loaded (`fmov`/`dup`) *inside* the A2 loop → missed
  hoisting, a free win;
- `str`/`ldr` of `h0..h3` to `[sp]` → register spills.

On an **Apple Silicon Mac** (real Arm NEON, full PMU via Instruments) you can
also run the phase profiler directly:

```bash
bash bench/profile/run_profile.sh mac-m1
```

Different core than Graviton, so treat the ratios as directional — but it's
instant and needs no cloud.

---

## Tier 2 — hardware counters, free on Oracle Ampere A1

Only needed if Tier 1 leaves *why* ambiguous (e.g. "recurrence is expensive" —
is that FMA latency or DRAM stalls?). Oracle's **Always-Free** tier gives a
4-core Ampere A1 (real Neoverse N1) VM with root, so full `perf`:

```bash
# on the A1 instance, after: sudo apt install linux-tools-generic linux-perf
sudo bash bench/profile/perf_ampere.sh ampere-a1
```

It runs `perf stat` (counters) + `perf record`/`annotate` (per-instruction) on
the single-threaded profiler binary and writes `perf-stat.txt` /
`perf-annotate.txt`. Interpretation:

- high `stalled-cycles-backend` + high `LLC-load-misses`/`L1-dcache-load-misses`
  → **memory-bound** → §4.1/§4.2 jump the queue;
- low stalls, cycles concentrated in `vexpq_f32` → **compute-bound** → §3.1/§3.2;
- cycles concentrated in the Pass-B FMA chain → **latency-bound** → §3.3.

**Caveat** (from `PROJECT_CONCEPT.md`): free A1 instances are often
un-provisionable due to regional capacity. That's why it's the escalation, not
the default — Tier 1 already answers most of the question.

Setup helpers live in `bench/setup_ampere.sh` (existing) for the instance, then
the perf script above.

---

## What each tier can and can't see

| | Tier 0 asm | Tier 1 (GH Actions) | Tier 2 (Oracle perf) |
|---|---|---|---|
| Cost | free | free | free |
| Needs local toolchain | yes (rustup) | **no** | on the VM |
| Real Neoverse timing | no | **yes** | yes (N1) |
| Per-phase split | no | **yes** | yes |
| Constant hoisting / spills | **yes** | yes (asm dumped) | via annotate |
| Compute vs memory bound | no | inferred | **measured** |
| Provisioning risk | none | none | A1 capacity |

Recommended order: **Tier 1 first** (answers ~80%), Tier 0 asm alongside it,
Tier 2 only for the compute-vs-memory confirmation on whatever Tier 1 flags.

---

## Files

- `kernel/arm-scan-core/src/neon/profile.rs` — instrumented `scan_profiled`
  (behind the `profiling` feature; aarch64 only).
- `kernel/arm-scan-core/examples/profile_phases.rs` — runs it over mamba shapes
  and prints the table.
- `.github/workflows/profile.yml` — Tier 1, the one-click free run.
- `bench/profile/run_profile.sh` — Tier 0/1 on any Arm host with cargo.
- `bench/profile/perf_ampere.sh` — Tier 2 hardware counters.
- `bench/profile/dump_asm.sh` — Tier 0 assembly audit, no Arm hardware needed.
