# BIDIRECTIONAL_LOG — the 1D bidirectional scan, step by step

A running record of the **1D bidirectional** topology: what changed, what it is
verified against, what broke along the way, and what is still unproven. Sibling
to [`OPTIMIZATION_LOG.md`](./OPTIMIZATION_LOG.md) (which tracks kernel *speed*);
this one tracks one axis of kernel *generality*. The 2D cross-scan (SS2D) gets
its own log when that work starts.

Plan: [`TOPOLOGY_IMPLEMENTATION_PLAN.md`](./TOPOLOGY_IMPLEMENTATION_PLAN.md) §2
(SS2D is §3 of the same plan). Every entry obeys the
[`CLAUDE.md`](./CLAUDE.md) rule **correctness gates speed** — nothing is
benchmarked, and no fusion work starts, until the correctness path is green.

**Convention used throughout:** the topology ships in two stages — *correct*
(Python rearrangement on top of the existing 1D op, zero new Rust) and then
*fast* (fused in Rust, via a kernel `reverse` flag). The correct stage lands
first so the fusion is justified by a measurement instead of an assumption.

> ## ✅ Bottom line (all CI green; full L sweep, `3c7a7c3`)
>
> **A bidirectional selective scan on Arm runs 5.2–5.6× faster than
> `torch.compile` (L ≥ 512) and 16–27× faster than PyTorch eager**, holding
> 4.3e-6 – 8.6e-6 max abs error vs the f64 reference (gate: 1e-4) all the way out
> to **L = 8192**.
>
> **And `torch.compile` cannot follow us to the lengths the applications need.**
> Compile time is **linear in L** — it converges to ~0.26 s *per timestep* — because
> the recurrence is unrolled into an L-step graph. L=8192 would take ~36 minutes to
> compile; the 131k-token genomics context would take **~9.5 hours**, if it did not
> OOM first. That is not a slow baseline; it is an absent one.
>
> The `reverse` kernel flag exists as the **substrate for the 2D cross-scan**, not
> as a bidirectional speedup — its own contribution is 1–7% with no stable pattern.
> Quote the vs-`torch.compile` row, and nothing else.

---

## Step 1 — correctness path (plan §2.1)

**Branch:** `feature/bidirectional-scan` · **Status:** ✅ **done and verified.**
Both gates green on CI. The kernel gate failed once on first run — a bug in the
test, not the kernel (see *Errors and surprises* §6) — and passed after the fix.

Full CI result (run `29323871999`, commit `bbc2722`):

| Job | Gate | Result |
|---|---|---|
| `test (linux-arm64)` | definition check (numpy) | ✅ |
| `test (macos-arm64)` | definition check (numpy) | ✅ |
| `test (linux-x86_64)` | definition check (numpy) | ✅ |
| `bench-op (linux-arm64)` | **correctness through the real kernel** | ✅ |

The `bench-op` row is the meaningful one: `bidirectional.py` running against the
real compiled NEON kernel on real Arm silicon, matching an f64 reference within
1e-4 across all 13 cases — including the NEON tail path (`state=13`), grouped
B/C, `L=1`, every merge mode, untied `reverse_params`, and `fwd == plain 1D
scan` (bit-identical). Wheels still build and golden-check on all three
platforms; op-level bench unregressed.

### What

`python/arm_scan/bidirectional.py` — `bidirectional_scan(...)`: runs the
recurrence over the sequence in both time directions and merges. Built entirely
on the existing 1D op — **no new Rust, no new FFI, no ABI bump**.

- Merge modes: `sum` (default), `mean`, `concat`, `none` (returns both
  directions unmerged, for a model-specific gated combine — the primitive
  deliberately does not guess at a learned merge).
- `reverse_params`: override A/D/delta/delta_bias/B/C for the backward pass, for
  untied models (Vim's `bimamba_type="v2"` has a separate `A_b`, `D_b`,
  `dt_proj_b`). Default `None` = weight-tied.
- `_scan_reverse` is deliberately **the single seam**: today it flips the
  time-varying inputs, calls the forward kernel, and flips the output back.
  When the kernel grows the `reverse` flag (plan §2.2), only that function's
  body changes — every caller inherits the win.
- Exposed as `arm_scan.bidirectional_scan` via the existing lazy-import
  mechanism, so numpy-only users still never import torch.

### Verification

Two gates, deliberately split by what they can prove:

| Gate | What it proves | Where it runs |
|---|---|---|
| `tests/check_bidirectional_math.py` | the **definition** is right | numpy only — **runs anywhere**, incl. this x86 Windows box |
| `tests/check_bidirectional.py` | the **code** is right, through the real kernel | needs torch + built cdylib → CI (`bench-op`, linux-arm64) |

The math gate is the load-bearing one, and it is green. It proves:

> flip → forward scan → flip  **==**  an explicitly-coded backward-in-time
> recurrence

**bit-identically** (not merely within tolerance) across 7 shapes — including
`state=13` (the NEON non-multiple-of-4 tail path), grouped B/C, `L=1`, and a
128-step sequence. It is written as an independent reverse-time loop sharing no
mechanism with the flip-based path, on top of the already-independent
`naive_scan_f64` from `verify_golden.py`.

It also guards against a **vacuous pass** — asserting the backward scan actually
differs from the forward one, so a no-op "backward" could not slip through every
equivalence check.

Both gates are wired into `.github/workflows/ci.yml`: the numpy one into the
`test` job (all three platforms — it needs no torch), the kernel one into
`bench-op` **before** the benchmark, per *correctness gates speed*.

### This file is also the spec for the Rust `reverse` flag

`naive_scan_backward_f64` in the math gate is exactly what `reverse=true` must
compute. Plan §2.2 now has an executable definition to implement against rather
than an assumption to re-derive — and the bit-identical result means the fused
path has no numerical excuse to differ.

### Errors and surprises encountered

**1. The f64 ground truth was silently f32.** ⚠ *Caught before it could mislead.*
The first draft of `check_bidirectional.py` passed **f32 inputs** with
`compute_dtype=torch.float64`. But the vendored `selective_scan_ref` ends with
`out = out.to(dtype=dtype_in)` — it casts back to the *input* dtype. So the
"float64 reference" would have come back as f32, and the 1e-4 gate would have
been comparing kernel-f32 against reference-f32: a far weaker check than
claimed, and one that could mask real error. `gen_golden.py` upcasts the inputs
(`f64 = lambda t: t.double()`) for exactly this reason. **Fixed** by upcasting
inputs to double, mirroring `gen_golden.py`.

**2. `tests/reference/` is a package, not a bare module.** The check originally
put `tests/reference/` on `sys.path` and did
`from selective_scan_ref import selective_scan_ref`. The repo convention (per
`gen_golden.py`) is to put `tests/` on the path and do
`from reference import selective_scan_ref`. **Fixed** to match.

**3. `D` is applied twice under a sum merge.** Not a bug — a real gotcha. With
`merge="sum"` and a shared `D`, the skip connection lands in **both** directions,
so the merged output carries `2·D·u`, not `D·u`. This *is* what real
bidirectional Mambas do (each direction's mixer applies its own D, then the
outputs are summed), but it is exactly the kind of thing that produces
plausible-looking, quietly-wrong output. Now **documented in the module and
pinned by an assertion** so it can never drift silently.

**4. Gating inside both passes is safe — proven, not assumed.** The module lets
the kernel apply the z-gate in *both* directions rather than once after the
merge. For any linear merge that is algebraically identical:

```
inside:  (y_f + D·u)·silu(z) + (y_b + D·u)·silu(z)
outside: ((y_f + D·u) + (y_b + D·u))·silu(z)
```

Asserted in the math gate rather than left as a claim in a docstring.

**5. The dev box could not run *anything*.** No Rust, no torch, no numpy, no
built cdylib. Rather than block on a multi-GB toolchain install, the work was
split so the *definitional* correctness could be proven with numpy alone (a
`.venv` at the repo root, which `bench/run_baseline.sh` already expects), and
the kernel-level check deferred to CI. This is why there are two gates and not
one — and it turned out to be a better structure anyway, since the math gate is
portable and catches the class of bug that kernel testing structurally cannot.

**6. The kernel gate failed on first CI run — and it was right to.** ⚠ *The most
instructive failure so far.* `no_softplus` came back at **max_abs = 3.3e-3**,
33× over the 1e-4 gate, while all 12 other cases passed.

Root cause was **the test, not the kernel**. `make_case` drew `delta` from a
normal distribution unconditionally. That is fine when `delta_softplus=True`
(delta is raw; the kernel applies softplus and the timestep comes out positive),
but with `delta_softplus=False` **delta *is* the timestep** and must already be
positive — HF's slow path pre-applies softplus, so no real Mamba ever passes a
negative one. `gen_golden.py` knows this and draws `uniform(1e-3, 0.1)` for its
own `no_softplus` case; my test drew `randn` and produced negative timesteps.

Why that blows up *now* specifically: with `delta < 0` and `A < 0`, the argument
`dt·A` goes **positive** — violating the precondition of the `vexpq_f32_nonpos`
optimization landed in [`OPTIMIZATION_LOG.md`](./OPTIMIZATION_LOG.md) Step 1,
whose docstring states it outright ("*positive input can overflow to inf because
the upper clamp is absent — the scan never passes positive*"). The 3.3e-3 is a
degree-reduced, unclamped exp being evaluated outside its fitted domain, exactly
as designed.

**Fixed** by drawing a positive delta when softplus is off (mirroring
`gen_golden.py`), and by making `make_case` *reject* `delta_bias` in that mode —
a bias could push a positive timestep negative and reintroduce the same
violation silently.

Two things worth taking from this:
- The precondition behind the exp optimization is **real and load-bearing**, and
  it now has a second, independent test exercising it. An unrelated workstream
  tripped over it within a day of it landing.
- `check_bidirectional_math.py` deliberately *keeps* drawing `randn` delta for
  its no_softplus case and still passes — numpy's exp is exact over the whole
  line, and the identity it tests (flip-forward-flip == backward recurrence) is
  a mathematical fact that holds for any delta. The two files differ on purpose;
  a comment in each says so, so nobody "fixes" one to match the other.

**Result after the fix:** 12 of 13 cases were already green on the first run,
including `state13_neon_tail`, `grouped_bc`, `edge_len1`, all merge modes, the
untied-`reverse_params` path, and `fwd == plain 1D scan` (bit-identical). Only
`no_softplus` failed, and only because it was asking the kernel a question no
model asks.

### Design boundary — which bidirectional models this is for

Two patterns exist in the wild and they are **not interchangeable**:

- **"outer"** (Caduceus's `BiMambaWrapper`, Vim): the *whole mixer* — causal
  conv, `x_proj`, `dt_proj` — is re-run on `x.flip(time)`. A causal conv over
  flipped input is **not** the flip of the conv over input, so the two
  directions' scan inputs are genuinely different tensors. Such a model does not
  need this module: it already calls `selective_scan` twice, and **both calls are
  ordinary forward scans**. A kernel `reverse` flag buys it *nothing*.
- **"inner"** (VMamba/SS2D-style cross-scan, and bidirectional variants that flip
  *after* the projections): the *same* projected tensors are traversed in both
  time directions, because flipping commutes with the time-pointwise
  projections. **This is what the module implements**, it is what a fused
  `reverse` flag actually accelerates, and it is the 1D case of the SS2D
  cross-scan.

This distinction cannot be settled until the application is chosen
(`APPLICATIONS.md` is still open) — it decides whether §2.2's fused `reverse`
flag is worth building at all for the app we ship. Documented at the top of the
module so nobody wires it into an outer-pattern model by mistake.

### Not yet done

- **Fast** is not done. *Correct* is. Each direction runs on the full fast NEON
  kernel, but the six flip copies around them are pure overhead that only the
  fused `reverse` flag (Step 3) removes. **No Rust was written for this step.**
- No HF integration (`patch.py` dispatch for a bidirectional mixer class) —
  blocked on the application decision.

---

## Step 2 — measure what the flips cost (the gate on Step 3)

**Status:** ✅ **measured. The answer is: the flips are noise. Do not fuse.**

Benchmark: `bench/bench_bidirectional.py`, wired into CI's `bench-op` job.

### The numbers (linux-arm64 CI runner, 4-core, torch 2.13.0, `--quick`, commit `5cb4fe8`)

| Shape | `scan_fwd` | `fused_estimate` | `bidirectional` | `flips_only` | headroom (ceiling) |
|---|---|---|---|---|---|
| B1 D768 L128 N16 | 0.801 ms | 1.641 ms | 1.781 ms | 0.057 ms | **1.085×** |
| B1 D768 L512 N16 | 2.694 ms | 5.492 ms | 5.631 ms | 0.108 ms | **1.025×** |

**Sanity check first:** `fused_estimate` ≈ 2 × `scan_fwd` at both shapes (1.641
vs 1.602; 5.492 vs 5.388), which is exactly what the proxy is supposed to do. The
comparison is sound.

### Verdict: **Step 3 is rejected on measurement.**

Three independent reasons, in order of weight:

**1. The ceiling is already low, and it *shrinks* with L.** 8.5% at L=128 →
2.5% at L=512. Not noise: flip traffic is O(B·D·L) memory copies while scan work
is O(B·D·L·N) compute, so the flips get relatively cheaper the longer the
sequence. Every application this topology targets (genomics at **131k tokens**,
long audio, multi-hour ECG) lives far to the right of L=512, where the flips are
negligible. The measurement gets *more* damning at the shapes we actually care
about, not less.

**2. Most of the apparent overhead is not even flips.** The two measurements
disagree — subtraction says 7.8% at L=128, but `flips_only` directly measures
**3.2%**. That gap (~0.14 ms, roughly *constant* across both shapes) is
Python-side wrapper overhead in `bidirectional_scan`, not memory traffic, and a
Rust `reverse` flag **would not remove it**. Having both measurements is what
caught this; a subtraction-only benchmark would have overstated the case for
fusion by ~2.5×. The honest ceiling on what fusion saves is **1.9–3.2%**.

**3. It may help zero models anyway.** A `reverse` flag only accelerates *inner*
bidirectional models. If the application lands on Caduceus or Vim (*outer* — see
the design boundary above), both of their scans are ordinary forward scans and
the flag accelerates **nothing at all**.

So the trade is: half a day of NEON chunk-reversal surgery + an FFI ABI bump +
new goldens + new parity tests, to win ~2% that trends to ~0% at the sequence
lengths that matter, for a model class that may not even benefit. **No.**

### Why this is a good outcome, not a wasted step

This is the **second** time measuring-before-optimizing has killed a plausible
idea in this project. The first was the plane transpose that everyone assumed was
a top-priority cost and profiled at **0.1%**
([`OPTIMIZATION_LOG.md`](./OPTIMIZATION_LOG.md)). Had §2.2 been built on
intuition — and the plan's own effort estimate said "half a day," which is
exactly the size of task that gets waved through — it would have bought ~2%.

Per [`IMPROVEMENT_IDEAS.md`](./IMPROVEMENT_IDEAS.md) §7.1, a **considered-and-
rejected decision backed by numbers** is worth publishing. This one goes in the
writeup: *we built the general path, measured what fusing it would buy, found it
was ~2%, and spent the time elsewhere.* That is a stronger technical signal than
a fused kernel nobody needed.

### Caveats on these numbers (stated, not buried)

- `--quick` mode: **reps=5, warmup=1**, on a *shared* 4-core CI runner. Noisy.
  The 1.085 vs 1.025 spread is within plausible run-to-run variation; the
  *trend* and the *magnitude* are what carry the conclusion, not the third
  decimal.
- Only two shapes, both short (L=128, L=512). The full `sweep-len` suite goes to
  L=8192 — worth one run on a dedicated Arm host before the number is quoted in
  `RESULTS.md`, since it would show the trend continuing rather than asserting it.
- `fusion_headroom` is a **ceiling**, not an achieved speedup (see the module
  docstring). A real fused kernel reads the sequence backward, which is less
  cache-friendly than the forward stream timed here, so it would land *under*
  these figures. Quoting it as a speedup would be dishonest.

Step 3 (the fused kernel) is deliberately **gated on this measurement rather
than scheduled**. Its entire value is deleting the six copies Step 1 pays for —
so if those copies are cheap, the fused kernel is worthless and should not be
built. The kernel-side profiling work already produced the cautionary example:
the plane transpose that everyone *assumed* was a top-priority cost measured
**0.1%** ([`OPTIMIZATION_LOG.md`](./OPTIMIZATION_LOG.md)). Assuming here would be
repeating a mistake this project has already made once and caught.

### What it times

| Series | What it is |
|---|---|
| `scan_fwd` | one forward scan — the floor |
| `fused_estimate` | two forward scans + merge, **zero flips** — the proxy for a fused `reverse` |
| `bidirectional` | the real thing today: flips + two scans + un-flip + merge |
| `flips_only` | the six copies alone, no scans — reads the cost directly instead of by subtraction |

**The decision number:** `fusion_headroom = bidirectional / fused_estimate`.

- `~1.0×` → the flips are noise. **Do not build Step 3.** Ship §2.1 and say so
  plainly in the writeup.
- `>1.15×` → the flips are real and the fused flag pays for itself.

### The proxy is a ceiling, and is reported as one

`fused_estimate` is an **upper bound** on what fusion could achieve, not a
measurement of a fused path (which does not exist). It runs identical scan work
with zero flip traffic, so a real fused kernel *cannot beat it* — a real one
also walks the sequence backward, which is less cache-friendly than the forward
stream being timed. So: a low ceiling is conclusive (don't fuse); a high ceiling
is permission to try, not a promise. It is never to be quoted as an achieved
speedup.

---

## Step 3 — fused `reverse` flag in Rust (plan §2.2)

**Status:** ✅ built. **But note the reversal of the Step-2 decision, and why.**

Step 2 rejected this **as a speedup**, and that rejection still stands: the
copies are worth ~2%, and nothing here changes that. It was built anyway, on a
different justification that Step 2 did not weigh:

> **The 2D cross-scan needs a backward traversal regardless.** SS2D's four
> directions are row-forward, row-**backward**, column-forward, and
> column-**backward**. Plan §3.2's design has the row directions reusing the 1D
> scan directly — which requires exactly this flag. So `reverse` is not a
> bidirectional optimization that failed to pay off; it is **the substrate SS2D
> is built on**, and it happens to also remove bidirectional's flip copies.

**The claim to make when this ships is therefore:** *"a fused backward traversal
— the substrate for the 2D cross-scan, which also removes the flip copies from
bidirectional (~2%)."* **Not** *"we made bidirectional faster."* The benchmark
prints that caveat on every run so nobody quotes it wrong.

### What changed

The implementation turned out **much cheaper than plan §2.2 predicted**, and for
an instructive reason. The plan claimed the NEON work needed SIMD lane-reversal
(`vrev64q_f32` / `vextq_f32` shuffles inside each chunk). That was wrong:

- **Pass A is pointwise in time** — no cross-timestep dependency. That is the
  entire reason the two-pass split exists. It needs *zero* direction awareness.
- **Pass B vectorizes across STATE, not time.** `h` lives in four q-registers and
  `t` is a plain scalar loop index. So reversing time is one subtraction
  (`t = tlen - 1 - i`) — no shuffles, no extra loads, no extra work.
- The B/C plane transpose, the epilogue, and `parallel.rs` are all
  direction-agnostic and were **not touched at all**.

Net: a new `chunks_in_scan_order()` iterator (visit chunks last-first when
reversed), one flipped index in each of the two NEON channel paths and the
scalar path, and the FFI/Python plumbing. The recurrence math is untouched.

| Layer | Change |
|---|---|
| `arm-scan-core/src/lib.rs` | `ScanInput::reverse` |
| `src/scalar.rs` | one flipped time index |
| `src/neon/mod.rs` | `chunks_in_scan_order()` + flipped `t` in Pass B (both paths) |
| `src/parallel.rs` | **none** — channel independence is direction-agnostic |
| `arm-scan-ffi/src/lib.rs` | `reverse` param; ABI bump (→ **4** after reconciling with `h0`, see Step 4) |
| `python/arm_scan/{_ffi,op,numpy_api}.py` | `reverse` threaded through |
| `python/arm_scan/bidirectional.py` | the seam closed: `reverse=True`, no flips |

### Correctness

`reverse=True` is *defined* as flip-forward-flip, and that definition is now
enforced at three independent levels:

1. **Rust, bit-for-bit** — `reverse_matches_flip_forward_flip` (property test,
   256 random shapes/flag combos). Asserts **bit-identity**, not tolerance: both
   paths apply the same arithmetic to the same values in the same order. Note the
   two paths land on *different chunk boundaries* (forward-on-flipped splits the
   flipped axis; reverse splits the original), so passing bit-exactly also proves
   chunking never leaks into the math.
2. **Rust, anti-vacuous** — `reverse_actually_reverses` asserts a reversed scan
   genuinely differs from a forward one, so a dropped flag cannot pass silently.
3. **numpy, independent** — `tests/check_bidirectional_math.py` proves the
   identity itself is sound against a separately-written backward recurrence,
   with no kernel involved.

Plus: `reverse` is now generated by the proptest `Case`, so **every** existing
property test (f64 agreement, NEON-vs-scalar parity, rayon bit-identity) sweeps
both directions for free. And `ffi_reverse_two_steps` hand-checks it across the
C ABI.

The benchmark refuses to report a speedup unless the fused output is
bit-identical to the flip-based one — a fast wrong answer is not a result.

### Still worth doing, and still not this

Step 2's measurement found ~0.14 ms of **Python-side wrapper overhead**, roughly
constant across shapes and *larger than the flip copies themselves* at L=512.
`reverse` does not touch it. It belongs with the trims in
[`IMPROVEMENT_IDEAS.md`](./IMPROVEMENT_IDEAS.md) §8 (ctypes call path, `_c()`'s
redundant `.float()` dispatch, per-call allocations) — cheap, and it helps
*every* caller rather than just bidirectional ones.

---

## Step 4 — the `reverse` / `h0` ABI collision (merging with main)

**Status:** resolved. Both features shipped. **ABI is now 4.**

While `reverse` was in flight, [`OPTIMIZATION_LOG.md`](./OPTIMIZATION_LOG.md)'s
workstream landed **`h0`** on main — a caller-supplied *initial SSM state* that
makes the scan resumable (run a prefix, feed its `last_state` back as `h0`,
continue). That is `IMPROVEMENT_IDEAS.md` §2.4/§7.6, the decode/streaming work.

**Both branches independently bumped the ABI 2 → 3, with different signatures.**

| | `h0` (main) | `reverse` (this branch) |
|---|---|---|
| Core API | new `selective_scan_with_state(...)`, `h0` as a **parameter** | `reverse` as a **field on `ScanInput`** |
| C ABI | `h0: *const f32` appended after `last_state` | `reverse: c_int` inserted after `delta_softplus` |
| Claimed ABI | **3** | **3** ← collision |

Reconciled to **ABI 4**, carrying both. They are genuinely orthogonal — `h0`
seeds the state, `reverse` picks the traversal direction — and they compose: a
*backward* scan resumed from a prior state is coherent, and is in fact what SS2D
will want for its column traversals. No redesign was needed, only plumbing.

### The dangerous part: what git merged *without* a conflict

Only 4 files conflicted (`scalar.rs`, `_ffi.py`, `numpy_api.py`, `op.py`) — the
ordinary "both added a parameter" kind, resolved by keeping both. **The damage
was in the files that auto-merged cleanly**, and it is worth internalizing:

**1. The ABI version silently stayed at 3.** Both branches wrote `3`, so git saw
identical text and merged happily — while the Python loader had been reconciled
to expect 4. A version check that exists precisely to catch ABI drift would
itself have been the thing that was wrong. Caught by reading the merged file
rather than trusting the absence of a conflict marker.

**2. Adding a struct field breaks the *other* branch's new construction sites —
with no conflict.** `ScanInput` gained `reverse` here; main added new
`ScanInput` literals in its streaming tests. Git merged both hunks cleanly and
produced code that does not compile. Same for the C ABI: main's three `h0` tests
were written against a signature with no `reverse`, and this branch's `reverse`
test against one with no `h0` — every one of those call sites was left short an
argument.

This bit **twice**: the FFI call sites were caught before pushing (by counting
arguments at each call), but property.rs's three new `ScanInput` literals were
not — because only the *conflicted* files were re-checked, and property.rs was
not one of them. CI's `cargo clippy --all-targets` caught them (note: `--all-targets`
is why clippy, not `cargo build`, was the step that failed — it is the only gate
that compiles the test targets).

**The rule to take from this:** after a merge that adds a field to a shared
struct or a parameter to a shared signature, *enumerate every construction and
call site in the whole workspace* — do not assume the conflicted files are the
complete set of affected ones. The absence of a conflict marker means git found
no textual overlap, not that the result is coherent.

### The bit-identity assertion was wrong — and finding out was worth it

⚠ **The most interesting numerics finding of this whole workstream.**

`reverse_matches_flip_forward_flip` originally asserted that a fused reverse scan
is **bit-identical** to flip-forward-flip. It passed on x86 (scalar) and **failed
on both Arm legs** (`last_state differs`, at `dims = {dim: 3, len: 31, state: 7}`).

**The kernel was right. The assertion was false.** Two NEON passes process four
timesteps at a time with a **scalar tail**:

```rust
while t + 4 <= tlen { ... vsoftplusq_f32 (NEON polynomial) ... }
while t < tlen      { ... dt.softplus()   (libm)            ... }
```

`discretize_chunk` (softplus) and `epilogue_row` (SiLU) both look like this. The
vector and tail branches compute the *same function by different means* — they
agree to ~1e-7, but not to the last bit. **Which branch a timestep takes depends
on its array POSITION**, and flipping the array moves timesteps across that
boundary.

At `len = 31`: the vector body covers positions 0–27, the tail 28–30. Scanned in
place, timestep 29 sits at position 29 → **scalar tail**. Scanned flipped, it
lands at position 1 → **vector body**. Same timestep, same value, ~1 ulp apart.
That propagates through the recurrence and shows up in `last_state`.

The scalar backend has one uniform code path per timestep, so it *is* bit-exact
— which is precisely why x86 passed and Arm did not.

**This is a property of the pre-existing forward kernel, not of `reverse`.**
Demanding bit-equality on NEON was asserting something false.

**Resolution — keep the strongest claim that is actually true, per backend:**

| Backend | Assertion | Why |
|---|---|---|
| `Scalar` | **bit-identical** | uniform per-timestep path; any difference is an indexing bug, not rounding — this is what pins the traversal down |
| `Auto` (NEON) | scale-relative `< 1e-5` | one SIMD-vs-libm transcendental apart, matching `auto_backend_matches_scalar`'s existing bar |

The scalar leg still delivers the guarantee that matters — that the *indexing* is
exactly right, including that chunk boundaries never leak into the math (the two
routes chunk the axis differently). The NEON leg confirms the fused traversal is
numerically sound without pretending to an equality that cannot hold.

The same over-strong gate was in `bench/bench_bidirectional.py` (`torch.equal`).
It would have passed anyway — every benchmarked length is a multiple of 4, so no
scalar tail exists — but that is luck, not correctness, and a shape-dependent
gate that silently holds is worse than one that states its tolerance. Relaxed to
the same scale-relative bar, and it still reports when the result *is* bit-exact.

**The lesson:** "bit-identical" is the right bar for a *reordering* of identical
arithmetic, and the wrong bar the moment a SIMD tail means the arithmetic is not
identical. The test was correct to fail; it caught an over-claim in the
documentation before it reached a judge.

### Verification after the merge

- Every `ScanInput` and `Channel` literal in the workspace sets `reverse`
  (swept exhaustively, not spot-checked).
- All 7 C-ABI call sites pass exactly 16 arguments.
- `try_neon` carries **both** `reverse` and `h0` into the NEON path — a drop
  there would silently ignore one feature on aarch64 only, which is the worst
  possible failure mode (correct on the x86 CI leg, wrong on the target).
- `op.py`'s custom op, its `register_fake`, and the FFI call all agree on
  argument order (`…, delta_bias, h0, delta_softplus, reverse`). A mismatch here
  breaks `torch.compile` composability, which is the entire point of the fake
  kernel.
- `profile.rs` needs no `h0` plumbing (it is a zero-initialized diagnostic) but
  does honor `reverse` — otherwise `scan_profiled(reverse: true)` would have
  silently profiled a *forward* scan, which is worse than a compile error.

---

## Step 5 — Results (all CI green, commits `6e92b03` / `605b056`)

Host: GitHub `ubuntu-24.04-arm` runner — 4-core aarch64, torch 2.13.0, `--quick`
(reps=5, warmup=1). **Provisional**: a shared runner. Headline figures still need
a dedicated Arm host. Two independent runs are reported, because the agreement
between them is what makes the numbers credible.

### The result (run `3c7a7c3` — full L sweep, 128 → 8192)

| L | eager | torch.compile | **kernel** | vs eager | **vs torch.compile** |
|---|---|---|---|---|---|
| 128 | 27.84 ms | 6.66 ms | **1.70 ms** | 16.4× | **3.92×** |
| 512 | 136.06 ms | 30.86 ms | **5.74 ms** | 23.7× | **5.38×** |
| 1024 | 262.57 ms | 56.70 ms | **10.95 ms** | 24.0× | **5.18×** |
| 2048 | 540.64 ms | 118.25 ms | **21.03 ms** | 25.7× | **5.62×** |
| 4096 | 1130.13 ms | *(compile capped)* | **41.87 ms** | 27.0× | — |
| 8192 | 2303.90 ms | *(compile capped)* | **87.67 ms** | 26.3× | — |

B=1, D=768, N=16. Correctness in the same run: kernel-vs-f64 max abs **4.3e-6 →
8.6e-6**, against the 1e-4 gate — and, importantly, **it does not drift with L**.
Accumulating 8192 sequential steps through a chunked scan with a degree-3 exp
could plausibly have degraded; it did not. All 13 cases of
`check_bidirectional.py` green.

### The ratio improves, then PLATEAUS — it does not keep growing

**Correcting an earlier claim.** From two points (128 → 512) I concluded the
advantage "grows with sequence length." With six points it clearly **plateaus**:

- **3.92×** at L=128, then **5.2–5.6×** from L=512 onward.

The jump is real but it is a *one-off*, not a trend. L=128 is **depressed** by the
kernel's fixed per-call overhead (ctypes dispatch, `_c()` contiguity checks,
output allocation — the ~0.14 ms measured in Step 2), which is a meaningful
fraction of a 1.7 ms call and a negligible one of a 21 ms call. Once it amortizes
(by L≈512), both the kernel and `torch.compile` scale linearly in L, so the ratio
settles.

**Honest headline: ~5.2–5.6× vs `torch.compile` at L ≥ 512; ~3.9× at very short L.**

### ⚠ Compile time is LINEAR in L — the withdrawn claim, reinstated with data

Step 5 previously **withdrew** the "compile time explodes" argument, on the basis
that 59 s → 134 s (L 128 → 512) is only 2.3× for 4× the length, i.e. sub-linear.
**That withdrawal was premature — it was a two-point fit.** With four points:

| L | compile | **seconds per timestep** |
|---|---|---|
| 128 | 63.4 s | 0.495 |
| 512 | 136.7 s | 0.267 |
| 1024 | 251.0 s | 0.245 |
| 2048 | 533.5 s | **0.260** |

The per-timestep cost **converges to ~0.26 s**. Compile time is **linear in L** —
doubling the sequence doubles the compilation. The apparent sub-linearity at
128→512 was just the fixed compiler startup cost washing out.

Extrapolating that constant:

| L | projected compile time |
|---|---|
| 8192 | **~36 minutes** |
| 131,072 (genomics context) | **~9.5 hours** |

…assuming it does not OOM first, which a 131k-step unrolled graph almost
certainly would.

**This is the moat, stated precisely.** At the sequence lengths our headline
applications actually use, `torch.compile` is not a slow baseline — it is an
**absent** one. `CLAUDE.md` says the kernel's argument is that "`torch.compile`
cannot restructure a sequential recurrence"; the linear compile cost is that claim
made measurable, and it now rests on four points rather than an assertion.

**Amortization is stable at ~5,450 iterations** for L ≥ 512 (compile and runtime
both scale linearly, so the ratio is constant). That is the number a skeptic
computes for themselves, so we publish it rather than let them derive it.

### Methodological note: I over-extrapolated from two points, twice

Both mistakes in this section came from the same error, in opposite directions:

1. *"The advantage grows with L"* — from two points. It plateaus.
2. *"Compile cost does not explode"* — from two points. It is linear, and at
   application scale that is fatal for the baseline.

Each was corrected only by measuring more shapes. **Two points define a line; they
do not establish a trend.** The same lesson the fusion number taught (Step 5,
"two runs is not reproducibility") — recorded here because it cost real time twice
and would have put a wrong claim in front of a judge both times.

**Why this number is believable:** the *forward* scan measures **3.74×** vs
`torch.compile` at the same shape ([`OPTIMIZATION_LOG.md`](./OPTIMIZATION_LOG.md)).
A bidirectional scan is two forward scans, so it should land in the same band —
and 3.93× does. That internal consistency is the real check; a bidirectional
figure wildly different from the forward one would have meant a broken comparison,
not a fast kernel.

`torch.compile` also cost **62.4 s of one-time compilation** for a single shape,
and only ran at L=128 (`--quick` caps `compile_max_len` at 128 — graph unrolling
makes compile time explode with L). **One `torch.compile` data point is thin.**
The full `sweep-len` suite is needed before this goes in `RESULTS.md`.

### `torch.compile`'s compile cost — measured, and it does NOT explode

I claimed the recurrence's unrolled graph makes compile time "explode" with L.
**The data says otherwise, and the claim is withdrawn.**

| L | compile | run/iter | iterations before compile pays for itself vs our kernel |
|---|---|---|---|
| 128 | 59.3 s | 6.13 ms | ~13,400 |
| 512 | 134.1 s | 25.99 ms | ~6,600 |

**2.3× compile time for 4× the length — sub-linear.** Extrapolating, L=8192 would
be roughly ten minutes: painful, not prohibitive. And it amortizes after ~6–13k
calls, which any long-running server clears trivially.

**So "compile cost" is a weak argument and this project should stop leaning on
it.** The strong argument is the one that survives the objection: we are
**3.6–4.7× faster than `torch.compile` after it has fully paid its compile cost**,
and the gap widens with L. Publish the compile cost honestly — including the
amortization column, which is the number a skeptic would compute themselves — and
let the steady-state ratio carry the case.

**One thread left open** (assert nothing until it is checked): the baseline uses
`dynamic=False`, so **every distinct sequence length should trigger a recompile**.
Under variable-length inference — i.e. real serving — that 134 s would be paid per
shape, not once, which *would* make compile cost a serious argument. It is worth
confirming before it is claimed.

### The fusion win: small, and NOT reproducible on a shared runner

⚠ **Third revision of this number. Recording the whole arc, because the
flip-flopping is itself the lesson.**

- **First read:** "it's noise" — the achieved speedup exceeded its own theoretical
  ceiling, which is impossible.
- **Second read:** "no, the *ceiling* is the broken thing" — `fusion_speedup`
  reproduced to ~0.001× across two runs while the ceiling swung wildly. That part
  stands: `fused_estimate` (two forward scans, no flips) was supposed to be a
  lower bound, the real fused path consistently beat it by 3–6%, and a bound the
  real thing beats is a broken proxy. It has been **demoted to a diagnostic** and
  the guard built on it removed.
- **Third run broke the rest of it:**

| | run 1 | run 2 | run 3 (`23995c7`) |
|---|---|---|---|
| L=128 | 1.064× | 1.065× | **1.070×** |
| L=512 | 1.151× | 1.136× | **1.027×** |

L=128 is genuinely stable (~7%, three runs). **L=512 swings 1.027–1.151×** — so
the "reproducible, and grows with L" conclusion was two runs of coincidence, and
is withdrawn.

**Final position: the fused reverse is worth ~3–7%, it is not reliably measurable
on a shared 4-core runner, and pinning it down is not worth another CI cycle.** It
was never the justification: `reverse` exists because **SS2D needs a backward
traversal** (plan §3.2's row-backward and column-backward directions). The number
to quote is 3.63–4.67× vs `torch.compile`. Nothing else.

**Two lessons worth keeping:**
1. A proxy is only trustworthy until the real thing exists. If the real thing
   outruns its own "lower bound," suspect the proxy before you blame the noise.
2. **Two runs is not reproducibility.** Two agreeing measurements produced a
   confident, mechanistic story ("grows with L, because of cold working sets")
   that the third run dismantled. The effect being chased (~3–15%) was simply
   smaller than the shared runner's variance the whole time.

### What the numbers confirmed along the way

- **The forward path did not regress.** `fwd == plain 1D scan` is bit-identical,
  and all 16 goldens hold at their recorded error floors — so the loop-invariant
  `if ch.reverse` branch added to Pass B did not disturb the existing kernel.
  (LLVM presumably unswitched it, as expected. The criterion ladder is the direct
  confirmation and is still worth reading.)
- **The SIMD-tail prediction was right.** `fused == flip-based` came out
  **bit-identical** at both shapes — exactly as predicted, because L=128 and
  L=512 are multiples of 4, so no scalar tail exists to diverge (see Step 4's
  bit-identity finding). At a length like 31 it would not have been, which is
  precisely why the gate is a tolerance and not `torch.equal`.
- **`flips_only` (0.064 / 0.110 ms) is far smaller than the flip path's total
  penalty** (0.108 / 0.882 ms over the fused path). The gap is not copy cost — it
  is the *second working set*: flipped tensors are freshly allocated, so the scan
  streams ~4.7 MB of cold memory instead of re-reading warm cache. A real effect,
  and another reason the naive "flips cost ~2%" framing understated things. Also
  another reason not to trust ±20%-noise numbers to arbitrate it.

### Next, to make this publishable

1. Full `sweep-len` (not `--quick`), reps ≥ 10, on a **dedicated** Arm host —
   Oracle Ampere A1 or a short Graviton session — so `torch.compile` is measured
   at more than one shape and the noise floor drops below the effect.
2. Tighten the benchmark's ceiling guard to a **per-shape** check so an inversion
   like L=128's is a hard error, not a silent oddity.
3. Then, and only then, put the vs-`torch.compile` row in `RESULTS.md`.

---

## Step 6 — `torch.compile` OOM-kills the runner at L=8192 (the hard wall)

**The linear-compile-cost finding (Step 5) has an endpoint, and we hit it.**

Dispatched the long-L job (`bench-baseline.yml`, run `29350634837`) with
`long_l_compile_max=8192`. After ~96 minutes it died with:

> *The hosted runner lost communication with the server. Anything in your
> workflow that terminates the runner process, starves it for CPU/Memory, or
> blocks its network access can cause this error.*

That is an **OOM**, not a timeout (the job timeout is 150 min and was not
reached; a timeout reports cleanly as one). `torch.compile` unrolls the L-step
recurrence into a single graph, and at L=8192 the graph exhausts the 4-core
arm64 runner's memory; the OOM reaper kills the VM. The ~96 min (vs a ~75 min
estimate) is consistent with the machine thrashing on swap before the kill.

**Where it died:** cumulative compile to L=4096 is ~34 min; L=8192 adds ~36 more.
At 96 min it was well past 4096 and inside the 8192 compile. So 4096 *did*
compile — but see below.

**This is the strongest form of the moat argument.** Not "`torch.compile` is
slower" — at the sequence lengths the headline applications use (genomics @131k,
long audio, multi-hour ECG), `torch.compile` **cannot build the graph at all**.
The kernel ran L=8192 in **89.7 ms** at 8.6e-6 error (Step 5), in constant
memory, on the same box that could not compile the baseline.

### Operational failure: we lost the data, and why

**No artifact was produced.** The per-shape JSON flush (added precisely to
survive a mid-sweep death) protects against a *process* kill — a `SIGKILL` of the
python process, after which later steps still run. It does **nothing** against a
*VM* death: when the machine itself is OOM-killed, there is no runner left to
execute the `if: always()` artifact-upload step. Both the flushed JSON on the
dead VM's disk and the upload step went down with it.

So the 4096 compile row — which had almost certainly completed — was lost. The
lesson: **`if: always()` is not a guarantee against OOM; it is a guarantee
against a failed *step*.** Different failure modes need different safety nets, and
"the whole machine dies" has essentially none within a single job.

### The fix, and the plan

To bank the L=4096 compile row cleanly, cap compile at **4096** (which fits the
runner's memory) so the job completes and uploads. The L=8192 OOM is already
established qualitatively by this run; it does not need re-running to be cited —
though a dedicated host with more RAM would let us find the *exact* L at which
the graph stops fitting, which is a sharper number than "≥ 8192 on 16 GB."
