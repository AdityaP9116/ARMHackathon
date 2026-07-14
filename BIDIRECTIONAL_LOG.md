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

---

## Step 1 — correctness path (plan §2.1)

**Branch:** `feature/bidirectional-scan` · **Status:** code complete; math gate
green locally, kernel gate pending first CI run.

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

- Kernel-level gate has **not run** (no local torch/cdylib) — first CI push proves it.
- No measurement yet of the flip-copy overhead the fused path would remove. Per
  plan §4.3 that measurement is the gate on whether §2.2 is worth the time.
- No HF integration (`patch.py` dispatch for a bidirectional mixer class) — blocked
  on the application decision.

---

## Step 2 — fused `reverse` flag in Rust (plan §2.2)

Not started — and deliberately **gated on a measurement**, not scheduled.

The fused path's whole value is deleting the flip copies (four tensors in, one
out) that Step 1 pays for. Per plan §4.3, the decision to spend the ~half day on
it should follow from measuring how much those copies actually cost at the
chosen application's real shapes — at short sequence lengths they may be noise,
in which case the correctness path is what ships and the fusion is future work
in the writeup.

Two prerequisites, both outside this log:
- **The application decision** (`APPLICATIONS.md`) — it determines whether the
  target model is *inner* or *outer* bidirectional (see the design boundary
  above). If it is outer, a `reverse` flag buys that model **nothing**, and this
  step should not be built for it at all.
- The overlap flagged in plan §2.2 with `IMPROVEMENT_IDEAS.md` §4.2
  (cache-blocking over L): both restructure the same chunk loop in
  `neon/mod.rs`. Whoever gets there first should leave it in a shape the other
  can build on.
