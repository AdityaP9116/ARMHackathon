# 8. How we prove correctness

**A fast wrong kernel is worth nothing.** This file is about the machinery that
makes the speed numbers mean something — and it is the part of the project that
took the most effort.

The governing rule, from `CLAUDE.md`:

> **Correctness gates speed. Always.** Every optimization layer must reproduce
> the previous layer's output within tolerance before anyone benchmarks it.
> **Never loosen a tolerance to make a test pass; find the bug.**

## The layered strategy

Each layer is checked against the one before it, so a bug is localized rather
than merely detected:

```
f64 reference  ──►  f32 scalar  ──►  tiled  ──►  NEON  ──►  threaded  ──►  through the C ABI
   (truth)         (portable)     (blocked)   (SIMD)    (rayon)       (what Python calls)
```

If NEON disagrees with scalar, the bug is in the SIMD code. If threaded disagrees
with single-threaded, threads are sharing state. **The layering is what makes
failures diagnosable.**

## Goldens

A **golden** is a recorded input/output pair from a trusted source, saved to disk
as `.npz` and committed.

```
tests/golden/          Mamba-1 goldens
tests/golden/mamba3/       10 cases, 7 output shapes  (34 MB)
tests/golden/mamba3_mimo/  MIMO cases                 (19 MB)
```

They ship in the repo — **53 MB, so no GPU is ever needed to check correctness.**
That is the entire point of capturing them.

### Where the truth comes from — and why it differs by generation

| | Mamba-1 | Mamba-3 |
|---|---|---|
| Source of truth | vendored f64 PyTorch reference | **captured from the official GPU kernels** |
| Why | upstream ships `selective_scan_ref` | **there is no CPU path in `mamba_ssm` at all** |

For Mamba-3 this was the project's first and hardest step, described in
[§3](03_mamba3_whats_new.md): rent a GPU, wrap the official kernel entry point,
record every call's tensors, commit them.

Two findings from that capture that shaped everything downstream:

**1. The released `mamba-ssm` wheel silently corrupts output on Blackwell GPUs.**
Upstream PR #997 (merged after the last PyPI release) fixes wrong results from
`mamba3_siso_fwd_kernel` with `num_stages` 2 or 3 — **no error, just wrong
numbers**. Capturing ground truth through that bug would have validated every
downstream Rust kernel against garbage *and looked green doing it*. Our capture
script refuses to run without the fix, and detects it by **inspecting the
source**, because patched and unpatched installs both report version
`2.3.2.post1`.

**2. The kernel is mixed-precision with no flag to change it.** `Q/K/V/Trap/
Angles/Z` are cast to **bf16** on entry; `ADT/DT` stay fp32 "for stability";
output is bf16. So the goldens record inputs **post-cast** — the values the kernel
actually consumed, not the ones we handed it. Recording pre-cast fp32 would make
a CPU reference diverge by an amount that **compounds over the sequence** and
reads as a kernel bug rather than a capture artifact.

## Tolerances, and why Mamba-3 needed a different instrument

For Mamba-1 the criterion is fixed:

```
max_abs(out_kernel − out_f64) < 1e-4
```

plus: a correct f32 kernel must land within a small factor of the case's recorded
**f32 error floor** — the error an ideal f32 implementation would have. Landing
*orders of magnitude* above that floor means a bug, even if you are under 1e-4.

For **Mamba-3**, `1e-4` is **unsatisfiable** — the ground truth's output is
**bf16**, whose relative epsilon is ~0.4%. Demanding 1e-4 against a bf16 number
is demanding precision the oracle does not have.

So Mamba-3's gate is measured in **ULPs of bf16** — Units in the Last Place, the
gap between adjacent representable numbers at a given magnitude. Our kernel lands
at **4.47 bf16 ULP** (bound 16), and MIMO at **1.90** through the C ABI.

**This is a different instrument, not a relaxed bound.** "Within 2 ULP of bf16" is
a *tighter* statement than a loose absolute tolerance would be. This is the one
documented exception to the never-loosen rule, and it is documented precisely
because exceptions to that rule are how projects fool themselves.

## Independent re-derivation

Checking a kernel against a reference you wrote has a hole: if you misunderstood
the operation, both will be wrong in the same way.

So `tests/verify_golden.py` **re-derives** the expected output in NumPy, written
independently, and replays it **through the real C ABI** rather than through Rust
tests. Two implementations, no shared code.

The strongest instance of this is the non-causal gate from
[§7](07_the_three_topologies.md): an **O(L²) dense algorithm reproducing the O(L)
scan to 2.99e-16 in f64**. Machine precision, no shared code. That does not just
check the kernel — it validates the mathematical derivation behind it.

## Property tests

Beyond fixed cases, `tests/property.rs` uses **proptest** to generate random
inputs and assert invariants that must hold for *every* input:

- threaded output is **bit-identical** to sequential
- `reverse` genuinely differs from forward
- f32 tracks f64
- bad shapes are rejected rather than read out of bounds

### A real bug this found, and what it teaches

`reverse_actually_reverses` failed intermittently. The counterexample: `u`, `B`
and `C` all zero. With zero input the output is identically zero **in both
directions** — so "reverse differs from forward" is *false*, not merely unproven.

**The test was asserting something untrue of a correct kernel.** The fix excludes
inputs where the property is vacuous; it does not weaken the assertion where the
property holds.

Two lessons worth keeping:

- Proptest reports the **shrunk** counterexample, not the original failure. The
  strategy draws from continuous ranges, so exact zeros essentially never arise
  by chance — proptest *walked* the failure into that state. The evidence that
  this was not a real bug came from elsewhere: the same test passed on two other
  platforms on the same commit.
- **A failure in one job skipped every downstream job**, hiding all seven Mamba-3
  gates behind an unrelated flake.

## Gates that lie — the failure mode to fear most

The dangerous test is not the one that fails. It is the one that **passes without
checking anything.** This project has hit three:

**1. The vacuous parity gate.** A comparison reported `max_abs = 0.000e+00` and
looked perfect. The cause: an untrained network whose zero-initialised output
layers made the result **independent of the scan entirely**. It would have passed
with the kernel deleted. Fixed by activating those layers first; it now reports
8.3e-7.

> **Heuristic: if a parity number is *exactly* zero, suspect the harness.**

**2. CI silently stopped running.** CI triggers on `pull_request`, and that
workflow builds a *merge ref* — which GitHub cannot create for a **conflicted**
PR. CI quietly stopped firing for a day while a different workflow kept going
green and made the pipeline look healthy. Check `gh pr view N --json mergeable`
if CI seems quiet.

**3. A check returned PASS when its input was missing.** An `r=1` collapse test
returned `True` if the goldens were absent. Those goldens are committed, so their
absence could only mean a broken tree — and "quietly green when the input
vanished" is the same shape as (2). Now a hard failure.

## Reproducibility traps

Two subtle ones, both discovered the hard way:

- **Avoid transcendental ufuncs in anything reproducible.** Golden draws moved
  off `torch.Generator` (unstable across torch versions) and then off float32
  `np.exp` — NumPy dispatches float32 transcendentals to per-architecture SIMD,
  **so x86 and aarch64 disagree in the last bit.** Only exact-rounded IEEE
  operations are portable.
- **x86 does not compile the aarch64 code at all.** A field added to a shared
  struct passes every local check and breaks only on the Arm runners. `make
  check-cross` typechecks the Arm path from an x86 box in about a second.

## Running the gates

```bash
make test            # kernel gates + goldens through the C ABI + re-derivation
make test-mamba3     # all 7 Mamba-3 gates (SISO, MIMO, 2D causal + non-causal)
make validate        # the judge path: ~5 min, no data, no AWS account
```

Correctness never needs credentials, a dataset, or a GPU. That is deliberate: a
judge must be able to verify the claims on their own laptop.

---

**Next:** [Benchmarking honestly](09_how_we_benchmark.md)
