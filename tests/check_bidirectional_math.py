"""The bidirectional scan's DEFINITION, verified in pure numpy — no kernel, no torch.

`bidirectional.py` and the Rust `reverse` flag (`ScanInput::reverse`) both rest
on one load-bearing claim:

    flipping the time axis, running an ordinary FORWARD scan, and flipping the
    output back  ==  running the recurrence BACKWARD in time.

If that equivalence is false, the whole bidirectional design is wrong — and no
amount of kernel testing would reveal it, because both paths would be
consistently wrong together. So it is checked here, independently: this file
implements the backward recurrence *directly* (an explicit reverse-time loop)
and compares it against flip-forward-flip built on the already-independent
`naive_scan_f64` from `verify_golden.py`.

This file therefore doubles as the **executable spec for the Rust `reverse`
flag**: `naive_scan_backward_f64` below is exactly what `reverse=true` computes.
The Rust side asserts the same identity bit-for-bit against the real kernel
(`reverse_matches_flip_forward_flip` in `kernel/arm-scan-core/tests/property.rs`);
this file proves the identity itself is sound, independently of any kernel.

Runs with numpy alone — no Rust toolchain, no built cdylib, no torch. The
kernel-level gate for `bidirectional.py` itself is `check_bidirectional.py`.

Usage: python tests/check_bidirectional_math.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from verify_golden import naive_scan_f64, silu  # noqa: E402

# Two float64 computations of the same recurrence, differing only in index
# order, should agree to round-off. They are in fact usually bit-identical.
RTOL, ATOL = 1e-12, 1e-14


def naive_scan_backward_f64(u, delta, A, B, C, D_skip=None, z=None,
                            delta_bias=None, delta_softplus=False):
    """The selective scan run BACKWARD in time — the spec for `reverse=true`.

    Identical to `naive_scan_f64` except the time loop runs from L-1 down to 0.
    The state starts at zero at the END of the sequence and accumulates toward
    the start; output for timestep t is still written at index t, and the
    pointwise D-skip and z-gate still apply at index t. Deliberately naive: an
    explicit loop, no flips anywhere, so it shares no mechanism with the
    flip-based path it is used to check.
    """
    u = u.astype(np.float64)
    delta = delta.astype(np.float64)
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    C = C.astype(np.float64)

    batch, dim, L = u.shape
    N = A.shape[1]
    grouped = B.ndim == 4
    group_size = dim // B.shape[1] if grouped else None

    if delta_bias is not None:
        delta = delta + delta_bias.astype(np.float64)[None, :, None]
    if delta_softplus:
        # same softplus as verify_golden (torch's beta=1, threshold=20)
        delta = np.where(delta > 20.0, delta,
                         np.log1p(np.exp(np.minimum(delta, 20.0))))

    out = np.empty((batch, dim, L), dtype=np.float64)
    last_state = np.empty((batch, dim, N), dtype=np.float64)
    for b in range(batch):
        for d in range(dim):
            g = d // group_size if grouped else None
            h = np.zeros(N, dtype=np.float64)
            for t in range(L - 1, -1, -1):          # <-- the only difference
                dt = delta[b, d, t]
                b_vec = B[b, g, :, t] if grouped else B[b, :, t]
                c_vec = C[b, g, :, t] if grouped else C[b, :, t]
                h = np.exp(dt * A[d, :]) * h + (dt * u[b, d, t]) * b_vec
                acc = 0.0
                for n in range(N):
                    acc += c_vec[n] * h[n]
                out[b, d, t] = acc
            last_state[b, d, :] = h     # state after consuming t=0

    if D_skip is not None:
        out = out + u * D_skip.astype(np.float64)[None, :, None]
    if z is not None:
        out = out * silu(z.astype(np.float64))
    return out, last_state


def flip_time(x):
    """Reverse the last axis — the time axis for every time-varying tensor
    (u/delta/z: (B,D,L); B/C: (B,[G,]N,L)). Mirrors bidirectional.py."""
    return None if x is None else np.flip(x, axis=-1).copy()


def backward_via_flip(u, delta, A, B, C, D_skip=None, z=None, delta_bias=None,
                      delta_softplus=False):
    """What `bidirectional.py` does today: flip the time-varying inputs, run
    the ordinary forward scan, flip the output back. Note A / D_skip /
    delta_bias have no time axis and are NOT flipped."""
    out, last = naive_scan_f64(
        flip_time(u), flip_time(delta), A, flip_time(B), flip_time(C),
        D_skip=D_skip, z=flip_time(z), delta_bias=delta_bias,
        delta_softplus=delta_softplus)
    return flip_time(out), last


def make_case(batch, dim, length, state, groups=None, seed=0,
              D=False, z=False, delta_bias=False, delta_softplus=True):
    """Note: `delta` is drawn from a normal (either sign) even in the
    no_softplus case, which DIFFERS from `check_bidirectional.py` on purpose.

    That file must draw a positive delta because the kernel's Pass-A2 exp is
    specialized for dt*A <= 0 and a negative timestep breaks its precondition.
    Here there is no kernel — numpy's exp is exact over the whole line — and the
    identity under test (flip-forward-flip == backward recurrence) is a
    mathematical fact that holds for ANY delta. Restricting the sign would only
    narrow coverage of the thing this file exists to prove. Do not "fix" it to
    match.
    """
    rng = np.random.default_rng(seed)
    f32 = lambda *s: rng.standard_normal(s).astype(np.float32)
    bc_shape = ((batch, groups, state, length) if groups
                else (batch, state, length))
    case = dict(
        u=f32(batch, dim, length),
        delta=f32(batch, dim, length),
        # A < 0 always (models parameterize it as -exp(A_log))
        A=(-rng.random((dim, state)) - 0.1).astype(np.float32),
        B=f32(*bc_shape),
        C=f32(*bc_shape),
        delta_softplus=delta_softplus,
    )
    if D:
        case["D_skip"] = f32(dim)
    if z:
        case["z"] = f32(batch, dim, length)
    if delta_bias:
        case["delta_bias"] = f32(dim)
    return case


CASES = [
    ("tiny",              dict(batch=1, dim=2, length=4,  state=16, seed=1)),
    ("full_opts",         dict(batch=2, dim=4, length=16, state=16, seed=2,
                               D=True, z=True, delta_bias=True)),
    ("no_softplus",       dict(batch=1, dim=2, length=8,  state=16, seed=3,
                               delta_softplus=False)),
    ("state13_neon_tail", dict(batch=1, dim=2, length=8,  state=13, seed=4,
                               D=True, z=True)),
    ("grouped_bc",        dict(batch=2, dim=4, length=12, state=16, groups=2,
                               seed=5, z=True)),
    ("edge_len1",         dict(batch=1, dim=2, length=1,  state=16, seed=6,
                               D=True, z=True)),
    ("longer_seq",        dict(batch=1, dim=2, length=128, state=16, seed=7,
                               D=True, z=True, delta_bias=True)),
]


def check_equivalence(name, case):
    """flip-forward-flip == an explicit backward-in-time recurrence."""
    direct, direct_last = naive_scan_backward_f64(**case)
    via_flip, flip_last = backward_via_flip(**case)

    err = np.abs(direct - via_flip).max()
    bit_identical = np.array_equal(direct, via_flip)
    last_err = np.abs(direct_last - flip_last).max()
    ok = (np.allclose(direct, via_flip, rtol=RTOL, atol=ATOL)
          and np.allclose(direct_last, flip_last, rtol=RTOL, atol=ATOL))

    note = "bit-identical" if bit_identical else f"max_abs={err:.3e}"
    print(f"  {name:20s} out {note:16s} last_state={last_err:.3e}  "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def check_backward_is_not_forward():
    """Guard against a vacuous pass: if the 'backward' scan silently computed
    the forward one, every equivalence check above would still succeed. It must
    actually differ from the forward scan."""
    case = make_case(batch=1, dim=2, length=16, state=16, seed=20, z=True)
    fwd, _ = naive_scan_f64(**case)
    bwd, _ = naive_scan_backward_f64(**case)
    diff = np.abs(fwd - bwd).max()
    ok = diff > 1e-3
    print(f"  {'backward != forward':20s} max_abs={diff:.3e}  "
          f"{'ok' if ok else 'FAIL (backward is a no-op?)'}")
    return ok


def check_gate_commutes_with_linear_merge():
    """`bidirectional.py` lets the kernel apply the z-gate inside BOTH
    directions rather than once after the merge. For a linear merge that is
    algebraically identical; this proves it, because if it were false the
    module's gating would be quietly wrong.

        inside:  (y_f + D·u)·silu(z)  +  (y_b + D·u)·silu(z)
        outside: ((y_f + D·u) + (y_b + D·u))·silu(z)
    """
    case = make_case(batch=1, dim=4, length=16, state=16, seed=21,
                     D=True, z=True, delta_bias=True)
    z = case.pop("z")

    gated_f, _ = naive_scan_f64(**case, z=z)
    gated_b, _ = naive_scan_backward_f64(**case, z=z)
    inside = gated_f + gated_b

    plain_f, _ = naive_scan_f64(**case)
    plain_b, _ = naive_scan_backward_f64(**case)
    outside = (plain_f + plain_b) * silu(z.astype(np.float64))

    err = np.abs(inside - outside).max()
    ok = np.allclose(inside, outside, rtol=RTOL, atol=ATOL)
    print(f"  {'gate commutes/sum':20s} max_abs={err:.3e}  "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def check_d_skip_is_applied_twice():
    """A documented gotcha, asserted so it can never drift silently.

    With merge='sum' and a shared D, the skip connection lands in BOTH
    directions, so the merged output carries 2·D·u — not D·u. That matches how
    real bidirectional Mambas behave (each direction's mixer applies its own D
    and the outputs are summed), but it surprises people, so it is pinned here.
    """
    case = make_case(batch=1, dim=4, length=8, state=16, seed=22, D=True)
    d_skip = case.pop("D_skip")

    with_d = (naive_scan_f64(**case, D_skip=d_skip)[0]
              + naive_scan_backward_f64(**case, D_skip=d_skip)[0])
    without_d = naive_scan_f64(**case)[0] + naive_scan_backward_f64(**case)[0]

    contribution = with_d - without_d
    expected_twice = 2.0 * case["u"].astype(np.float64) * d_skip.astype(
        np.float64)[None, :, None]
    err = np.abs(contribution - expected_twice).max()
    ok = np.allclose(contribution, expected_twice, rtol=RTOL, atol=ATOL)
    print(f"  {'D applied twice':20s} max_abs={err:.3e}  "
          f"{'ok (2·D·u, as documented)' if ok else 'FAIL'}")
    return ok


def main():
    print("bidirectional definition check (pure numpy, no kernel/torch)")
    print("proving: flip -> forward scan -> flip  ==  backward-in-time recurrence\n")

    results = [check_equivalence(name, make_case(**kw)) for name, kw in CASES]
    print()
    results.append(check_backward_is_not_forward())
    results.append(check_gate_commutes_with_linear_merge())
    results.append(check_d_skip_is_applied_twice())

    if not all(results):
        print("\nBIDIRECTIONAL MATH CHECK FAILED")
        sys.exit(1)
    print("\nequivalence holds — flip-forward-flip and a backward-in-time "
          "recurrence are the same function")
    print("(this is the spec the Rust `reverse` flag implements; it is checked "
          "against it bit-for-bit by reverse_matches_flip_forward_flip)")


if __name__ == "__main__":
    main()
