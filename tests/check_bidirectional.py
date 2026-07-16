"""Correctness gate for the bidirectional scan (TOPOLOGY_IMPLEMENTATION_PLAN.md §2.3).

Ground truth is the VENDORED reference (`tests/reference/selective_scan_ref.py`)
run at float64 on explicitly flipped inputs — never our own kernel. That keeps
this an independent check of `arm_scan.bidirectional_scan` rather than a
tautology, exactly as `check_ffi.py` does for the 1D op.

Acceptance is the project-wide gate: max_abs(kernel_f32 - reference_f64) < 1e-4.

Usage:
    cargo build --release -p arm-scan-ffi   # (in kernel/)
    python tests/check_bidirectional.py
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tests"))

import arm_scan  # noqa: E402
from reference import selective_scan_ref  # noqa: E402

MAX_ABS = 1e-4


def _flip(t):
    return None if t is None else torch.flip(t, dims=(-1,))


def reference_bidirectional(u, delta, A, B, C, D=None, z=None,
                            delta_bias=None, delta_softplus=False,
                            merge="sum", reverse_params=None):
    """Bidirectional scan computed in float64 through the vendored reference.

    The backward direction is *defined* as: flip every time-varying tensor,
    run the ordinary forward reference, flip the output back. This is the
    definition `arm_scan.bidirectional_scan` claims to implement, computed
    independently of it.

    The INPUTS are upcast to double — `selective_scan_ref` casts its result
    back to the input dtype on the way out, so passing f32 with
    compute_dtype=float64 would silently hand back an f32 answer and make this
    a far weaker gate than it claims. `gen_golden.py` upcasts for the same
    reason; this mirrors it.
    """
    p = reverse_params or {}
    f64 = lambda t: None if t is None else t.double()
    kw = dict(delta_softplus=delta_softplus, compute_dtype=torch.float64)

    out_f = selective_scan_ref(
        f64(u), f64(delta), f64(A), f64(B), f64(C),
        D=f64(D), z=f64(z), delta_bias=f64(delta_bias), **kw)

    out_b = selective_scan_ref(
        f64(_flip(u)), f64(_flip(p.get("delta", delta))), f64(p.get("A", A)),
        f64(_flip(p.get("B", B))), f64(_flip(p.get("C", C))),
        D=f64(p.get("D", D)), z=f64(_flip(z)),
        delta_bias=f64(p.get("delta_bias", delta_bias)), **kw)
    out_b = _flip(out_b)

    if merge == "sum":
        return out_f + out_b
    if merge == "mean":
        return (out_f + out_b) * 0.5
    if merge == "concat":
        return torch.cat((out_f, out_b), dim=1)
    return out_f, out_b


def make_case(batch, dim, length, state, groups=1, seed=0, **opts):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32)
    softplus = opts.get("delta_softplus", True)

    # The kernel's Pass-A2 exp is `vexpq_f32_nonpos`, specialized for the
    # scan's always-non-positive argument dt*A. That holds because A < 0 and
    # the timestep dt >= 0 — and dt >= 0 is only guaranteed if delta is either
    # raw (softplus applied inside the kernel) or ALREADY positive. So when
    # delta_softplus=False, delta *is* the timestep and must be drawn positive,
    # exactly as gen_golden.py does for its own no_softplus case. Feeding a
    # negative delta here violates the kernel's documented precondition and is
    # a regime no real Mamba ever produces (HF's slow path pre-applies
    # softplus). See BIDIRECTIONAL_LOG.md.
    if softplus:
        delta = r(batch, dim, length)                      # raw, any sign
    else:
        delta = (torch.rand(batch, dim, length, generator=g,
                            dtype=torch.float32) * 0.099) + 1e-3  # (0.001, 0.1]

    case = dict(
        u=r(batch, dim, length),
        delta=delta,
        # A must be negative (the model parameterizes it as -exp(A_log)).
        A=-torch.rand(dim, state, generator=g, dtype=torch.float32) - 0.1,
        B=r(batch, groups, state, length),
        C=r(batch, groups, state, length),
    )
    if opts.get("D"):
        case["D"] = r(dim)
    if opts.get("z"):
        case["z"] = r(batch, dim, length)
    if opts.get("delta_bias"):
        if not softplus:
            # a bias could push the raw positive timestep negative, breaking
            # the same precondition; gen_golden.py forces bias=None here too.
            raise ValueError("delta_bias with delta_softplus=False would "
                             "violate the dt >= 0 precondition")
        case["delta_bias"] = r(dim)
    case["delta_softplus"] = softplus
    return case


CASES = [
    # (name, case kwargs, bidirectional_scan kwargs)
    ("tiny_sum", dict(batch=1, dim=4, length=8, state=16, seed=1), {}),
    ("full_opts_sum",
     dict(batch=2, dim=8, length=32, state=16, seed=2,
          D=True, z=True, delta_bias=True),
     {}),
    ("no_softplus",
     dict(batch=1, dim=4, length=16, state=16, seed=3, delta_softplus=False),
     {}),
    ("state13_neon_tail",
     dict(batch=1, dim=4, length=16, state=13, seed=4, D=True, z=True),
     {}),
    ("grouped_bc",
     dict(batch=2, dim=8, length=24, state=16, groups=2, seed=5, z=True),
     {}),
    ("edge_len1", dict(batch=1, dim=4, length=1, state=16, seed=6), {}),
    ("long_seq",
     dict(batch=1, dim=8, length=512, state=16, seed=7, D=True, z=True,
          delta_bias=True),
     {}),
    ("merge_mean",
     dict(batch=1, dim=4, length=16, state=16, seed=8, z=True),
     dict(merge="mean")),
    ("merge_concat",
     dict(batch=1, dim=4, length=16, state=16, seed=9, z=True),
     dict(merge="concat")),
]


def check(name, case, kwargs):
    got = arm_scan.bidirectional_scan(**case, **kwargs)
    want = reference_bidirectional(**case, **kwargs)
    err = (got.double() - want).abs().max().item()
    ok = err < MAX_ABS
    print(f"  {name:22s} max_abs={err:.3e}  {'ok' if ok else 'FAIL'}")
    return ok


def check_untied():
    """Backward direction with its own weights (Vim `bimamba_type=v2` style):
    a separate A, D and delta must actually be used, and must NOT be flipped
    by the wrapper (the caller supplies them in forward-time order)."""
    case = make_case(batch=1, dim=4, length=16, state=16, seed=10,
                     D=True, z=True)
    g = torch.Generator().manual_seed(99)
    rev = dict(
        A=-torch.rand(4, 16, generator=g, dtype=torch.float32) - 0.1,
        D=torch.randn(4, generator=g, dtype=torch.float32),
        delta=torch.randn(1, 4, 16, generator=g, dtype=torch.float32),
    )
    got = arm_scan.bidirectional_scan(**case, reverse_params=rev)
    want = reference_bidirectional(**case, reverse_params=rev)
    err = (got.double() - want).abs().max().item()
    ok = err < MAX_ABS
    print(f"  {'untied_reverse_params':22s} max_abs={err:.3e} "
          f"{'ok' if ok else 'FAIL'}")

    # A wrapper that silently ignored reverse_params would still pass the
    # check above only if the tied result happened to match — prove it does not.
    tied = arm_scan.bidirectional_scan(**case)
    differs = (got - tied).abs().max().item() > 1e-3
    print(f"  {'untied != tied':22s} "
          f"{'ok' if differs else 'FAIL (params ignored?)'}")
    return ok and differs


def check_merge_none():
    """merge='none' returns both directions unmerged, and summing them by hand
    must reproduce merge='sum' — i.e. the escape hatch is consistent."""
    case = make_case(batch=1, dim=4, length=16, state=16, seed=11, z=True)
    out_f, out_b = arm_scan.bidirectional_scan(**case, merge="none")
    summed = arm_scan.bidirectional_scan(**case, merge="sum")
    err = (out_f + out_b - summed).abs().max().item()
    ok = err == 0.0
    print(f"  {'merge_none':22s} max_abs={err:.3e}  {'ok' if ok else 'FAIL'}")
    return ok


def check_forward_direction_unchanged():
    """The forward half of a bidirectional scan must be bit-identical to a
    plain 1D scan — a regression here would mean the wrapper perturbed the
    path the rest of the project already validates."""
    case = make_case(batch=1, dim=4, length=16, state=16, seed=12, D=True,
                     z=True, delta_bias=True)
    out_f, _ = arm_scan.bidirectional_scan(**case, merge="none")
    plain = arm_scan.selective_scan(
        case["u"], case["delta"], case["A"], case["B"], case["C"],
        D=case.get("D"), z=case.get("z"), delta_bias=case.get("delta_bias"),
        delta_softplus=case["delta_softplus"])
    ok = torch.equal(out_f, plain)
    print(f"  {'fwd == plain 1D scan':22s} {'ok (bit-identical)' if ok else 'FAIL'}")
    return ok


def main():
    print(f"kernel library: {arm_scan.lib_path()}")
    failures = []

    for name, case_kw, kwargs in CASES:
        case = make_case(**case_kw)
        if not check(name, case, kwargs):
            failures.append(name)

    if not check_untied():
        failures.append("untied_reverse_params")
    if not check_merge_none():
        failures.append("merge_none")
    if not check_forward_direction_unchanged():
        failures.append("fwd_vs_plain")

    if failures:
        print(f"\nBIDIRECTIONAL CHECK FAILED: {failures}")
        sys.exit(1)
    print(f"\nall bidirectional cases pass (max_abs < {MAX_ABS:g} vs f64 reference)")


if __name__ == "__main__":
    main()
