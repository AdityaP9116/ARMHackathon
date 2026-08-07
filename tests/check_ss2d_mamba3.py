"""C1 gate: the 2D Mamba-3 cross-scan, checked layer by layer.

`arm_scan.ss2d_scan_mamba3` adds no kernel code — it is pure layout over
`mamba3_scan_pair`. So the things that can be wrong are layout things, and each
gets its own check rather than being folded into one end-to-end number:

  1. reference reverse == kernel reverse        the primitive the rest builds on
  2. view builders round-trip                   grid -> views -> grid is exact
  3. the two orderings are genuinely different  a transpose that silently
                                                no-ops would pass everything else
  4. kernel vs reference, per direction         the real correctness gate,
                                                BEFORE merge so a direction bug
                                                cannot hide in a sum
  5. angles accumulate per traversal order      the one real correctness trap:
                                                running the pre-pass on the grid
                                                would give both orderings the
                                                row-major theta, and nothing
                                                would raise
  6. merge modes are consistent                 sum/mean/none agree
  7. thread invariance                          identical under RAYON_NUM_THREADS

THERE IS NO AUTHORITATIVE ORACLE FOR 2D, AND THAT IS STATED NOT BURIED.
For 1D we captured ground truth from the official GPU kernels. No 2D Mamba-3
implementation exists to capture from — VNCT's code is unreleased. So this
validates our Rust against *our own reading of the paper*: it proves the kernel
implements the reference, NOT that the reference implements VNCT as intended.

Usage: python tests/check_ss2d_mamba3.py
"""

import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arm_scan.mamba3 import mamba3_scan  # noqa: E402
from arm_scan.ss2d_mamba3 import (  # noqa: E402
    grid_to_views_time_major, ss2d_scan_mamba3, views_to_grid_time_major)
from reference.mamba3_ref import mamba3_siso_ref  # noqa: E402

# (name, batch, H, W, heads, dv, dqk) — grid coverage mirrors the 1D edge
# philosophy: non-square, odd extents, and a degenerate row.
CASES = [
    ("square",     1, 4, 4, 4, 16, 32),
    ("nonsquare",  2, 3, 5, 4, 16, 32),
    ("odd",        1, 5, 7, 2,  8, 16),   # H, W both odd and coprime
    ("wide",       1, 2, 9, 4, 16, 32),
    ("degenerate", 1, 1, 6, 2,  8, 16),   # H = 1: rows and cols coincide
]


def make_case(b, hh, ww, h, dv, dqk, seed=0):
    """Grid-shaped Mamba-3 inputs, in f64. Ranges match the 1D generators'."""
    g = torch.Generator().manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float64)

    r = dqk // 4  # rope_fraction=0.5 => r == dqk//4 angle pairs
    dt = torch.nn.functional.softplus(rn(b, h, hh, ww) * 0.5 - 4.5)
    return dict(
        q=rn(b, hh, ww, 1, dqk), k=rn(b, hh, ww, 1, dqk),
        v=rn(b, hh, ww, h, dv), z=rn(b, hh, ww, h, dv),
        adt=-torch.exp(rn(b, h, hh, ww) * 0.5) * dt,
        dt=dt, trap=rn(b, h, hh, ww),
        q_bias=rn(h, dqk), k_bias=rn(h, dqk),
        angles=rn(b, hh, ww, h, r), D=rn(h),
    )


def ref_scan_pair(q, k, v, adt, dt, trap, q_bias, k_bias, angles=None,
                  D=None, z=None):
    """`mamba3_scan_pair`'s signature, backed by the f64 reference."""
    common = dict(Q_bias=q_bias, K_bias=k_bias, Angles=angles, D=D, Z=z)
    args = dict(Q=q, K=k, V=v, ADT=adt, DT=dt, Trap=trap)
    return (mamba3_siso_ref(**args, **common, reverse=False),
            mamba3_siso_ref(**args, **common, reverse=True))


def _f32(d):
    """Kernel inputs are f32; the reference stays f64."""
    return {kk: (vv.float() if torch.is_tensor(vv) else vv)
            for kk, vv in d.items()}


def _rel(a, b_):
    a, b_ = a.double(), b_.double()
    return float((a - b_).abs().max()) / max(float(b_.abs().max()), 1e-30)


def check_reference_reverse():
    """The reference's new reverse mode must match the kernel's.

    Everything below builds on `ref_scan_pair`, so if this is wrong the rest of
    the gate is measuring the wrong thing and would still look green.
    """
    c = make_case(2, 4, 4, 4, 16, 32, seed=11)
    flat = {kk: (vv.reshape(2, 16, *vv.shape[3:]) if vv.dim() == 5
                 else vv.reshape(2, 4, 16) if vv.dim() == 4 else vv)
            for kk, vv in c.items()}
    f = _f32(flat)
    ker = mamba3_scan(f["q"], f["k"], f["v"], f["adt"], f["dt"], f["trap"],
                      f["q_bias"], f["k_bias"], angles=f["angles"],
                      D=f["D"], z=f["z"], reverse=True)
    ref = mamba3_siso_ref(
        Q=flat["q"], K=flat["k"], V=flat["v"], ADT=flat["adt"],
        DT=flat["dt"], Trap=flat["trap"], Q_bias=flat["q_bias"],
        K_bias=flat["k_bias"], Angles=flat["angles"], D=flat["D"],
        Z=flat["z"], reverse=True)
    err = _rel(ker, ref)
    ok = err < 1e-5
    print(f"  reference reverse == kernel reverse: rel={err:.3e}  "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def check_roundtrip():
    """grid -> views -> grid must be exact, for both layout families."""
    ok = True
    for _, b, hh, ww, h, dv, dqk in CASES:
        t = torch.randn(b, hh, ww, h, dv, dtype=torch.float64)
        v = grid_to_views_time_major(t)
        if tuple(v.shape) != (2 * b, hh * ww, h, dv):
            print(f"  round-trip {hh}x{ww}: view shape {tuple(v.shape)} FAIL")
            ok = False
            continue
        rows, cols = views_to_grid_time_major(v[:b], v[b:], hh, ww)
        if not (torch.equal(rows, t) and torch.equal(cols, t)):
            print(f"  round-trip {hh}x{ww}: not exact  FAIL")
            ok = False
    print(f"  view builders round-trip exactly: {'ok' if ok else 'FAIL'}")
    return ok


def check_orderings_differ():
    """Row-major and column-major must be genuinely different token orders.

    Guards a real failure mode: if the transpose were dropped, every other
    check here would still pass -- the four directions would just silently
    become two, and a 'sum' merge would hide it completely.
    """
    ok = True
    for name, b, hh, ww, h, dv, dqk in CASES:
        t = torch.arange(b * hh * ww * h * dv, dtype=torch.float64).reshape(
            b, hh, ww, h, dv)
        v = grid_to_views_time_major(t)
        same = torch.equal(v[:b], v[b:])
        expect_same = (hh == 1 or ww == 1)  # a 1-D grid has one ordering
        if same != expect_same:
            print(f"  orderings {name} ({hh}x{ww}): same={same}, "
                  f"expected={expect_same}  FAIL")
            ok = False
    print(f"  row/col orderings genuinely differ: {'ok' if ok else 'FAIL'}")
    return ok


def check_angles_per_ordering():
    """theta must accumulate along EACH traversal order, not row-major twice.

    A POSITIVE test, deliberately. The obvious version -- "assert the row and
    col planes differ" -- is far too weak: it passes for the wrong reason as
    soon as any *other* input still reaches the column view correctly, and the
    negative control confirmed exactly that (dropping the transpose from the
    angle path alone left this green because `dt` still arrived transposed).

    So instead: build the column-major sequence independently, run the pre-pass
    on it directly, and require the module's column plane to equal it.
    """
    from arm_scan.mamba3 import angles_to_cos_sin

    from arm_scan.ss2d import grid_to_views
    b, hh, ww, h, dqk = 1, 3, 5, 2, 16
    half, r = dqk // 2, dqk // 4
    ang = torch.randn(b, hh, ww, h, r, dtype=torch.float32)
    dt = torch.nn.functional.softplus(
        torch.randn(b, h, hh, ww, dtype=torch.float32))

    cos_v, sin_v = angles_to_cos_sin(grid_to_views_time_major(ang),
                                     grid_to_views(dt), half)

    # Independently: transpose the grid to column-major, flatten, pre-pass.
    ang_col = ang.transpose(1, 2).reshape(b, ww * hh, h, r)
    dt_col = dt.transpose(2, 3).reshape(b, h, ww * hh)
    cos_col, sin_col = angles_to_cos_sin(ang_col, dt_col, half)

    ok = (torch.allclose(cos_v[b:], cos_col, atol=1e-6)
          and torch.allclose(sin_v[b:], sin_col, atol=1e-6))
    # And the row half must equal the row-major pre-pass, not the column one.
    cos_row, _ = angles_to_cos_sin(ang.reshape(b, hh * ww, h, r),
                                   dt.reshape(b, h, hh * ww), half)
    ok = ok and torch.allclose(cos_v[:b], cos_row, atol=1e-6)
    # Sanity: the two orderings must actually be distinguishable here, or the
    # test above is vacuous.
    distinguishable = not torch.allclose(cos_row, cos_col, atol=1e-6)
    print(f"  angles accumulate per traversal order: "
          f"{'ok' if ok and distinguishable else 'FAIL'}"
          f"{'' if distinguishable else ' (orderings indistinguishable — '
             'test would be vacuous)'}")
    return ok and distinguishable


def check_kernel_vs_reference():
    """The real gate: kernel vs f64 reference, per direction, before merge."""
    ok, worst = True, 0.0
    print(f"  {'case':>11}  {'grid':>7}  {'row_f':>9}  {'row_b':>9}  "
          f"{'col_f':>9}  {'col_b':>9}")
    for i, (name, b, hh, ww, h, dv, dqk) in enumerate(CASES):
        c = make_case(b, hh, ww, h, dv, dqk, seed=i)
        f = _f32(c)
        got = ss2d_scan_mamba3(
            f["q"], f["k"], f["v"], f["adt"], f["dt"], f["trap"],
            f["q_bias"], f["k_bias"], f["angles"], D=f["D"], z=f["z"],
            merge="none")
        want = ss2d_scan_mamba3(
            c["q"], c["k"], c["v"], c["adt"], c["dt"], c["trap"],
            c["q_bias"], c["k_bias"], c["angles"], D=c["D"], z=c["z"],
            merge="none", scan_pair=ref_scan_pair)
        errs = [_rel(g, w) for g, w in zip(got, want)]
        worst = max(worst, max(errs))
        if max(errs) >= 1e-4:
            ok = False
        print(f"  {name:>11}  {f'{hh}x{ww}':>7}  " +
              "  ".join(f"{e:9.2e}" for e in errs))
    print(f"  worst per-direction relative error: {worst:.3e}  "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def check_merges():
    """sum == the four planes added; mean == sum/4."""
    c = make_case(1, 3, 5, 4, 16, 32, seed=7)
    f = _f32(c)
    args = (f["q"], f["k"], f["v"], f["adt"], f["dt"], f["trap"],
            f["q_bias"], f["k_bias"], f["angles"])
    kw = dict(D=f["D"], z=f["z"])
    planes = ss2d_scan_mamba3(*args, **kw, merge="none")
    s = ss2d_scan_mamba3(*args, **kw, merge="sum")
    m = ss2d_scan_mamba3(*args, **kw, merge="mean")
    e1 = _rel(s, sum(planes))
    e2 = _rel(m, s * 0.25)
    ok = e1 == 0.0 and e2 == 0.0
    print(f"  merge sum/mean/none consistent: {e1:.1e}, {e2:.1e}  "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def check_threads():
    """Output must not depend on RAYON_NUM_THREADS."""
    script = (
        "import sys, torch; sys.path.insert(0, r'%s'); "
        "sys.path.insert(0, r'%s'); "
        "from check_ss2d_mamba3 import make_case, _f32; "
        "from arm_scan.ss2d_mamba3 import ss2d_scan_mamba3; "
        "c=_f32(make_case(1,3,5,4,16,32,seed=3)); "
        "o=ss2d_scan_mamba3(c['q'],c['k'],c['v'],c['adt'],c['dt'],c['trap'],"
        "c['q_bias'],c['k_bias'],c['angles'],D=c['D'],z=c['z'],merge='sum'); "
        "print(float(o.double().sum()).hex())"
        % (ROOT / "python", Path(__file__).resolve().parent))
    outs = {}
    for n in ("1", "2", "8"):
        env = dict(os.environ, RAYON_NUM_THREADS=n)
        r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                           text=True, env=env, cwd=str(ROOT))
        if r.returncode != 0:
            print(f"  threads={n}: subprocess failed\n{r.stderr[-400:]}")
            return False
        outs[n] = r.stdout.strip()
    ok = len(set(outs.values())) == 1
    print(f"  bit-identical across RAYON_NUM_THREADS 1/2/8: "
          f"{'ok' if ok else 'FAIL ' + str(outs)}")
    return ok


def main():
    print("2D Mamba-3 cross-scan (C1)\n")
    checks = [
        check_reference_reverse,
        check_roundtrip,
        check_orderings_differ,
        check_angles_per_ordering,
        check_kernel_vs_reference,
        check_merges,
        check_threads,
    ]
    results = [c() for c in checks]
    print()
    if all(results):
        print("SS2D MAMBA-3 CHECK: PASS")
        return 0
    print(f"SS2D MAMBA-3 CHECK: FAIL ({results.count(False)} of "
          f"{len(results)} checks)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
