"""Independently re-derive the 2D goldens, and replay them through the kernel.

Three jobs, matching what `verify_golden.py` + `check_ffi.py` do for 1D:

  1. REDRAW — regenerate every case's inputs from its name alone and compare
     them to the stored ones bit-for-bit, so the committed npz files are known
     to be reproducible rather than merely present. Numpy-only (the draws live
     in `golden_inputs.py` for exactly that reason), so it runs on every
     platform including the torch-free tier.

  2. VERIFY — recompute every direction plane in plain numpy, from the stored
     inputs, with no torch and no reference import. The goldens must not rest
     on the strength of one implementation; this is the second one.

  3. REPLAY (optional) — run the same inputs through `arm_scan.ss2d.ss2d_scan`
     (i.e. the real cdylib, via the fused bidirectional C ABI) and check each
     plane against the f64 ground truth at the standing 1e-4 gate, reporting
     each case's recorded f32 floor so a result that is merely "under the
     gate" but orders of magnitude above the floor is visible as the
     regression it would be. Skipped with a clear message when the library
     isn't built.

Usage:
    python tests/verify_golden_2d.py            # verify + replay if possible
    python tests/verify_golden_2d.py --no-kernel
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

GOLDEN_DIR = Path(__file__).parent / "golden" / "2d"
GATE = 1e-4


def softplus(x):
    # log1p(exp(-|x|)) + max(x, 0): the standard overflow-safe form.
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def scan_1d(u, delta, A, B, C, D, delta_bias):
    """Reference recurrence, numpy, float64. u/delta (b,d,l); A (d,n);
    B/C (b,n,l); D/delta_bias (d,) -> (b,d,l)."""
    b, d, ln = u.shape
    n = A.shape[1]
    dt = softplus(delta + delta_bias[None, :, None])
    dA = np.exp(dt[..., None] * A[None, :, None, :])       # (b,d,l,n)
    dBu = dt[..., None] * B.transpose(0, 2, 1)[:, None, :, :] * u[..., None]
    h = np.zeros((b, d, n), dtype=np.float64)
    out = np.empty((b, d, ln), dtype=np.float64)
    for t in range(ln):
        h = dA[:, :, t] * h + dBu[:, :, t]
        out[:, :, t] = np.einsum("bdn,bn->bd", h, C[:, :, t])
    return out + u * D[None, :, None]


def cross_scan_numpy(z):
    """The four direction planes from stored inputs, independently derived."""
    u, delta = z["u"].astype(np.float64), z["delta"].astype(np.float64)
    A, B, C = (z["A"].astype(np.float64), z["B"].astype(np.float64),
               z["C"].astype(np.float64))
    D, bias = z["D"].astype(np.float64), z["delta_bias"].astype(np.float64)
    b, d, h, w = u.shape

    def views(t):
        return (t.reshape(t.shape[0], t.shape[1], h * w),
                t.transpose(0, 1, 3, 2).reshape(t.shape[0], t.shape[1], w * h))

    def pair(uu, dd, BB, CC):
        fwd = scan_1d(uu, dd, A, BB, CC, D, bias)
        bwd = scan_1d(uu[..., ::-1], dd[..., ::-1], A, BB[..., ::-1],
                      CC[..., ::-1], D, bias)[..., ::-1]
        return fwd, bwd

    u_r, u_c = views(u)
    d_r, d_c = views(delta)
    B_r, B_c = views(B)
    C_r, C_c = views(C)
    row_f, row_b = pair(u_r, d_r, B_r, C_r)
    col_f, col_b = pair(u_c, d_c, B_c, C_c)
    return (row_f.reshape(b, d, h, w), row_b.reshape(b, d, h, w),
            col_f.reshape(b, d, w, h).transpose(0, 1, 3, 2),
            col_b.reshape(b, d, w, h).transpose(0, 1, 3, 2))


KEYS = ("row_fwd", "row_bwd", "col_fwd", "col_bwd")


def check_determinism(manifest):
    """Redrawing every case must reproduce the stored inputs bit-for-bit.

    Numpy-only, and it never skips: this set previously had no determinism
    check at all while drawing through `torch.Generator`, whose stream torch
    does not hold stable across releases — so `golden/2d/*.npz` could not have
    been regenerated under a different torch and nothing would have said so.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from golden_inputs import CORE_CASES_2D, INPUT_DRAW_SPEC, draw_inputs_2d

    ok = True
    # A manifest entry with a different (or missing) spec was drawn by code
    # that no longer exists here; say that rather than report a bit mismatch.
    spec = {m["name"]: m.get("input_draw") for m in manifest}
    stale = sorted(n for n, v in spec.items() if v != INPUT_DRAW_SPEC)
    if stale:
        print(f"   FAIL: {len(stale)} case(s) drawn by a different input spec "
              f"than {INPUT_DRAW_SPEC}: {', '.join(stale)}")
        print("   (regenerate with python tests/gen_golden_2d.py)")
        return False

    known = {name for name, *_ in CORE_CASES_2D}
    orphans = sorted(set(spec) - known)
    if orphans:
        print(f"   FAIL: manifest case(s) absent from CORE_CASES_2D, so their "
              f"inputs cannot be redrawn: {', '.join(orphans)}")
        ok = False

    for name, b, d, h, w, n in CORE_CASES_2D:
        path = GOLDEN_DIR / f"{name}.npz"
        if not path.exists():
            print(f"   {name:16s} MISSING {path.name}")
            ok = False
            continue
        z = np.load(path)
        redrawn = draw_inputs_2d(name, b, d, h, w, n)
        bad = [k for k, arr in redrawn.items()
               if k not in z or not np.array_equal(z[k], arr)]
        ok &= not bad
        print(f"   {name:16s} {len(redrawn)} arrays  "
              f"{'PASS' if not bad else 'FAIL differs: ' + ', '.join(bad)}")
    if ok:
        print(f"   all redrawn bit-identical ({INPUT_DRAW_SPEC}, no torch)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-kernel", action="store_true")
    args = ap.parse_args()

    manifest = json.loads((GOLDEN_DIR / "manifest.json").read_text())
    print(f"{len(manifest)} 2D golden cases\n")

    print("1. input determinism — redraw from the case name, numpy only")
    ok = check_determinism(manifest)

    print("\n2. independent numpy re-derivation vs stored f64 ground truth")
    for case in manifest:
        z = np.load(GOLDEN_DIR / f"{case['name']}.npz")
        planes = cross_scan_numpy(z)
        worst = max(float(np.abs(p - z[f"out_{k}"]).max())
                    for p, k in zip(planes, KEYS))
        # Two f64 implementations of the same recurrence: agreement should be
        # at f64 round-off, ~1e-12, not merely under the f32 gate.
        good = worst < 1e-9
        ok &= good
        print(f"   {case['name']:16s} max_abs {worst:.3e}  "
              f"{'PASS' if good else 'FAIL'}")

    if args.no_kernel:
        print("\n3. kernel replay SKIPPED (--no-kernel)")
        print("\nRESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    print("\n3. kernel replay through arm_scan.ss2d (real C ABI)")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
        import torch
        from arm_scan._ffi import load
        from arm_scan.ss2d import ss2d_scan
        load()
    except Exception as exc:  # noqa: BLE001
        print(f"   SKIPPED — arm_scan unavailable: {exc}")
        print("   (build it: cargo build --release -p arm-scan-ffi)")
        print("\nRESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    for case in manifest:
        z = np.load(GOLDEN_DIR / f"{case['name']}.npz")
        t = {k: torch.from_numpy(np.ascontiguousarray(z[k]))
             for k in ("u", "delta", "A", "B", "C", "D", "delta_bias")}
        planes = ss2d_scan(
            t["u"], t["delta"], t["A"], t["B"], t["C"], D=t["D"],
            delta_bias=t["delta_bias"], delta_softplus=True, merge="none")
        worst = max(float(np.abs(p.numpy().astype(np.float64)
                                 - z[f"out_{k}"]).max())
                    for p, k in zip(planes, KEYS))
        floor = case["f32_max_abs_err"]
        good = worst < GATE
        ok &= good
        ratio = worst / floor if floor > 0 else float("inf")
        print(f"   {case['name']:16s} max_abs {worst:.3e}  "
              f"floor {floor:.3e}  ({ratio:5.1f}x floor)  "
              f"{'PASS' if good else 'FAIL'}")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
