"""B2 gate: the Rust MIMO kernel, through the real C ABI, vs ground truth.

`verify_golden_mamba3_mimo.py` checks the *PyTorch reference* against the
official TileLang kernel. This checks the *Rust kernel* — the thing that will
actually run on Arm — against the same goldens, and against the reference, all
the way through ctypes and the v7 ABI.

What this covers that neither the Rust unit tests nor the reference gate do:
the angle pre-pass, the `(heads, rank, dv)` projection layouts, the head-major
-> time-major output permute, pointer marshalling for fifteen inputs, and the
all-or-nothing MIMO projection contract at the C boundary.

Usage: python tests/check_mamba3_mimo_op.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arm_scan.mamba3 import mamba3_mimo_scan  # noqa: E402
from reference.mamba3_ref import mamba3_mimo_ref  # noqa: E402
from verify_golden_mamba3 import bf16_ulp_at_scale  # noqa: E402

GOLD = ROOT / "tests" / "golden" / "mamba3_mimo"
KEYS = ("Q", "K", "V", "ADT", "DT", "Trap", "Q_bias", "K_bias", "MIMO_V",
        "MIMO_Z", "MIMO_Out", "Angles", "D", "Z")


def load(name):
    z = np.load(GOLD / f"{name}.npz")
    kw = {k: torch.from_numpy(z[f"kw_{k}"]) for k in KEYS if f"kw_{k}" in z.files}
    return kw, z["out"]


def call(kw):
    return mamba3_mimo_scan(
        kw["Q"], kw["K"], kw["V"], kw["ADT"], kw["DT"], kw["Trap"],
        kw["Q_bias"], kw["K_bias"],
        psi=kw["MIMO_V"], zeta=kw["MIMO_Z"], phi=kw["MIMO_Out"],
        angles=kw["Angles"], D=kw.get("D"), z=kw.get("Z"))


def check_vs_goldens():
    """Kernel vs the official TileLang output, at the bf16 bound."""
    man = json.loads((GOLD / "manifest.json").read_text())
    print(f"  {'case':>6}  {'rank':>4}  {'shape':>20}  {'max ULP':>9}  "
          f"{'rel':>9}")
    worst, ok = 0.0, True
    for c in man["cases"]:
        kw, golden = load(c["name"])
        got = call(kw).numpy().astype(np.float64)
        g = golden.astype(np.float64)
        diff = np.abs(got - g)
        ulps = diff.max() / bf16_ulp_at_scale(g)
        rel = diff.max() / max(np.abs(g).max(), 1e-30)
        worst = max(worst, ulps)
        if ulps > 12.0:
            ok = False
        print(f"  {c['name'][-6:]:>6}  {kw['Q'].shape[2]:>4}  "
              f"{str(tuple(g.shape)):>20}  {ulps:9.2f}  {rel:9.2e}")
    print(f"  worst vs official kernel: {worst:.2f} bf16 ULP  "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def check_vs_reference():
    """Kernel vs the f64 reference — a much tighter bound than the goldens.

    Both implement the same recurrence, so the only gap is fp32 vs f64. This
    is the check that would catch a Rust-side transcription slip that happened
    to sit inside the bf16 noise of the golden comparison.
    """
    kw, _ = load("mamba3_mimo_combined_04")
    got = call(kw).numpy().astype(np.float64)
    want = mamba3_mimo_ref(**kw).numpy()
    rel = np.abs(got - want).max() / max(np.abs(want).max(), 1e-30)
    ok = rel < 1e-5
    print(f"  kernel vs f64 reference: rel={rel:.3e}  {'ok' if ok else 'FAIL'}")
    return ok


def check_rank1_matches_siso_when_unrotated():
    """The free check that survives: rank-1 MIMO == SISO with no rotation.

    The plan predicted r=1 would reproduce SISO outright. It does not — the
    families rotate different lane pairs. But with the rotation removed they
    must agree exactly, and that still exercises the whole rank-r machinery
    against an independently-validated path. Run through BOTH kernels, so this
    is a Rust-vs-Rust check, not a Python one.
    """
    from arm_scan.mamba3 import mamba3_scan
    kw, _ = load("mamba3_mimo_combined_04")
    b, l, r, _, n = kw["Q"].shape
    h, dv = kw["V"].shape[2], kw["V"].shape[3]
    if r != 1:
        # Build a rank-1 slice of the case rather than needing a rank-1 golden.
        kw = {**kw,
              "Q": kw["Q"][:, :, :1], "K": kw["K"][:, :, :1],
              "Q_bias": kw["Q_bias"][:, :1], "K_bias": kw["K_bias"][:, :1],
              "MIMO_V": torch.ones(h, 1, dv), "MIMO_Z": torch.ones(h, 1, dv),
              "MIMO_Out": torch.ones(h, 1, dv)}
    zero = torch.zeros_like(kw["Angles"])
    mimo = mamba3_mimo_scan(
        kw["Q"], kw["K"], kw["V"], kw["ADT"], kw["DT"], kw["Trap"],
        kw["Q_bias"], kw["K_bias"], psi=kw["MIMO_V"], zeta=kw["MIMO_Z"],
        phi=kw["MIMO_Out"], angles=zero, D=kw.get("D"), z=kw.get("Z"))
    siso = mamba3_scan(
        kw["Q"][:, :, 0], kw["K"][:, :, 0], kw["V"], kw["ADT"], kw["DT"],
        kw["Trap"], kw["Q_bias"][:, 0], kw["K_bias"][:, 0], angles=zero,
        D=kw.get("D"), z=kw.get("Z"))
    rel = float((mimo - siso).abs().max()) / max(float(siso.abs().max()), 1e-30)
    ok = rel < 1e-6
    print(f"  rank-1 MIMO == SISO (rotation removed): rel={rel:.3e}  "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def check_partial_projections_rejected():
    """Two of three projections must be an error, not a guess."""
    kw, _ = load("mamba3_mimo_combined_04")
    from arm_scan import _ffi
    b, l, r, _, n = kw["Q"].shape
    h, dv = kw["V"].shape[2], kw["V"].shape[3]
    dims = _ffi.ArmMamba3Dims(b, h, dv, n, l, r)
    out = torch.empty((b, h, l, dv), dtype=torch.float32)
    cos = torch.ones(b, l, h, n // 2)
    sin = torch.zeros(b, l, h, n // 2)
    qf = kw["Q"][:, :, :, 0, :].contiguous()
    kf = kw["K"][:, :, :, 0, :].contiguous()
    ptrs = [qf.data_ptr(), kf.data_ptr(), kw["V"].data_ptr(),
            kw["ADT"].data_ptr(), kw["DT"].data_ptr(), kw["Trap"].data_ptr(),
            kw["Q_bias"].data_ptr(), kw["K_bias"].data_ptr(),
            cos.data_ptr(), sin.data_ptr(), 0, 0,
            kw["MIMO_V"].data_ptr(), kw["MIMO_Z"].data_ptr(), 0]  # phi missing
    try:
        _ffi.mamba3_raw(dims, ptrs, 0, 0, 0, out.data_ptr(), 0, 0)
        print("  partial MIMO projections rejected: FAIL (call succeeded)")
        return False
    except RuntimeError as e:
        ok = "null pointer" in str(e)
        print(f"  partial MIMO projections rejected: "
              f"{'ok' if ok else 'FAIL (' + str(e)[:40] + ')'}")
        return ok


def check_threads():
    """Output must be BIT-identical across thread counts, not merely close.

    `parallel::for_each_head` splits over (batch, head) and each head owns its
    whole recurrence, so there is no cross-thread reduction and no excuse for
    drift. Anything but bit-identity means work is being shared that should not
    be. Run in subprocesses because rayon reads the env once per process.
    """
    import os
    import subprocess
    script = (
        "import sys; sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s'); "
        "import torch; from check_mamba3_mimo_op import load, call; "
        "kw,_ = load('mamba3_mimo_combined_00'); "
        "print(float(call(kw).double().sum()).hex())"
        % (ROOT / "python", Path(__file__).resolve().parent))
    outs = {}
    for n in ("1", "2", "8"):
        r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                           text=True, cwd=str(ROOT),
                           env=dict(os.environ, RAYON_NUM_THREADS=n))
        if r.returncode != 0:
            print(f"  threads={n}: subprocess failed\n{r.stderr[-300:]}")
            return False
        outs[n] = r.stdout.strip()
    ok = len(set(outs.values())) == 1
    print(f"  bit-identical across RAYON_NUM_THREADS 1/2/8: "
          f"{'ok' if ok else 'FAIL ' + str(outs)}")
    return ok


def main():
    if not (GOLD / "manifest.json").is_file():
        raise SystemExit(f"No MIMO goldens at {GOLD}")
    print("Mamba-3 MIMO kernel through the C ABI (v7)\n")
    results = [
        check_vs_goldens(),
        check_vs_reference(),
        check_rank1_matches_siso_when_unrotated(),
        check_partial_projections_rejected(),
        check_threads(),
    ]
    print()
    if all(results):
        print("MAMBA-3 MIMO OP CHECK: PASS")
        return 0
    print(f"MAMBA-3 MIMO OP CHECK: FAIL ({results.count(False)} of "
          f"{len(results)})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
