"""B1 gate: does the CPU MIMO reference reproduce the official TileLang kernel?

The MIMO counterpart of `verify_golden_mamba3.py`. Same instrument, different
tolerance, and the difference is worth stating rather than inheriting:

  SISO's Triton kernel hardcodes `.to(torch.bfloat16)` on Q/K/V/Trap/Angles/Z,
  so its goldens are bf16-limited no matter how the model was loaded.
  MIMO's TileLang kernel instead **types its tiles on the caller's dtype**, and
  we capture at bf16 because fp32 does not fit consumer Blackwell's shared
  memory. So MIMO goldens are also bf16 — but for a different reason, and one
  that would change on a datacenter card where fp32 fits.

Usage:  python tests/verify_golden_mamba3_mimo.py [--dir tests/golden/mamba3_mimo]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference.mamba3_ref import mamba3_mimo_ref  # noqa: E402
from verify_golden_mamba3 import bf16_ulp_at_scale  # noqa: E402

KEYS = ("Q", "K", "V", "ADT", "DT", "Trap", "Q_bias", "K_bias", "MIMO_V",
        "MIMO_Z", "MIMO_Out", "Angles", "D", "Z")


def check_case(path):
    z = np.load(path)
    kw = {k: torch.from_numpy(z[f"kw_{k}"]) for k in KEYS if f"kw_{k}" in z.files}
    golden = z["out"].astype(np.float64)
    ref = mamba3_mimo_ref(**kw).numpy()

    ref_bf16 = torch.from_numpy(ref).to(torch.bfloat16).float().numpy()
    diff = np.abs(ref_bf16 - golden)
    ulp = bf16_ulp_at_scale(golden)
    scale = max(float(np.abs(golden).max()), 1e-30)
    return {
        "max_ulps": float(diff.max() / ulp),
        "mean_ulps": float(diff.mean() / ulp),
        "exact_frac": float((diff == 0).mean()),
        "rel": float(diff.max() / scale),
        "shape": tuple(golden.shape),
        "rank": int(kw["Q"].shape[2]),
    }


def check_rank1_collapse(siso_dir="tests/golden/mamba3"):
    """Does r=1 MIMO collapse to SISO? NO — and the reason is a real finding.

    `THREE_PATHS_INTEGRATION.md` predicted "r=1 reproduces the SISO goldens
    bit-for-bit -- a free correctness check, exactly like lambda=1 collapsing
    the trapezoid". That prediction is **wrong**, and it is worth keeping the
    disproof rather than quietly dropping the check.

    At r=1 with unit Psi/Phi/Zeta, every MIMO-specific term degenerates: the
    rank sum has one term, the rank-by-rank contraction is 1x1, `x = v`, and
    `silu(z * 1) = silu(z)`. So the two recurrences ARE algebraically the same
    there -- except for one thing: **SISO rotates interleaved lanes (2i, 2i+1)
    and MIMO rotates split halves (i, i + n/2)**. Two kernels of the same model
    family, different rotation conventions.

    This test proves that is the *only* difference, by running the same
    comparison twice: once as-is (must differ), and once with the angles zeroed
    so both rotations become the identity (must then agree to f64 precision).
    A single "they differ" assertion would not distinguish a convention
    difference from a genuine bug in the rank-r generalisation.
    """
    from reference.mamba3_ref import mamba3_siso_ref

    p = Path(siso_dir) / "mamba3_siso_combined_00.npz"
    if not p.is_file():
        print(f"\n(skipping r=1 collapse: {p} not found)")
        return True
    z = np.load(p)
    kw = {k: torch.from_numpy(z[f"kw_{k}"]) for k in
          ("Q", "K", "V", "ADT", "DT", "Trap", "Q_bias", "K_bias", "Angles",
           "D", "Z") if f"kw_{k}" in z.files}
    h, dqk = kw["Q_bias"].shape
    dv = kw["V"].shape[-1]
    one = torch.ones(h, 1, dv, dtype=torch.float64)

    def both(angles):
        siso = mamba3_siso_ref(**{**kw, "Angles": angles})
        mimo = mamba3_mimo_ref(
            Q=kw["Q"].unsqueeze(2), K=kw["K"].unsqueeze(2), V=kw["V"],
            ADT=kw["ADT"], DT=kw["DT"], Trap=kw["Trap"],
            Q_bias=kw["Q_bias"].unsqueeze(1), K_bias=kw["K_bias"].unsqueeze(1),
            MIMO_V=one, MIMO_Z=one, MIMO_Out=one, Angles=angles,
            D=kw.get("D"), Z=kw.get("Z"))
        return float((siso - mimo).abs().max()) / max(
            float(siso.abs().max()), 1e-30)

    with_rot = both(kw["Angles"])
    no_rot = both(torch.zeros_like(kw["Angles"]))

    print(f"\nr=1 collapse to SISO (the plan predicted bit-for-bit):")
    print(f"  as captured, rotation active : rel {with_rot:.3e}  "
          f"-> {'DIFFER (expected)' if with_rot > 1e-6 else 'agree'}")
    print(f"  with angles zeroed           : rel {no_rot:.3e}  "
          f"-> {'agree' if no_rot < 1e-12 else 'DIFFER'}")
    ok = with_rot > 1e-6 and no_rot < 1e-12
    if ok:
        print("  CONFIRMED: r=1 MIMO equals SISO in every term EXCEPT the "
              "rotation convention (SISO interleaved, MIMO split-halves).\n"
              "  The plan's 'bit-for-bit' prediction does not hold, and this "
              "is why. B2 needs BOTH conventions in the kernel.")
    else:
        print("  UNEXPECTED: the two differ for some reason other than RoPE, "
              "or agree when they should not. Investigate before trusting "
              "the rank-r generalisation.")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="tests/golden/mamba3_mimo")
    ap.add_argument("--max-ulps", type=float, default=12.0,
                    help="gate: worst-case bf16 ULPs at tensor scale. Looser "
                         "than SISO's 8 because MIMO sums r rank-1 terms into "
                         "one shared state and then reduces r gated streams on "
                         "output, so it accumulates more bf16 roundings per "
                         "result. Still tight: a structurally wrong recurrence "
                         "-- wrong RoPE convention, gamma on D, the rank "
                         "contraction transposed -- lands in the hundreds or "
                         "thousands, not at 15.")
    args = ap.parse_args()

    d = Path(args.dir)
    man = json.loads((d / "manifest.json").read_text())
    print(f"MIMO reference vs official kernel  ({man.get('device')}, "
          f"capture dtype {man.get('capture_dtype')}, "
          f"model {man.get('model')})\n")
    print(f"{'case':>6}  {'rank':>4}  {'shape':>20}  {'max ULP':>9}  "
          f"{'mean ULP':>9}  {'exact':>7}  {'rel':>9}")

    worst, n = 0.0, 0
    for c in man["cases"]:
        r = check_case(d / f"{c['name']}.npz")
        worst = max(worst, r["max_ulps"])
        n += 1
        print(f"{c['name'][-6:]:>6}  {r['rank']:>4}  {str(r['shape']):>20}  "
              f"{r['max_ulps']:9.2f}  {r['mean_ulps']:9.3f}  "
              f"{r['exact_frac']:6.1%}  {r['rel']:9.2e}")

    print(f"\nworst case across {n} goldens: {worst:.2f} bf16 ULP")
    collapse_ok = check_rank1_collapse()
    print()
    if worst <= args.max_ulps and collapse_ok:
        print(f"B1 MIMO GATE: PASS — the rank-r recurrence reproduces the "
              f"official TileLang kernel to <= {args.max_ulps} bf16 ULP on "
              f"every case.")
        return 0
    if worst <= args.max_ulps:
        print("B1 MIMO GATE: FAIL — the goldens match, but the r=1 collapse "
              "behaved unexpectedly; see above.")
        return 1
    print(f"B1 MIMO GATE: FAIL — worst {worst:.2f} ULP exceeds "
          f"{args.max_ulps}.\nDo NOT loosen this to make it pass; a reference "
          f"that is merely close is not an oracle.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
