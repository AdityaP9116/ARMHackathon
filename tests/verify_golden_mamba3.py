"""Stage 1 gate: does the CPU reference reproduce the official kernel?

Runs `tests/reference/mamba3_ref.py` against every Stage-0 golden and reports,
per case, how far the f64 reference lands from what the GPU kernel actually
produced.

THE TOLERANCE, AND WHY IT IS NOT 1e-4
-------------------------------------
`MAMBA3_IMPLEMENTATION_PLAN.md` originally set this gate at "< 1e-4 at f64".
That is unsatisfiable and was corrected: the kernel emits **bf16**, whose
relative epsilon is ~0.4% (8 mantissa bits) — four orders of magnitude above
1e-4. Holding to it would mean hunting a bug that does not exist.

The honest gate compares like with like: round the f64 reference to bf16 and
require agreement to ~1 ULP of bf16. That is *tighter* than a loose absolute
bound, because it says the reference reproduces the kernel to the full
precision the kernel actually carries.

We report both a ULP figure and a relative error so a near-miss is diagnosable
rather than just red.

Usage:  python tests/verify_golden_mamba3.py [--dir tests/golden/mamba3]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference.mamba3_ref import mamba3_siso_ref  # noqa: E402

# Kernel kwarg name -> golden array key.
KEYS = ("Q", "K", "V", "ADT", "DT", "Trap", "Q_bias", "K_bias", "Angles",
        "D", "Z")


def bf16_ulp(x):
    """Size of one bf16 ULP at each magnitude in `x`.

    bf16 has 8 mantissa bits, so the gap between representable neighbours is
    2^(exponent-7). Comparing against this makes "as close as bf16 can express"
    a precise statement instead of a hand-waved tolerance.
    """
    x = np.abs(np.asarray(x, dtype=np.float64))
    exp = np.where(x > 0, np.floor(np.log2(np.maximum(x, 1e-300))), 0.0)
    return np.power(2.0, exp - 7)


def check_case(path):
    z = np.load(path)
    kw = {k: torch.from_numpy(z[f"kw_{k}"]) for k in KEYS if f"kw_{k}" in z.files}
    golden = z["out"].astype(np.float64)
    ref = mamba3_siso_ref(**kw).numpy()

    # Like-for-like: what the reference would be if it were stored as bf16.
    ref_bf16 = torch.from_numpy(ref).to(torch.bfloat16).float().numpy()
    diff = np.abs(ref_bf16 - golden)
    ulp = bf16_ulp(golden)
    ulps = diff / np.maximum(ulp, 1e-300)

    scale = max(np.abs(golden).max(), 1e-30)
    return {
        "max_ulps": float(ulps.max()),
        "mean_ulps": float(ulps.mean()),
        "exact_frac": float((diff == 0).mean()),
        "max_abs": float(diff.max()),
        "rel": float(diff.max() / scale),
        "shape": tuple(golden.shape),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="tests/golden/mamba3")
    ap.add_argument("--max-ulps", type=float, default=1.0,
                    help="gate: worst-case bf16 ULPs the reference may differ")
    args = ap.parse_args()

    d = Path(args.dir)
    man = json.loads((d / "manifest.json").read_text())
    print(f"reference vs official kernel  ({man.get('device')}, "
          f"capture dtype {man.get('capture_dtype')}, "
          f"PR#997 fix {man.get('blackwell_fix_997')})\n")
    print(f"{'case':>6}  {'shape':>20}  {'max ULP':>9}  {'mean ULP':>9}  "
          f"{'exact':>7}  {'rel':>9}")

    worst, rows = 0.0, []
    for c in man["cases"]:
        r = check_case(d / f"{c['name']}.npz")
        rows.append((c["name"], r))
        worst = max(worst, r["max_ulps"])
        print(f"{c['name'][-6:]:>6}  {str(r['shape']):>20}  "
              f"{r['max_ulps']:9.2f}  {r['mean_ulps']:9.3f}  "
              f"{r['exact_frac']:6.1%}  {r['rel']:9.2e}")

    print(f"\nworst case across {len(rows)} goldens: {worst:.2f} bf16 ULP")
    if worst <= args.max_ulps:
        print(f"STAGE 1 GATE: PASS — the paper-form recurrence reproduces the "
              f"official kernel to <= {args.max_ulps} bf16 ULP on every case.")
        return 0
    print(f"STAGE 1 GATE: FAIL — worst {worst:.2f} ULP exceeds "
          f"{args.max_ulps}.\nDo NOT loosen this to make it pass; a reference "
          f"that is merely close is not an oracle.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
