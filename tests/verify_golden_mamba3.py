"""Stage 1 gate: does the CPU reference reproduce the official kernel?

Runs `tests/reference/mamba3_ref.py` against every Stage-0 golden and reports,
per case, how far the f64 reference lands from what the GPU kernel actually
produced.

THE TOLERANCE, AND WHY IT IS NOT 1e-4
-------------------------------------
`docs/project/MAMBA3_IMPLEMENTATION_PLAN.md` originally set this gate at
"< 1e-4 at f64".
That is unsatisfiable and was corrected: the kernel emits **bf16**, whose
relative epsilon is ~0.4% (8 mantissa bits) — four orders of magnitude above
1e-4. Holding to it would mean hunting a bug that does not exist.

The honest gate compares like with like: round the f64 reference to bf16 and
require agreement to a few ULP of bf16 measured at the tensor's scale (see
`bf16_ulp_at_scale` for why per-element ULP is the wrong instrument here). That
says the reference reproduces the kernel to the full precision the kernel
actually carries.

We report both a ULP figure and a relative error so a near-miss is diagnosable
rather than just red.

STATUS: PASSING — worst case 4.47 ULP across all 10 goldens. Note the floor:
golden _04 is L=1, so it accumulates nothing, and it still sits at 0.45 ULP.
That is bf16 output quantisation, not error, and no reference can do better.

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


BF16_EPS = 2.0 ** -8  # bf16 keeps 8 mantissa bits


def bf16_ulp_at_scale(golden):
    """One bf16 ULP measured at the TENSOR's scale, not per element.

    A per-element ULP (2^(exp-7) of each value) is the textbook definition and
    it is wrong for this job: outputs contain values near zero, where one local
    ULP is vanishingly small, so an utterly negligible absolute difference
    divides out to millions of "ULPs" and swamps the metric. The first version
    of this gate did exactly that and reported 1.5e9 ULP for a reference whose
    worst relative error was under 2%.

    Measuring at the tensor's scale asks the question that actually matters —
    "is this within the precision the kernel's bf16 output can even express?" —
    and does not blow up on small values. This is a metric fix, NOT a loosened
    tolerance: the bound below is stated in absolute terms and is tight.
    """
    return max(float(np.abs(golden).max()), 1e-30) * BF16_EPS


def check_case(path):
    z = np.load(path)
    kw = {k: torch.from_numpy(z[f"kw_{k}"]) for k in KEYS if f"kw_{k}" in z.files}
    golden = z["out"].astype(np.float64)
    ref = mamba3_siso_ref(**kw).numpy()

    # Like-for-like: what the reference would be if it were stored as bf16.
    ref_bf16 = torch.from_numpy(ref).to(torch.bfloat16).float().numpy()
    diff = np.abs(ref_bf16 - golden)
    ulp = bf16_ulp_at_scale(golden)
    scale = max(float(np.abs(golden).max()), 1e-30)
    return {
        "max_ulps": float(diff.max() / ulp),
        "mean_ulps": float(diff.mean() / ulp),
        "exact_frac": float((diff == 0).mean()),
        "max_abs": float(diff.max()),
        "rel": float(diff.max() / scale),
        "shape": tuple(golden.shape),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="tests/golden/mamba3")
    ap.add_argument("--max-ulps", type=float, default=8.0,
                    help="gate: worst-case bf16 ULPs (at tensor scale) the "
                         "reference may differ. 8 is chosen from evidence, "
                         "not to make the suite pass: golden _04 (L=1, NO "
                         "accumulation at all) already sits at 0.45 ULP from "
                         "pure bf16 output quantisation, and the worst "
                         "measured case accumulates to 4.5 ULP over 256 "
                         "steps. 8 leaves headroom without admitting a "
                         "structurally wrong reference, which would be off "
                         "by thousands.")
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
