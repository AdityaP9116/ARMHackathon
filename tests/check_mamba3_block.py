"""Path A gate: does our plain-PyTorch mixer reproduce the real one?

This is the checkpoint-level counterpart to `verify_golden_mamba3.py`. That one
asks "does the reference recurrence match the official kernel?" — a question
about *maths*. This one asks "does the block wrapped around that recurrence
match the official block, driven by the published weights?" — a question about
*plumbing*: projection split order, the heavy-tail decay, the softplus, the B/C
norms, the Q<-C / K<-B mapping, the bias squeeze, and `out_proj`.

WHY THIS EXISTS RATHER THAN JUST COMPARING LOGITS
-------------------------------------------------
An end-to-end logits mismatch says "something is wrong somewhere in twelve
identical layers". This says which of the eight parameters or six steps is
wrong, on layer 0, in one forward pass. Stage 1 already paid for this lesson
once: the RoPE convention was wrong (split-halves instead of interleaved) and
only a boundary-level oracle localised it.

THE TOLERANCE
-------------
The goldens were captured from a model run in fp32, but `mamba3_siso_combined`
downcasts Q/K/V/Trap/Angles/Z to **bf16** internally, so `hidden_out` carries
bf16-scale error no matter what we do. The gate is therefore the same
instrument Stage 1 uses — bf16 ULPs measured at the tensor's scale — and NOT
1e-4, which is unsatisfiable here for exactly the reason documented in
`verify_golden_mamba3.py`.

Usage:  python tests/check_mamba3_block.py [--dir tests/golden/mamba3]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "apps"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mamba3_lm.block import Mamba3Mixer  # noqa: E402
from verify_golden_mamba3 import bf16_ulp_at_scale  # noqa: E402


def unpack_param(arr, name, bf16_params):
    """Reverse `_pack_param`: int16-encoded bf16 back to a real tensor.

    The published weights ARE bfloat16, so the goldens store them as such
    (halving what goes into git forever) with the bits carried as int16 because
    numpy has no bfloat16 dtype. `bf16_params` comes from the manifest — an
    int16 array is otherwise indistinguishable from a genuine one.
    """
    t = torch.from_numpy(arr)
    if name in bf16_params:
        if t.dtype != torch.int16:
            raise ValueError(
                f"manifest says {name} is bf16-encoded but the array is "
                f"{t.dtype}; the golden and its manifest disagree")
        return t.view(torch.bfloat16).float()
    return t.float()


def build_from_golden(z, bf16_params=()):
    """Construct a mixer whose config is INFERRED from the golden's shapes.

    Deliberately derived rather than hardcoded: if a golden is ever recaptured
    at a different width, a config mismatch should surface as a shape error
    here instead of as a silent numerical difference.
    """
    bf16_params = set(bf16_params)
    d_in_proj, d_model = z["param_in_proj.weight"].shape
    d_inner = z["param_out_proj.weight"].shape[1]
    nheads = z["param_D"].shape[0]
    d_state = z["param_B_norm.weight"].shape[0]
    headdim = d_inner // nheads

    mixer = Mamba3Mixer(d_model=d_model, d_state=d_state,
                        expand=d_inner // d_model, headdim=headdim)
    assert sum(mixer.split_sizes) == d_in_proj, (
        f"in_proj width {d_in_proj} does not match the split "
        f"{mixer.split_sizes} summing to {sum(mixer.split_sizes)} — the "
        f"projection layout has changed")

    sd = {k[len("param_"):]: unpack_param(z[k], k[len("param_"):], bf16_params)
          for k in z.files if k.startswith("param_")}
    # strict=True: a renamed or missing parameter must fail loudly here rather
    # than leave an initialised-but-unloaded tensor in the graph.
    mixer.load_state_dict(sd, strict=True)
    return mixer


def check_case(path, bf16_params=()):
    z = np.load(path)
    mixer = build_from_golden(z, bf16_params).eval()
    hidden_in = torch.from_numpy(z["hidden_in"])
    golden = z["hidden_out"].astype(np.float64)

    with torch.no_grad():
        got = mixer(hidden_in).numpy().astype(np.float64)

    if got.shape != golden.shape:
        raise SystemExit(f"shape mismatch: got {got.shape}, want {golden.shape}")

    diff = np.abs(got - golden)
    ulp = bf16_ulp_at_scale(golden)
    scale = max(float(np.abs(golden).max()), 1e-30)
    return {
        "max_ulps": float(diff.max() / ulp),
        "mean_ulps": float(diff.mean() / ulp),
        "max_abs": float(diff.max()),
        "rel": float(diff.max() / scale),
        "shape": tuple(golden.shape),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="tests/golden/mamba3")
    ap.add_argument("--max-ulps", type=float, default=16.0,
                    help="gate: worst-case bf16 ULPs at tensor scale. Looser "
                         "than verify_golden_mamba3's 8 because this measures "
                         "AFTER out_proj, which sums 1536 bf16-derived terms "
                         "per output and so accumulates more than the scan "
                         "output does. Still tight enough that a structurally "
                         "wrong block — a mis-split projection, a swapped B/C "
                         "— lands in the thousands, not at 20.")
    args = ap.parse_args()

    d = Path(args.dir)
    man = json.loads((d / "manifest.json").read_text())
    cases = man.get("block_io", [])
    if not cases:
        raise SystemExit(
            f"No block-level goldens in {d}/manifest.json. Regenerate with "
            f"`python tools/capture_mamba3_goldens.py` on a GPU box — this "
            f"gate cannot run without them.")

    print(f"our mixer vs the real block  ({man.get('model')}, "
          f"{man.get('device')}, capture dtype {man.get('capture_dtype')})\n")
    print(f"{'case':>12}  {'layer':>5}  {'shape':>18}  {'max ULP':>9}  "
          f"{'mean ULP':>9}  {'rel':>9}")

    worst = 0.0
    for c in cases:
        r = check_case(d / f"{c['name']}.npz", c.get("bf16_params", ()))
        worst = max(worst, r["max_ulps"])
        print(f"{c['name']:>12}  {c['layer_idx']:>5}  {str(r['shape']):>18}  "
              f"{r['max_ulps']:9.2f}  {r['mean_ulps']:9.3f}  {r['rel']:9.2e}")

    print(f"\nworst case across {len(cases)} blocks: {worst:.2f} bf16 ULP")
    if worst <= args.max_ulps:
        print(f"PATH A BLOCK GATE: PASS — the plain-PyTorch mixer reproduces "
              f"the official block to <= {args.max_ulps} bf16 ULP.")
        return 0
    print(f"PATH A BLOCK GATE: FAIL — worst {worst:.2f} ULP exceeds "
          f"{args.max_ulps}.\nDo NOT loosen this. A plumbing bug (wrong split "
          f"order, B/C swapped, missing squeeze) is off by orders of "
          f"magnitude; a real precision issue is not.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
