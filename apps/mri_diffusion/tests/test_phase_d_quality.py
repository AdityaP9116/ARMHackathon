"""Phase-D QUALITY gate: does the diffusion prior actually beat zero-filling?

This is the app's quality claim, and the **only** thing in the repo that
measures it. Everything about the machinery — FFT, mask, data consistency, the
Heun integrator, the kernel in the sampling loop — is gated separately and
prior-independently by `test_phase_d_pipeline.py`. Keeping them apart means a
weak prior can no longer hide a kernel regression, and a kernel regression can
no longer be mistaken for a weak prior.

**This test REQUIRES a trained prior** and skips cleanly without one. It used
to train a 200-step prior inline and assert against it; that prior was ~2
orders of magnitude short of what the task needs, so the gate could not pass
and its failure said nothing useful. A skip is more honest than a red mark
whose real cause is "we have not trained a model yet".

    python apps/mri_diffusion/tests/test_phase_d_quality.py --checkpoint p.pt
    PRIOR_CKPT=p.pt python apps/mri_diffusion/tests/test_phase_d_quality.py

**The >1 dB bar is deliberately not softened.** It is the entire quality claim.
If a prior cannot clear it, the honest outcome is to report that — together
with the accuracy target from PHASE_D_DIAGNOSIS.md §4 (denoiser NRMSE below the
data-calibrated bar at every sigma on the ladder). Check a candidate with
`tools/prior_report.py` BEFORE spending sampling time on it.

Evaluation uses Shepp-Logan phantoms, not the smooth bump data: on smooth
images a centre-keeping mask discards ~0.2% of the energy, so zero-filling is
already near-optimal and no prior can demonstrate value. Phantoms discard ~21%
at R=4. See apps/mri_diffusion/data.py.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from apps.mri_diffusion.data import phantom_batch  # noqa: E402
from apps.mri_diffusion.sampling.posterior import (  # noqa: E402
    cartesian_mask, effective_R, heun_posterior, measure, psnr, zero_filled)
from apps.mri_diffusion.tests import _edm  # noqa: E402
from arm_scan.op import kernel_calls  # noqa: E402
from arm_scan.ss2d import use_arm_scan  # noqa: E402

EDMPrecond, EDMLoss, construct, EDM_SOURCE = _edm.load()

BAR_DB = 1.0  # reconstruction must beat zero-filled by this much at R=4


def nmse(a, b):
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("PRIOR_CKPT"))
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--steps", type=int, default=12)
    # None on purpose: a self-describing checkpoint supplies these, and
    # an argparse default would silently override the embedded config.
    ap.add_argument("--model-channels", type=int, default=None)
    ap.add_argument("--blocks", type=int, default=None)
    ap.add_argument("--d-state", type=int, default=None)
    args = ap.parse_args()

    if not args.checkpoint:
        print("PHASE D QUALITY GATE: SKIPPED — no trained prior.")
        print("  Pass --checkpoint PATH (or set PRIOR_CKPT).")
        print("  Train one with tools/train_prior.py, then")
        print("  check it with tools/prior_report.py before running this.")
        return 0
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        print(f"PHASE D QUALITY GATE: FAIL — no such checkpoint: {ckpt}")
        return 1

    torch.manual_seed(0)
    print(f"EDM source: {EDM_SOURCE}")
    from apps.mri_diffusion.checkpoint import build_prior
    overrides = {k: v for k, v in (
        ("img_resolution", args.res), ("model_channels", args.model_channels),
        ("num_blocks_per_level", args.blocks), ("d_state", args.d_state),
    ) if v is not None}
    net, _cfg, meta = build_prior(construct, ckpt, **overrides)
    n_blocks = use_arm_scan(net)
    if meta.get("step"):
        print(f"checkpoint: step {meta['step']}"
              + (f", data {meta['data']}" if meta.get("data") else ""))
    print(f"prior: {ckpt} "
          f"({sum(p.numel() for p in net.parameters())/1e3:.0f}K params, "
          f"{n_blocks} SS2D blocks on arm_scan)")
    print(f"sampler sigma_max: {getattr(net, 'sigma_max_trained', 80.0):.2f} "
          f"(read from the prior's trained support)\n")

    truth = phantom_batch(1, args.res, np.random.default_rng(99))
    results = {}
    for R in (2, 4, 8):
        mask = cartesian_mask(args.res, args.res, R, acs=8, seed=R)
        y = measure(truth, mask)
        zf = zero_filled(y, mask)
        c0 = kernel_calls()
        rec = heun_posterior(net, y, mask, num_steps=args.steps)
        engaged = kernel_calls() - c0
        p_zf, p_re = psnr(zf, truth), psnr(rec, truth)
        results[R] = (p_zf, p_re)
        print(f"R={R} (effective {effective_R(mask):.2f}): "
              f"zero-filled {p_zf:6.2f} dB -> recon {p_re:6.2f} dB "
              f"({p_re - p_zf:+.2f} dB)  NMSE {nmse(rec, truth):.4f}  "
              f"[{engaged} kernel calls]")
        assert torch.isfinite(rec).all(), f"non-finite reconstruction at R={R}"
        assert engaged > 0, f"kernel never engaged at R={R}"

    gain = results[4][1] - results[4][0]
    print(f"\nprior contributes {gain:+.2f} dB at R=4 (bar: >{BAR_DB:.1f} dB)")
    if gain <= BAR_DB:
        print("PHASE D QUALITY GATE: FAIL — the prior does not beat "
              "zero-filled by the required margin.")
        print("  This is a PRIOR-QUALITY result, not a kernel or sampler "
              "fault: test_phase_d_pipeline.py")
        print("  proves the machinery is exact (oracle denoiser -> ~150 dB).")
        print("  Check the prior with tools/prior_report.py — it must clear "
              "the data-calibrated")
        print("  NRMSE bar at every sigma. Do NOT relax this bar.")
        return 1

    print("\nPHASE D QUALITY GATE: PASS — diffusion prior beats zero-filled "
          f"by {gain:.2f} dB at R=4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
