"""How accurate must the denoiser be, on THIS data, to be worth having?

`tools/prior_report.py` checks a checkpoint against a threshold. This is where
that threshold comes from — and it must be re-derived whenever the data or the
sampling pattern changes, because it encodes how much information the mask
actually destroys.

METHOD (no training required)
-----------------------------
Replace the denoiser with an **oracle** that returns the truth plus controlled
error `eps`, reconstruct, and find the `eps` at which the reconstruction stops
beating zero-filling by 1 dB. Reported relative to the data's RMS, so the
number transfers across resolutions.

The curve is a clean -20*log10 line offset by how good zero-filling already is:

    gain over zero-filled (dB) ~= -20 * log10(NRMSE) - OFFSET

so this prints both the crossing point and the fitted OFFSET, which are exactly
the two constants at the top of `prior_report.py`.

WHY THIS IS NOT OPTIONAL WHEN THE DATA CHANGES
----------------------------------------------
On Shepp-Logan phantoms the bar is NRMSE ~0.62 (offset 3.1 dB). On the smooth
bump data it is ~0.10, because zero-filling already reaches 38.6 dB there and
there is almost nothing left to recover. Same code, same sampler — an eightfold
difference in what "good enough" means. Using a phantom-derived bar to judge a
prior trained on knee MRI would be meaningless.

    python tools/calibrate_prior_bar.py                       # phantoms
    python tools/calibrate_prior_bar.py --data fastmri --cache data/knee.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.mri_diffusion.data import phantom_batch, toy_batch  # noqa: E402
from apps.mri_diffusion.sampling.posterior import (  # noqa: E402
    cartesian_mask, effective_R, heun_posterior, measure, psnr, zero_filled)

EPS_GRID = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64)


class Oracle:
    """A perfect denoiser degraded by additive error of scale `eps`."""

    def __init__(self, truth, eps=0.0, seed=0):
        self.truth, self.eps = truth, eps
        self.g = torch.Generator().manual_seed(seed)
        self.sigma_max_trained = 80.0  # an oracle is correct at every sigma

    def __call__(self, x, sigma, labels=None):
        out = self.truth.expand_as(x).clone()
        if self.eps:
            out = out + self.eps * torch.randn(x.shape, generator=self.g)
        return out


def load_images(kind, res, cache):
    if kind == "phantom":
        return phantom_batch(1, res, np.random.default_rng(99))
    if kind == "bumps":
        return toy_batch(1, res)
    if kind == "fastmri":
        if not cache:
            raise SystemExit("--data fastmri needs --cache "
                             "(see tools/prepare_fastmri.py)")
        from apps.mri_diffusion.fastmri_data import load_cache
        imgs = load_cache(cache)
        return imgs[:1]
    raise SystemExit(f"unknown --data {kind}")


def sweep(truth, res, R, steps, acs):
    rms = float((truth ** 2).mean().sqrt())
    mask = cartesian_mask(res, res, R, acs=acs, seed=R)
    y = measure(truth, mask)
    p_zf = psnr(zero_filled(y, mask), truth)
    bar = p_zf + 1.0
    print(f"\n=== {res}px, R={R} (effective {effective_R(mask):.2f}) ===")
    print(f"data RMS {rms:.4f} | zero-filled {p_zf:.2f} dB | bar {bar:.2f} dB")

    prev, cross, offsets = None, None, []
    for eps in EPS_GRID:
        p = psnr(heun_posterior(Oracle(truth, eps=eps), y, mask,
                                num_steps=steps), truth)
        nrmse = eps / rms
        # gain = -20log10(NRMSE) - OFFSET  ->  OFFSET = -20log10(NRMSE) - gain
        offsets.append(-20 * np.log10(nrmse) - (p - p_zf))
        print(f"  eps {eps:6.3f}  NRMSE {nrmse:6.3f}  recon {p:7.2f} dB "
              f"(gain {p - p_zf:+6.2f})  {'PASS' if p > bar else 'fail'}")
        if prev and prev[1] > bar >= p and cross is None:
            f = (prev[1] - bar) / (prev[1] - p)
            cross = float(np.exp(np.log(prev[0])
                                 + f * (np.log(eps) - np.log(prev[0]))))
        prev = (eps, p)

    offset = float(np.median(offsets))
    if cross:
        print(f"  --> NRMSE_BAR = {cross/rms:.3f}   "
              f"GAIN_OFFSET_DB = {offset:.2f}")
    return (cross / rms if cross else None), offset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="phantom",
                    choices=("phantom", "bumps", "fastmri"))
    ap.add_argument("--cache", default=None, help="fastMRI cache (.pt)")
    ap.add_argument("--res", default="32,64")
    ap.add_argument("--R", default="4,8")
    ap.add_argument("--acs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(0)
    bars, offsets = [], []
    for res in [int(r) for r in args.res.split(",")]:
        truth = load_images(args.data, res, args.cache)
        if truth.shape[-1] != res:
            truth = torch.nn.functional.interpolate(
                truth, size=(res, res), mode="bilinear", align_corners=False)
        for R in [int(r) for r in args.R.split(",")]:
            b, o = sweep(truth, res, R, args.steps, args.acs)
            if b:
                bars.append(b)
                offsets.append(o)

    if bars:
        print(f"\n{'='*62}\nPut these in tools/prior_report.py for "
              f"--data {args.data}:")
        print(f"  NRMSE_BAR      = {np.median(bars):.2f}   "
              f"(spread {min(bars):.2f}-{max(bars):.2f})")
        print(f"  GAIN_OFFSET_DB = {np.median(offsets):.1f}")
        print("A tight spread across resolutions and R means the bar is a "
              "property of the DATA,\nnot of one configuration — which is "
              "what makes it reusable.")


if __name__ == "__main__":
    main()
