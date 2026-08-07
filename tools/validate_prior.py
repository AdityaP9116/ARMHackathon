"""Validate a trained prior: is it accurate, does it reconstruct, is the kernel still exact?

One entry point that answers all three questions a trained checkpoint has to
answer, and writes the table that goes in the results.

  1. ACCURACY   — denoiser NRMSE across the sampler's sigma ladder, against the
                  data-calibrated bar. Cheap, and it predicts (3) before any
                  sampling runs.
  2. QUALITY    — PSNR / SSIM / NMSE at R = 2, 4, 8 versus zero-filling, on
                  HELD-OUT data. This is the app's actual claim.
  3. PARITY     — the same reconstruction through the arm_scan kernel and
                  through the torch reference. Quality-independent: it is the
                  kernel claim, and it must hold whether or not the prior is
                  any good.

Keeping (3) in here matters. A prior that fails (2) says nothing about the
kernel, and this report makes that separation visible instead of leaving a
reader to infer it from a single red number.

    python tools/validate_prior.py --checkpoint data/prior_knee128.pt \\
        --cache data/knee_128.pt --json results/prior_validation.json

Without `--cache` it validates against synthetic phantoms — useful for a smoke
run, but remember the NRMSE bar is data-dependent (`--nrmse-bar`, derived by
`tools/calibrate_prior_bar.py`).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.mri_diffusion.checkpoint import build_prior  # noqa: E402
from apps.mri_diffusion.data import phantom_batch  # noqa: E402
from apps.mri_diffusion.demo import magnitude, montage, ssim, write_png  # noqa: E402
from apps.mri_diffusion.sampling.posterior import (  # noqa: E402
    cartesian_mask, effective_R, heun_posterior, measure, psnr, zero_filled)
from apps.mri_diffusion.tests import _edm  # noqa: E402
from arm_scan.op import kernel_calls  # noqa: E402
from arm_scan.ss2d import use_arm_scan  # noqa: E402
from tools.prior_report import GAIN_OFFSET_DB, NRMSE_BAR, sigma_ladder  # noqa: E402

EDMPrecond, EDMLoss, construct, EDM_SOURCE = _edm.load()


def nmse(a, b):
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache", default=None,
                    help="fastMRI cache; evaluation uses its HELD-OUT split")
    ap.add_argument("--res", type=int, default=None)
    ap.add_argument("--steps", type=int, default=12, help="Heun steps")
    ap.add_argument("--R", default="2,4,8")
    ap.add_argument("--acs", type=int, default=8)
    ap.add_argument("--nrmse-bar", type=float, default=NRMSE_BAR)
    ap.add_argument("--bar-db", type=float, default=1.0,
                    help="required dB gain over zero-filled at R=4")
    ap.add_argument("--json", default=None)
    ap.add_argument("--png", default="demo_out/validation.png")
    ap.add_argument("--skip-parity", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)

    # ---- data (held out, never trained on) ---------------------------
    if args.cache:
        from apps.mri_diffusion.fastmri_data import load_cache
        pool = load_cache(args.cache, "eval")
        res = args.res or pool.shape[-1]
        truth = pool[:1]
        source = f"fastMRI held-out volumes ({args.cache})"
    else:
        res = args.res or 64
        truth = phantom_batch(1, res, np.random.default_rng(99))
        source = "synthetic phantoms"

    net, cfg, meta = build_prior(construct, args.checkpoint,
                                 img_resolution=res)
    n_blocks = use_arm_scan(net)
    smax = float(getattr(net, "sigma_max_trained", 80.0))
    print(f"checkpoint : {args.checkpoint}"
          + (f"  (step {meta['step']})" if meta.get("step") else ""))
    print(f"trained on : {meta.get('data', 'unknown')}")
    print(f"model      : {sum(p.numel() for p in net.parameters())/1e6:.2f}M "
          f"params, {n_blocks} SS2D blocks on arm_scan")
    print(f"evaluating : {source}, {res}px, sigma_max {smax:.2f}\n")

    out = {"kind": "prior_validation", "checkpoint": str(args.checkpoint),
           "step": meta.get("step"), "trained_on": meta.get("data"),
           "eval_source": source, "res": res, "nrmse_bar": args.nrmse_bar}

    # ---- 1. accuracy --------------------------------------------------
    print("1. denoiser accuracy across the sampler's ladder")
    rms = float((truth ** 2).mean().sqrt())
    rungs, worst = [], 0.0
    with torch.no_grad():
        for s in sigma_ladder(10, smax):
            noisy = truth + float(s) * torch.randn_like(truth)
            den = net(noisy, torch.full((truth.shape[0],), float(s)), None)
            nr = float(((den - truth) ** 2).mean().sqrt()) / rms
            worst = max(worst, nr)
            rungs.append({"sigma": float(s), "nrmse": nr,
                          "pass": nr < args.nrmse_bar})
    failing = sum(1 for r in rungs if not r["pass"])
    predicted = -20 * np.log10(max(worst, 1e-9)) - GAIN_OFFSET_DB
    out.update({"rungs": rungs, "worst_nrmse": worst,
                "predicted_gain_db": predicted, "failing_rungs": failing})
    print(f"   worst NRMSE {worst:.3f} (bar {args.nrmse_bar}), "
          f"{failing}/10 rungs failing")
    print(f"   -> predicts the reconstruction can be no better than "
          f"{predicted:+.1f} dB\n")

    # ---- 2. reconstruction quality ------------------------------------
    print("2. reconstruction vs zero-filling (held-out data)")
    quality, panels = [], [magnitude(truth)]
    for R in [int(r) for r in args.R.split(",")]:
        mask = cartesian_mask(res, res, R, acs=args.acs, seed=R)
        y = measure(truth, mask)
        zf = zero_filled(y, mask)
        t0 = time.perf_counter()
        rec = heun_posterior(net, y, mask, num_steps=args.steps)
        wall = time.perf_counter() - t0
        mt, mz, mr = magnitude(truth), magnitude(zf), magnitude(rec)
        row = {"R": R, "effective_R": effective_R(mask),
               "zf_psnr": psnr(mz, mt), "psnr": psnr(mr, mt),
               "zf_ssim": ssim(mz, mt), "ssim": ssim(mr, mt),
               "nmse": nmse(rec, truth), "zf_nmse": nmse(zf, truth),
               "wall_s": wall}
        row["gain_db"] = row["psnr"] - row["zf_psnr"]
        quality.append(row)
        print(f"   R={R} (eff {row['effective_R']:.2f}): "
              f"zero-filled {row['zf_psnr']:6.2f} dB / SSIM "
              f"{row['zf_ssim']:.3f}  ->  recon {row['psnr']:6.2f} dB / SSIM "
              f"{row['ssim']:.3f}   ({row['gain_db']:+.2f} dB)")
        if R == 4:
            panels += [mz, mr]
    out["quality"] = quality

    # ---- 3. kernel parity (independent of how good the prior is) ------
    if not args.skip_parity:
        print("\n3. kernel vs reference scan, same weights and measurement")
        mask = cartesian_mask(res, res, 4, acs=args.acs, seed=4)
        y = measure(truth, mask)
        use_arm_scan(net, enable=False)
        ref = heun_posterior(net, y, mask, num_steps=6)
        use_arm_scan(net)
        c0 = kernel_calls()
        kern = heun_posterior(net, y, mask, num_steps=6)
        engaged = kernel_calls() - c0
        diff = float((kern - ref).abs().max())
        scale = float(ref.abs().max())
        out["parity"] = {"max_abs": diff, "scale": scale,
                         "kernel_calls": engaged}
        print(f"   max_abs {diff:.3e} (scale {scale:.2f}), "
              f"{engaged} kernel calls")
        parity_ok = diff < 1e-3 * max(1.0, scale) and engaged > 0
    else:
        parity_ok = True

    # ---- verdict ------------------------------------------------------
    at_r4 = next((q for q in quality if q["R"] == 4), None)
    gain = at_r4["gain_db"] if at_r4 else float("nan")
    quality_ok = at_r4 is not None and gain > args.bar_db
    out.update({"quality_pass": quality_ok, "parity_pass": parity_ok,
                "accuracy_pass": failing == 0})

    Path(args.png).parent.mkdir(parents=True, exist_ok=True)
    write_png(args.png, montage(panels))
    print(f"\nwrote {args.png} (ground truth | zero-filled | reconstruction "
          f"at R=4)")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json}")

    print("\n" + "=" * 66)
    print(f"  accuracy : {'PASS' if failing == 0 else 'FAIL'}  "
          f"(worst NRMSE {worst:.3f} vs bar {args.nrmse_bar})")
    print(f"  quality  : {'PASS' if quality_ok else 'FAIL'}  "
          f"({gain:+.2f} dB at R=4, bar >{args.bar_db:.1f})")
    print(f"  parity   : {'PASS' if parity_ok else 'FAIL'}  "
          f"(kernel vs reference)")
    if not quality_ok and parity_ok:
        print("\n  A quality failure with parity green is a PRIOR problem, "
              "not a kernel one.")
        print("  Train longer (tools/train_prior.py --resume), and confirm "
              "the bar is calibrated")
        print("  for THIS data (tools/calibrate_prior_bar.py). Do not relax "
              "the dB bar.")
    return 0 if (quality_ok and parity_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
