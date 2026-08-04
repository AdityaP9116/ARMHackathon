"""Is this prior good enough to reconstruct with? Answer before you sample.

Prior quality was previously a vibe: train "a while", run the reconstruction,
see if it looked bad. That wasted a whole cycle on Phase D, because a failing
reconstruction does not say *why* it failed.

It is now a measured property with a threshold. From PHASE_D_DIAGNOSIS.md §2.2,
an oracle-denoiser sweep put the requirement at **denoiser RMSE < 0.28**, and
§2.3 showed the failing 300-step prior sitting at 0.29–0.53 across the six
highest sigmas of the sampler's ladder — accurate where it did not matter,
useless where it did.

So: this walks the exact sigma ladder `heun_posterior` will walk, measures the
denoiser's RMSE at each, and reports pass/fail per rung. Use it as a **stop
condition** for training (stop when it clears, not at some fixed step count)
and as a gate before spending sampling time on a checkpoint.

    python tools/prior_report.py --checkpoint prior.pt
    python tools/prior_report.py --checkpoint prior.pt --json report.json

Exit code 0 if every sigma on the ladder clears the bar, 1 otherwise.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.mri_diffusion.data import phantom_batch  # noqa: E402
from apps.mri_diffusion.tests import _edm  # noqa: E402

EDMPrecond, EDMLoss, construct, EDM_SOURCE = _edm.load()

# The measured requirement, expressed RELATIVE to the data's own RMS so it
# transfers across resolutions and image families.
#
# An oracle-denoiser sweep (no training needed — inject controlled error into a
# perfect denoiser and reconstruct) crosses the ">1 dB better than zero-filled"
# bar at **NRMSE ~= 0.95** — measured at 0.96, 0.96, 0.95, 0.93 across
# phantom 32px/64px x R=4/8, and 1.00 on the old smooth-bump setup. The
# earlier absolute figure of 0.28 was that same threshold for one dataset
# (bumps, RMS 0.262 -> 0.28/0.262 ~= 1.07); it does not transfer, which is why
# it is stated relatively now.
#
# Clearing 0.95 only buys the MINIMUM passing reconstruction. The sweep is a
# clean -20*log10 line, so the denoiser's error predicts the gain:
#     expected PSNR gain over zero-filled ~= -20 * log10(NRMSE)
# i.e. NRMSE 0.5 -> ~+6 dB, 0.25 -> ~+12 dB, 0.125 -> ~+18 dB.
NRMSE_BAR = 0.95     # minimum to clear the >1 dB reconstruction bar
NRMSE_TARGET = 0.50  # a prior worth showing: ~+6 dB or better


def sigma_ladder(num_steps, sigma_max, sigma_min=0.002, rho=7.0):
    """The EDM sigma schedule the sampler actually traverses."""
    i = np.arange(num_steps, dtype=np.float64)
    return (sigma_max ** (1 / rho) + i / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=10,
                    help="ladder length; match the sampler you will use")
    ap.add_argument("--sigma-max", type=float, default=None,
                    help="default: the prior's own trained support")
    ap.add_argument("--model-channels", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--d-state", type=int, default=16)
    ap.add_argument("--bar", type=float, default=NRMSE_BAR,
                    help="NRMSE (RMSE / data RMS) that must not be exceeded")
    ap.add_argument("--json", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    net = construct(
        class_name="training.networks.EDMPrecond", model_type="MambaSS2DNet",
        img_resolution=args.res, img_channels=2, label_dim=0,
        model_channels=args.model_channels,
        num_blocks_per_level=args.blocks, d_state=args.d_state,
        use_fp16=False, sigma_data=0.5)
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    net.eval()

    smax = args.sigma_max or float(getattr(net, "sigma_max_trained", 80.0))
    ladder = sigma_ladder(args.steps, smax)
    clean = phantom_batch(args.batch, args.res,
                          np.random.default_rng(args.seed))
    rms = float((clean ** 2).mean().sqrt())

    print(f"prior      : {args.checkpoint}")
    print(f"params     : {sum(p.numel() for p in net.parameters())/1e3:.0f}K")
    print(f"ladder     : {args.steps} steps, sigma_max={smax:.2f} "
          f"({'declared by the prior' if not args.sigma_max else 'overridden'})")
    print(f"data       : {args.batch} phantoms at {args.res}px, "
          f"RMS = {rms:.4f}")
    print(f"bar        : NRMSE < {args.bar} at EVERY sigma; target "
          f"< {NRMSE_TARGET} (PHASE_D_DIAGNOSIS.md §4)\n")
    print(f"{'sigma':>10} {'RMSE':>9} {'NRMSE':>8} {'pred. gain':>11}   status")

    rows, failed, weak = [], 0, 0
    with torch.no_grad():
        for s in ladder:
            noisy = clean + float(s) * torch.randn_like(clean)
            den = net(noisy, torch.full((args.batch,), float(s)), None)
            rmse = float(((den - clean) ** 2).mean().sqrt())
            nrmse = rmse / rms
            ok = nrmse < args.bar
            failed += 0 if ok else 1
            weak += 1 if ok and nrmse >= NRMSE_TARGET else 0
            # -20*log10(NRMSE) predicts the reconstruction's dB gain over
            # zero-filled; below the bar it is a deficit, so show it signed.
            gain = -20 * np.log10(max(nrmse, 1e-9))
            rows.append({"sigma": float(s), "rmse": rmse, "nrmse": nrmse,
                         "predicted_gain_db": gain, "pass": ok})
            status = ("ok" if nrmse < NRMSE_TARGET
                      else ("marginal" if ok else "TOO HIGH"))
            print(f"{s:10.3f} {rmse:9.4f} {nrmse:8.3f} {gain:+10.1f} dB   "
                  f"{status}")

    ok = failed == 0
    limiting = max(rows, key=lambda r: r["nrmse"])
    print(f"\n{len(ladder) - failed}/{len(ladder)} rungs clear the bar "
          f"({len(ladder) - failed - weak} with margin)")
    print(f"limiting rung: sigma={limiting['sigma']:.3f}, NRMSE "
          f"{limiting['nrmse']:.3f} -> the reconstruction can be no better "
          f"than about {limiting['predicted_gain_db']:+.1f} dB")
    if ok and weak == 0:
        print("PRIOR REPORT: PASS — good enough to reconstruct with. "
              "Stop training.")
    elif ok:
        print("PRIOR REPORT: PASS (marginal) — clears the bar, but "
              f"{weak} rung(s) sit above the {NRMSE_TARGET} target. "
              "Expect a thin margin.")
    else:
        print("PRIOR REPORT: FAIL — keep training.")
        if rows[0]["nrmse"] >= rows[-1]["nrmse"]:
            print("  error is concentrated at the HIGH sigmas — the steps "
                  "that fix global structure.")
            print("  Either train longer/bigger, or lower the sampler's "
                  "sigma_max to the range actually")
            print("  trained (see edm_min.trained_sigma_max).")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({
            "kind": "prior_report", "checkpoint": str(args.checkpoint),
            "sigma_max": smax, "steps": args.steps, "nrmse_bar": args.bar,
            "nrmse_target": NRMSE_TARGET, "data_rms": rms, "res": args.res,
            "pass": ok, "marginal_rungs": weak,
            "limiting_nrmse": limiting["nrmse"],
            "predicted_gain_db": limiting["predicted_gain_db"],
            "rungs": rows}, indent=2))
        print(f"\nwrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
