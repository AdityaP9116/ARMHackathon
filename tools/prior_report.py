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
# Derived by oracle sweep (no training: inject controlled error into a perfect
# denoiser and reconstruct). Reconstruction PSNR falls as a clean -20*log10
# line in the denoiser's relative error, offset by how good zero-filling
# already is on this data:
#
#     gain over zero-filled (dB) ~= -20 * log10(NRMSE) - OFFSET
#
# Measured on phantoms with the CENTRED FFT and true-R masks: OFFSET = 3.1 dB
# (consistent to +-0.1 across NRMSE 0.13 .. 1.05), so the >1 dB bar sits at
# NRMSE ~= 0.62 — measured at 0.62, 0.62, 0.63, 0.63 across 32/64px x R=4/8.
#
# !! RE-DERIVE THESE FOR A NEW DATASET. Both numbers encode how much energy the
# mask discards on THIS data, so they move when the data moves. On the smooth
# bump set the same sweep gives a bar of 0.10, because zero-filling already
# reaches 38.6 dB there and almost nothing is left to recover. Run
# `tools/calibrate_prior_bar.py` after switching to fastMRI.
#
# (Earlier revisions of this file quoted 0.28 absolute, then 0.95 relative.
# The first did not transfer across datasets; the second was measured before
# the FFT centring fix, when the mask was sampling Nyquist instead of DC.)
NRMSE_BAR = 0.62     # minimum to clear the >1 dB reconstruction bar
NRMSE_TARGET = 0.35  # a prior worth showing: ~+6 dB or better
GAIN_OFFSET_DB = 3.1


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
    # Default None on purpose: a self-describing checkpoint supplies these,
    # and an argparse default would silently override the embedded config.
    ap.add_argument("--model-channels", type=int, default=None)
    ap.add_argument("--blocks", type=int, default=None)
    ap.add_argument("--d-state", type=int, default=None)
    ap.add_argument("--bar", type=float, default=NRMSE_BAR,
                    help="NRMSE (RMSE / data RMS) that must not be exceeded")
    ap.add_argument("--cache", default=None,
                    help="fastMRI cache to evaluate against instead of "
                         "phantoms; use the SAME data the prior was trained "
                         "for, and re-derive --bar with "
                         "tools/calibrate_prior_bar.py")
    ap.add_argument("--json", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    # Checkpoints from tools/train_prior.py embed their architecture, so the
    # --model-channels/--blocks/--d-state flags are only needed for bare
    # state_dicts written by older runs.
    from apps.mri_diffusion.checkpoint import build_prior
    overrides = {k: v for k, v in (
        ("img_resolution", args.res), ("model_channels", args.model_channels),
        ("num_blocks_per_level", args.blocks), ("d_state", args.d_state),
    ) if v is not None}
    net, cfg, meta = build_prior(construct, args.checkpoint, **overrides)
    if meta.get("step"):
        print(f"checkpoint : step {meta['step']}"
              + (f", trained on {meta['data']}" if meta.get("data") else ""))

    smax = args.sigma_max or float(getattr(net, "sigma_max_trained", 80.0))
    ladder = sigma_ladder(args.steps, smax)
    if args.cache:
        from apps.mri_diffusion.fastmri_data import load_cache
        pool = load_cache(args.cache, "eval")
        clean = pool[:args.batch]
        source = f"fastMRI eval split ({args.cache})"
    else:
        clean = phantom_batch(args.batch, args.res,
                              np.random.default_rng(args.seed))
        source = "synthetic phantoms"
    rms = float((clean ** 2).mean().sqrt())

    print(f"prior      : {args.checkpoint}")
    print(f"params     : {sum(p.numel() for p in net.parameters())/1e3:.0f}K")
    print(f"ladder     : {args.steps} steps, sigma_max={smax:.2f} "
          f"({'declared by the prior' if not args.sigma_max else 'overridden'})")
    print(f"data       : {clean.shape[0]} images from {source}, "
          f"{clean.shape[-1]}px, RMS = {rms:.4f}")
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
            # Predicts the reconstruction's dB gain over zero-filled; below
            # the bar it is a deficit, so show it signed.
            gain = -20 * np.log10(max(nrmse, 1e-9)) - GAIN_OFFSET_DB
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
