"""The diffusion application's own benchmark: per-NFE latency and $/reconstruction.

The README promises "diffusion end-to-end per-NFE latency and $/reconstruction"
and "PSNR/SSIM/NMSE parity at R=2-8". Nothing measured either: `bench_ss2d.py`
times a single SS2D *block*, and `bench_op.py` times the isolated scan. This
measures the actual denoiser and the actual reconstruction — the workload the
whole pitch is about.

TWO INDEPENDENT MODES, deliberately separated
---------------------------------------------
**Latency/cost** needs no trained prior. A randomly-initialised network of the
same shape costs exactly the same to evaluate, so timing is a property of the
architecture and the kernel, not of the weights. This runs today.

**Quality** (`--checkpoint`) needs a real prior, and is skipped without one
rather than reporting numbers from an untrained net.

Keeping them apart means a rented Arm session can produce the headline latency
and cost table even if the prior is not ready — the kernel claim never has to
wait on the model.

WHY RECONSTRUCTION TIME IS PROJECTED, NOT MEASURED
--------------------------------------------------
Diffusion reconstruction calls the denoiser 18-256 times. At 384x320 a single
call is seconds, so NFE=256 is most of an hour — too slow to run per shape on a
metered instance, and it would tell us nothing that `per_nfe x NFE` does not.
So per-NFE is measured and the reconstruction column is projected, labelled as
such. That linearity is the honest part: the sampler is a fixed number of
denoiser calls plus negligible FFT work.

Usage:
    python bench/bench_diffusion.py --tag graviton-c8g --json out.json
    python bench/bench_diffusion.py --usd-per-hour 2.90 --instance c8g.16xlarge
    python bench/bench_diffusion.py --checkpoint prior.pt --quality
"""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.mri_diffusion.tests import _edm  # noqa: E402
from arm_scan.op import kernel_calls  # noqa: E402
from arm_scan.ss2d import use_arm_scan  # noqa: E402

EDMPrecond, EDMLoss, construct, EDM_SOURCE = _edm.load()

# NFE counts worth projecting: EDM's default 18-step Heun (35 NFE), a 35-step
# run (69 NFE), and the 256 the README cites as the upper end.
NFE_POINTS = (18, 69, 256)


def peak_rss_mb():
    """Peak resident set size, or None where unavailable (Windows)."""
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB, macOS reports bytes.
        return kb / 1024 if sys.platform != "darwin" else kb / 1024 / 1024
    except Exception:  # noqa: BLE001
        return None


def build_net(res, model_channels, blocks, d_state):
    """The locked backbone recipe (MRI_DIFFUSION_IMPLEMENTATION_PLAN §3.2)."""
    return construct(
        class_name="training.networks.EDMPrecond", model_type="MambaSS2DNet",
        img_resolution=res, img_channels=2, label_dim=0,
        model_channels=model_channels, num_blocks_per_level=blocks,
        d_state=d_state, use_fp16=False, sigma_data=0.5).eval()


def time_nfe(net, res, reps, warmup=1):
    """Median seconds for ONE denoiser call (one NFE) at this grid."""
    h, w = res
    x = torch.randn(1, 2, h, w)
    sigma = torch.tensor([1.0])
    times = []
    with torch.no_grad():
        for i in range(warmup + reps):
            t0 = time.perf_counter()
            net(x, sigma, None)
            if i >= warmup:
                times.append(time.perf_counter() - t0)
    return statistics.median(times)


def nmse(a, b):
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


def run_quality(net, res, ckpt, Rs, steps, seed=0):
    """PSNR/SSIM/NMSE vs zero-filled at each acceleration. Needs a prior."""
    from apps.mri_diffusion.data import phantom_batch
    from apps.mri_diffusion.demo import magnitude, ssim
    from apps.mri_diffusion.sampling.posterior import (
        cartesian_mask, effective_R, heun_posterior, measure, psnr,
        zero_filled)

    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.eval()
    rng = np.random.default_rng(seed)
    truth = phantom_batch(1, res[0], rng)
    out = []
    for R in Rs:
        mask = cartesian_mask(res[0], res[1], R, acs=8, seed=R)
        y = measure(truth, mask)
        zf = zero_filled(y, mask)
        t0 = time.perf_counter()
        rec = heun_posterior(net, y, mask, num_steps=steps)
        wall = time.perf_counter() - t0
        mt, mz, mr = magnitude(truth), magnitude(zf), magnitude(rec)
        out.append({
            "R": R, "sampling_fraction": float(mask.mean()),
            "effective_R": effective_R(mask),
            "zf_psnr": psnr(mz, mt), "psnr": psnr(mr, mt),
            "ssim": ssim(mr, mt), "nmse": nmse(mr, mt),
            "zf_nmse": nmse(mz, mt), "wall_s": wall, "heun_steps": steps,
        })
        print(f"  R={R} (effective {1/float(mask.mean()):.2f}): "
              f"zero-filled {psnr(mz, mt):6.2f} dB -> recon "
              f"{psnr(mr, mt):6.2f} dB "
              f"({psnr(mr, mt) - psnr(mz, mt):+.2f} dB)  "
              f"SSIM {ssim(mr, mt):.4f}  NMSE {nmse(mr, mt):.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=platform.node())
    ap.add_argument("--json", default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--model-channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=2,
                    help="SS2D blocks per resolution level")
    ap.add_argument("--d-state", type=int, default=16)
    ap.add_argument("--grids", default="384x320,192x160,128x128,64x64",
                    help="comma-separated HxW list")
    ap.add_argument("--usd-per-hour", type=float, default=None,
                    help="instance on-demand price, for the cost table")
    ap.add_argument("--instance", default=None)
    ap.add_argument("--checkpoint", default=None,
                    help="trained prior; enables the quality rows")
    ap.add_argument("--quality-res", type=int, default=128)
    ap.add_argument("--quality-steps", type=int, default=12)
    ap.add_argument("--quality-R", default="2,4,8")
    args = ap.parse_args()

    torch.manual_seed(0)
    grids = []
    for g in args.grids.split(","):
        h, w = g.lower().split("x")
        grids.append((int(h), int(w)))

    out = {
        "kind": "diffusion", "tag": args.tag, "host": platform.platform(),
        "machine": platform.machine(), "torch": torch.__version__,
        "threads": torch.get_num_threads(), "reps": args.reps,
        "edm_source": EDM_SOURCE,
        "prior": args.checkpoint or "untrained (timing only)",
        "cases": [],
    }
    print(f"host {platform.platform()} / {platform.machine()}, torch "
          f"{torch.__version__}, {torch.get_num_threads()} threads")
    print(f"backbone: model_channels={args.model_channels}, "
          f"{args.blocks} blocks/level, d_state={args.d_state}\n")
    print("Latency is prior-independent: an untrained net of the same shape "
          "costs the same to\nevaluate. Reconstruction columns are PROJECTED "
          "as per_nfe x NFE.\n")

    for res in grids:
        net = build_net(max(res), args.model_channels, args.blocks,
                        args.d_state)
        n_params = sum(p.numel() for p in net.parameters())
        n_blocks = use_arm_scan(net)
        c0 = kernel_calls()
        per_nfe = time_nfe(net, res, args.reps)
        engaged = kernel_calls() - c0
        assert engaged > 0, "kernel never engaged — check ARM_SCAN_LIB"

        case = {
            "res": f"{res[0]}x{res[1]}", "height": res[0], "width": res[1],
            "tokens": res[0] * res[1], "params": n_params,
            "ss2d_blocks": n_blocks, "kernel_calls_per_nfe": engaged,
            "per_nfe_s": per_nfe, "peak_rss_mb": peak_rss_mb(),
            "projected_s": {str(n): per_nfe * n for n in NFE_POINTS},
        }
        if args.usd_per_hour:
            case["usd"] = {str(n): per_nfe * n * args.usd_per_hour / 3600
                           for n in NFE_POINTS}
        out["cases"].append(case)

        proj = "  ".join(f"NFE={n}: {per_nfe * n:7.1f}s" for n in NFE_POINTS)
        print(f"{case['res']:>9s}  L={res[0]*res[1]:6d}  "
              f"{n_params/1e6:4.1f}M params  {n_blocks} blocks  "
              f"per-NFE {per_nfe:7.3f}s   {proj}")

    if args.usd_per_hour:
        out["cost"] = {"usd_per_hour": args.usd_per_hour,
                       "instance": args.instance or "(unspecified)"}
        print(f"\ncost at ${args.usd_per_hour}/h "
              f"({args.instance or 'unspecified instance'}):")
        for c in out["cases"]:
            usd = "  ".join(f"NFE={n}: ${c['usd'][str(n)]:.4f}"
                            for n in NFE_POINTS)
            print(f"{c['res']:>9s}  {usd}")

    if args.checkpoint:
        print(f"\nquality at {args.quality_res}x{args.quality_res} "
              f"from {args.checkpoint}:")
        qnet = build_net(args.quality_res, args.model_channels, args.blocks,
                         args.d_state)
        use_arm_scan(qnet)
        Rs = [int(r) for r in args.quality_R.split(",")]
        out["quality"] = run_quality(
            qnet, (args.quality_res, args.quality_res), args.checkpoint, Rs,
            args.quality_steps)
    else:
        print("\nquality rows SKIPPED — pass --checkpoint with a trained "
              "prior.\n(An untrained net would produce numbers, and they "
              "would be meaningless.)")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nresults written to {args.json}")


if __name__ == "__main__":
    main()
