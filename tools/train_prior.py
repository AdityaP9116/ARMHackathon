"""Train the SS2D-Mamba EDM prior. The real trainer, not the demo's inline loop.

`demo.py`'s `train_prior()` is a smoke path: it never moves anything to CUDA
(so on a rented GPU box it trains on the CPU while the GPU idles), keeps no
EMA, and saves only at the very end. This is what you point at a GPU.

    # phantoms, no download needed — proves the loop works
    python tools/train_prior.py --steps 200 --res 64 --out demo_out/p.pt

    # the real run
    python tools/train_prior.py --cache data/knee_128.pt --res 128 \
        --steps 20000 --device cuda --amp --out data/prior_knee128.pt

WHAT IT DOES THAT MATTERS
-------------------------
**EMA.** Diffusion models are sampled from an exponential moving average of the
training trajectory, not the live weights. This is worth several dB and is
standard in EDM; the checkpoint's primary weights are the EMA copy.

**A stop condition, not a step count.** Every `--eval-every` steps it measures
denoiser NRMSE across the sampler's sigma ladder — the same measurement
`tools/prior_report.py` makes — and stops as soon as every rung clears the bar.
Training longer than that buys nothing you can see in the reconstruction, and
GPU time is the scarce resource. `--no-early-stop` disables it.

**Resume.** Checkpoints are written every `--save-every` steps and carry the
optimiser and EMA state, so a spot interruption costs minutes rather than the
run. `--resume` picks up where it stopped.

**Self-describing checkpoints.** The architecture is embedded, so
`prior_report.py`, `demo.py` and the quality gate rebuild the right network
without being handed flags that have to match.

THE BAR IS DATA-DEPENDENT
-------------------------
`--nrmse-bar` defaults to the phantom-calibrated 0.62. **Re-derive it for
fastMRI** with `tools/calibrate_prior_bar.py --data fastmri --cache ...` and
pass the result; the bar encodes how much energy the mask destroys on that
data, and it moves by ~8x between image families.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.mri_diffusion.checkpoint import EMA, save_prior  # noqa: E402
from apps.mri_diffusion.data import phantom_batch  # noqa: E402
from apps.mri_diffusion.tests import _edm  # noqa: E402
from tools.prior_report import sigma_ladder  # noqa: E402

EDMPrecond, EDMLoss, construct, EDM_SOURCE = _edm.load()


def evaluate(net, images, steps, sigma_max, bar):
    """Denoiser NRMSE across the sampler's ladder -> (worst, n_failing)."""
    rms = float((images ** 2).mean().sqrt())
    worst, failing = 0.0, 0
    net.eval()
    with torch.no_grad():
        for s in sigma_ladder(steps, sigma_max):
            noisy = images + float(s) * torch.randn_like(images)
            den = net(noisy, torch.full((images.shape[0],), float(s),
                                        device=images.device), None)
            nrmse = float(((den - images) ** 2).mean().sqrt()) / rms
            worst = max(worst, nrmse)
            failing += int(nrmse >= bar)
    net.train()
    return worst, failing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/prior.pt")
    ap.add_argument("--cache", default=None,
                    help="fastMRI cache; omit to train on synthetic phantoms")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--ema-decay", type=float, default=0.9995)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--model-channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--d-state", type=int, default=16)
    ap.add_argument("--sigma-data", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--amp", action="store_true",
                    help="bf16 autocast (GPU only; large speedup)")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--nrmse-bar", type=float, default=0.62,
                    help="phantom-calibrated; RE-DERIVE for fastMRI")
    ap.add_argument("--no-early-stop", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but no CUDA device is visible")

    # ---- data -------------------------------------------------------
    if args.cache:
        from apps.mri_diffusion.fastmri_data import batcher, load_cache
        train = load_cache(args.cache, "train", device=dev)
        if train.shape[-1] != args.res:
            raise SystemExit(f"cache is {train.shape[-1]}px, --res is "
                             f"{args.res}; rebuild or change --res")
        try:
            holdout = load_cache(args.cache, "eval", device=dev)[:16]
        except SystemExit:
            holdout = train[:16]
        sampler = batcher(train, args.batch)
        source = f"fastMRI knee singlecoil ({train.shape[0]} slices)"
    else:
        rng = np.random.default_rng(args.seed)
        sampler = None
        holdout = phantom_batch(16, args.res, rng).to(dev)
        source = "synthetic phantoms"

    # ---- model ------------------------------------------------------
    cfg = dict(img_resolution=args.res, img_channels=2, label_dim=0,
               model_channels=args.model_channels,
               num_blocks_per_level=args.blocks, d_state=args.d_state,
               sigma_data=args.sigma_data)
    net = construct(class_name="training.networks.EDMPrecond",
                    model_type="MambaSS2DNet", use_fp16=False, **cfg).to(dev)
    loss_fn = EDMLoss(sigma_data=args.sigma_data)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    ema = EMA(net, decay=args.ema_decay)
    start = 0

    if args.resume:
        blob = torch.load(args.resume, map_location=dev, weights_only=False)
        net.load_state_dict(blob["model_raw"])
        if "optimizer" in blob:
            opt.load_state_dict(blob["optimizer"])
        if "ema" in blob:
            ema.shadow = {k: v.to(dev) for k, v in blob["ema"].items()}
        start = int(blob.get("step", 0))
        print(f"resumed from {args.resume} at step {start}")

    n_params = sum(p.numel() for p in net.parameters())
    sigma_max = float(getattr(net, "sigma_max_trained", 80.0))
    print(f"EDM source : {EDM_SOURCE}")
    print(f"data       : {source}, {args.res}px")
    print(f"model      : {n_params/1e6:.2f}M params, "
          f"{args.model_channels}ch x {args.blocks} blocks/level")
    print(f"device     : {dev}"
          f"{' (bf16 autocast)' if args.amp else ''}")
    print(f"schedule   : {args.steps} steps, batch {args.batch}, lr {args.lr}, "
          f"EMA {args.ema_decay}")
    print(f"stop when  : NRMSE < {args.nrmse_bar} at every sigma "
          f"(ladder to {sigma_max:.1f})"
          f"{' [disabled]' if args.no_early_stop else ''}\n")
    if not args.cache:
        print("NOTE: training on phantoms. For the real result pass --cache "
              "from tools/prepare_fastmri.py.\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    amp_ctx = (torch.autocast(device_type=dev.type, dtype=torch.bfloat16)
               if args.amp else torch.autocast(device_type=dev.type,
                                               enabled=False))

    losses, t0, seen = [], time.time(), 0
    for step in range(start, args.steps):
        imgs = (next(sampler) if sampler
                else phantom_batch(args.batch, args.res, rng).to(dev))
        # Linear warmup: EDM's loss weights are large at small sigma and a
        # cold Adam at full lr can diverge in the first few dozen steps.
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (step + 1) / max(1, args.warmup))

        with amp_ctx:
            loss = loss_fn(net=net, images=_edm.pad_for_loss(imgs),
                           labels=None).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        opt.step()
        ema.update(net, step)
        losses.append(loss.item())
        seen += imgs.shape[0]

        if step % args.log_every == 0:
            el = time.time() - t0
            rate = seen / max(el, 1e-9)
            eta = (args.steps - step) * (el / max(step - start + 1, 1))
            print(f"  step {step:6d}/{args.steps}  loss "
                  f"{np.mean(losses[-args.log_every:]):.4f}  "
                  f"{rate:6.1f} img/s  eta {eta/60:5.1f} min", flush=True)

        done = step + 1 >= args.steps
        if (step + 1) % args.eval_every == 0 or done:
            worst, failing = evaluate(net, holdout, 10, sigma_max,
                                      args.nrmse_bar)
            gain = -20 * np.log10(max(worst, 1e-9)) - 3.1
            print(f"  [eval] worst NRMSE {worst:.3f} over the ladder "
                  f"({failing} rung(s) failing) -> reconstruction can be no "
                  f"better than {gain:+.1f} dB", flush=True)
            if failing == 0 and not args.no_early_stop:
                print("  [eval] every rung clears the bar — stopping early.")
                done = True

        if (step + 1) % args.save_every == 0 or done:
            save_prior(args.out, net, cfg, step + 1,
                       ema_state=ema.state_dict(net),
                       meta={"data": source, "loss": float(np.mean(
                           losses[-100:])), "device": str(dev),
                           "nrmse_bar": args.nrmse_bar})
            torch.save({"model_raw": net.state_dict(),
                        "optimizer": opt.state_dict(),
                        "ema": ema.shadow, "step": step + 1},
                       str(args.out) + ".resume")
            print(f"  saved {args.out} (step {step+1})", flush=True)
        if done:
            break

    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {args.out}")
    print(f"check it:  python tools/prior_report.py --checkpoint {args.out}"
          + (f" --cache {args.cache}" if args.cache else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
