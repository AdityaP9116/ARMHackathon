"""The demo: undersampled MRI reconstruction by an SS2D-Mamba diffusion prior,
running on CPU through the arm_scan kernel.

Produces one image — ground truth | zero-filled | reconstruction — plus the
numbers that belong beside it (PSNR/SSIM, per-NFE latency, kernel engagement,
host/thread/torch provenance). This is the artifact the video is shot around,
so it must run BEFORE any rented hardware is running.

Credential-free by construction: the phantom track synthesises a Shepp-Logan
image and trains a small prior in-process. Nothing is downloaded, no dataset
registration, no AWS account, no external clone. `--checkpoint` swaps in a
properly trained prior when one exists.

Dependency-free by construction too: numpy + torch only. The PNG is written by
a ~20-line encoder below rather than pulling in matplotlib, so `make validate`
never grows a plotting dependency.

Usage:
    python apps/mri_diffusion/demo.py                       # R=4, phantom
    python apps/mri_diffusion/demo.py --R 8 --steps 18
    python apps/mri_diffusion/demo.py --compare-reference   # kernel vs torch
"""

import argparse
import json
import platform
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from apps.mri_diffusion.data import phantom_batch, shepp_logan  # noqa: E402,F401
from apps.mri_diffusion.sampling.posterior import (  # noqa: E402
    cartesian_mask, data_consistency, effective_R, measure, psnr, to_2ch,
    zero_filled)
from apps.mri_diffusion.tests import _edm  # noqa: E402
from arm_scan.op import kernel_calls  # noqa: E402
from arm_scan.ss2d import use_arm_scan  # noqa: E402

EDMPrecond, EDMLoss, construct, EDM_SOURCE = _edm.load()

def ssim(a, b, window=7, C1=1e-4, C2=9e-4):
    """Mean SSIM over a uniform window — no skimage dependency.

    The default constants are the standard `(0.01*L)^2` / `(0.03*L)^2` for a
    dynamic range `L = 1`, which is what the phantom magnitudes span. Pass
    your own for data on a different scale, and never compare SSIM values
    computed at different `L`.
    """
    a, b = a.double(), b.double()
    pad = window // 2
    k = torch.ones(1, 1, window, window, dtype=torch.float64) / window ** 2
    f = lambda t: torch.nn.functional.conv2d(t, k, padding=pad)  # noqa: E731
    mu_a, mu_b = f(a), f(b)
    saa, sbb, sab = f(a * a) - mu_a ** 2, f(b * b) - mu_b ** 2, f(a * b) - mu_a * mu_b
    num = (2 * mu_a * mu_b + C1) * (2 * sab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2)
    return float((num / den).mean())


def magnitude(x2ch):
    return (x2ch[:, 0] ** 2 + x2ch[:, 1] ** 2).sqrt()[:, None]


def write_png(path, img_u8):
    """Minimal greyscale PNG encoder. `img_u8`: (h, w) uint8."""
    h, w = img_u8.shape
    raw = b"".join(b"\x00" + img_u8[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9))
           + chunk(b"IEND", b""))
    Path(path).write_bytes(png)


def montage(panels, gap=6):
    """Side-by-side uint8 montage.

    Each panel is normalised INDEPENDENTLY (standard for MRI display, since
    zero-filled and reconstructed images live on different scales). That means
    relative brightness across panels carries no information — read the PSNR
    and SSIM numbers, not the pixels, for anything quantitative.
    """
    out = []
    for p in panels:
        p = p.detach().cpu().numpy().squeeze()
        lo, hi = float(p.min()), float(p.max())
        out.append(((p - lo) / (hi - lo + 1e-12) * 255).astype(np.uint8))
    h = max(p.shape[0] for p in out)
    sep = np.full((h, gap), 255, dtype=np.uint8)
    strip = []
    for i, p in enumerate(out):
        if i:
            strip.append(sep)
        strip.append(p)
    return np.concatenate(strip, axis=1)


def train_prior(net, res, steps, rng, lr=3e-3, log_every=100, cache=None):
    """Train the prior on synthetic phantoms, or on a fastMRI cache.

    `cache` is a file built by `tools/prepare_fastmri.py`. Real k-space beats
    phantoms as a demonstration, but it is optional by design: the phantom path
    needs no download, no credentials and no data agreement, so `make demo`
    keeps working for anyone.
    """
    sampler = None
    if cache:
        from apps.mri_diffusion.fastmri_data import batcher, load_cache
        imgs = load_cache(cache)
        if imgs.shape[-1] != res:
            raise SystemExit(
                f"cache is {imgs.shape[-1]}px but --res is {res}; rebuild the "
                f"cache with --res {res} or pass --res {imgs.shape[-1]}")
        print(f"   training data: {cache} ({imgs.shape[0]} slices, "
              f"fastMRI knee single-coil)")
        sampler = batcher(imgs, 8)

    loss_fn = EDMLoss()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    t0 = time.time()
    for step in range(steps):
        imgs = next(sampler) if sampler else phantom_batch(8, res, rng)
        loss = loss_fn(net=net, images=_edm.pad_for_loss(imgs),
                       labels=None).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0:
            print(f"   prior step {step:4d}/{steps}  loss {loss.item():.4f}",
                  flush=True)
    net.eval()
    print(f"   trained in {time.time()-t0:.0f}s")
    return net


def reconstruct(net, y, mask, steps, seed=0):
    """Heun + hard data consistency, timing every denoiser call (NFE)."""
    if steps < 2:
        raise ValueError("--steps must be >= 2 (the sigma ladder interpolates "
                         "between the first and last step)")
    b, _, h, w = zero_filled(y, mask).shape
    g = torch.Generator().manual_seed(seed)
    rho, s_max, s_min = 7, 80.0, 0.002
    t = (s_max ** (1 / rho) + torch.arange(steps) / (steps - 1)
         * (s_min ** (1 / rho) - s_max ** (1 / rho))) ** rho
    t = torch.cat([t, torch.zeros(1)])
    x = torch.randn(b, 2, h, w, generator=g) * t[0]
    nfe = []

    with torch.no_grad():
        for i in range(steps):
            t0 = time.perf_counter()
            d = data_consistency(net(x, t[i].repeat(b), None), y, mask)
            nfe.append(time.perf_counter() - t0)
            dx = (x - d) / t[i]
            x1 = x + (t[i + 1] - t[i]) * dx
            if i < steps - 1:
                t1 = time.perf_counter()
                d2 = data_consistency(net(x1, t[i + 1].repeat(b), None),
                                      y, mask)
                nfe.append(time.perf_counter() - t1)
                x1 = x + (t[i + 1] - t[i]) * 0.5 * (dx + (x1 - d2) / t[i + 1])
            x = x1
    return data_consistency(x, y, mask), nfe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--R", type=int, default=4, help="acceleration factor")
    ap.add_argument("--steps", type=int, default=8, help="Heun steps")
    # Training runs on the REFERENCE scan (the kernel registers no autograd),
    # which is a Python loop over L: ~13 s/step at res=32 on a 16-core x86 box.
    # Keep the default modest and reuse a saved prior for repeat runs.
    ap.add_argument("--train-steps", type=int, default=60)
    ap.add_argument("--checkpoint", default=None,
                    help="trained prior (.pt state_dict); else trained here")
    ap.add_argument("--save-prior", default=None,
                    help="write the trained prior here for reuse "
                         "(--checkpoint on later runs)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--require-gain", type=float, default=None,
                    help="exit non-zero unless the reconstruction beats "
                         "zero-filled by this many dB (use in gates)")
    ap.add_argument("--data-cache", default=None,
                    help="fastMRI cache from tools/prepare_fastmri.py; "
                         "trains and evaluates on real knee data instead of "
                         "synthetic phantoms")
    ap.add_argument("--out", default="demo_out/reconstruction.png")
    ap.add_argument("--compare-reference", action="store_true",
                    help="also reconstruct on the torch reference scan")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    host = (f"{platform.platform()} / {platform.machine()}, "
            f"torch {torch.__version__}, {torch.get_num_threads()} threads")
    print(f"host: {host}\nEDM source: {EDM_SOURCE}\n")

    net = construct(
        class_name="training.networks.EDMPrecond", model_type="MambaSS2DNet",
        img_resolution=args.res, img_channels=2, label_dim=0,
        model_channels=32, num_blocks_per_level=1, d_state=16,
        use_fp16=False, sigma_data=0.5)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"1. prior: MambaSS2DNet under EDMPrecond, {n_params/1e3:.0f}K params")

    if args.checkpoint:
        net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        net.eval()
        print(f"   loaded {args.checkpoint}")
    else:
        print(f"   no --checkpoint: training in-process "
              f"({args.train_steps} steps on synthetic phantoms). This is a "
              f"SMOKE-quality prior — it proves the pipeline, not the science.")
        train_prior(net, args.res, args.train_steps, rng,
                    log_every=args.log_every, cache=args.data_cache)
        if args.save_prior:
            Path(args.save_prior).parent.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), args.save_prior)
            print(f"   saved prior -> {args.save_prior} "
                  f"(reuse with --checkpoint)")

    if args.data_cache:
        # Held-out volumes: never seen in training (fastmri_data splits by
        # volume, not by slice, so adjacent slices cannot leak).
        from apps.mri_diffusion.fastmri_data import load_cache
        truth = load_cache(args.data_cache, "eval")[args.seed:args.seed + 1]
    else:
        truth = phantom_batch(1, args.res,
                              np.random.default_rng(args.seed + 99))
    mask = cartesian_mask(args.res, args.res, args.R, acs=8, seed=args.seed)
    y = measure(truth, mask)
    zf = zero_filled(y, mask)
    sampled = float(mask.mean())
    print(f"\n2. measurement: R={args.R} Cartesian, "
          f"{sampled*100:.0f}% of k-space lines kept "
          f"(effective R={effective_R(mask):.2f})")

    n_blocks = use_arm_scan(net)
    calls0 = kernel_calls()
    t0 = time.time()
    recon, nfe = reconstruct(net, y, mask, args.steps, seed=args.seed)
    wall = time.time() - t0
    engaged = kernel_calls() - calls0
    print(f"3. reconstruction on arm_scan: {n_blocks} SS2D blocks, "
          f"{len(nfe)} NFE, {engaged} kernel calls")
    print(f"   wall {wall:.1f}s   per-NFE median "
          f"{np.median(nfe)*1e3:.0f} ms")

    m_truth, m_zf, m_rec = magnitude(truth), magnitude(zf), magnitude(recon)
    metrics = {
        "zero_filled": {"psnr": psnr(m_zf, m_truth),
                        "ssim": ssim(m_zf, m_truth)},
        "reconstruction": {"psnr": psnr(m_rec, m_truth),
                           "ssim": ssim(m_rec, m_truth)},
    }
    print("\n4. quality (magnitude images)")
    for k, v in metrics.items():
        print(f"   {k:16s} PSNR {v['psnr']:6.2f} dB   SSIM {v['ssim']:.4f}")
    gain = metrics["reconstruction"]["psnr"] - metrics["zero_filled"]["psnr"]
    metrics["psnr_gain_db"] = gain
    print(f"   prior contributes {gain:+.2f} dB over zero-filled")
    if gain <= 0:
        print("   *** WARNING: the reconstruction is WORSE than the "
              "zero-filled input. ***")
        print("   The prior has not learned enough to help. This is the "
              "expected outcome for a")
        print("   short in-process run — it is not a kernel or sampler "
              "fault (the kernel-vs-")
        print("   reference cross-check below is the thing that validates "
              "those). Raise")
        print("   --train-steps substantially, or pass a properly trained "
              "--checkpoint.")

    panels = [m_truth, m_zf, m_rec]
    if args.compare_reference:
        use_arm_scan(net, enable=False)
        t0 = time.time()
        recon_ref, nfe_ref = reconstruct(net, y, mask, args.steps,
                                         seed=args.seed)
        wall_ref = time.time() - t0
        m_ref = magnitude(recon_ref)
        parity = float((m_rec - m_ref).abs().max())
        metrics["reference_scan"] = {
            "psnr": psnr(m_ref, m_truth), "ssim": ssim(m_ref, m_truth),
            "wall_s": wall_ref, "per_nfe_ms": float(np.median(nfe_ref) * 1e3)}
        metrics["kernel_vs_reference_max_abs"] = parity
        print(f"\n5. reference-scan cross-check: per-NFE "
              f"{np.median(nfe_ref)*1e3:.0f} ms "
              f"({np.median(nfe_ref)/np.median(nfe):.1f}x the kernel), "
              f"recon max_abs {parity:.3e}")
        panels.append(m_ref)
        use_arm_scan(net)

    metrics.update({
        "host": host, "R": args.R, "res": args.res, "heun_steps": args.steps,
        "nfe": len(nfe), "kernel_calls": engaged, "wall_s": wall,
        "per_nfe_ms": float(np.median(nfe) * 1e3),
        "panels": ["ground_truth", "zero_filled", "reconstruction"]
                  + (["reference_scan"] if args.compare_reference else []),
    })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_png(out, montage(panels))
    out.with_suffix(".json").write_text(json.dumps(metrics, indent=2))
    print(f"\nwrote {out}  (panels: {', '.join(metrics['panels'])})")
    print(f"wrote {out.with_suffix('.json')}")

    if args.require_gain is not None and gain < args.require_gain:
        print(f"\nFAIL: prior gain {gain:+.2f} dB < required "
              f"{args.require_gain:+.2f} dB")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
