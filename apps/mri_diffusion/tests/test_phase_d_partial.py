"""Phase-D gate: partial (undersampled) reconstruction on CPU with the
kernel in the loop.

Phantom track (credential-free): train a tiny MambaSS2DNet prior on toy
smooth fields, simulate R={2,4} Cartesian undersampling, reconstruct via
Heun + hard data-consistency, and require:
  1. recon PSNR beats zero-filled PSNR by >1 dB at R=4 (prior adds value);
  2. kernel-path vs reference-scan-path reconstructions agree (parity);
  3. everything finite; kernel engagement counter grew.

Usage: python -u apps/mri_diffusion/tests/test_phase_d_partial.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
REF = Path(r"C:\Users\Adity\Claude\Projects\reference\ambient-diffusion-mri")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(REF))

import dnnlib  # noqa: E402
import training.networks as tn  # noqa: E402
from training.loss import EDMLoss  # noqa: E402
from apps.mri_diffusion.backbone.mamba_ss2d import MambaSS2DNet  # noqa: E402
from apps.mri_diffusion.tests.test_backbone_bringup import toy_batch  # noqa: E402
from apps.mri_diffusion.sampling.posterior import (  # noqa: E402
    cartesian_mask, heun_posterior, measure, psnr, zero_filled)
from arm_scan.op import kernel_calls  # noqa: E402
from arm_scan.ss2d import use_arm_scan  # noqa: E402

torch.manual_seed(0)
RES = 32


def main():
    tn.MambaSS2DNet = MambaSS2DNet
    net = dnnlib.util.construct_class_by_name(
        class_name="training.networks.EDMPrecond", model_type="MambaSS2DNet",
        img_resolution=RES, img_channels=2, label_dim=0, model_channels=32,
        num_blocks_per_level=1, d_state=16, use_fp16=False, sigma_data=0.5)

    loss_fn, opt = EDMLoss(), torch.optim.Adam(net.parameters(), lr=3e-3)
    for step in range(200):
        imgs = toy_batch(8)
        padded = torch.cat([torch.zeros_like(imgs), imgs], dim=-1)
        loss = loss_fn(net=net, images=padded, labels=None).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            print(f"prior training step {step} loss {loss.item():.4f}",
                  flush=True)
    net.eval()

    clean = toy_batch(1)
    results = {}
    for R in (2, 4):
        mask = cartesian_mask(RES, RES, R, acs=6, seed=R)
        y = measure(clean, mask)
        zf = zero_filled(y, mask)
        use_arm_scan(net)
        c0 = kernel_calls()
        recon = heun_posterior(net, y, mask, num_steps=10)
        engaged = kernel_calls() - c0
        p_zf, p_re = psnr(zf, clean), psnr(recon, clean)
        results[R] = (p_zf, p_re)
        print(f"R={R}: zero-filled {p_zf:.2f} dB -> recon {p_re:.2f} dB "
              f"(kernel calls {engaged})", flush=True)
        assert torch.isfinite(recon).all() and engaged > 0

    assert results[4][1] > results[4][0] + 1.0, \
        "recon does not beat zero-filled by >1dB at R=4"

    # parity: same measurement, reference-scan path
    mask = cartesian_mask(RES, RES, 4, acs=6, seed=4)
    y = measure(clean, mask)
    use_arm_scan(net)
    r_kern = heun_posterior(net, y, mask, num_steps=8)
    use_arm_scan(net, enable=False)
    r_ref = heun_posterior(net, y, mask, num_steps=8)
    diff = (r_kern - r_ref).abs().max().item()
    scale = r_ref.abs().max().item()
    print(f"parity kernel-vs-reference: max_abs={diff:.3e} "
          f"(scale {scale:.2f})", flush=True)
    assert diff < 1e-3 * max(1.0, scale), "parity FAILED"

    print("\nPHASE D GATE: PASS — partial reconstruction (R=2/4) on CPU "
          "through arm_scan, beats zero-filled, parity verified", flush=True)


if __name__ == "__main__":
    main()
