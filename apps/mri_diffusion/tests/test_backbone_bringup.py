"""Phase-B bring-up gate: MambaSS2DNet denoises under EDMPrecond and
round-trips through EDM persistence.

Checks, in order:
  1. INTERFACE — constructs under EDMPrecond via construct_class_by_name
     (exactly how prior.py/train.py build networks) with img_channels=2.
  2. TRAINS — a tiny net on a toy dataset (random smooth 2-channel fields)
     with the real EDMLoss: loss must drop substantially.
  3. DENOISES — D_theta(clean+noise; sigma) must beat the noisy input by a
     clear margin (relative MSE), at two sigma levels.
  4. PERSISTS — pickle via the EDM/CSI persistence machinery, reload,
     identical outputs.

Runs on CPU in ~2-4 min, on a clean checkout with no external clone: the EDM
preconditioning and loss come from `apps/mri_diffusion/edm_min.py`. Set
`ADM_REF` to a clone of ambient-diffusion-mri to run the identical checks
against the CSI classes instead (see tests/_edm.py).

Usage: python apps/mri_diffusion/tests/test_backbone_bringup.py
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP.parent.parent))  # repo root (apps package)

from apps.mri_diffusion.data import toy_batch as _toy_batch  # noqa: E402
from apps.mri_diffusion.tests import _edm  # noqa: E402

EDMPrecond, EDMLoss, construct, EDM_SOURCE = _edm.load()

torch.manual_seed(0)
RES, CH, BATCH = 32, 2, 8


def toy_batch(n, res=RES, device="cpu"):
    """Smooth 2-channel fields. Fine here — this test only asks whether the
    network learns to DENOISE. Reconstruction evaluation must not use these;
    see apps/mri_diffusion/data.py for why."""
    return _toy_batch(n, res=res, device=device)


def main():
    # --- 1. interface: construct exactly like train.py/prior.py ---------
    # EDMPrecond looks up model_type by NAME (in EDM's own module globals);
    # _edm.load() has already performed that injection for the CSI path.
    print(f"0. EDM source: {EDM_SOURCE}")
    net = construct(
        class_name="training.networks.EDMPrecond",
        model_type="MambaSS2DNet",
        img_resolution=RES, img_channels=CH, label_dim=0,
        model_channels=32, num_blocks_per_level=1, d_state=8,
        use_fp16=False, sigma_data=0.5,
    )
    n_params = sum(p.numel() for p in net.parameters())
    print(f"1. interface ok: EDMPrecond(MambaSS2DNet), "
          f"{n_params/1e3:.0f}K params")
    x = torch.randn(2, CH, RES, RES)
    d = net(x, torch.tensor([1.7, 0.3]), None)
    assert d.shape == x.shape and torch.isfinite(d).all()
    print(f"   forward ok: D(x;sigma) -> {tuple(d.shape)}, finite")

    # --- 2. train on the toy target with the real EDMLoss ---------------
    loss_fn = EDMLoss()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    losses = []
    t0 = time.time()
    for step in range(300):
        imgs = toy_batch(BATCH)
        # CSI's EDMLoss hardcodes the MRI width crop images[:,:,:,32:352]
        # (384->320), so under ADM_REF the batch is pre-padded with 32 zero
        # columns to land the unmodified crop on our toy content — a Phase-A
        # finding worth keeping visible. edm_min has no crop; identity there.
        loss = loss_fn(net=net, images=_edm.pad_for_loss(imgs),
                       labels=None).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 75 == 0:
            print(f"   step {step:4d} loss {loss.item():.4f}")
    first, last = np.mean(losses[:20]), np.mean(losses[-20:])
    print(f"2. training: loss {first:.4f} -> {last:.4f} "
          f"({time.time()-t0:.0f}s)")
    assert last < 0.6 * first, "loss did not drop enough"

    # --- 3. denoising beats the noisy input -----------------------------
    net.eval()
    clean = toy_batch(16)
    report = []
    with torch.no_grad():
        for sigma in (0.3, 1.0):
            noisy = clean + sigma * torch.randn_like(clean)
            den = net(noisy, torch.full((16,), sigma), None)
            mse_in = ((noisy - clean) ** 2).mean().item()
            mse_out = ((den - clean) ** 2).mean().item()
            report.append((sigma, mse_in, mse_out))
            print(f"3. sigma={sigma}: noisy MSE {mse_in:.4f} -> "
                  f"denoised {mse_out:.4f} ({mse_in/mse_out:.1f}x better)")
            assert mse_out < 0.5 * mse_in, f"weak denoising at sigma={sigma}"

    # --- 4. EDM persistence round-trip -----------------------------------
    blob = pickle.dumps(net)  # persistence embeds class source
    net2 = pickle.loads(blob)
    with torch.no_grad():
        torch.manual_seed(7)
        probe = torch.randn(2, CH, RES, RES)
        o1 = net(probe, torch.full((2,), 0.5), None)
        o2 = net2(probe, torch.full((2,), 0.5), None)
    assert torch.equal(o1, o2), "persistence round-trip changed outputs"
    print(f"4. persistence: pickled {len(blob)/1e6:.1f}MB, reloaded, "
          f"outputs identical")

    print("\nPHASE B BRING-UP: PASS")


if __name__ == "__main__":
    main()
