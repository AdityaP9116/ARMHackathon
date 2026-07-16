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

Runs on CPU in ~2-4 min. Uses the CSI reference repo for EDMPrecond,
EDMLoss, and persistence — the same classes the real training uses.

Usage: python apps/mri_diffusion/tests/test_backbone_bringup.py
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

APP = Path(__file__).resolve().parents[1]
REF = Path(r"C:\Users\Adity\Claude\Projects\reference\ambient-diffusion-mri")
sys.path.insert(0, str(APP.parent.parent))  # repo root (apps package)
sys.path.insert(0, str(REF))  # dnnlib, torch_utils, training

import dnnlib  # noqa: E402
from training.loss import EDMLoss  # noqa: E402

torch.manual_seed(0)
RES, CH, BATCH = 32, 2, 8


def toy_batch(n, res=RES, device="cpu"):
    """Random smooth 2-channel fields (superposed gaussian bumps)."""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, res),
                            torch.linspace(-1, 1, res), indexing="ij")
    imgs = []
    g = torch.Generator().manual_seed(int(torch.randint(0, 1 << 31, (1,))))
    for _ in range(n):
        img = torch.zeros(CH, res, res)
        for _ in range(4):
            cx, cy = torch.rand(2, generator=g) * 1.6 - 0.8
            s = 0.15 + 0.25 * torch.rand(1, generator=g)
            amp = torch.randn(CH, generator=g)
            bump = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
            img += amp[:, None, None] * bump
        imgs.append(img)
    return torch.stack(imgs).to(device)


def main():
    # --- 1. interface: construct exactly like train.py/prior.py ---------
    # EDMPrecond looks up model_type in ITS OWN globals(); inject ours the
    # way the --arch=ss2dmamba branch will (documented insertion point).
    import training.networks as tn
    from apps.mri_diffusion.backbone.mamba_ss2d import MambaSS2DNet
    tn.MambaSS2DNet = MambaSS2DNet
    net = dnnlib.util.construct_class_by_name(
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
        # (384->320). Pre-pad 32 zero columns so the unmodified crop lands
        # exactly on our 32-wide toy content — a Phase-A finding worth
        # keeping visible here.
        padded = torch.cat([torch.zeros_like(imgs), imgs], dim=-1)
        loss = loss_fn(net=net, images=padded, labels=None).mean()
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
