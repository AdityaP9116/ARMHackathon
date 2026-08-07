"""Synthetic image sources for the phantom track. No downloads, no credentials.

Two families, and **which one you use decides whether an experiment can even
succeed**:

`toy_batch` — superposed Gaussian bumps. Smooth, so essentially all of their
    energy sits in low spatial frequencies. Fine for asking "does the network
    learn to denoise?", which is all `test_backbone_bringup.py` needs.

`phantom_batch` — Shepp-Logan ellipses. **Sharp boundaries**, so real energy
    lives in high frequencies. This is what reconstruction evaluation must use.

WHY THAT DISTINCTION IS LOAD-BEARING
------------------------------------
A Cartesian mask with a centre ACS block keeps the low frequencies and throws
away high ones. On smooth data that discards almost nothing, so zero-filling is
already near-optimal and there is nothing left for a generative prior to
restore — any detail it invents is a net loss under PSNR. Measured on the
bump data: sweeping the sampler's sigma_max only ever crept *up toward*
zero-filled and never crossed it, no matter how much prior influence was
dialled in.

That is not a weak prior. It is an evaluation that cannot be won. Sharp edges
put recoverable information in the discarded band, which is the regime where a
diffusion prior is supposed to help. See PHASE_D_DIAGNOSIS.md §3 D-alpha.
"""

import numpy as np
import torch

CH = 2  # complex MRI carried as 2 real channels

# Standard Shepp-Logan ellipses: (intensity, a, b, x0, y0, phi_degrees).
_SL = [
    (1.00, .69, .92, 0., 0., 0), (-.80, .6624, .8740, 0., -.0184, 0),
    (-.20, .1100, .3100, .22, 0., -18), (-.20, .1600, .4100, -.22, 0., 18),
    (0.10, .2100, .2500, 0., .35, 0), (0.10, .0460, .0460, 0., .1, 0),
    (0.10, .0460, .0460, 0., -.1, 0), (0.10, .0460, .0230, -.08, -.605, 0),
    (0.10, .0230, .0230, 0., -.606, 0), (0.10, .0230, .0460, .06, -.605, 0),
]


def shepp_logan(n, jitter=0.0, rng=None):
    """Shepp-Logan phantom on an n x n grid.

    `jitter` perturbs each ellipse's size, position and rotation so a family of
    related images can be drawn for training without them being identical.
    """
    yy, xx = np.mgrid[-1:1:complex(0, n), -1:1:complex(0, n)]
    img = np.zeros((n, n), dtype=np.float64)
    for inten, a, b, x0, y0, phi in _SL:
        if jitter and rng is not None:
            a = a * (1 + jitter * rng.normal())
            b = b * (1 + jitter * rng.normal())
            x0 = x0 + jitter * rng.normal() * .1
            y0 = y0 + jitter * rng.normal() * .1
            phi = phi + jitter * rng.normal() * 20
        t = np.deg2rad(phi)
        xr = (xx - x0) * np.cos(t) + (yy - y0) * np.sin(t)
        yr = -(xx - x0) * np.sin(t) + (yy - y0) * np.cos(t)
        img[(xr / a) ** 2 + (yr / b) ** 2 <= 1] += inten
    return img


def phantom_batch(n_imgs, res, rng=None, jitter=0.06, phase=True):
    """`(n, 2, res, res)` Shepp-Logan phantoms with a smooth phase ramp.

    Real MRI is complex-valued, and a purely real image would make the second
    channel degenerate — so a smooth phase is applied, which is also what the
    2-channel backbone expects to see.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    yy, xx = np.mgrid[-1:1:complex(0, res), -1:1:complex(0, res)]
    ph = (0.6 * np.sin(2.1 * xx + 0.4) + 0.5 * np.cos(1.7 * yy - 0.2)
          if phase else np.zeros((res, res)))
    out = []
    for _ in range(n_imgs):
        mag = shepp_logan(res, jitter=jitter, rng=rng)
        out.append(np.stack([mag * np.cos(ph), mag * np.sin(ph)]))
    return torch.from_numpy(np.asarray(out, dtype=np.float32))


def toy_batch(n, res=32, device="cpu", generator=None):
    """`(n, 2, res, res)` smooth fields: superposed Gaussian bumps.

    Denoising-only test data. Do NOT use for reconstruction evaluation — see
    the module docstring.
    """
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, res),
                            torch.linspace(-1, 1, res), indexing="ij")
    g = generator
    if g is None:
        g = torch.Generator().manual_seed(
            int(torch.randint(0, 1 << 31, (1,))))
    imgs = []
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


def highfreq_energy_fraction(imgs, keep_fraction=0.25):
    """Fraction of k-space energy OUTSIDE the central `keep_fraction` columns.

    The one-line justification for using phantoms over bumps: it reports how
    much energy a centre-keeping mask would actually discard, i.e. how much
    there is for a prior to restore. Near zero means the evaluation cannot
    distinguish a good prior from no prior at all.
    """
    x = torch.complex(imgs[:, 0], imgs[:, 1])
    k = torch.fft.fft2(x, norm="ortho")
    k = torch.fft.fftshift(k, dim=(-2, -1))
    w = k.shape[-1]
    lo = int(w / 2 - keep_fraction * w / 2)
    hi = int(w / 2 + keep_fraction * w / 2)
    total = (k.abs() ** 2).sum()
    centre = (k[..., lo:hi].abs() ** 2).sum()
    return float((total - centre) / total)
