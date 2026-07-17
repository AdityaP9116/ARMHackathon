"""Partial (undersampled) MRI reconstruction: EDM Heun sampling with hard
data-consistency (phantom track: single-coil Cartesian, no maps).

y = M * F(x): 2-channel real <-> complex; centered orthonormal FFT (CSI
convention). DC projects the denoised estimate's k-space onto the measured
lines each step — the csgm/DPS-family measurement step in its simplest
robust form. Multi-coil A-DPS (maps, --l_ss guidance) is the CSI-inherited
upgrade path per MRI_DIFFUSION_IMPLEMENTATION_PLAN §7.2.
"""

import torch


def to_cplx(x):  # (b,2,h,w) -> (b,1,h,w) complex
    return torch.complex(x[:, 0], x[:, 1])[:, None]


def to_2ch(c):
    return torch.cat([c.real, c.imag], dim=1)


def fft(x):
    return torch.fft.fft2(x, dim=(-2, -1), norm="ortho")


def ifft(x):
    return torch.fft.ifft2(x, dim=(-2, -1), norm="ortho")


def cartesian_mask(h, w, R, acs=8, seed=0):
    """Random column undersampling at acceleration R with an ACS block."""
    g = torch.Generator().manual_seed(seed)
    m = (torch.rand(w, generator=g) < (1.0 / R)).float()
    m[w // 2 - acs // 2:w // 2 + acs // 2] = 1.0
    return m[None, None, None, :].expand(1, 1, h, w)


def measure(x2ch, mask):
    return mask * fft(to_cplx(x2ch))


def zero_filled(y, mask):
    return to_2ch(ifft(mask * y))


def data_consistency(x2ch, y, mask):
    k = fft(to_cplx(x2ch))
    k = mask * y + (1 - mask) * k
    return to_2ch(ifft(k))


def heun_posterior(net, y, mask, num_steps=12, sigma_max=80.0,
                   sigma_min=0.002, rho=7, seed=0):
    """Deterministic Heun sampling with per-step hard DC on the denoised
    estimate. Returns the reconstruction (b,2,h,w)."""
    b, _, h, w = zero_filled(y, mask).shape
    g = torch.Generator().manual_seed(seed)
    t = (sigma_max ** (1 / rho) + torch.arange(num_steps) / (num_steps - 1)
         * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t = torch.cat([t, torch.zeros(1)])
    x = torch.randn(b, 2, h, w, generator=g) * t[0]

    with torch.no_grad():
        for i in range(num_steps):
            sig = t[i].repeat(b)
            d = data_consistency(net(x, sig, None), y, mask)
            dx = (x - d) / t[i]
            x1 = x + (t[i + 1] - t[i]) * dx
            if i < num_steps - 1:
                d2 = data_consistency(net(x1, t[i + 1].repeat(b), None),
                                      y, mask)
                x1 = x + (t[i + 1] - t[i]) * 0.5 * (dx + (x1 - d2) / t[i + 1])
            x = x1
    return data_consistency(x, y, mask)


def psnr(a, b):
    mse = ((a - b) ** 2).mean().item()
    peak = (b.max() - b.min()).item()
    return 10 * torch.log10(torch.tensor(peak ** 2 / mse)).item()
