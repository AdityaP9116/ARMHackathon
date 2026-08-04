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
    """CENTRED orthonormal 2D FFT: DC lands at the middle of the array.

    The shifts are not cosmetic. Without them `torch.fft.fft2` puts DC at
    index 0 and Nyquist at N/2 — so `cartesian_mask`'s centre "ACS" block was
    sampling the HIGHEST frequencies, i.e. the least informative columns in
    the whole of k-space. Measured on a 64px phantom at R=4: the unshifted
    layout captured **12.6%** of k-space energy where the centred one captures
    **64.7%**, and zero-filled reconstruction went 23.00 -> 26.94 dB.

    Centred is also the MRI convention and what fastMRI uses, so raw k-space
    loaded from a `.h5` needs no re-shuffling to line up with our masks.
    """
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)),
                       dim=(-2, -1), norm="ortho"), dim=(-2, -1))


def ifft(x):
    """Inverse of `fft`; see there for why the shifts matter."""
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)),
                        dim=(-2, -1), norm="ortho"), dim=(-2, -1))


def cartesian_mask(h, w, R, acs=8, seed=0):
    """Random column undersampling at acceleration R, ACS block INSIDE the budget.

    Keeps `round(w / R)` columns in total, of which `acs` form the centre
    autocalibration block and the rest are drawn uniformly from the periphery
    without replacement. So `mask.mean() == 1/R` and the nominal acceleration
    is the real one.

    This used to force the ACS block on top of an independent `rand < 1/R`
    draw, which oversampled by `(acs/w)*(1 - 1/R)`: at 32 columns with acs=6,
    a nominal R=4 actually kept 37.5% of k-space — an **effective R of 2.67**.
    Every R-labelled number produced before this fix was optimistic, and the
    inflated sampling also flattered the zero-filled baseline that a
    reconstruction has to beat. See PHASE_D_DIAGNOSIS.md §3 D-beta.
    """
    if not 1 <= R:
        raise ValueError(f"R must be >= 1, got {R}")
    g = torch.Generator().manual_seed(seed)
    acs = min(int(acs), w)
    n_total = min(w, max(acs, int(round(w / R))))

    m = torch.zeros(w)
    lo = max(0, w // 2 - acs // 2)
    hi = min(w, lo + acs)
    m[lo:hi] = 1.0

    n_random = n_total - (hi - lo)
    if n_random > 0:
        rest = torch.cat([torch.arange(0, lo), torch.arange(hi, w)])
        pick = rest[torch.randperm(rest.numel(), generator=g)[:n_random]]
        m[pick] = 1.0
    return m[None, None, None, :].expand(1, 1, h, w)


def effective_R(mask):
    """The acceleration a mask actually delivers: 1 / (sampled fraction).

    Report this, never the nominal R you asked for. They agree now, but a
    small grid with a large ACS block has a hard floor — 32 columns with
    acs=6 cannot exceed R=5.33 however high you set R, because the ACS block
    alone is 6/32 of k-space.
    """
    return 1.0 / float(mask.mean())


def measure(x2ch, mask):
    return mask * fft(to_cplx(x2ch))


def zero_filled(y, mask):
    return to_2ch(ifft(mask * y))


def data_consistency(x2ch, y, mask):
    k = fft(to_cplx(x2ch))
    k = mask * y + (1 - mask) * k
    return to_2ch(ifft(k))


def heun_posterior(net, y, mask, num_steps=12, sigma_max=None,
                   sigma_min=0.002, rho=7, seed=0):
    """Deterministic Heun sampling with per-step hard DC on the denoised
    estimate. Returns the reconstruction (b,2,h,w).

    `sigma_max=None` (default) reads `net.sigma_max_trained` if the network
    carries it, falling back to EDM's textbook 80.0. That default matters: a
    sampler run far above the sigma range the prior was trained on spends its
    first steps — the ones that fix global structure — asking the denoiser
    questions it has never seen. Pass an explicit value only when you mean to
    override the model's own declaration. See edm_min.trained_sigma_max.
    """
    if sigma_max is None:
        sigma_max = float(getattr(net, "sigma_max_trained", 80.0))
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
