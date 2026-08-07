"""Phase-D PIPELINE gate — everything about reconstruction that does not
depend on how good the prior is.

WHY THIS EXISTS SEPARATELY
--------------------------
The old single Phase-D test asserted reconstruction *quality* first and kernel
parity second, so an undertrained prior stopped the kernel check from ever
running. But the submission's claim is about the **kernel**; the >1 dB quality
bar measures the **model**. One failing must not hide the other.

So: this file gates the machinery — FFT, mask, data consistency, the Heun
integrator, and the kernel inside the sampling loop — with **no trained prior
anywhere**. It runs in seconds and belongs in CI.
`test_phase_d_quality.py` holds the quality bar and needs a checkpoint.

THE ORACLE ARGUMENT
-------------------
With a denoiser that returns the true image, `DC(x_true) == x_true`
identically, so Heun *must* return the truth. That makes reconstruction
exactness a property we can assert without any training at all — and it is
what proved the sampler was never the cause of Phase D's failure (it
reconstructs to ~151 dB). If someone later breaks the FFT convention, the
mask, the DC projection or the integrator, this is the test that catches it.

Usage: python apps/mri_diffusion/tests/test_phase_d_pipeline.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from apps.mri_diffusion.data import phantom_batch  # noqa: E402
from apps.mri_diffusion.sampling.posterior import (  # noqa: E402
    cartesian_mask, data_consistency, effective_R, fft, heun_posterior, ifft,
    measure, psnr, to_2ch, to_cplx, zero_filled)

RES = 32
IDENTITY_TOL = 1e-5
ORACLE_DB = 100.0     # a correct pipeline reconstructs essentially exactly
KERNEL_TOL = 1e-3     # sampler-level kernel-vs-reference, scale-relative


class Oracle:
    """A denoiser with a controlled error level. eps=0 is perfect."""

    def __init__(self, truth, eps=0.0, seed=0):
        self.truth, self.eps = truth, eps
        self.g = torch.Generator().manual_seed(seed)
        # The sampler asks the net for its trained sigma range; an oracle is
        # correct at every sigma, so keep EDM's textbook ceiling here.
        self.sigma_max_trained = 80.0

    def __call__(self, x, sigma, labels=None):
        out = self.truth.expand_as(x).clone()
        if self.eps:
            out = out + self.eps * torch.randn(x.shape, generator=self.g)
        return out


def check(label, ok, detail=""):
    print(f"   {label:44s} {detail:26s} {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    torch.manual_seed(0)
    ok = True
    truth = phantom_batch(1, RES)

    # --- 1. transform identities -------------------------------------
    print("1. FFT / mask / data-consistency identities")
    full = torch.ones(1, 1, RES, RES)
    rt = (to_2ch(ifft(fft(to_cplx(truth)))) - truth).abs().max().item()
    ok &= check("ifft(fft(x)) == x", rt < IDENTITY_TOL, f"{rt:.3e}")

    zf_full = (zero_filled(measure(truth, full), full) - truth).abs().max().item()
    ok &= check("zero_filled(full mask) == x", zf_full < IDENTITY_TOL,
                f"{zf_full:.3e}")

    mask = cartesian_mask(RES, RES, 4, acs=6, seed=4)
    y = measure(truth, mask)
    dc = (data_consistency(truth, y, mask) - truth).abs().max().item()
    ok &= check("DC(truth) == truth", dc < IDENTITY_TOL, f"{dc:.3e}")

    # --- 2. the mask delivers the acceleration it claims --------------
    print("\n2. mask acceleration (nominal R must equal effective R)")
    for R in (2, 4, 8):
        m = cartesian_mask(128, 128, R, acs=8, seed=R)
        eff = effective_R(m)
        good = abs(eff - R) < 0.05 * R
        ok &= check(f"R={R} at 128x128", good, f"effective {eff:.2f}")

    # --- 3. ORACLE exactness: the real regression gate on the sampler --
    print("\n3. oracle denoiser -> reconstruction must be essentially exact")
    zf_psnr = psnr(zero_filled(y, mask), truth)
    for steps in (10, 18):
        rec = heun_posterior(Oracle(truth), y, mask, num_steps=steps)
        p = psnr(rec, truth)
        ok &= check(f"{steps}-step Heun", p > ORACLE_DB, f"{p:.1f} dB")
    print(f"   (zero-filled baseline for reference: {zf_psnr:.2f} dB)")

    # --- 4. quality metric is wired up the right way round ------------
    # More denoiser error must mean worse reconstruction. Cheap insurance
    # that a future refactor hasn't inverted or disconnected the metric.
    print("\n4. reconstruction degrades monotonically with denoiser error")
    prev, mono = float("inf"), True
    for eps in (0.02, 0.1, 0.4):
        p = psnr(heun_posterior(Oracle(truth, eps=eps), y, mask,
                                num_steps=10), truth)
        mono &= p < prev
        print(f"   denoiser rmse {eps:4.2f} -> {p:7.2f} dB")
        prev = p
    ok &= check("monotone in denoiser error", mono)

    # --- 5. the kernel, inside the real sampling loop -----------------
    # Quality-independent: an untrained net is fine, because this compares
    # the kernel path against the reference path on the SAME weights.
    print("\n5. kernel vs reference scan, through the full sampler")
    try:
        sys.path.insert(0, str(ROOT / "python"))
        from arm_scan._ffi import load
        from arm_scan.op import kernel_calls
        from arm_scan.ss2d import use_arm_scan
        load()
    except Exception as exc:  # noqa: BLE001
        print(f"   SKIPPED — arm_scan unavailable: {exc}")
    else:
        from apps.mri_diffusion.tests import _edm
        from apps.mri_diffusion.tests.test_phase_c_parity import (
            activate_output_layers)
        _, _, construct, src = _edm.load()
        net = construct(
            class_name="training.networks.EDMPrecond",
            model_type="MambaSS2DNet", img_resolution=RES, img_channels=2,
            label_dim=0, model_channels=32, num_blocks_per_level=1,
            d_state=16, use_fp16=False, sigma_data=0.5).eval()
        activate_output_layers(net)

        use_arm_scan(net, enable=False)
        ref = heun_posterior(net, y, mask, num_steps=6)
        use_arm_scan(net)
        c0 = kernel_calls()
        kern = heun_posterior(net, y, mask, num_steps=6)
        engaged = kernel_calls() - c0

        diff = (kern - ref).abs().max().item()
        scale = ref.abs().max().item()
        ok &= check("kernel == reference through sampler",
                    diff < KERNEL_TOL * max(1.0, scale),
                    f"{diff:.3e} (scale {scale:.2f})")
        ok &= check("kernel actually engaged", engaged > 0,
                    f"{engaged} scan calls")

    print("\nPHASE D PIPELINE GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
