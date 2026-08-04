"""Phase-C gate: full (R=1) sampling on CPU with the arm_scan kernel in the
loop, output-parity-verified against the pure-torch reference scan.

  1. BACKBONE PARITY — same weights, same input: F_theta through arm_scan
     vs through the torch reference scan (scale-relative fp32 tolerance;
     the NEON/scalar kernel is disclosed-approximate, not bit-exact).
  2. SAMPLING PARITY + ENGAGEMENT — a short deterministic Heun loop under
     EDMPrecond on both paths: outputs agree, kernel call counter grew.
  3. TIMING (informational) — per-NFE wall time both paths.

Usage: python apps/mri_diffusion/tests/test_phase_c_parity.py
"""

import sys
import time
from pathlib import Path

import torch

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parent.parent
sys.path.insert(0, str(ROOT))

from apps.mri_diffusion.tests import _edm  # noqa: E402
from arm_scan.ss2d import use_arm_scan  # noqa: E402
from arm_scan.op import kernel_calls  # noqa: E402

EDMPrecond, EDMLoss, construct, EDM_SOURCE = _edm.load()

torch.manual_seed(0)
RES, CH = 32, 2
TOL = 1e-4  # scale-relative, same bound family as the kernel gates


def activate_output_layers(net):
    """Break the identity-at-init residual before comparing anything.

    `SS2DBlock.out_proj` and `MambaSS2DNet.head` are deliberately zero-init, so
    a FRESHLY CONSTRUCTED net returns `c_skip * x` no matter what the scan
    computed — every scan output is multiplied by zero on its way out. A parity
    check on an untrained net is therefore vacuously green (it reported
    max_abs=0.000e+00 for exactly this reason). Give those layers real weights
    so the scan actually reaches the output and the comparison has teeth.
    """
    touched = 0
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, torch.nn.Conv2d) and not m.weight.any():
                m.weight.normal_(0.0, 0.1)
                if m.bias is not None:
                    m.bias.normal_(0.0, 0.05)
                touched += 1
    assert touched, "expected zero-init output layers to activate"
    return touched


def heun_sample(net, latents, num_steps=4):
    t = torch.linspace(80 ** (1 / 7), 0.002 ** (1 / 7), num_steps) ** 7
    t = torch.cat([t, torch.zeros(1)])
    x, times = latents * t[0], []
    with torch.no_grad():
        for i in range(num_steps):
            t0 = time.perf_counter()
            d = net(x, t[i].repeat(x.shape[0]), None)
            times.append(time.perf_counter() - t0)
            dx = (x - d) / t[i]
            x1 = x + (t[i + 1] - t[i]) * dx
            if i < num_steps - 1:
                d2 = net(x1, t[i + 1].repeat(x.shape[0]), None)
                times.append(time.perf_counter() - t0)
                x1 = x + (t[i + 1] - t[i]) * 0.5 * (dx + (x1 - d2) / t[i + 1])
            x = x1
    return x, times


def main():
    print(f"0. EDM source: {EDM_SOURCE}")
    net = construct(
        class_name="training.networks.EDMPrecond", model_type="MambaSS2DNet",
        img_resolution=RES, img_channels=CH, label_dim=0,
        model_channels=32, num_blocks_per_level=1, d_state=16,
        use_fp16=False, sigma_data=0.5).eval()
    n_act = activate_output_layers(net)
    print(f"   activated {n_act} zero-init output layers so the scan reaches "
          f"the network output")

    x = torch.randn(2, CH, RES, RES)
    sig = torch.tensor([1.3, 0.4])

    with torch.no_grad():
        ref = net(x, sig, None)
    n_blocks = use_arm_scan(net)
    calls0 = kernel_calls()
    with torch.no_grad():
        kern = net(x, sig, None)
    scale = ref.abs().max().item()
    diff = (kern - ref).abs().max().item()
    print(f"1. backbone parity: {n_blocks} blocks on arm_scan, "
          f"max_abs={diff:.3e} (scale {scale:.2f}, tol {TOL}*scale)")
    assert diff < TOL * max(1.0, scale), "backbone parity FAILED"
    engaged = kernel_calls() - calls0
    assert engaged > 0, "kernel never engaged"
    print(f"   kernel engaged: {engaged} scan calls in one forward")

    lat = torch.randn(1, CH, RES, RES)
    use_arm_scan(net, enable=False)
    s_ref, t_ref = heun_sample(net, lat)
    use_arm_scan(net)
    s_kern, t_kern = heun_sample(net, lat)
    sdiff = (s_kern - s_ref).abs().max().item()
    sscale = s_ref.abs().max().item()
    print(f"2. sampling parity (R=1, 4-step Heun, 7 NFE): "
          f"max_abs={sdiff:.3e} (scale {sscale:.2f})")
    assert sdiff < 10 * TOL * max(1.0, sscale), "sampling parity FAILED"
    assert torch.isfinite(s_kern).all()

    import numpy as np
    print(f"3. per-NFE: torch-ref {np.median(t_ref)*1e3:.0f} ms vs "
          f"arm_scan {np.median(t_kern)*1e3:.0f} ms (x86 scalar backend; "
          f"informational)")
    print("\nPHASE C GATE: PASS — full sampling on CPU through arm_scan, "
          "parity verified")


if __name__ == "__main__":
    main()
