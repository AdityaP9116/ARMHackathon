"""SS2D Stage 1 (TOPOLOGY_IMPLEMENTATION_PLAN §3.1): route 2D cross-scan
directions through the 1D kernel, unfused.

`scan_fn_arm` is drop-in for the MRI backbone's `scan_fn` seam
(apps/mri_diffusion/backbone/torch_scan.selective_scan_torch signature).
Inference-only (the kernel registers no autograd); use the torch reference
for training. `use_arm_scan(module)` swaps every SS2DBlock in a model.
"""


def scan_fn_arm(u, delta, A, B, C, D=None, delta_bias=None,
                delta_softplus=True):
    from .op import selective_scan
    return selective_scan(u, delta, A, B, C, D=D, delta_bias=delta_bias,
                          delta_softplus=delta_softplus)


def use_arm_scan(module, enable=True):
    """Swap the scan implementation on every SS2D block in `module`.
    Returns the number of blocks switched."""
    n = 0
    for m in module.modules():
        if hasattr(m, "scan_fn"):
            if enable:
                m.scan_fn = scan_fn_arm
            else:
                from apps.mri_diffusion.backbone.torch_scan import \
                    selective_scan_torch
                m.scan_fn = selective_scan_torch
            n += 1
    return n
