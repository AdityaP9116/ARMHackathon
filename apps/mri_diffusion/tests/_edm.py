"""Which EDM implementation the app tests run against.

Default: `apps.mri_diffusion.edm_min` — our own transcription of the published
EDM equations. No external clone, no credentials, MIT-clean, so every app test
runs on a fresh checkout (and in CI, and on a judge's laptop).

Optional: set `ADM_REF` to a clone of `utcsilab/ambient-diffusion-mri` and the
same tests run against THEIR `EDMPrecond`/`EDMLoss` instead — the exact classes
their pretrained checkpoints were trained under. That path is for work touching
their snapshots; it is not required for anything on the phantom track.

    ADM_REF=/path/to/ambient-diffusion-mri python apps/.../test_x.py

Note their `EDMLoss` hardcodes the MRI width crop `images[:, :, :, 32:352]`
(the 384->320 crop Phase A found), so callers must pre-pad when `uses_ref()`
is true. `pad_for_loss()` does that conditionally.
"""

import os
import sys
from pathlib import Path

import torch

_REF = os.environ.get("ADM_REF")


def uses_ref():
    """True when running against the CSI reference tree."""
    return bool(_REF) and Path(_REF).is_dir()


def load():
    """Return `(EDMPrecond, EDMLoss, construct, source_name)`.

    `construct(class_name=..., model_type=..., **kw)` mirrors dnnlib's
    `construct_class_by_name` so test bodies read the same either way.
    """
    if uses_ref():
        sys.path.insert(0, str(Path(_REF)))
        import dnnlib  # noqa: F401 - import proves the tree is usable
        import training.networks as tn
        from training.loss import EDMLoss

        from apps.mri_diffusion.backbone.mamba_ss2d import MambaSS2DNet
        tn.MambaSS2DNet = MambaSS2DNet  # the --arch=ss2dmamba injection point

        def construct(class_name="training.networks.EDMPrecond", **kw):
            return dnnlib.util.construct_class_by_name(
                class_name=class_name, **kw)

        return tn.EDMPrecond, EDMLoss, construct, f"CSI ref ({_REF})"

    from apps.mri_diffusion.edm_min import EDMLoss, EDMPrecond

    def construct(class_name=None, **kw):
        kw.pop("class_name", None)
        return EDMPrecond(**kw)

    return EDMPrecond, EDMLoss, construct, "edm_min (in-repo)"


def pad_for_loss(imgs):
    """Pre-pad width only when the CSI loss's hardcoded 384->320 crop is in
    play; `edm_min.EDMLoss` has no crop, so this is the identity there."""
    if uses_ref():
        return torch.cat([torch.zeros_like(imgs), imgs], dim=-1)
    return imgs
