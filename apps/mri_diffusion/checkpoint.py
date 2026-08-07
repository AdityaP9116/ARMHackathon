"""Prior checkpoints that carry their own architecture.

A bare `state_dict` does not say what shape of network it came from, so loading
one with the wrong `model_channels` fails with a shape mismatch several layers
deep — a confusing error for something that is really "you passed the wrong
flags". These checkpoints embed the construction config, so a loader can
rebuild the exact network without being told.

Format (a dict):
    model        state_dict of the EMA weights — what you sample from
    model_raw    state_dict of the live weights (for resuming training)
    config       everything needed to reconstruct the network
    step, meta   provenance: steps trained, data source, loss, timestamp

Bare `state_dict` files still load, so anything saved by earlier runs (e.g.
`demo.py --save-prior`) keeps working — you just have to pass the architecture
flags yourself.
"""

import time

import torch

CONFIG_KEYS = ("img_resolution", "img_channels", "model_channels",
               "num_blocks_per_level", "d_state", "sigma_data", "label_dim")


def save_prior(path, net, config, step, ema_state=None, meta=None):
    """Write a self-describing checkpoint.

    `ema_state` is what gets stored as the primary weights: diffusion models
    are sampled from an EMA of the trajectory, not the live weights.
    """
    torch.save({
        "model": ema_state if ema_state is not None else net.state_dict(),
        "model_raw": net.state_dict(),
        "config": {k: config[k] for k in CONFIG_KEYS if k in config},
        "step": step,
        "meta": {"saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime()), **(meta or {})},
    }, path)


def load_prior(path, map_location="cpu"):
    """-> (state_dict, config_or_None, meta). Accepts bare state_dicts too."""
    blob = torch.load(path, map_location=map_location, weights_only=False)
    if not (isinstance(blob, dict) and "model" in blob
            and isinstance(blob.get("config", None), dict)):
        # A bare state_dict from an earlier run: usable, just not self-describing.
        return blob, None, {}
    meta = dict(blob.get("meta", {}))
    meta["step"] = blob.get("step")
    return blob["model"], blob["config"], meta


def build_prior(construct, path, map_location="cpu", **overrides):
    """Rebuild the exact network a checkpoint came from, and load it.

    `overrides` win over the embedded config, and are REQUIRED for a bare
    state_dict — with a clear message rather than a shape mismatch.
    """
    state, config, meta = load_prior(path, map_location)
    if config is None:
        missing = [k for k in ("img_resolution", "model_channels",
                               "num_blocks_per_level", "d_state")
                   if k not in overrides]
        if missing:
            raise SystemExit(
                f"{path} is a bare state_dict with no embedded architecture.\n"
                f"Pass these explicitly: {', '.join(missing)}\n"
                f"(Checkpoints written by tools/train_prior.py carry their own "
                f"config.)")
        config = {}
    cfg = {"img_channels": 2, "label_dim": 0, "sigma_data": 0.5,
           **config, **overrides}
    net = construct(class_name="training.networks.EDMPrecond",
                    model_type="MambaSS2DNet", use_fp16=False,
                    **{k: v for k, v in cfg.items() if k in CONFIG_KEYS})
    net.load_state_dict(state)
    net.eval()
    return net, cfg, meta


class EMA:
    """Exponential moving average of the weights, EDM-style.

    Diffusion models are sampled from an EMA of the training trajectory, not
    the live weights — it is worth several dB and is standard practice, not a
    refinement. The warmup ramp (`decay` rises with step count) stops the
    average being dominated by the random initialisation early on.
    """

    def __init__(self, net, decay=0.9995, warmup=True):
        self.decay, self.warmup = decay, warmup
        self.shadow = {k: v.detach().clone().float()
                       for k, v in net.state_dict().items()
                       if v.dtype.is_floating_point}
        self.buffers = {k: v.detach().clone()
                        for k, v in net.state_dict().items()
                        if not v.dtype.is_floating_point}

    def update(self, net, step):
        d = self.decay
        if self.warmup:
            d = min(d, (1 + step) / (10 + step))
        with torch.no_grad():
            for k, v in net.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
                else:
                    self.buffers[k] = v.detach().clone()

    def state_dict(self, reference):
        """EMA weights cast back to the reference network's dtypes."""
        ref = reference.state_dict()
        return {k: (self.shadow[k].to(ref[k].dtype) if k in self.shadow
                    else self.buffers[k]) for k in ref}
