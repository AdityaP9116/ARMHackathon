"""The Mamba-3 language model around the mixer — plain PyTorch, CPU-runnable.

Transcribed from `mamba_ssm.modules.block.Block` and
`mamba_ssm.models.mixer_seq_simple.{MixerModel, MambaLMHeadModel}`, keeping the
parameter names byte-identical to the published checkpoint so `load.py` needs no
key remapping at all.

THE ONE PLACE THIS DEPARTS FROM UPSTREAM, AND WHY IT IS NOT A DEPARTURE
-----------------------------------------------------------------------
The checkpoint's config sets `fused_add_norm: true`, which routes upstream
through Triton's `layer_norm_fn`. We take the `fused_add_norm=False` branch —
which upstream also ships and which is *mathematically the same computation*:
`layer_norm_fn(..., prenorm=True, residual_in_fp32=...)` is defined to return
exactly `(norm(h + residual), (h + residual).float())`. The fusion is a memory
optimisation, not a change of semantics. We keep `residual_in_fp32=True`
because that one DOES change results.

RESIDUALS IN FP32 — THE DETAIL MOST LIKELY TO BE DROPPED
--------------------------------------------------------
`residual_in_fp32: true` means the residual stream accumulates in fp32 even
when activations are bf16. Over twelve layers that is the difference between a
faithful reproduction and a slow drift. It is one `.float()` and it is easy to
lose; the config is the authority, not the default.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import Mamba3Mixer, RMSNorm


class GatedMLP(nn.Module):
    """SwiGLU MLP. `fc1` emits value and gate concatenated, value FIRST.

    The order is not a guess — upstream is `y, gate = y.chunk(2, dim=-1)` and
    then `y * silu(gate)`. Flipping it produces a model that still runs and
    still emits plausible-looking logits, which is exactly why it is worth
    pinning down here rather than discovering at the logits gate.
    """

    def __init__(self, in_features: int, hidden_features: int,
                 multiple_of: int = 128):
        super().__init__()
        # Upstream rounds the hidden width UP to a multiple of 128, so the
        # config's `d_intermediate` is not always the layer's actual width.
        # SISO-187M never showed this (1536 is already a multiple); MIMO-187M
        # asks for 1264 and the checkpoint carries 1280. Caught by
        # `strict=True` rather than by silently loading a mis-shaped layer.
        hidden_features = ((hidden_features + multiple_of - 1)
                           // multiple_of * multiple_of)
        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=False)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(y * F.silu(gate))


class Block(nn.Module):
    """Pre-norm mixer + pre-norm MLP, both returning the running residual."""

    def __init__(self, d_model: int, d_intermediate: int, layer_idx: int,
                 ssm_cfg: dict, residual_in_fp32: bool = True,
                 norm_eps: float = 1e-5):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.norm = RMSNorm(d_model, eps=norm_eps)
        self.mixer = Mamba3Mixer(d_model=d_model, layer_idx=layer_idx,
                                 **ssm_cfg)
        self.norm2 = RMSNorm(d_model, eps=norm_eps)
        self.mlp = GatedMLP(d_model, d_intermediate)

    def forward(self, hidden_states, residual=None):
        residual = (hidden_states + residual) if residual is not None \
            else hidden_states
        hidden_states = self.norm(residual.to(self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.float()
        hidden_states = self.mixer(hidden_states)

        residual = hidden_states + residual
        hidden_states = self.norm2(residual.to(self.norm2.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.float()
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class MixerModel(nn.Module):
    def __init__(self, d_model, n_layer, d_intermediate, vocab_size, ssm_cfg,
                 residual_in_fp32=True, norm_eps=1e-5):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            Block(d_model, d_intermediate, i, ssm_cfg,
                  residual_in_fp32=residual_in_fp32, norm_eps=norm_eps)
            for i in range(n_layer)
        ])
        self.norm_f = RMSNorm(d_model, eps=norm_eps)

    def forward(self, input_ids):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        residual = (hidden_states + residual) if residual is not None \
            else hidden_states
        return self.norm_f(residual.to(self.norm_f.weight.dtype))


class Mamba3LMHeadModel(nn.Module):
    """`backbone` + tied LM head.

    `tie_embeddings: true` and `model_shape.json` carries no separate
    `lm_head.weight`, so the head IS the embedding matrix. Allocating an
    untied head would load nothing into it and emit noise.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        ssm_cfg = dict(config.get("ssm_cfg", {}))
        if ssm_cfg.get("is_outproj_norm", False):
            raise ValueError(
                "is_outproj_norm=True routes Z through a gated RMSNorm after "
                "the scan instead of into the kernel. The published SISO "
                "checkpoints set it False and carry no mixer `norm.weight`.")
        ssm_cfg.pop("layer", None)          # selects the class; not a mixer arg
        for k in ("dt_min", "dt_max", "dt_init_floor", "is_outproj_norm"):
            ssm_cfg.pop(k, None)            # init-only, or checked above
        # `is_mimo` and `mimo_rank` ARE mixer args and are passed through.
        # Upstream forces rank 1 when is_mimo is False; the mixer mirrors that,
        # so a config carrying a stale mimo_rank alongside is_mimo=False still
        # builds a SISO block rather than a silently mis-shaped one.

        vocab_size = config["vocab_size"]
        pad = config.get("pad_vocab_size_multiple", 1)
        if vocab_size % pad:
            vocab_size += pad - vocab_size % pad

        self.backbone = MixerModel(
            d_model=config["d_model"], n_layer=config["n_layer"],
            d_intermediate=config["d_intermediate"], vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            residual_in_fp32=config.get("residual_in_fp32", True))
        self.tie_embeddings = config.get("tie_embeddings", True)
        if not self.tie_embeddings:
            raise ValueError(
                "Untied embeddings are unsupported: the published checkpoint "
                "has no lm_head.weight, so there would be nothing to load.")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.backbone(input_ids)
        return F.linear(hidden_states, self.backbone.embedding.weight)
