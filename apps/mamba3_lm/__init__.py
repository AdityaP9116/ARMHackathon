"""Mamba-3 SISO language model on Arm CPU, with the scan on our NEON kernel.

The published `state-spaces/mamba3-*` checkpoints cannot run on a CPU-only
machine at all: `mamba_ssm.modules.mamba3` imports Triton / TileLang / CuTe at
module scope, and the package will not install without `nvcc`. This package
rebuilds the model in plain PyTorch and routes only the recurrence through
`arm_scan.mamba3_scan`.

    from mamba3_lm import load_model
    model = load_model()                       # 187M, fp32, CPU
    logits = model(input_ids)                  # (b, l, vocab)

Gated by `tests/check_mamba3_block.py` (mixer vs the real block) and
`tests/check_mamba3_model.py` (logits vs the real model).
"""

from .block import Mamba3Mixer, RMSNorm, heavy_tail_activation
from .load import DEFAULT_MODEL, load_model
from .model import Block, GatedMLP, Mamba3LMHeadModel, MixerModel

__all__ = [
    "Block",
    "DEFAULT_MODEL",
    "GatedMLP",
    "Mamba3LMHeadModel",
    "Mamba3Mixer",
    "MixerModel",
    "RMSNorm",
    "heavy_tail_activation",
    "load_model",
]
