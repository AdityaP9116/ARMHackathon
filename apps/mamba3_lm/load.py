"""Load a published Mamba-3 SISO checkpoint into the CPU model.

No key remapping: `model.py` names every module to match the checkpoint, so
this is `torch.load` -> `load_state_dict(strict=True)`. Keep it that way — a
rename here is a rename that has to be mirrored in two files and silently
mis-loads if it is not.

DTYPE
-----
The published weights are **bfloat16** on disk. We upcast to fp32 by default,
because our kernel computes in fp32 and because `verify_golden_mamba3.py`'s
ground truth was captured from a model loaded the same way. Loading bf16
directly would compound the checkpoint's own quantisation with the kernel's,
and the logits gate would then be measuring both at once.
"""

import json
from pathlib import Path

import torch

from .model import Mamba3LMHeadModel

DEFAULT_MODEL = "state-spaces/mamba3-siso-187m"


def resolve_snapshot(model_id_or_path: str) -> Path:
    """A local directory if it is one, otherwise the HF cache (download if needed).

    Tries `huggingface_hub` first so a cold machine works, and falls back to
    reading the cache layout directly so an offline box with the model already
    pulled does not need the dependency at all.
    """
    p = Path(model_id_or_path)
    if p.is_dir():
        return p

    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(model_id_or_path))
    except Exception:  # noqa: BLE001 — fall through to the cache layout
        pass

    cache = (Path.home() / ".cache" / "huggingface" / "hub" /
             f"models--{model_id_or_path.replace('/', '--')}" / "snapshots")
    snaps = sorted(cache.glob("*")) if cache.is_dir() else []
    if not snaps:
        raise FileNotFoundError(
            f"{model_id_or_path} is not a local directory and is not in the "
            f"HF cache at {cache}. Install `huggingface_hub` to download it, "
            f"or pass a path to an already-downloaded snapshot.")
    return snaps[-1]


def load_config(snapshot: Path) -> dict:
    return json.loads((snapshot / "config.json").read_text())


def load_model(model_id_or_path: str = DEFAULT_MODEL,
               dtype: torch.dtype = torch.float32,
               snapshot: Path = None) -> Mamba3LMHeadModel:
    snapshot = snapshot or resolve_snapshot(model_id_or_path)
    config = load_config(snapshot)

    weights = snapshot / "pytorch_model.bin"
    if not weights.is_file():
        cands = sorted(snapshot.glob("*.bin")) + \
            sorted(snapshot.glob("*.safetensors"))
        if not cands:
            raise FileNotFoundError(f"no weight file in {snapshot}")
        weights = cands[0]

    if weights.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(weights))
    else:
        # weights_only=True: this is a pickle fetched from the network, so it
        # is parsed as tensors rather than executed.
        sd = torch.load(str(weights), map_location="cpu", weights_only=True)

    # `lm_head.weight` is the SAME tensor as the embedding (verified: tied by
    # identity in the published checkpoint). Our model has no separate head
    # parameter, so drop the alias rather than let strict=True reject it.
    head = sd.pop("lm_head.weight", None)
    if head is not None and not torch.equal(head, sd["backbone.embedding.weight"]):
        raise ValueError(
            "lm_head.weight differs from backbone.embedding.weight, so this "
            "checkpoint is NOT tied — but config says tie_embeddings and the "
            "model has no untied head. Refusing to silently drop a real head.")

    model = Mamba3LMHeadModel(config)
    sd = {k: v.to(dtype) for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.to(dtype).eval()
