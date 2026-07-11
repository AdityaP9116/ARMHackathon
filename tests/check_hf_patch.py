"""End-to-end drop-in check: HF Mamba with arm_scan.patch() engaged.

Proves the Phase-4 integration story on a real model:
  1. EQUIVALENCE — patched logits match the unpatched slow path within the
     kernel tolerance on a full forward pass.
  2. ENGAGEMENT — the native kernel actually ran on every layer (counter,
     not assumption), and the torch custom op composes with the fallback
     logic (cache/decode calls fall back cleanly during generate()).
  3. GENERATION — patched greedy generation runs and (informationally)
     matches unpatched token-for-token.

Needs torch + transformers + the mamba-130m-hf weights (cached by
tests/check_hf_slow_path.py). Run locally; too heavy for CI.

Usage: python tests/check_hf_patch.py
"""

import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import arm_scan  # noqa: E402

MODEL_ID = "state-spaces/mamba-130m-hf"
PROMPT = ("The Arm architecture powers most of the world's phones, and "
          "state-space models like Mamba promise linear-time sequence "
          "modeling on exactly that kind of hardware.")
TOL = 1e-4  # kernel acceptance tolerance vs the reference path


def main():
    from transformers import AutoTokenizer, MambaForCausalLM

    print(f"kernel library: {arm_scan.lib_path()}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = MambaForCausalLM.from_pretrained(MODEL_ID)
    model.eval()
    inputs = tok(PROMPT, return_tensors="pt")
    n_layers = model.config.num_hidden_layers

    # --- reference forward (unpatched slow path) -----------------------
    with torch.no_grad():
        t0 = time.perf_counter()
        ref_logits = model(**inputs).logits
        t_ref = time.perf_counter() - t0

    # --- patched forward ------------------------------------------------
    targets = arm_scan.patch()
    print(f"patched: {targets}")
    with torch.no_grad():
        t0 = time.perf_counter()
        fast_logits = model(**inputs).logits
        t_fast = time.perf_counter() - t0

    stats = arm_scan.stats()
    print(f"engagement: {stats}")
    assert stats["fast_calls"] == n_layers, (
        f"kernel path ran on {stats['fast_calls']}/{n_layers} layers")
    assert stats["kernel_calls"] >= n_layers

    diff = (fast_logits - ref_logits).abs().max().item()
    scale = ref_logits.abs().max().item()
    print(f"logits: max_abs_diff={diff:.3e} (scale {scale:.1f}, tol {TOL})")
    assert diff < TOL * max(1.0, scale), "patched logits diverge"
    print(f"forward wall time: unpatched {t_ref:.3f}s -> patched {t_fast:.3f}s"
          f" (informational; proper benches live in bench/)")

    # --- generation: prefill uses the kernel, decode falls back ----------
    with torch.no_grad():
        fast_ids = model.generate(
            **inputs, max_new_tokens=16, do_sample=False)
    stats2 = arm_scan.stats()
    assert stats2["fast_calls"] > stats["fast_calls"], "prefill not engaged"
    assert stats2["fallback_calls"] > 0, "decode fallback never used"

    arm_scan.unpatch()
    with torch.no_grad():
        ref_ids = model.generate(
            **inputs, max_new_tokens=16, do_sample=False)
    same = torch.equal(fast_ids, ref_ids)
    print(f"greedy generation: tokens identical to unpatched: {same}")
    print(f"  patched : {tok.decode(fast_ids[0][-16:])}")
    if not same:
        print(f"  original: {tok.decode(ref_ids[0][-16:])}")

    print("\nHF PATCH CHECK PASSED")


if __name__ == "__main__":
    main()
