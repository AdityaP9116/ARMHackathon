"""Phase-0 integration check against Hugging Face Mamba on CPU.

Three things are established here, all exit criteria from INTEGRATION_PLAN.md:

  1. ROUTING — with mamba_ssm absent, transformers' MambaMixer.forward takes
     `slow_forward` on CPU (asserted with a call counter, not assumed). This
     is the exact function arm_scan.patch() will target in Phase 4.
  2. EQUIVALENCE — the vendored selective_scan_ref, fed the tensors a real
     mixer produces, reproduces the mixer's actual output. This proves our
     ground-truth function computes the same math as the HF slow path, so a
     kernel validated against our goldens is valid for HF models.
  3. REAL-DISTRIBUTION GOLDEN — the captured layer-0 tensors are saved as an
     extra golden case (hf_mixer_layer0.npz) so the Rust kernel is also
     tested on genuine trained-model value distributions, not just synthetic
     ones.

Mapping from HF slow_forward internals to selective_scan_ref arguments:
    u          = hidden_states after conv1d + SiLU          (B, D, L)
    delta      = softplus(dt_proj(time_step))               (B, D, L)
                 -> pass delta_softplus=False (already applied; dt_proj's
                    bias plays the delta_bias role and is inside the linear)
    A          = -exp(A_log.float())                        (D, N)
    B, C       = x_proj splits, transposed (B,L,N)->(B,N,L)
    D_skip     = mixer.D
    z          = gate (HF applies self.act == SiLU, same as the reference)

Usage: python tests/check_hf_slow_path.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from reference import selective_scan_ref

MODEL_ID = "state-spaces/mamba-130m-hf"
PROMPT = ("The Arm architecture powers most of the world's phones, and "
          "state-space models like Mamba promise linear-time sequence "
          "modeling on exactly that kind of hardware.")
GOLDEN_DIR = Path(__file__).parent / "golden"
TOL_MAX_ABS = 1e-4  # kernel acceptance tolerance from INTEGRATION_PLAN.md


def main():
    from transformers import AutoTokenizer, MambaForCausalLM
    from transformers.models.mamba import modeling_mamba

    # --- 1. routing ---------------------------------------------------
    calls = {"n": 0}
    orig_slow = modeling_mamba.MambaMixer.slow_forward

    def counting_slow(self, *a, **k):
        calls["n"] += 1
        return orig_slow(self, *a, **k)

    modeling_mamba.MambaMixer.slow_forward = counting_slow

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = MambaForCausalLM.from_pretrained(MODEL_ID)
    model.eval()

    # transformers 5.x sets these globals lazily in MambaMixer.__init__
    # (lazy_load_kernel), so the check must come after model construction.
    fast_path = all(
        getattr(modeling_mamba, name, None) is not None
        for name in ("selective_state_update", "selective_scan_fn",
                     "causal_conv1d_fn", "causal_conv1d_update",
                     "mamba_inner_fn"))
    print(f"fast path available: {fast_path} (expected False on CPU-only install)")
    assert not fast_path, "mamba_ssm/causal_conv1d unexpectedly installed"
    inputs = tok(PROMPT, return_tensors="pt")
    L_tokens = inputs["input_ids"].shape[1]

    captured = {}

    def hook(module, args, kwargs, output):
        captured["in"] = (args[0] if args else kwargs["hidden_states"]).detach().clone()
        captured["out"] = output.detach().clone()

    mixer = model.backbone.layers[0].mixer
    handle = mixer.register_forward_hook(hook, with_kwargs=True)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    modeling_mamba.MambaMixer.slow_forward = orig_slow

    n_layers = model.config.num_hidden_layers
    print(f"forward pass: {L_tokens} tokens, slow_forward called "
          f"{calls['n']}/{n_layers} layers")
    assert calls["n"] == n_layers, "slow_forward was not used for every layer"

    # --- 2. equivalence: recompute the mixer via the vendored reference
    x = captured["in"]
    with torch.no_grad():
        proj = mixer.in_proj(x).transpose(1, 2)
        hs, gate = proj.chunk(2, dim=1)
        hs = mixer.act(mixer.conv1d(hs)[..., :x.shape[1]])
        ssm_p = mixer.x_proj(hs.transpose(1, 2))
        ts, Bm, Cm = torch.split(
            ssm_p,
            [mixer.time_step_rank, mixer.ssm_state_size, mixer.ssm_state_size],
            dim=-1)
        delta = F.softplus(mixer.dt_proj(ts)).transpose(1, 2)
        A = -torch.exp(mixer.A_log.float())
        Bm = Bm.transpose(1, 2).contiguous()
        Cm = Cm.transpose(1, 2).contiguous()

        ref_scan = selective_scan_ref(
            hs, delta, A, Bm, Cm, D=mixer.D, z=gate, delta_softplus=False)
        ref_out = mixer.out_proj(ref_scan.transpose(1, 2))

    diff = (ref_out - captured["out"]).abs()
    scale = captured["out"].abs().max()
    print(f"mixer-output agreement: max_abs={diff.max():.3e} "
          f"(output scale {scale:.3f}, tolerance {TOL_MAX_ABS})")
    assert diff.max() < TOL_MAX_ABS, "vendored reference != HF slow_forward"

    # --- 3. save real-distribution golden case ------------------------
    f64 = lambda t: t.double()
    out_f64, last_f64 = selective_scan_ref(
        f64(hs), f64(delta), f64(A), f64(Bm), f64(Cm), f64(mixer.D.data),
        f64(gate), delta_softplus=False, return_last_state=True,
        compute_dtype=torch.float64)
    out_f32, last_f32 = selective_scan_ref(
        hs, delta, A, Bm, Cm, mixer.D.data, gate, delta_softplus=False,
        return_last_state=True, compute_dtype=torch.float32)

    meta = {
        "name": "hf_mixer_layer0", "batch": 1, "dim": hs.shape[1],
        "len": hs.shape[2], "state": A.shape[1], "groups": None,
        "delta_softplus": False, "has_z": True, "has_D": True,
        "has_delta_bias": False,
        "seed": None, "source": f"{MODEL_ID} layer-0 mixer, real forward pass",
        "torch_version": torch.__version__,
        "f32_max_abs_err": float((out_f32.double() - out_f64).abs().max()),
    }
    arrays = {
        "u": hs.numpy(), "delta": delta.numpy(), "A": A.numpy(),
        "B": Bm.numpy(), "C": Cm.numpy(), "D_skip": mixer.D.data.numpy(),
        "z": gate.numpy(),
        "out_f64": out_f64.numpy(), "last_state_f64": last_f64.numpy(),
        "out_f32": out_f32.numpy(), "last_state_f32": last_f32.numpy(),
        "meta_json": np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8).copy(),
    }
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GOLDEN_DIR / "hf_mixer_layer0.npz", **arrays)

    manifest_path = GOLDEN_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    manifest = [m for m in manifest if m["name"] != "hf_mixer_layer0"] + [meta]
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"saved golden hf_mixer_layer0.npz "
          f"(D={meta['dim']} L={meta['len']} N={meta['state']}, "
          f"f32 floor={meta['f32_max_abs_err']:.3e})")
    print("\nHF slow-path check PASSED")


if __name__ == "__main__":
    main()
