"""Drop-in acceleration for Hugging Face transformers Mamba on CPU.

`patch()` replaces `MambaMixer.slow_forward` with a version whose selective
scan runs on the Arm kernel. Everything else — projections, conv, gating
semantics, weights, checkpoints — is untouched; the replacement transcribes
the upstream code and routes only the recurrence through
`arm_scan::selective_scan`, with softplus fused into the kernel.

Upstream's slow_forward signature and cache API changed across transformers
releases, so the patch adapts at patch time by inspecting the original:

  - "cache_position" API (~4.44 – 5.1):
      slow_forward(self, x, cache_params, cache_position, attention_mask);
      prefill iff cache_position spans the conv kernel;
      conv cache: update_conv_state(layer_idx, conv_state, cache_position);
      ssm cache:  ssm_states[layer_idx].copy_(state)
  - "has_previous_state" API (5.2+ / main):
      slow_forward(self, x, cache_params, attention_mask);
      prefill iff not cache_params.has_previous_state(layer_idx);
      conv cache: update_conv_state(conv_state, layer_idx);
      ssm cache:  update_recurrent_state(state, layer_idx)

Unknown future signatures are left untouched (patch() reports it) rather
than risking a mis-wired forward. The wrapper's fallback always passes the
original arguments through verbatim, so decode steps and unpatched paths
can never see an argument-shape mismatch.

The kernel handles cache-less forwards and the cached PREFILL call of
`generate()` (fresh cache -> zero initial SSM state, the kernel's
contract); it seeds the conv and SSM caches exactly like upstream so
decoding continues seamlessly. Decode steps (single-token, nonzero state)
and mamba.py training mode fall back to the original.

`unpatch()` restores the original. `stats()` reports engagement.
"""

import inspect

import torch
import torch.nn.functional as F

from .op import kernel_calls, selective_scan

_STATE = {"orig": None, "api": None, "arg_names": (),
          "fast_calls": 0, "fallback_calls": 0}


def _detect_api(orig):
    names = list(inspect.signature(orig).parameters)
    if "cache_position" in names:
        return "cache_position"
    if "cache_params" in names:
        return "has_previous_state"
    return None


def _mixer_scan_forward(self, input_states, cache_params, cache_position,
                        attention_mask, api):
    """Transcription of MambaMixer.slow_forward (no-cache and cached-prefill
    branches) with the scan section replaced by the Arm kernel."""
    seq_len = input_states.shape[1]
    dtype = input_states.dtype

    # 1. gated MLP linear projection
    projected_states = self.in_proj(input_states).transpose(1, 2)
    hidden_states, gate = projected_states.chunk(2, dim=1)
    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)

    # 2. convolution sequence transformation (+ seed the conv cache at
    # prefill, exactly as upstream does for each API generation)
    if cache_params is not None:
        conv_state = F.pad(
            hidden_states,
            (self.conv_kernel_size - hidden_states.shape[-1], 0))
        if api == "cache_position":
            cache_params.update_conv_state(
                self.layer_idx, conv_state, cache_position)
        else:
            cache_params.update_conv_state(conv_state, self.layer_idx)
    hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])
    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)

    # 3. selection + discretization + recurrence — the kernel's territory.
    ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
    time_step, B, C = torch.split(
        ssm_parameters,
        [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
        dim=-1,
    )
    # raw dt_proj output: softplus is fused inside the kernel
    discrete_time_step = self.dt_proj(time_step).transpose(1, 2)
    A = -torch.exp(self.A_log.float())

    scan_output, last_state = selective_scan(
        hidden_states,
        discrete_time_step,
        A,
        B.transpose(1, 2),
        C.transpose(1, 2),
        D=self.D,
        z=gate,
        delta_softplus=True,
        return_last_state=True,
    )

    # seed the SSM cache for the decode steps that follow prefill
    if cache_params is not None:
        if api == "cache_position":
            cache_params.ssm_states[self.layer_idx].copy_(last_state)
        else:
            cache_params.update_recurrent_state(last_state, self.layer_idx)

    # 4. final linear projection
    return self.out_proj(scan_output.transpose(1, 2).to(dtype))


def patch():
    """Route HF Mamba's CPU slow path through the Arm kernel.

    Returns the list of patched targets ([] if the installed transformers
    has an unrecognized slow_forward signature). Idempotent.
    """
    if _STATE["orig"] is not None:
        return ["transformers MambaMixer.slow_forward (already patched)"]

    from transformers.models.mamba import modeling_mamba

    orig = modeling_mamba.MambaMixer.slow_forward
    api = _detect_api(orig)
    if api is None:
        import warnings

        warnings.warn(
            "arm_scan.patch(): unrecognized MambaMixer.slow_forward "
            "signature in this transformers version; leaving it unpatched")
        return []
    # positional names after (self, input_states), for arg normalization
    arg_names = tuple(inspect.signature(orig).parameters)[2:]

    def slow_forward(self, input_states, *args, **kwargs):
        bound = dict(zip(arg_names, args))
        bound.update(kwargs)
        cache_params = bound.get("cache_params")
        cache_position = bound.get("cache_position")
        attention_mask = bound.get("attention_mask")

        use_mambapy = getattr(self, "use_mambapy", False)
        if api == "cache_position":
            # prefill iff cache_position spans the conv kernel (upstream's
            # own check); fresh cache -> zero SSM state
            is_prefill = cache_params is None or (
                cache_position is not None
                and cache_position.shape[0] == self.conv_kernel_size
            )
        else:
            is_prefill = cache_params is None or not (
                cache_params.has_previous_state(self.layer_idx))

        if (use_mambapy and self.training) or not is_prefill:
            _STATE["fallback_calls"] += 1
            return orig(self, input_states, *args, **kwargs)
        _STATE["fast_calls"] += 1
        return _mixer_scan_forward(self, input_states, cache_params,
                                   cache_position, attention_mask, api)

    modeling_mamba.MambaMixer.slow_forward = slow_forward
    _STATE["orig"] = orig
    _STATE["api"] = api
    _STATE["arg_names"] = arg_names
    return [f"transformers MambaMixer.slow_forward ({api} API)"]


def unpatch():
    """Restore the original implementation (for A/B benchmarking)."""
    if _STATE["orig"] is None:
        return False
    from transformers.models.mamba import modeling_mamba

    modeling_mamba.MambaMixer.slow_forward = _STATE["orig"]
    _STATE["orig"] = None
    _STATE["api"] = None
    return True


def stats():
    """Engagement counters: how often the fast path vs fallback ran, and
    total native-kernel invocations."""
    return {
        "fast_calls": _STATE["fast_calls"],
        "fallback_calls": _STATE["fallback_calls"],
        "kernel_calls": kernel_calls(),
        "patched": _STATE["orig"] is not None,
        "api": _STATE["api"],
    }
