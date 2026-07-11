"""Drop-in acceleration for Hugging Face transformers Mamba on CPU.

`patch()` replaces `MambaMixer.slow_forward` with a version whose selective
scan runs on the Arm kernel. Everything else — projections, conv, gating
semantics, weights, checkpoints — is untouched; the replacement transcribes
the upstream no-cache path (transformers 4.x/5.x `slow_forward`) and routes
only the recurrence through `arm_scan::selective_scan`, with softplus fused
into the kernel.

The kernel handles both cache-less forwards and the cached PREFILL call of
`generate()` (fresh cache -> zero initial SSM state, which is the kernel's
contract); it writes the conv state and final SSM state into the cache
exactly like upstream so decoding continues seamlessly.

Falls back to the original implementation for:
  - decode steps (single-token calls carrying a nonzero SSM state — cheap
    matmuls where an O(len) scan kernel has nothing to win),
  - mamba.py training mode (`use_mambapy and self.training`).

`unpatch()` restores the original. `stats()` reports engagement.
"""

import torch
import torch.nn.functional as F

from .op import kernel_calls, selective_scan

_STATE = {"orig": None, "fast_calls": 0, "fallback_calls": 0}


def _mixer_scan_forward(self, input_states, cache_params=None,
                        cache_position=None, attention_mask=None):
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
    # prefill, exactly as upstream does)
    if cache_params is not None:
        conv_state = F.pad(
            hidden_states,
            (self.conv_kernel_size - hidden_states.shape[-1], 0))
        cache_params.update_conv_state(
            self.layer_idx, conv_state, cache_position)
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
        cache_params.ssm_states[self.layer_idx].copy_(last_state)

    # 4. final linear projection
    return self.out_proj(scan_output.transpose(1, 2).to(dtype))


def patch():
    """Route HF Mamba's CPU slow path through the Arm kernel.

    Returns the list of patched targets. Idempotent.
    """
    if _STATE["orig"] is not None:
        return ["transformers MambaMixer.slow_forward (already patched)"]

    from transformers.models.mamba import modeling_mamba

    orig = modeling_mamba.MambaMixer.slow_forward

    def slow_forward(self, input_states, cache_params=None,
                     cache_position=None, attention_mask=None):
        use_mambapy = getattr(self, "use_mambapy", False)
        # A cached call is a PREFILL (fresh cache, zero SSM state — the
        # kernel's contract) iff cache_position spans the conv kernel;
        # upstream uses this exact check. Decode steps fall back.
        is_prefill = cache_params is None or (
            cache_position is not None
            and cache_position.shape[0] == self.conv_kernel_size
        )
        if (use_mambapy and self.training) or not is_prefill:
            _STATE["fallback_calls"] += 1
            return orig(self, input_states, cache_params, cache_position,
                        attention_mask)
        _STATE["fast_calls"] += 1
        return _mixer_scan_forward(self, input_states, cache_params,
                                   cache_position, attention_mask)

    modeling_mamba.MambaMixer.slow_forward = slow_forward
    _STATE["orig"] = orig
    return ["transformers MambaMixer.slow_forward"]


def unpatch():
    """Restore the original implementation (for A/B benchmarking)."""
    if _STATE["orig"] is None:
        return False
    from transformers.models.mamba import modeling_mamba

    modeling_mamba.MambaMixer.slow_forward = _STATE["orig"]
    _STATE["orig"] = None
    return True


def stats():
    """Engagement counters: how often the fast path vs fallback ran, and
    total native-kernel invocations."""
    return {
        "fast_calls": _STATE["fast_calls"],
        "fallback_calls": _STATE["fallback_calls"],
        "kernel_calls": kernel_calls(),
        "patched": _STATE["orig"] is not None,
    }
