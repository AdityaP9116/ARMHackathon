"""The Mamba-3 SISO mixer, in plain PyTorch, with the scan on our Arm kernel.

WHY THIS FILE EXISTS
--------------------
`mamba_ssm.modules.mamba3` cannot be imported on a CPU-only machine. It pulls in
Triton / TileLang / CuTe at module scope, and the package will not even install
without `nvcc`. So running the published checkpoint on Arm means rebuilding the
block around the scan.

THE DISCIPLINE, WHICH IS THE SAME ONE `python/arm_scan/patch.py` FOLLOWS
-----------------------------------------------------------------------
**Transcribe the non-scan code verbatim; replace only the recurrence.** Every
line below is a direct translation of `Mamba3.forward`'s SISO branch, in the
same order, with the same casts. Where upstream uses `einops.rearrange` this
uses the equivalent `reshape`/`permute` (einops is not in the CPU tier's
requirements). Nothing here is a reinterpretation of the paper — the paper was
already settled in `tests/reference/mamba3_ref.py`, and this is plumbing.

WHAT UPSTREAM DOES *NOT* HAVE, AND IT IS WORTH SAYING OUT LOUD
--------------------------------------------------------------
There is **no `conv1d`** in Mamba-3. Mamba-1 and Mamba-2 both open the mixer
with a short depthwise convolution; Mamba-3 dropped it. Do not add one back by
analogy — the checkpoint has no weights for it (the mixer has exactly 8
parameters, listed below), so it would be inventing state.

THE CHECKPOINT'S MIXER PARAMETERS, ALL OF THEM
----------------------------------------------
    in_proj.weight   (3432, 768)   dt_bias  (24,)     D       (24,)
    out_proj.weight  (768, 1536)   B_bias   (24, 1, 128)
    B_norm.weight    (128,)        C_bias   (24, 1, 128)
    C_norm.weight    (128,)

Note `A` is absent: unlike Mamba-1's learned `A_log`, Mamba-3's decay is
**data-dependent** — it comes out of `in_proj` and through `heavy_tail`.

TWO NAMING TRAPS, BOTH LIVE
---------------------------
1. The checkpoint says `B`/`C`; the kernel says `K`/`Q`. The mapping is
   **Q <- C** and **K <- B**, not the alphabetical pairing.
2. `B_bias`/`C_bias` are `(nheads, mimo_rank, d_state)`; the SISO kernel wants
   `(nheads, d_state)`. Upstream squeezes dim 1. Getting this wrong broadcasts
   silently rather than raising.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import arm_scan


def heavy_tail_activation(x: torch.Tensor) -> torch.Tensor:
    """f(x) = 1 + x for x >= 0, else 1 / (1 - x).

    Transcribed branchlessly exactly as upstream writes it. Note this is NOT a
    softplus and not any other standard activation — substituting one changes
    the decay of every head. The two clamps also keep both sub-expressions
    finite, so no branch can evaluate `1/0`.
    """
    neg = x.clamp_max(0)
    pos = x.clamp_min(0)
    return pos + torch.reciprocal(1 - neg)


class RMSNorm(nn.Module):
    """Plain RMSNorm over the last axis, upcasting to fp32.

    Upstream uses `RMSNormGated` but calls it with `z=None`, which reduces to
    exactly this. `eps` is inside the sqrt and the accumulation is fp32
    regardless of input dtype — both matter for matching.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rstd = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x * rstd * self.weight.float()).to(dtype)


class Mamba3Mixer(nn.Module):
    """One Mamba-3 SISO mixer. Parameter names match the checkpoint exactly."""

    def __init__(self, d_model=768, d_state=128, expand=2, headdim=64,
                 ngroups=1, rope_fraction=0.5, A_floor=1e-4, chunk_size=64,
                 layer_idx=None, is_mimo=False, mimo_rank=1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.headdim = headdim
        self.A_floor = A_floor
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.is_mimo = is_mimo

        self.d_inner = int(expand * d_model)
        assert self.d_inner % headdim == 0
        self.nheads = self.d_inner // headdim
        self.num_bc_heads = ngroups
        # Upstream forces rank 1 when is_mimo is False, and the rank axis then
        # collapses so SISO and MIMO share every reshape below.
        self.mimo_rank = mimo_rank if is_mimo else 1

        # RoPE: at rope_fraction=0.5 only the first half of the head dimension
        # rotates, so there are d_state/4 angles, not d_state/2.
        assert rope_fraction in (0.5, 1.0)
        split_tensor_size = int(d_state * rope_fraction)
        if split_tensor_size % 2:
            split_tensor_size -= 1
        self.num_rope_angles = split_tensor_size // 2
        assert self.num_rope_angles > 0

        # Order is fixed by the checkpoint: [z, x, B, C, dd_dt, dd_A, trap, angles]
        bc = d_state * self.num_bc_heads * self.mimo_rank
        self.split_sizes = [self.d_inner, self.d_inner, bc, bc,
                            self.nheads, self.nheads, self.nheads,
                            self.num_rope_angles]
        d_in_proj = sum(self.split_sizes)

        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.B_bias = nn.Parameter(
            torch.ones(self.nheads, self.mimo_rank, d_state))
        self.C_bias = nn.Parameter(
            torch.ones(self.nheads, self.mimo_rank, d_state))
        self.B_norm = RMSNorm(d_state, eps=1e-5)
        self.C_norm = RMSNorm(d_state, eps=1e-5)

        if is_mimo:
            # Psi / Zeta / Phi — the input, gate and output projections. All
            # (nheads, rank, headdim) and applied ELEMENTWISE over the head
            # dimension; they are per-rank reweightings, not matmuls. Names
            # match the checkpoint (`mimo_x`/`mimo_z`/`mimo_o`), which is what
            # keeps `load.py` free of any remapping.
            shape = (self.nheads, self.mimo_rank, headdim)
            self.mimo_x = nn.Parameter(torch.ones(*shape) / self.mimo_rank)
            self.mimo_z = nn.Parameter(torch.ones(*shape))
            self.mimo_o = nn.Parameter(torch.ones(*shape) / self.mimo_rank)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (batch, seqlen, d_model) -> same shape."""
        b, seqlen, _ = u.shape
        h, p, n = self.nheads, self.headdim, self.d_state

        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            self.in_proj(u), self.split_sizes, dim=-1)

        z = z.reshape(b, seqlen, h, p)
        x = x.reshape(b, seqlen, h, p)
        # (b, l, r*g*n) -> (b, l, r, g, n); SISO has r = g = 1.
        B = B.reshape(b, seqlen, self.mimo_rank, self.num_bc_heads, n)
        C = C.reshape(b, seqlen, self.mimo_rank, self.num_bc_heads, n)
        trap = trap.permute(0, 2, 1)  # (b, l, h) -> (b, h, l), PRE-sigmoid

        # Data-dependent decay. The fp32 cast is upstream's and is load-bearing:
        # the reciprocal branch of heavy_tail is where a bf16 `dd_A` would lose
        # most of its resolution.
        _A = -heavy_tail_activation(dd_A.float())
        _A = torch.clamp(_A, max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)
        ADT = _A * DT
        DT = DT.permute(0, 2, 1)      # (b, l, h) -> (b, h, l)
        ADT = ADT.permute(0, 2, 1)

        # The angles are projected ONCE and shared across every head — this is
        # a broadcast, not a per-head projection. `expand` matches upstream and
        # keeps it a view; the kernel's pre-pass reads it read-only.
        angles = angles.unsqueeze(-2).expand(-1, -1, h, -1).float()

        B = self.B_norm(B)
        C = self.C_norm(C)

        # The recurrence, on our kernel. Q<-C and K<-B in BOTH variants; see the
        # module docstring for why that mapping is not the alphabetical one.
        if self.is_mimo:
            # MIMO keeps the rank axis (upstream squeezes only for SISO) and
            # passes the biases unsqueezed at (nheads, rank, d_state).
            y = arm_scan.mamba3_mimo_scan(
                q=C, k=B, v=x, adt=ADT, dt=DT, trap=trap,
                q_bias=self.C_bias, k_bias=self.B_bias,
                psi=self.mimo_x, zeta=self.mimo_z, phi=self.mimo_o,
                angles=angles, D=self.D, z=z,
            )
        else:
            y = arm_scan.mamba3_scan(
                q=C.squeeze(2), k=B.squeeze(2), v=x,
                adt=ADT, dt=DT, trap=trap,
                q_bias=self.C_bias.squeeze(1), k_bias=self.B_bias.squeeze(1),
                angles=angles, D=self.D, z=z,
            )

        y = y.reshape(b, seqlen, h * p)
        return self.out_proj(y.to(x.dtype))
