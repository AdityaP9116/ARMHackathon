"""MambaSS2DNet — the SS2D-Mamba denoiser backbone F_theta for EDM.

Satisfies the stock EDM backbone contract confirmed in Phase A
(ambient-diffusion-mri training/networks.py, EDMPrecond.forward):

    F_x = model((c_in * x), c_noise.flatten(), class_labels=..., ...)
    forward(x, noise_labels, class_labels=None, augment_labels=None)

Locked recipe (MRI_DIFFUSION_IMPLEMENTATION_PLAN.md §3.2):
  - plain 4-direction SS2D cross-scan (rows fwd/back, cols fwd/back), VMamba
    style, summed over directions;
  - U-Net-shaped (2 resolution levels with skip), matching CSI's multiscale
    bias without importing their code;
  - EDM PositionalEmbedding + 2-layer MLP for sigma conditioning, injected
    per block adaLN-style (scale/shift after GroupNorm);
  - img_channels=2 (complex MRI as 2 real channels); real-valued throughout.

The scan itself goes through `scan_fn` (default: the pure-torch reference in
torch_scan.py). Phase C swaps in arm_scan behind the same signature.

EDM persistence: decorated when torch_utils.persistence is importable (i.e.
running inside the EDM/CSI repo context, which training and pickling always
do); falls back to a no-op decorator so the file also imports standalone.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .torch_scan import selective_scan_torch

try:  # inside EDM/CSI repo context
    from torch_utils import persistence
    _persist = persistence.persistent_class
except ImportError:  # standalone import (tests, docs)
    def _persist(cls):
        return cls


class SigmaEmbedding(nn.Module):
    """EDM PositionalEmbedding + 2-layer MLP (networks.py map_layer0/1)."""

    def __init__(self, num_channels, emb_dim):
        super().__init__()
        self.num_channels = num_channels
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim), nn.SiLU(),
        )

    def forward(self, noise_labels):
        freqs = torch.arange(self.num_channels // 2, dtype=torch.float32,
                             device=noise_labels.device)
        freqs = (1 / 10000) ** (freqs / (self.num_channels // 2))
        x = noise_labels.float().ger(freqs)
        emb = torch.cat([x.cos(), x.sin()], dim=1)
        return self.mlp(emb)


class SS2DBlock(nn.Module):
    """One SS2D-Mamba residual block: GroupNorm + adaLN sigma conditioning,
    depthwise local conv, 4-direction cross-scan, SiLU gate, projection."""

    def __init__(self, dim, emb_dim, d_state=16, dt_rank=None, expand=1.5):
        super().__init__()
        inner = int(dim * expand)
        self.inner, self.d_state = inner, d_state
        self.dt_rank = dt_rank or max(8, dim // 16)

        self.norm = nn.GroupNorm(min(32, dim), dim)
        self.affine = nn.Linear(emb_dim, dim * 2)  # adaLN scale/shift
        self.in_proj = nn.Conv2d(dim, inner * 2, 1)  # x branch + gate z
        self.local = nn.Conv2d(inner, inner, 3, padding=1, groups=inner)
        # shared SSM parameterization applied per direction (DiM-style)
        self.x_proj = nn.Linear(inner, self.dt_rank + 2 * d_state)
        self.dt_proj = nn.Linear(self.dt_rank, inner)
        a = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(a)[None].repeat(inner, 1))
        self.D = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Conv2d(inner, dim, 1)
        nn.init.zeros_(self.out_proj.weight)  # identity-at-init residual
        nn.init.zeros_(self.out_proj.bias)
        self.scan_fn = selective_scan_torch  # Phase-C swap point

    def _scan_dir(self, seq):
        """seq: (b, inner, L) -> scanned (b, inner, L)."""
        b, c, length = seq.shape
        proj = self.x_proj(seq.transpose(1, 2))  # (b, L, rank+2n)
        dt, Bm, Cm = torch.split(
            proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = self.dt_proj(dt).transpose(1, 2)  # (b, inner, L) raw
        A = -torch.exp(self.A_log)
        return self.scan_fn(seq, delta, A, Bm.transpose(1, 2),
                            Cm.transpose(1, 2), D=self.D,
                            delta_softplus=True)

    def forward(self, x, emb):
        b, d, h, w = x.shape
        scale, shift = self.affine(emb).chunk(2, dim=1)
        y = self.norm(x) * (1 + scale[:, :, None, None]) \
            + shift[:, :, None, None]
        xz = self.in_proj(y)
        s, z = xz.chunk(2, dim=1)
        s = F.silu(self.local(s))

        # P0-1 (SS2D_REPOSITIONING_PLAN §5): all 4 directions in ONE scan
        # call — 4x the rayon rows, 1/4 the FFI crossings. x_proj/dt_proj
        # are shared across directions, so batching them is exact.
        rows = s.flatten(2)                                   # row-major
        cols = s.transpose(2, 3).flatten(2)                   # col-major
        seqs = torch.cat(
            [rows, rows.flip(-1), cols, cols.flip(-1)], dim=0)
        o1, o2, o3, o4 = self._scan_dir(seqs).chunk(4, dim=0)
        out = o1 + o2.flip(-1)
        oc = o3 + o4.flip(-1)
        merged = (out.view(b, self.inner, h, w)
                  + oc.view(b, self.inner, w, h).transpose(2, 3))

        gated = merged * F.silu(z)
        return x + np.sqrt(0.5) * self.out_proj(gated)


@_persist
class MambaSS2DNet(nn.Module):
    """EDM-contract SS2D-Mamba backbone (see module docstring)."""

    def __init__(self, img_resolution, in_channels, out_channels,
                 label_dim=0, augment_dim=0, model_channels=64,
                 num_blocks_per_level=2, d_state=16, emb_channels=None,
                 **unused_kwargs):
        super().__init__()
        self.img_resolution = img_resolution
        self.label_dim = label_dim
        emb_dim = emb_channels or model_channels * 4
        self.sigma_emb = SigmaEmbedding(model_channels, emb_dim)

        c1, c2 = model_channels, model_channels * 2
        self.stem = nn.Conv2d(in_channels, c1, 3, padding=1)
        self.enc = nn.ModuleList(
            [SS2DBlock(c1, emb_dim, d_state)
             for _ in range(num_blocks_per_level)])
        self.down = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.mid = nn.ModuleList(
            [SS2DBlock(c2, emb_dim, d_state)
             for _ in range(num_blocks_per_level)])
        self.up = nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1)
        self.dec = nn.ModuleList(
            [SS2DBlock(c1, emb_dim, d_state)
             for _ in range(num_blocks_per_level)])
        self.skip_join = nn.Conv2d(c1 * 2, c1, 1)
        self.head_norm = nn.GroupNorm(min(32, c1), c1)
        self.head = nn.Conv2d(c1, out_channels, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, noise_labels, class_labels=None,
                augment_labels=None):
        emb = self.sigma_emb(noise_labels)
        h1 = self.stem(x)
        for blk in self.enc:
            h1 = blk(h1, emb)
        h2 = self.down(h1)
        for blk in self.mid:
            h2 = blk(h2, emb)
        h = self.up(h2)
        h = self.skip_join(torch.cat([h, h1], dim=1))
        for blk in self.dec:
            h = blk(h, emb)
        return self.head(F.silu(self.head_norm(h)))
