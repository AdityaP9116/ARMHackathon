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

The scan itself goes through `scan_pair_fn` (default: the pure-torch reference
in torch_scan.py), driven by `arm_scan.ss2d.ss2d_scan`. Phase C swaps in
arm_scan behind the same signature. The four cross-scan directions run as two
traversal-order PAIRS, so the kernel evaluates the direction-independent
Pass A (discretize + exp, ~85% of its time) twice per block instead of four
times; `_cross_scan_legacy` retains the older four-forward-scans formulation
as the correctness oracle.

EDM persistence: decorated when torch_utils.persistence is importable (i.e.
running inside the EDM/CSI repo context, which training and pickling always
do); falls back to a no-op decorator so the file also imports standalone.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from arm_scan.ss2d import ss2d_scan  # repo `python/` is on sys.path via
from .torch_scan import (selective_scan_pair_torch,  # apps.mri_diffusion
                         selective_scan_torch)

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
    depthwise local conv, 4-direction cross-scan, SiLU gate, projection.

    The cross-scan runs as TWO traversal-order pairs (rows fwd/bwd, cols
    fwd/bwd) through `arm_scan.ss2d.ss2d_scan`, so the kernel computes the
    direction-independent Pass A once per pair instead of once per direction.
    `_cross_scan_legacy` keeps the older four-separate-directions formulation
    as the parity oracle — it is what `forward` is gated against, and the two
    are algebraically identical (see `tests/test_ss2d_pair_parity.py`).
    """

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
        # Phase-C swap points. `scan_pair_fn` is what forward() uses;
        # `scan_fn` is retained for the legacy oracle. use_arm_scan() swaps
        # both together so the two paths always share a backend.
        self.scan_pair_fn = selective_scan_pair_torch
        self.scan_fn = selective_scan_torch
        self.legacy_cross_scan = False  # True -> run the oracle formulation

    def _pre(self, x, emb):
        """Norm + sigma conditioning + input projection + local mixing."""
        scale, shift = self.affine(emb).chunk(2, dim=1)
        y = self.norm(x) * (1 + scale[:, :, None, None]) \
            + shift[:, :, None, None]
        s, z = self.in_proj(y).chunk(2, dim=1)
        return F.silu(self.local(s)), z

    def _post(self, x, merged, z):
        """SiLU gate + output projection + scaled residual."""
        return x + np.sqrt(0.5) * self.out_proj(merged * F.silu(z))

    def _project_grid(self, s):
        """`s: (b, inner, h, w)` -> `(delta, B, C, A)`, all grid-shaped.

        `x_proj`/`dt_proj` are token-wise, so projecting the grid once and
        reordering afterwards is identical to reordering first and projecting
        per direction — and costs one pass instead of four.
        """
        b, _, h, w = s.shape
        proj = self.x_proj(s.flatten(2).transpose(1, 2))  # (b, hw, rank+2n)
        dt, Bm, Cm = torch.split(
            proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = self.dt_proj(dt).transpose(1, 2).reshape(b, self.inner, h, w)
        Bg = Bm.transpose(1, 2).reshape(b, self.d_state, h, w)
        Cg = Cm.transpose(1, 2).reshape(b, self.d_state, h, w)
        return delta, Bg, Cg, -torch.exp(self.A_log)

    def _cross_scan(self, s):
        """4-direction cross-scan as two traversal-order PAIRS.

        Each pair is one kernel call that shares Pass A (discretize + exp,
        ~85% of kernel time) between its two directions.
        """
        delta, Bg, Cg, A = self._project_grid(s)
        return ss2d_scan(s, delta, A, Bg, Cg, D=self.D, delta_softplus=True,
                         merge="sum", scan_pair=self.scan_pair_fn)

    def _scan_dir(self, seq):
        """seq: (b, inner, L) -> scanned (b, inner, L). Legacy oracle path."""
        proj = self.x_proj(seq.transpose(1, 2))  # (b, L, rank+2n)
        dt, Bm, Cm = torch.split(
            proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = self.dt_proj(dt).transpose(1, 2)  # (b, inner, L) raw
        A = -torch.exp(self.A_log)
        return self.scan_fn(seq, delta, A, Bm.transpose(1, 2),
                            Cm.transpose(1, 2), D=self.D,
                            delta_softplus=True)

    def _cross_scan_legacy(self, s):
        """The P0-1 formulation: four directions stacked into ONE forward
        call. Kept as the correctness oracle for `_cross_scan` — NOT the
        shipping path, because it recomputes Pass A four times and flips a
        full-size tensor four times per block.
        """
        b, _, h, w = s.shape
        rows = s.flatten(2)                                   # row-major
        cols = s.transpose(2, 3).flatten(2)                   # col-major
        seqs = torch.cat(
            [rows, rows.flip(-1), cols, cols.flip(-1)], dim=0)
        o1, o2, o3, o4 = self._scan_dir(seqs).chunk(4, dim=0)
        out = o1 + o2.flip(-1)
        oc = o3 + o4.flip(-1)
        return (out.view(b, self.inner, h, w)
                + oc.view(b, self.inner, w, h).transpose(2, 3))

    def forward(self, x, emb):
        s, z = self._pre(x, emb)
        merged = (self._cross_scan_legacy(s) if self.legacy_cross_scan
                  else self._cross_scan(s))
        return self._post(x, merged, z)


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
