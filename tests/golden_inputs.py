"""The golden cases' input draws — pure numpy, no torch.

This module owns two things the generators used to own: the case tables
(`CORE_CASES` for 1D, `CORE_CASES_2D` for the SS2D grids) and the input draws
(`draw_inputs`, `draw_inputs_2d`). It is deliberately torch-free so that both
the generators (dev tier) and the verifiers' determinism checks
(requirements.txt's torch-free tier, which is all CI's `test` job installs) can
call the same code.

Why not `torch.Generator`, which is what this used to draw with: torch does
not promise a stable RNG stream across releases, and it broke one — goldens
generated under torch 2.11 do not redraw bit-for-bit under 2.13, so the
determinism check failed for everyone with the full requirements installed
while silently skipping in CI (see `verify_golden.check_determinism`). The 2D
set had it worse: it drew the same way and had no determinism check at all, so
nothing would have reported that `golden/2d/*.npz` was unregenerable.

What this draws from instead, in decreasing order of how much it is
promised to us:

  1. PCG64's raw 64-bit output stream, via `BitGenerator.random_raw`.
     numpy's compatibility policy guarantees a bit generator's stream for a
     given seed; PCG64 and `SeedSequence` are both covered.
  2. uniforms and normals derived from that stream *here*, by the two
     textbook constructions below, rather than by `Generator.standard_normal`
     et al. — whose streams numpy explicitly does NOT guarantee (the
     ziggurat tables are an implementation detail it may change).

So the draws are pinned to an algorithm written out in this file, not to any
library's version. Regenerating under a different numpy — or a from-scratch
reimplementation in another language — reproduces them.

Values are computed in float64 and cast to float32, which is the dtype the
kernel is validated at.
"""

import hashlib

import numpy as np

# 1 / 2**53: scales a 53-bit mantissa draw into the unit interval.
_TWO_POW_M53 = 2.0 ** -53

# Recorded per case in golden/manifest.json. Bump it if the draws below ever
# change, so a stale golden set fails with that fact rather than with an
# unexplained bit mismatch.
#
# v2: `A` now takes its exp in float64. v1 took it in float32, which numpy
#     dispatches to per-architecture SIMD, so v1 goldens do not redraw on Arm.
INPUT_DRAW_SPEC = "pcg64-raw/box-muller/v2"


def case_seed(name: str) -> int:
    """Per-case seed, so cases are independent and order does not matter."""
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")


class Draw:
    """Deterministic draws from PCG64's raw stream. See the module docstring."""

    def __init__(self, seed: int):
        self._bits = np.random.PCG64(seed)

    def _u01(self, n: int) -> np.ndarray:
        """n uniforms in (0, 1]. The +1 keeps 0 out, so log() below is safe."""
        raw = self._bits.random_raw(n).astype(np.uint64)
        return ((raw >> np.uint64(11)).astype(np.float64) + 1.0) * _TWO_POW_M53

    def uniform(self, lo: float, hi: float, *shape) -> np.ndarray:
        return self.uniform64(lo, hi, *shape).astype(np.float32)

    def uniform64(self, lo: float, hi: float, *shape) -> np.ndarray:
        """As `uniform`, but left in float64.

        Exists so callers that feed the result to a transcendental can do that
        in float64 — see `draw_inputs`'s `A`, and the note there on why the
        dtype of a ufunc's *input* decides whether the draw is portable.
        """
        n = int(np.prod(shape)) if shape else 1
        return (lo + (hi - lo) * self._u01(n)).reshape(shape)

    def randn(self, *shape) -> np.ndarray:
        """Standard normals by Box-Muller, two per pair of uniforms."""
        n = int(np.prod(shape)) if shape else 1
        pairs = (n + 1) // 2
        u = self._u01(2 * pairs)
        radius = np.sqrt(-2.0 * np.log(u[0::2]))
        angle = 2.0 * np.pi * u[1::2]
        z = np.empty(2 * pairs, dtype=np.float64)
        z[0::2] = radius * np.cos(angle)
        z[1::2] = radius * np.sin(angle)
        return z[:n].reshape(shape).astype(np.float32)


def draw_inputs(name, B, D, L, N, *, groups=None, with_z=True, with_D=True,
                with_bias=True, softplus=True, delta_style="normal"):
    """Draw one case's f32 inputs. Returns numpy arrays, never torch tensors."""
    g = Draw(case_seed(name))

    u = g.randn(B, D, L)
    # A: negative, magnitudes spanning the trained-Mamba range (init is
    # -[1..N] per channel; training spreads it out).
    # Log-uniform over [0.5, 16], computed in float64 and cast at the end.
    #
    # This used to run `np.exp` on the float32 array `uniform` returns, and that
    # was NOT portable: numpy dispatches float32 transcendentals to
    # architecture-specific SIMD (AVX on x86, NEON on aarch64), so the two
    # disagree in the last bit. The determinism check passed on the x86 dev box
    # and failed on both Arm CI platforms — 22 cases, every one of them on `A`
    # and on nothing else, because every other field goes through Box-Muller,
    # whose log/cos/sin already operate in float64.
    #
    # Bounds are written as literals rather than `np.log(0.5)` / `np.log(16.0)`
    # so that not even the endpoints depend on a library's scalar log.
    _LOG_HALF = -0.6931471805599453  # ln(0.5)
    _LOG_16 = 2.772588722239781  # ln(16)
    A = (-np.exp(g.uniform64(_LOG_HALF, _LOG_16, D, N))).astype(np.float32)

    if softplus:
        # raw (pre-softplus) delta; bias chosen so softplus(delta+bias)
        # lands in the realistic ~[1e-3, 0.1] region.
        delta = g.randn(B, D, L) * 0.5
        delta_bias = g.uniform(-6.0, -3.0, D) if with_bias else None
        if not with_bias:
            delta = delta - 4.5
        if delta_style == "extreme":
            # stress exp underflow: softplus(delta) up to ~10 -> exp(delta*A)
            # down to exp(-160) == 0.0 in f32
            delta = g.uniform(-8.0, 10.0, B, D, L)
    else:
        # delta used directly as the (positive) timestep
        delta = g.uniform(1e-3, 0.1, B, D, L)
        delta_bias = None

    bc_batch_shape = (B, groups, N, L) if groups else (B, N, L)
    Bmat = g.randn(*bc_batch_shape)
    Cmat = g.randn(*bc_batch_shape)
    D_skip = g.randn(D) if with_D else None
    z = g.randn(B, D, L) if with_z else None
    return u, delta, A, Bmat, Cmat, D_skip, z, delta_bias


def draw_inputs_2d(name, b, d, h, w, n):
    """Draw one grid-shaped (SS2D) case's f32 inputs, as a name -> array dict.

    The keys are the npz keys `gen_golden_2d.py` stores and
    `verify_golden_2d.py` reads back, so the determinism check can compare the
    two without a separate key list.

    Same value distributions this case set has always used; only the RNG
    underneath moved (see the module docstring). Draw ORDER is part of the
    spec — reordering these changes every case, so bump INPUT_DRAW_SPEC if it
    ever has to happen.
    """
    g = Draw(case_seed(name))

    u = g.randn(b, d, h, w)
    # A negative, magnitudes over the trained-Mamba range (init -[1..N]).
    A = -g.uniform(0.5, float(n), d, n)
    # delta pre-softplus, chosen so softplus(delta) lands in ~[1e-3, 0.1].
    delta = g.uniform(-7.0, -2.0, b, d, h, w)
    Bmat = g.randn(b, n, h, w)
    Cmat = g.randn(b, n, h, w)
    D_skip = g.randn(d)
    delta_bias = g.uniform(-1.0, 1.0, d)
    return dict(u=u, delta=delta, A=A, B=Bmat, C=Cmat, D=D_skip,
                delta_bias=delta_bias)


CORE_CASES = [
    # (name, B, D, L, N, kwargs) — full Mamba config (z, D, bias, softplus)
    # unless overridden
    ("tiny",                1, 4,    8,    16, {}),
    ("small",               2, 8,    32,   16, {}),
    ("medium",              2, 64,   128,  16, {}),
    ("channels",            1, 256,  64,   16, {}),
    ("long_seq",            1, 16,   1024, 16, {}),
    ("edge_L1",             2, 8,    1,    16, {}),
    ("edge_D1",             1, 1,    32,   16, {}),
    ("state8",              2, 8,    32,   8,  {}),
    ("state13_neon_tail",   2, 8,    32,   13, {}),
    ("no_z",                2, 8,    32,   16, {"with_z": False}),
    ("no_D",                2, 8,    32,   16, {"with_D": False}),
    ("no_bias",             2, 8,    32,   16, {"with_bias": False}),
    ("no_softplus",         2, 8,    32,   16, {"softplus": False}),
    ("extreme_delta",       2, 8,    64,   16, {"delta_style": "extreme"}),
    ("grouped_BC",          2, 8,    32,   16, {"groups": 2}),
]

LARGE_CASES = [
    # benchmark-shaped; regenerate on demand, never committed
    ("large_mamba130m", 1, 1536, 512,  16, {}),
    ("large_batch",     4, 768,  1024, 16, {}),
]

CORE_CASES_2D = [
    # (name, batch, dim, height, width, state) — grid coverage mirrors the 1D
    # edge philosophy; see gen_golden_2d.py's docstring for what each buys.
    ("grid_square",      1, 4, 8, 8,  16),
    ("grid_nonsquare",   2, 4, 6, 10, 16),
    ("grid_odd",         1, 4, 7, 5,  16),   # H, W not multiples of 4
    ("grid_wide",        1, 8, 4, 12, 16),
    ("grid_state13",     1, 4, 5, 6,  13),   # state not a multiple of 4
    ("grid_l1",          1, 4, 1, 9,  16),   # degenerate height
]
