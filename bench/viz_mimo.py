"""Draw what MIMO changes: a rank-r state update instead of a rank-1 one.

WHY THIS EXISTS
---------------
MIMO is the hardest part of this project to explain and the easiest to
mis-read, because its benchmark ratios are large for the wrong reason (see the
MIMO section of `README.md`). Its actual contribution is structural, and the
structure is drawable.

SISO absorbs **one** outer product into the state per step:

    S <- alpha*S + scale * (x (x) k)

MIMO absorbs **r** of them into the *same* state:

    S <- alpha*S + scale * sum_r (x_r (x) k_r)

Everything else — alpha, gamma, the forward-looking scale, the discretization —
is identical. That single change is what the panels show:

  1. the update itself     rank-1 is separable and looks it; rank-4 does not.
  2. singular values       a sum of r outer products has EXACTLY r non-zero
                           singular values. This is the rank, measured, and it
                           is the panel that proves the structure rather than
                           illustrating it.
  3. arithmetic intensity  the state loaded per step is the SAME size for every
                           rank, while the arithmetic done with it scales with
                           r. That ratio is the argument for MIMO on a
                           memory-bound CPU — and it is why our scalar-only
                           MIMO path is penalised rather than helped.
  4. r=1 collapses to SISO with the rotation removed, a rank-1 MIMO scan is
                           a SISO scan. Structural proof that the
                           generalisation is right, not merely close.

Panel 4 mirrors `check_rank1_matches_siso_when_unrotated` in
`tests/check_mamba3_mimo_op.py`, which runs the same comparison kernel-to-kernel
in CI. Here it runs through the references so the figure needs no cdylib.

    python bench/viz_mimo.py          # -> bench/results/mimo_rank.png
"""

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tests"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    raise SystemExit("This figure needs matplotlib:\n    pip install matplotlib")

from reference.mamba3_ref import mamba3_mimo_ref, mamba3_siso_ref  # noqa: E402


def state_update(rank, dv=32, dqk=32, seed=0):
    """One step's contribution to the state: `sum_r x_r (x) k_r`, shape (dv, dqk).

    Written from the definition in `kernel/arm-scan-core/src/mamba3/mimo.rs`
    (`S[p][n] += scale * sum_r x_r[p] * k_r[n]`), which is what makes the rank
    claim checkable rather than asserted: the matrix below is a sum of `rank`
    outer products by construction, so panel 2 must find exactly `rank`
    non-zero singular values.
    """
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dv, generator=g)
    psi = torch.randn(rank, dv, generator=g)      # per-rank input projection
    k = torch.randn(rank, dqk, generator=g)       # per-rank key
    x = psi * v                                    # x_r[p] = psi[r][p] * v[p]
    return torch.einsum("rp,rn->pn", x, k)


def rank1_vs_siso(length=64, dv=8, dqk=8, seed=0):
    """A rank-1 MIMO scan and a SISO scan on the same inputs, angles zeroed."""
    g = torch.Generator().manual_seed(seed)
    b, h, nr = 1, 1, 1

    def rn(*s):
        return torch.randn(*s, generator=g)

    dt = torch.nn.functional.softplus(rn(b, h, length) * 0.5 - 1.0)
    common = dict(ADT=-torch.exp(rn(b, h, length) * 0.3) * dt, DT=dt,
                  Trap=rn(b, h, length), dtype=torch.float64)
    q, k, v = rn(b, length, nr, 1, dqk), rn(b, length, nr, 1, dqk), rn(b, length, h, dv)
    qb, kb = rn(h, nr, dqk), rn(h, nr, dqk)
    ang = torch.zeros(b, length, h, dqk // 4)

    mimo = mamba3_mimo_ref(
        Q=q, K=k, V=v, Q_bias=qb, K_bias=kb,
        MIMO_V=torch.ones(h, nr, dv), MIMO_Z=torch.ones(h, nr, dv),
        MIMO_Out=torch.ones(h, nr, dv), Angles=ang, **common)
    siso = mamba3_siso_ref(
        Q=q[:, :, 0], K=k[:, :, 0], V=v, Q_bias=qb[:, 0], K_bias=kb[:, 0],
        Angles=ang, **common)
    return mimo[0, :, 0, 0].numpy(), siso[0, :, 0, 0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/results/mimo_rank.png")
    args = ap.parse_args()

    ranks = (1, 2, 4, 8)
    cmap = LinearSegmentedColormap.from_list(
        "upd", ["#1b3a6b", "#4a7fc1", "#f7f9fc", "#e0864a", "#7a2f12"])

    fig = plt.figure(figsize=(13.4, 8.6))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 4, hspace=0.30, wspace=0.34,
                          left=0.07, right=0.985, top=0.86, bottom=0.16,
                          height_ratios=[1.05, 1.25])

    # 1. the update matrix, per rank -------------------------------------
    for i, r in enumerate(ranks):
        ax = fig.add_subplot(gs[0, i])
        m = state_update(r)
        lim = m.abs().max()
        ax.imshow(m, cmap=cmap, vmin=-lim, vmax=lim)
        ax.set_title(f"rank {r}" + ("   (SISO)" if r == 1 else ""),
                     fontsize=10.5)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("1 · one step's state update\n$\\sum_r x_r \\otimes k_r$",
                          fontsize=10)

    # 2. singular values: the rank, measured ------------------------------
    ax = fig.add_subplot(gs[1, :2])
    for r, colour in zip(ranks, ("#2f7fd1", "#e8a33d", "#d1495b", "#5b3f8f")):
        sv = torch.linalg.svdvals(state_update(r))
        sv = (sv / sv[0]).clamp(min=1e-18)
        ax.semilogy(range(1, 17), sv[:16], "o-", ms=4, lw=1.4,
                    color=colour, label=f"rank {r}")
        ax.axvline(r + 0.5, color=colour, ls=":", lw=1, alpha=0.5)
    ax.set_title("2 · exactly r non-zero singular values — the rank, measured",
                 fontsize=11)
    ax.set_xlabel("singular value index")
    ax.set_ylabel("magnitude, relative (log)")
    ax.legend(fontsize=8.5, frameon=False, ncol=4)
    ax.spines[["top", "right"]].set_visible(False)

    # 3. arithmetic intensity ---------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    dv = dqk = 64
    state_bytes = dv * dqk * 4
    flops = [2 * r * dv * dqk for r in ranks]
    xs = range(len(ranks))
    ax.bar([x - 0.2 for x in xs], [state_bytes / 1e3] * len(ranks), width=0.4,
           color="#9aa5b1", label="state bytes loaded")
    ax.bar([x + 0.2 for x in xs], [f / 1e3 for f in flops], width=0.4,
           color="#d1495b", label="arithmetic (kFLOP)")
    ax.set_xticks(list(xs)); ax.set_xticklabels([f"r={r}" for r in ranks])
    ax.set_title("3 · same load, r× the work", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # 4. rank-1 collapses to SISO -----------------------------------------
    ax = fig.add_subplot(gs[1, 3])
    mimo, siso = rank1_vs_siso()
    rel = abs(mimo - siso).max() / max(abs(siso).max(), 1e-30)
    ax.plot(siso, color="#2f7fd1", lw=2.4, label="SISO")
    ax.plot(mimo, color="#d1495b", lw=1.0, ls="--", label="MIMO, r=1")
    ax.set_title(f"4 · r=1 collapses to SISO\nmax rel diff {rel:.1e}",
                 fontsize=11)
    ax.set_xlabel("position")
    ax.legend(fontsize=8.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "MIMO — a rank-r state update, and why it is a CPU-shaped idea",
        fontsize=13)
    fig.text(0.5, 0.012,
             "The discretization is identical to SISO. Only the state update changes: "
             "r outer products instead of one, into the same state.\n"
             "Panel 3 is why that should suit a memory-bound CPU — and why our "
             "scalar-only MIMO path cannot cash it in: scalar code executes r× the "
             "operations without exploiting the density.",
             ha="center", fontsize=9.5, color="#333")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"wrote {out}   (rank-1 vs SISO: {rel:.3e})")


if __name__ == "__main__":
    main()
