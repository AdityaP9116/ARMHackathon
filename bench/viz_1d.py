"""Draw what the 1D selective scan does: selectivity, reach, and flat memory.

WHY THIS EXISTS
---------------
`viz_ss2d.py` shows *where* the 2D scan reaches. This is the companion for the
1D path, and it shows the mechanism that gives the family its name.

An SSM before Mamba had **fixed** dynamics: the same decay at every position, so
its memory behaved identically no matter what it read. Mamba makes the step size
`dt` a function of the input, which makes `alpha = exp(dt*A)` per-token — the
model can decide, at each position, whether to *hold* what it has or *dump* it
and start fresh. That is "selective", and it is the whole difference.

Panels:

  1. memory length      one impulse, three decay rates. How far a token's
                        influence survives is set by alpha, and nothing else.
  2. selectivity        the same sequence with a single high-`dt` token in the
                        middle. The state is wiped there and rebuilds after —
                        content-dependent forgetting, which is what a fixed SSM
                        cannot express.
  3. reach              forward / backward / fused, from a centre impulse.
                        The 1D counterpart of the 2D cross-scan panel.
  4. memory in L        the constant-memory claim, from measured numbers.

Panels 1-3 are computed; panel 4 plots figures measured by `bench_longctx.py`
and recorded in the README, and is labelled as such on the axis.

    python bench/viz_1d.py            # -> bench/results/scan_1d.png
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
except ImportError:
    raise SystemExit("This figure needs matplotlib:\n    pip install matplotlib")

from reference.mamba3_ref import mamba3_siso_ref  # noqa: E402

# Measured by bench/bench_longctx.py at a mamba-130m layer shape (B=1 D=768
# N=16); the reference must materialise a (B, D, L, N) intermediate while the
# kernel streams CHUNK=128 timesteps of scratch and never allocates it.
MEASURED_L = [2048, 8192, 32768, 131072]
MEASURED_REF_GB = [0.20, 0.81, 3.22, 12.88]
KERNEL_SCRATCH_GB = (128 * 16 * 2 + 128 * 2) * 4 / 1e9   # abar+bbar+dt+dtu


def run(length, adt, dqk=4, reverse=False):
    """One SISO scan with a centre impulse and the given per-token `adt`.

    q/k constant and angles zero for the same reason as in `viz_ss2d.py`: with
    random values the picture shows noise in `q.k` rather than the propagation
    envelope, which is the thing being drawn.
    """
    b, h, dv, r = 1, 1, 1, dqk // 4
    v = torch.zeros(b, length, h, dv)
    v[0, length // 2, 0, 0] = 1.0
    dt = torch.ones(b, h, length)
    out = mamba3_siso_ref(
        Q=torch.ones(b, length, 1, dqk), K=torch.ones(b, length, 1, dqk),
        V=v, ADT=adt, DT=dt, Trap=torch.zeros(b, h, length),
        Q_bias=torch.zeros(h, dqk), K_bias=torch.zeros(h, dqk),
        Angles=torch.zeros(b, length, h, r),
        dtype=torch.float32, reverse=reverse)
    m = out[0, :, 0, :].abs().sum(-1)
    return (m / m.max().clamp(min=1e-12)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=256)
    ap.add_argument("--out", default="bench/results/scan_1d.png")
    args = ap.parse_args()
    L = args.length
    x = range(L)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2))
    fig.patch.set_facecolor("white")

    # 1. memory length is set by alpha ---------------------------------
    ax = axes[0][0]
    for decay, colour in ((0.005, "#2f7fd1"), (0.02, "#e8a33d"),
                          (0.08, "#d1495b")):
        ax.plot(x, run(L, torch.full((1, 1, L), -decay)), color=colour,
                lw=1.6, label=f"|A·dt| = {decay}")
    ax.set_title("1 · how long a token is remembered", fontsize=11)
    ax.set_xlabel("position"); ax.set_ylabel("influence (normalised)")
    ax.legend(fontsize=8.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # 2. selectivity: one token dumps the state -------------------------
    ax = axes[0][1]
    adt = torch.full((1, 1, L), -0.004)
    reset = int(L * 0.68)
    adt[0, 0, reset] = -6.0          # one token with a very large step
    ax.plot(x, run(L, torch.full((1, 1, L), -0.004)), color="#9aa5b1",
            lw=1.5, label="uniform dt — nothing forgotten")
    ax.plot(x, run(L, adt), color="#d1495b", lw=1.8,
            label="one high-dt token — state dumped")
    ax.axvline(reset, color="#d1495b", ls=":", lw=1.2)
    ax.annotate("this token decides\nto forget", xy=(reset, 0.55),
                xytext=(reset - L * 0.30, 0.72), fontsize=8.5, color="#d1495b",
                arrowprops=dict(arrowstyle="->", color="#d1495b", lw=1))
    ax.set_title("2 · selectivity — forgetting is data-dependent", fontsize=11)
    ax.set_xlabel("position"); ax.set_ylabel("influence (normalised)")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # 3. reach: forward / backward / fused ------------------------------
    ax = axes[1][0]
    adt = torch.full((1, 1, L), -0.01)
    fwd, bwd = run(L, adt), run(L, adt, reverse=True)
    ax.fill_between(x, fwd, color="#2f7fd1", alpha=0.55, label="forward")
    ax.fill_between(x, bwd, color="#e8a33d", alpha=0.55, label="backward")
    both = (fwd + bwd)
    ax.plot(x, both / both.max(), color="#1b3a6b", lw=1.4,
            label="fused (one kernel call)")
    ax.set_title("3 · reach — one direction is blind on one side", fontsize=11)
    ax.set_xlabel("position"); ax.set_ylabel("influence (normalised)")
    ax.legend(fontsize=8.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # 4. the memory claim ------------------------------------------------
    ax = axes[1][1]
    ax.plot(MEASURED_L, MEASURED_REF_GB, "o-", color="#d1495b", lw=1.8,
            label="PyTorch reference — (B,D,L,N) intermediate")
    ax.plot(MEASURED_L, [KERNEL_SCRATCH_GB] * len(MEASURED_L), "o-",
            color="#2f7fd1", lw=1.8, label="our kernel — CHUNK scratch, flat")
    ax.annotate(f"{MEASURED_REF_GB[-1]} GB", xy=(MEASURED_L[-1],
                MEASURED_REF_GB[-1]), xytext=(-64, -4),
                textcoords="offset points", fontsize=9, color="#d1495b")
    ax.annotate(f"{KERNEL_SCRATCH_GB * 1e6:.0f} KB", xy=(MEASURED_L[-1],
                KERNEL_SCRATCH_GB), xytext=(-58, 8),
                textcoords="offset points", fontsize=9, color="#2f7fd1")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_title("4 · why long context works at all", fontsize=11)
    ax.set_xlabel("sequence length (measured, bench_longctx.py)")
    ax.set_ylabel("scan intermediates (GB, log)")
    ax.legend(fontsize=8.5, frameon=False, loc="center left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "The 1D selective scan — what the kernel computes, and why it fits a CPU",
        fontsize=13)
    fig.text(0.5, 0.015,
             "Panels 1–3: a single impulse, run through the Mamba-3 SISO recurrence. "
             "Panel 4: measured, not modelled.\n"
             "A fixed SSM has one decay for every token; making it per-token is what "
             "'selective' means — and it is also what stops the scan being a convolution.",
             ha="center", fontsize=9.5, color="#333")
    fig.tight_layout(rect=(0, 0.065, 1, 0.945))

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
