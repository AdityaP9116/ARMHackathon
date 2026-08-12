"""Draw what the 2D cross-scan actually does, using the real kernel.

WHY THIS EXISTS
---------------
Every other artifact in this repo is a number. This one is a picture, and it
answers the question a reader has after "four-direction cross-scan": *four
directions of what, and why four?*

It needs **no weights, no dataset and no downstream task.** That is not a
compromise — it is the point. A scan is an operator, and the honest way to
characterise an operator is its **impulse response**: put a 1 in one cell, zero
everywhere else, run it, and look at where the energy goes. That shows exactly
which positions each direction can reach, which is the property the four-scan
construction exists to fix.

(This matters practically too: no 2D Mamba-3 weights have ever been published,
so a segmentation or classification demo is not available to us. The impulse
response is not a substitute for one — it is a *better* tool for showing what
the operator computes.)

WHAT IS HELD FIXED, AND WHY
---------------------------
`q` and `k` are set to a constant vector and `angles` to zero, so `q·k` is the
same at every position and RoPE contributes no rotation. With random q/k the
picture is dominated by the noise in their dot products and shows nothing. Held
constant, what remains is the **propagation envelope** — the decay the state
undergoes as it is carried along each traversal — which is what we mean to
show. The kernel is entirely unmodified; only its inputs are chosen to make one
property legible.

    python bench/viz_ss2d.py                    # -> bench/results/ss2d_scans.png
    python bench/viz_ss2d.py --grid 64 --decay 0.03
"""

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "python"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    raise SystemExit("This figure needs matplotlib:\n    pip install matplotlib")

from arm_scan.mamba3_noncausal import ss2d_noncausal_mamba3  # noqa: E402
from arm_scan.ss2d_mamba3 import ss2d_scan_mamba3  # noqa: E402

DIRS = ("row →", "row ←", "col ↓", "col ↑")


def reference_pair(q, k, v, adt, dt, trap, q_bias, k_bias, angles=None,
                   D=None, z=None):
    """`mamba3_scan_pair`'s signature, backed by the PyTorch reference.

    So the figure can be regenerated from a fresh clone with no Rust toolchain
    and no built cdylib. `ss2d_scan_mamba3` takes the two-direction primitive as
    an argument precisely so a reference can be substituted with the identical
    dataflow, and `tests/check_ss2d_mamba3.py` gates the two against each other
    — so this produces the same picture, more slowly.
    """
    sys.path.insert(0, str(REPO / "tests"))
    from reference.mamba3_ref import mamba3_siso_ref

    common = dict(Q_bias=q_bias, K_bias=k_bias, Angles=angles, D=D, Z=z,
                  dtype=torch.float32)
    args = dict(Q=q, K=k, V=v, ADT=adt, DT=dt, Trap=trap)
    return (mamba3_siso_ref(**args, **common, reverse=False),
            mamba3_siso_ref(**args, **common, reverse=True))


def make_inputs(grid, decay, v, dqk=4):
    """Kernel inputs on a `grid x grid` token grid, with `v` as the signal.

    Everything except `v` is uniform: constant q/k, zero angles, one head, one
    channel. So the output is the propagation envelope and nothing else.
    """
    h = w = grid
    b, nh, dv, r = 1, 1, 1, dqk // 4
    ones = torch.ones(b, h, w, 1, dqk)
    dt = torch.full((b, nh, h, w), 1.0)
    return dict(
        q=ones, k=ones.clone(), v=v,
        # adt = A*dt, and must be <= 0: it is the log of the per-step decay.
        # Smaller |adt| -> slower decay -> longer visible trails.
        adt=torch.full((b, nh, h, w), -float(decay)),
        dt=dt,
        trap=torch.zeros(b, nh, h, w),
        q_bias=torch.zeros(nh, dqk), k_bias=torch.zeros(nh, dqk),
        angles=torch.zeros(b, h, w, nh, r),
    )


def impulse(grid):
    """A single 1 at the centre; zeros elsewhere. Shape (1, H, W, 1, 1)."""
    v = torch.zeros(1, grid, grid, 1, 1)
    v[0, grid // 2, grid // 2, 0, 0] = 1.0
    return v


def shapes_image(grid):
    """A small synthetic scene — a disc, a bar and a corner square.

    Deliberately not a photograph: the point is to read directional smear off
    hard edges, and a synthetic scene keeps the script self-contained with no
    asset to ship or license.
    """
    yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid),
                            indexing="ij")
    img = torch.zeros(grid, grid)
    img[((yy - grid * 0.35) ** 2 + (xx - grid * 0.35) ** 2)
        < (grid * 0.13) ** 2] = 1.0
    img[int(grid * 0.62):int(grid * 0.70), int(grid * 0.20):int(grid * 0.80)] = 1.0
    img[int(grid * 0.15):int(grid * 0.28), int(grid * 0.70):int(grid * 0.85)] = 1.0
    return img.reshape(1, grid, grid, 1, 1)


def magnitude(plane):
    """(b, H, W, heads, dv) -> (H, W) energy, normalised to [0, 1]."""
    m = plane[0].abs().sum(dim=(-1, -2))
    return (m / m.max().clamp(min=1e-12)).numpy()


def draw_order(ax, which, n=8):
    """The traversal path itself, on a small grid.

    Both orderings are plain raster scans, not boustrophedon: `grid_to_views_
    time_major` reshapes row-major and column-major, so each line jumps back to
    the start of the next line. Drawing it accurately matters — a snake would
    imply a locality the kernel does not have.
    """
    pts = [(r, c) for r in range(n) for c in range(n)]
    if which in (2, 3):                     # column-major
        pts = [(r, c) for c in range(n) for r in range(n)]
    if which in (1, 3):                     # reverse traversal
        pts = pts[::-1]
    ys = [p[0] for p in pts]
    xs = [p[1] for p in pts]
    ax.plot(xs, ys, lw=0.7, color="#5b8def", alpha=0.85, zorder=1)
    ax.scatter([xs[0]], [ys[0]], s=26, color="#2ecc71", zorder=3)
    ax.scatter([xs[-1]], [ys[-1]], s=26, color="#e74c3c", zorder=3)
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(n - 0.3, -0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=48)
    ap.add_argument("--decay", type=float, default=0.015,
                    help="per-step |adt|; smaller = longer trails")
    ap.add_argument("--out", default="bench/results/ss2d_scans.png")
    ap.add_argument("--reference", action="store_true",
                    help="use the PyTorch reference instead of the kernel "
                         "(same picture, no cdylib needed)")
    args = ap.parse_args()

    g = args.grid
    pair = reference_pair if args.reference else None
    print("scan primitive:",
          "PyTorch reference" if args.reference else "Arm kernel")
    cmap = LinearSegmentedColormap.from_list(
        "trail", ["#0b1021", "#1b3a6b", "#2f7fd1", "#63c7f0", "#f5f9ff"])

    # --- impulse: what each direction can reach -------------------------
    imp = make_inputs(g, args.decay, impulse(g))
    planes = ss2d_scan_mamba3(**imp, merge="none", scan_pair=pair)
    causal = ss2d_scan_mamba3(**imp, merge="sum", scan_pair=pair)
    # ss2d_noncausal_mamba3 takes no scan_pair hook -- it calls the kernel
    # directly. Rather than re-deriving forward+backward-diagonal here, where a
    # mistake would render a plausible but wrong picture with no gate on it,
    # the panel is simply omitted when there is no kernel to ask.
    noncausal = None if args.reference else ss2d_noncausal_mamba3(
        **imp, merge="sum")

    # --- a scene, one direction vs all four ------------------------------
    scn = make_inputs(g, args.decay, shapes_image(g))
    scn_planes = ss2d_scan_mamba3(**scn, merge="none", scan_pair=pair)
    scn_all = ss2d_scan_mamba3(**scn, merge="sum", scan_pair=pair)

    fig, axes = plt.subplots(3, 4, figsize=(13.2, 10.4))
    fig.patch.set_facecolor("white")

    for i, ax in enumerate(axes[0]):
        draw_order(ax, i)
        ax.set_title(f"{DIRS[i]}   traversal", fontsize=10)
    axes[0][0].set_ylabel("the four orders", fontsize=11)

    for i, ax in enumerate(axes[1]):
        ax.imshow(magnitude(planes[i]), cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"{DIRS[i]}   reach", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    axes[1][0].set_ylabel("impulse response", fontsize=11)

    panels = [
        (magnitude(causal), "all four, summed"),
        (None if noncausal is None else magnitude(noncausal), "non-causal"),
        (magnitude(scn_planes[0]), "scene · row → only"),
        (magnitude(scn_all), "scene · all four"),
    ]
    for ax, (data, title) in zip(axes[2], panels):
        if data is None:
            ax.text(0.5, 0.5,
                    "non-causal panel\nneeds the kernel\n(drop --reference)",
                    ha="center", va="center", fontsize=9, color="#888",
                    transform=ax.transAxes)
            ax.set_facecolor("#f4f4f6")
        else:
            ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    axes[2][0].set_ylabel("combined", fontsize=11)

    who = "the PyTorch reference" if args.reference else "the Arm kernel"
    fig.suptitle(
        f"SS2D cross-scan on the Mamba-3 recurrence — computed by {who}\n"
        f"{g}×{g} token grid · impulse at centre · no weights, no dataset, no task",
        fontsize=12.5)
    fig.text(0.5, 0.020,
             "Each direction reaches only what precedes it in its own traversal — "
             "that blind spot is why four scans exist.\n"
             "The faint offset band is the raster wrap: state survives the jump from "
             "the end of one line to the start of the next.\n"
             "Green = first token, red = last. Energy is per-position, normalised.",
             ha="center", fontsize=9.5, color="#333")
    fig.tight_layout(rect=(0, 0.085, 1, 0.94))

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
