"""fastMRI knee single-coil: raw `.h5` -> a compact training cache.

**Licensing — read before running anything here.** The fastMRI Data Sharing
Agreement forbids redistributing the data or the download links, limits use to
internal research/education, and requires the Knoll et al. citation in any
publication. Nothing under `data/` is ever committed (see `.gitignore`), and no
signed URL belongs in this repo, a commit message, an issue, or a screen
recording. Cite per https://fastmri.med.nyu.edu/.

WHY KNEE SINGLE-COIL AND NOT BRAIN
----------------------------------
Our forward model is `y = M . F(x)` — one coil, no sensitivity maps. fastMRI
only ever released emulated-single-coil (ESC) data for **knee**; every brain
volume is multi-coil, which would need ESPIRiT maps and a different sampler.
Knee ESC is complex-valued, so it matches the 2-channel backbone exactly.

WHAT THE FILES CONTAIN
----------------------
Each single-coil volume `.h5` has:
  `kspace`             complex64, (slices, H, W) — typically (~36, 640, 368)
  `reconstruction_esc` float32,   (slices, 320, 320) — magnitude ground truth
  attrs: acquisition, max, norm, patient_id

We use `kspace`, because we want the **complex** image (phase included), which
`reconstruction_esc` has already discarded.

THE THREE THINGS THAT MATTER
----------------------------
1. **Centred FFT.** fastMRI k-space is stored DC-centred, which is the
   convention `sampling/posterior.py` now uses. No re-shuffling needed — but it
   is why that convention had to be fixed first.

2. **Per-volume normalisation is mandatory.** EDM's preconditioning assumes the
   data's standard deviation is ~`sigma_data`. Raw fastMRI scale varies by
   orders of magnitude between scans; feed it unnormalised and `c_skip`/`c_out`
   are wrong for every sample, which looks exactly like "the model is bad".
   Each volume is scaled to unit RMS, then globally to `sigma_data`.

3. **Split by VOLUME, never by slice.** Adjacent slices of one knee are nearly
   identical; splitting by slice leaks the evaluation set into training and
   produces meaningless metrics.
"""

import sys
from pathlib import Path

import numpy as np
import torch

CROP = 320  # fastMRI's standard knee evaluation crop


def _require_h5py():
    try:
        import h5py
    except ImportError:  # pragma: no cover
        raise SystemExit(
            "h5py is required to read fastMRI .h5 files:\n"
            "    pip install h5py") from None
    return h5py


def list_volumes(root):
    """Every `.h5` under `root`, sorted for a reproducible split."""
    return sorted(Path(root).rglob("*.h5"))


def center_crop(x, size=CROP):
    """Centre-crop the last two axes to `size` (fastMRI's eval convention)."""
    h, w = x.shape[-2:]
    size_h, size_w = min(size, h), min(size, w)
    top, left = (h - size_h) // 2, (w - size_w) // 2
    return x[..., top:top + size_h, left:left + size_w]


def volume_to_images(path, res=None, crop=CROP, sigma_data=0.5,
                     max_slices=None, skip_edge_frac=0.2):
    """One `.h5` volume -> `(n, 2, res, res)` float32, normalised.

    Edge slices are dropped by default: the first and last ~20% of a knee
    volume are mostly noise and anatomy-free background, which teaches a
    generative prior very little and skews the normalisation.
    """
    h5py = _require_h5py()
    with h5py.File(path, "r") as f:
        if "kspace" not in f:
            raise ValueError(f"{path.name}: no 'kspace' — is this "
                             f"single-coil data?")
        k = np.asarray(f["kspace"])
    if k.ndim != 3:
        raise ValueError(f"{path.name}: expected (slices, H, W) single-coil "
                         f"kspace, got {k.shape}. Multi-coil is not supported "
                         f"— see the module docstring.")

    n = k.shape[0]
    lo, hi = int(n * skip_edge_frac), int(np.ceil(n * (1 - skip_edge_frac)))
    idx = np.arange(lo, max(hi, lo + 1))
    if max_slices:
        idx = idx[:: max(1, len(idx) // max_slices)][:max_slices]

    kt = torch.from_numpy(k[idx])
    # fastMRI k-space is DC-centred, matching sampling/posterior.fft.
    img = torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(kt, dim=(-2, -1)),
                        dim=(-2, -1), norm="ortho"), dim=(-2, -1))
    img = center_crop(img, crop)
    x = torch.stack([img.real, img.imag], dim=1).float()  # (n, 2, H, W)

    if res and res != x.shape[-1]:
        x = torch.nn.functional.interpolate(
            x, size=(res, res), mode="bilinear", align_corners=False)

    # Per-VOLUME normalisation (see module docstring, point 2).
    rms = x.pow(2).mean().sqrt().clamp_min(1e-12)
    return x * (sigma_data / rms)


def build_cache(root, out, res=128, limit_volumes=None, max_slices=None,
                holdout=8, sigma_data=0.5, verbose=True, eval_root=None,
                eval_volumes=None):
    """Preprocess volumes into one compact tensor file.

    The raw download is transient: after this you can delete the `.h5` files.
    A few thousand slices at 128px in float16 is a few hundred MB.

    Two ways to get an evaluation split, and the first is preferable:

    `eval_root` — a SEPARATE directory (fastMRI's own `val` split against its
        `train` split). The conventional setup: nothing about the evaluation
        data shares a split with training, so no caveat attaches to the
        numbers.

    `holdout` — reserve N volumes from `root` when only one split was
        downloaded. Legitimate (the split is by volume, so no patient leaks
        between the two) but non-standard, and it should be stated plainly
        wherever the metrics are reported.
    """
    vols = list_volumes(root)
    if not vols:
        raise SystemExit(f"no .h5 files under {root}")
    if limit_volumes:
        vols = vols[:limit_volumes]

    eval_vols = []
    if eval_root:
        eval_vols = list_volumes(eval_root)
        if not eval_vols:
            raise SystemExit(f"no .h5 files under {eval_root}")
        if eval_volumes:
            eval_vols = eval_vols[:eval_volumes]
        eval_idx = set()  # nothing held back from `root`
        if verbose:
            print(f"  splits: {len(vols)} train volumes from {root}, "
                  f"{len(eval_vols)} eval volumes from {eval_root}")
    else:
        if holdout >= len(vols):
            raise SystemExit(f"holdout={holdout} but only {len(vols)} volumes")
        # Split by VOLUME (module docstring, point 3). Deterministic: sorted
        # order plus a fixed stride, so the same volumes are always held out
        # regardless of how many are processed.
        eval_idx = set(np.linspace(0, len(vols) - 1,
                                   holdout).astype(int).tolist())
        if verbose:
            print(f"  splits: {len(vols) - len(eval_idx)} train / "
                  f"{len(eval_idx)} holdout volumes, both from {root}")
            print("  (single-split mode — state this when reporting metrics; "
                  "pass --eval-root for")
            print("   the conventional train/val setup)")

    train, evalset, skipped = [], [], []
    for i, v in enumerate(vols):
        try:
            x = volume_to_images(v, res=res, sigma_data=sigma_data,
                                 max_slices=max_slices)
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort
            skipped.append((v.name, str(exc)[:80]))
            continue
        (evalset if i in eval_idx else train).append(x.half())
        if verbose and (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(vols)} volumes "
                  f"({sum(t.shape[0] for t in train)} train slices)",
                  flush=True)

    for v in eval_vols:
        try:
            evalset.append(volume_to_images(
                v, res=res, sigma_data=sigma_data,
                max_slices=max_slices).half())
        except Exception as exc:  # noqa: BLE001
            skipped.append((v.name, str(exc)[:80]))

    if not train:
        raise SystemExit("no usable volumes — are these single-coil files?")
    blob = {
        "train": torch.cat(train), "eval": torch.cat(evalset) if evalset
        else torch.empty(0), "res": res, "sigma_data": sigma_data,
        "n_volumes": len(vols), "n_holdout_volumes": len(eval_idx),
        "n_eval_volumes": len(eval_vols),
        # Recorded so the provenance travels with the data: whether the
        # evaluation set came from fastMRI's own val split or was carved out
        # of train changes what the metrics may be compared against.
        "split_mode": "separate-val" if eval_root else "holdout-from-train",
        "eval_root": str(eval_root) if eval_root else None,
        "source": "fastMRI knee singlecoil (NYU Langone; Knoll et al.)",
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, out)

    if verbose:
        mb = Path(out).stat().st_size / 1e6
        print(f"\ncache: {out}  ({mb:.0f} MB)  mode={blob['split_mode']}")
        print(f"  train {blob['train'].shape[0]} slices from "
              f"{len(vols) - len(eval_idx)} volumes")
        n_ev = len(eval_vols) if eval_root else len(eval_idx)
        print(f"  eval  {blob['eval'].shape[0]} slices from {n_ev} volumes"
              + ("" if eval_root else " held out of the same split"))
        if skipped:
            print(f"  skipped {len(skipped)} file(s):")
            for name, why in skipped[:5]:
                print(f"    {name}: {why}")
    return blob


def load_cache(path, split="train", device="cpu"):
    """Load a cache built by `build_cache` as float32 `(n, 2, res, res)`."""
    blob = torch.load(path, map_location="cpu")
    x = blob[split] if isinstance(blob, dict) else blob
    if x.numel() == 0:
        raise SystemExit(f"{path}: split '{split}' is empty")
    return x.float().to(device)


def batcher(images, batch_size, generator=None):
    """Infinite random batches — a drop-in for `phantom_batch` in training."""
    g = generator or torch.Generator().manual_seed(0)
    n = images.shape[0]
    while True:
        idx = torch.randint(0, n, (batch_size,), generator=g)
        yield images[idx]


if __name__ == "__main__":  # quick inspection of one volume
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m apps.mri_diffusion.fastmri_data "
                         "<volume.h5>")
    x = volume_to_images(Path(sys.argv[1]), res=128)
    print(f"{x.shape} float32, RMS {x.pow(2).mean().sqrt():.4f}, "
          f"|x| max {x.abs().max():.4f}")
