"""Turn a fastMRI knee single-coil download into a small training cache.

The raw archive is transient. This extracts what we need, builds a compact
cache, and then tells you exactly what to delete — steady-state footprint is a
few hundred MB rather than tens of GB.

    # 1. download (your signed URL — never paste it into this repo)
    curl -C - "<knee_singlecoil_val URL>" --output knee_singlecoil_val.tar.xz

    # 2. extract (peak disk is roughly 2.5x the archive, briefly)
    tar -xJf knee_singlecoil_val.tar.xz -C data/raw

    # 3. build the cache
    python tools/prepare_fastmri.py --root data/raw --out data/knee_128.pt \
        --res 128 --volumes 60 --max-slices 20

    # 4. delete the raw data (the script prints the exact commands)

`--volumes 60 --max-slices 20` gives ~1,200 slices, which is ample for a
demo-scale prior and processes in a couple of minutes. Drop the limits for
more.

**Licensing:** fastMRI's Data Sharing Agreement forbids redistributing the data
or the links. Nothing under `data/` is committed. Cite Knoll et al. — see
`apps/mri_diffusion/fastmri_data.py`.

`--self-test` needs no data at all: it synthesises fastMRI-shaped `.h5` files
and runs the whole path, so the pipeline can be validated before a download
finishes.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.mri_diffusion.fastmri_data import (  # noqa: E402
    build_cache, list_volumes, load_cache, volume_to_images)


def make_fake_volume(path, slices=12, h=640, w=368, seed=0):
    """A `.h5` shaped exactly like fastMRI knee single-coil.

    Lets the whole path be exercised — and CI-gated — without the real data,
    which cannot be committed. The content is a smooth blob plus noise; only
    the SHAPES and dtypes need to be faithful.
    """
    import h5py
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[-1:1:complex(0, h), -1:1:complex(0, w)]
    img = np.exp(-(xx ** 2 + yy ** 2) / 0.3) * (1 + 0.3 * rng.normal(size=(h, w)))
    vol = np.stack([img * (1 + 0.1 * i) for i in range(slices)])
    phase = 0.5 * np.sin(3 * xx) + 0.4 * np.cos(2 * yy)
    cplx = (vol * np.exp(1j * phase)).astype(np.complex64)
    # Store CENTRED k-space, as fastMRI does.
    k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(cplx, axes=(-2, -1)),
                                    axes=(-2, -1), norm="ortho"),
                        axes=(-2, -1)).astype(np.complex64)
    # Real files carry a scale that varies wildly between volumes; reproduce
    # that, because per-volume normalisation is the thing being tested.
    k *= 10.0 ** rng.uniform(-3, 3)
    with h5py.File(path, "w") as f:
        f.create_dataset("kspace", data=k)
        f.create_dataset("reconstruction_esc",
                         data=np.abs(cplx[:, 160:480, 24:344]).astype(np.float32))
        f.attrs["acquisition"] = "CORPD_FBK"
        f.attrs["patient_id"] = f"fake{seed}"


def inspect(path):
    """Check a REAL `.h5` against the assumptions the loader is built on.

    The self-test proves the loader's logic against files we generate — which
    cannot prove that real fastMRI files are shaped the way we assumed. Run
    this on the first volume that finishes downloading, before trusting
    anything downstream.

    The load-bearing assumption is that k-space is stored **DC-centred**. If it
    is not, every image comes out quadrant-swapped — and it would still have
    plausible shapes, statistics and RMS, so it would sail through every other
    check and train on nonsense. The centring test below is decisive: centred
    k-space has its energy peak in the middle of the array, un-centred has it
    at the corners.
    """
    from apps.mri_diffusion.fastmri_data import _require_h5py
    h5py = _require_h5py()
    print(f"inspecting {path}\n")
    ok = True
    with h5py.File(path, "r") as f:
        print("datasets:")
        for k in f:
            d = f[k]
            print(f"  {k:22s} {str(d.shape):24s} {d.dtype}")
        print("attrs:")
        for k, v in f.attrs.items():
            print(f"  {k:22s} {v}")

        if "kspace" not in f:
            print("\nFAIL: no 'kspace' dataset — the loader needs it.")
            return 1
        k = np.asarray(f["kspace"][: min(4, f["kspace"].shape[0])])

    print()
    if k.ndim == 3:
        print(f"OK  : kspace is 3-D {k.shape} -> single-coil, as expected")
    elif k.ndim == 4:
        print(f"FAIL: kspace is 4-D {k.shape} -> MULTI-COIL. Only knee "
              f"singlecoil is supported;")
        print("      brain/multicoil would need ESPIRiT maps and a different "
              "forward operator.")
        return 1
    else:
        print(f"FAIL: unexpected kspace rank {k.ndim}")
        return 1

    if np.iscomplexobj(k):
        print(f"OK  : complex ({k.dtype}) -> phase is preserved")
    else:
        print(f"FAIL: kspace is real ({k.dtype}); expected complex")
        ok = False

    # THE decisive check: where does k-space energy live?
    mag = np.abs(k[0])
    h, w = mag.shape
    ch, cw = h // 2, w // 2
    q = max(4, min(h, w) // 16)
    centre = mag[ch - q:ch + q, cw - q:cw + q].mean()
    corners = np.mean([mag[:q, :q].mean(), mag[:q, -q:].mean(),
                       mag[-q:, :q].mean(), mag[-q:, -q:].mean()])
    ratio = centre / max(corners, 1e-30)
    print(f"\nk-space energy: centre {centre:.3e} vs corners {corners:.3e} "
          f"(ratio {ratio:.1f}x)")
    if ratio > 10:
        print("OK  : DC is at the ARRAY CENTRE -> centred storage, which is "
              "what the loader")
        print("      and sampling/posterior.fft both assume. No reshuffling "
              "needed.")
    elif ratio < 0.1:
        print("FAIL: DC is at the CORNERS -> k-space is NOT centred.")
        print("      The loader would produce quadrant-swapped images that "
              "still look")
        print("      statistically plausible. Fix: apply fftshift to kspace "
              "before the")
        print("      iFFT in fastmri_data.volume_to_images.")
        ok = False
    else:
        print("WARN: ambiguous (ratio near 1). Inspect a reconstructed slice "
              "by eye before trusting this.")
        ok = False

    # Reconstruct one slice and sanity-check it looks like an anatomy image.
    img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k[0]), norm="ortho"))
    a = np.abs(img)
    cropped = a[max(0, h // 2 - 160):h // 2 + 160,
                max(0, w // 2 - 160):w // 2 + 160]
    edge = np.mean([a[:8].mean(), a[-8:].mean()])
    print(f"\nreconstructed slice: {a.shape}, centre-crop mean "
          f"{cropped.mean():.3e}, border mean {edge:.3e}")
    if cropped.mean() > 3 * edge:
        print("OK  : signal is concentrated centrally, background is dark — "
              "looks like anatomy")
    else:
        print("WARN: no clear centre/background contrast. Check a rendered "
              "slice by eye.")

    print("\nINSPECT:", "PASS — assumptions hold, proceed" if ok
          else "PROBLEM — see above; do NOT train until resolved")
    return 0 if ok else 1


def self_test():
    print("self-test: synthesising fastMRI-shaped volumes (no real data)\n")
    tmp = Path(tempfile.mkdtemp(prefix="fastmri-selftest-"))
    try:
        raw = tmp / "raw"
        raw.mkdir()
        for i in range(6):
            make_fake_volume(raw / f"file{i}.h5", slices=10, seed=i)
        print(f"  wrote {len(list_volumes(raw))} volumes")

        x = volume_to_images(raw / "file0.h5", res=64)
        rms = float(x.pow(2).mean().sqrt())
        print(f"  one volume -> {tuple(x.shape)}, RMS {rms:.4f}")
        assert x.shape[1] == 2, "expected 2 channels (complex as real/imag)"
        assert x.shape[-1] == 64, "resize failed"
        assert abs(rms - 0.5) < 1e-3, f"normalisation off: RMS {rms}"
        assert x.abs().sum() > 0 and torch.isfinite(x).all()
        print("  normalisation OK (unit-RMS -> sigma_data=0.5), finite")

        # Volumes differ in raw scale by ~1e6; all must land on the same RMS.
        rms_all = [float(volume_to_images(p, res=64).pow(2).mean().sqrt())
                   for p in list_volumes(raw)]
        spread = max(rms_all) - min(rms_all)
        print(f"  per-volume RMS across 6 volumes: spread {spread:.2e}")
        assert spread < 1e-3, "per-volume normalisation is not working"

        out = tmp / "cache.pt"
        blob = build_cache(raw, out, res=64, holdout=2, verbose=False)
        tr, ev = blob["train"].shape[0], blob["eval"].shape[0]
        print(f"  cache: {tr} train / {ev} eval slices, "
              f"{out.stat().st_size/1e6:.1f} MB")
        assert tr > 0 and ev > 0, "empty split"
        assert blob["train"].dtype == torch.float16, "cache should be float16"

        loaded = load_cache(out, "train")
        assert loaded.dtype == torch.float32 and loaded.shape[0] == tr
        print(f"  load_cache -> {tuple(loaded.shape)} float32")

        # A multi-coil file must be refused with a clear message, not
        # silently mis-read as single-coil.
        import h5py
        with h5py.File(raw / "multicoil.h5", "w") as f:
            f.create_dataset("kspace",
                             data=np.zeros((4, 15, 640, 368), np.complex64))
        try:
            volume_to_images(raw / "multicoil.h5", res=64)
        except ValueError as exc:
            assert "Multi-coil" in str(exc) or "single-coil" in str(exc)
            print("  multi-coil input rejected with a clear error")
        else:
            raise AssertionError("multi-coil file was not rejected")

        print("\nSELF-TEST: PASS — the fastMRI path works end to end.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="directory holding extracted .h5 files")
    ap.add_argument("--out", default="data/knee_128.pt")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--volumes", type=int, default=None,
                    help="cap the number of volumes (default: all)")
    ap.add_argument("--max-slices", type=int, default=None,
                    help="cap slices taken per volume")
    ap.add_argument("--holdout", type=int, default=8,
                    help="volumes reserved for evaluation (never trained on)")
    ap.add_argument("--sigma-data", type=float, default=0.5)
    ap.add_argument("--self-test", action="store_true",
                    help="validate the pipeline on synthetic data, no download")
    ap.add_argument("--inspect", metavar="FILE.h5", default=None,
                    help="check a REAL volume against the loader's "
                         "assumptions — run this first")
    args = ap.parse_args()

    if args.inspect:
        return inspect(Path(args.inspect))
    if args.self_test:
        return self_test()
    if not args.root:
        ap.error("--root is required (or use --self-test)")

    print(f"fastMRI knee single-coil -> {args.out}")
    print(f"  source {args.root}, {args.res}px, "
          f"holdout {args.holdout} volumes\n")
    build_cache(args.root, args.out, res=args.res,
                limit_volumes=args.volumes, max_slices=args.max_slices,
                holdout=args.holdout, sigma_data=args.sigma_data)

    print("\nThe raw data is no longer needed. To reclaim the space:")
    print(f"    rm -rf {args.root}")
    print("    rm -f knee_singlecoil_*.tar.xz")
    print("\nNext:")
    print(f"    python tools/calibrate_prior_bar.py --data fastmri "
          f"--cache {args.out}")
    print("    # then train, and check with tools/prior_report.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
