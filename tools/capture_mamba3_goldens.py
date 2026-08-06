"""Capture Mamba-3 ground truth from the OFFICIAL kernels. Run this on a CUDA box.

WHY THIS EXISTS
---------------
Our entire correctness method is "diff the fast implementation against a trusted
reference." For Mamba-1 that reference ships with upstream as a pure-PyTorch
`selective_scan_ref`. Mamba-3 has no such thing: `mamba_ssm/modules/mamba3.py`
imports Triton/TileLang/CuTe kernels and asserts if they are missing — there is
no CPU path anywhere in the file.

The obvious substitute — the community re-implementation
`rishikksh20/mamba3-pytorch` — **does not agree with the paper's recurrence**.
Measured on random inputs (states are O(1), so this is not rounding):

    community vs paper, same gate ........................ max_abs 1.06
    best gate remapping (1 - gate/2) ..................... max_abs 0.145
    structural difference alone (decay on the prev term) . max_abs 0.363

The paper carries the state decay on the PREVIOUS input term:

    h_t = a_t h_{t-1} + (1-L_t) dt_t a_t (B_{t-1}x_{t-1}) + L_t dt_t (B_t x_t)

the community version does not:

    h_t = a_t h_{t-1} + dt_t (tr_t/2)(B_{t-1}x_{t-1}) + dt_t (1-tr_t/2)(B_t x_t)

We cannot tell which is right from outside, and it does not matter: the only
authoritative answer is what the official kernels compute, because that is what
the published checkpoints were trained against. So we capture from those.

WHAT IT PRODUCES
----------------
`.npz` files holding the scan's INPUTS and OUTPUTS, plus a manifest — the same
shape of artifact as `tests/golden/` for Mamba-1. Those files get committed and
the GPU is never needed again: the Rust kernel is developed and validated on
CPU against them.

    # on the GPU box
    pip install mamba-ssm --no-build-isolation
    python tools/capture_mamba3_goldens.py --out tests/golden/mamba3

    # then, on any CPU
    python tests/verify_golden_mamba3.py

Also captures full-model logits for a fixed prompt, so end-to-end parity can be
checked later without re-running anything on a GPU.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def find_scan_fn():
    """Locate the official SISO scan without hardcoding a module path.

    The import path has moved between releases, so search rather than assume,
    and report what was found so the capture is auditable.
    """
    import importlib
    import pkgutil

    import mamba_ssm
    candidates = []
    for mod in pkgutil.walk_packages(mamba_ssm.__path__,
                                     mamba_ssm.__name__ + "."):
        try:
            m = importlib.import_module(mod.name)
        except Exception:
            continue
        for attr in ("mamba3_siso_combined", "mamba3_mimo_combined"):
            fn = getattr(m, attr, None)
            if callable(fn):
                candidates.append((mod.name, attr, fn))
    return candidates


class Capture:
    """Wrap a kernel entry point and record every call's tensors."""

    def __init__(self, fn, limit=64):
        self.fn, self.limit, self.calls = fn, limit, []

    def __call__(self, *args, **kwargs):
        out = self.fn(*args, **kwargs)
        if len(self.calls) < self.limit:
            def snap(v):
                if torch.is_tensor(v):
                    return v.detach().float().cpu().numpy()
                return v
            self.calls.append({
                "args": [snap(a) for a in args],
                "kwargs": {k: snap(v) for k, v in kwargs.items()},
                "out": (tuple(snap(o) for o in out)
                        if isinstance(out, tuple) else snap(out)),
            })
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/golden/mamba3")
    ap.add_argument("--model", default="state-spaces/mamba3-siso-187m")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--max-calls", type=int, default=8,
                    help="scan calls to record (one per layer per forward)")
    ap.add_argument("--skip-model", action="store_true",
                    help="capture from a freshly-built block only")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device. This script must run on a GPU box — capturing "
            "ground truth is the ONE step that needs one. Everything "
            "downstream runs on CPU against the files it writes.")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"torch : {torch.__version__}\n")

    found = find_scan_fn()
    if not found:
        raise SystemExit(
            "Could not find mamba3_siso_combined in mamba_ssm. Check the "
            "install, and report the version — the entry point may have been "
            "renamed.")
    print("official kernel entry points found:")
    for mod, attr, _ in found:
        print(f"  {mod}.{attr}")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {"device": torch.cuda.get_device_name(0),
                "torch": torch.__version__, "cases": []}

    # --- capture at the kernel boundary --------------------------------
    import importlib
    caps = []
    for mod, attr, fn in found:
        m = importlib.import_module(mod)
        cap = Capture(fn, limit=args.max_calls)
        setattr(m, attr, cap)
        caps.append((mod, attr, cap))

    # --- drive it with the real model ----------------------------------
    logits = None
    if not args.skip_model:
        try:
            from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
            print(f"\nloading {args.model} ...")
            model = MambaLMHeadModel.from_pretrained(
                args.model, device="cuda", dtype=torch.bfloat16)
            model.eval()
            torch.manual_seed(0)
            ids = torch.randint(0, 1000, (1, args.seq), device="cuda")
            with torch.no_grad():
                logits = model(ids).logits.float().cpu().numpy()
            np.savez_compressed(outdir / "model_forward.npz",
                                input_ids=ids.cpu().numpy(), logits=logits)
            manifest["model"] = args.model
            manifest["model_forward"] = {
                "seq": args.seq, "logits_shape": list(logits.shape)}
            print(f"  logits {logits.shape} -> model_forward.npz")
        except Exception as exc:  # noqa: BLE001
            print(f"\nmodel load failed ({type(exc).__name__}: {exc})")
            print("continuing with block-level capture only")

    # --- and with a freshly built block, for shape coverage -------------
    if not any(c.calls for _, _, c in caps):
        from mamba_ssm.modules.mamba3 import Mamba3
        print("\nno calls captured from the model; driving a fresh block")
        blk = Mamba3(d_model=768, d_state=128, headdim=64,
                     is_mimo=False).to(torch.bfloat16).cuda().eval()
        with torch.no_grad():
            blk(torch.randn(1, args.seq, 768, device="cuda",
                            dtype=torch.bfloat16))

    # --- write the goldens ---------------------------------------------
    n = 0
    for mod, attr, cap in caps:
        for i, call in enumerate(cap.calls):
            arrays = {}
            for j, a in enumerate(call["args"]):
                if isinstance(a, np.ndarray):
                    arrays[f"arg{j}"] = a
            for k, v in call["kwargs"].items():
                if isinstance(v, np.ndarray):
                    arrays[f"kw_{k}"] = v
            out = call["out"]
            if isinstance(out, tuple):
                for j, o in enumerate(out):
                    if isinstance(o, np.ndarray):
                        arrays[f"out{j}"] = o
            elif isinstance(out, np.ndarray):
                arrays["out"] = out
            if not arrays:
                continue
            name = f"{attr}_{i:02d}"
            np.savez_compressed(outdir / f"{name}.npz", **arrays)
            manifest["cases"].append({
                "name": name, "kernel": f"{mod}.{attr}",
                "arrays": {k: list(v.shape) for k, v in arrays.items()},
            })
            n += 1
            print(f"  {name}: " + ", ".join(
                f"{k}{list(v.shape)}" for k, v in arrays.items()))

    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {n} golden case(s) + manifest.json to {outdir}")
    if n == 0:
        raise SystemExit(
            "No kernel calls were captured. The scan may be dispatched under "
            "a different name; print the traceback inside Mamba3.forward to "
            "find the real entry point.")
    print("\nCommit these files. The GPU is not needed again — the Rust "
          "kernel is built and validated on CPU against them.")


if __name__ == "__main__":
    main()
