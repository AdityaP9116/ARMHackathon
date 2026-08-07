"""Mamba-3 through the PyTorch op, end to end, against real ground truth.

Everything below the torch layer has its own gate already: the Rust kernel is
checked against `tests/golden/mamba3/` by `cargo test`, and the C ABI by the
FFI unit tests. What is NOT covered by those is the layer this file exercises —
the angle pre-pass, the head-major -> time-major permute, the pointer
marshalling, and the `torch.compile` composability that the project's baseline
depends on. Each is a place where a shape or stride mistake produces plausible
numbers.

Checks, in order of what they would catch:

  1. op vs the captured official-kernel goldens (the real thing)
  2. `torch.compile` composes without a graph break (the fake kernel works)
  3. reverse == flip-time / scan / flip-back, through the torch layer
  4. `mamba3_scan_pair` agrees with two separate calls

Usage: python tests/check_mamba3_op.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from arm_scan.mamba3 import (  # noqa: E402
    angles_to_cos_sin, mamba3_scan, mamba3_scan_pair)

GOLDEN = ROOT / "tests" / "golden" / "mamba3"
BF16_EPS = 2.0 ** -8
MAX_ULPS = 8.0


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def load(name):
    z = np.load(GOLDEN / f"{name}.npz")
    t = {k[3:]: torch.from_numpy(z[k]).float()
         for k in z.files if k.startswith("kw_")}
    return t, torch.from_numpy(z["out"]).float()


def check_goldens():
    man = json.loads((GOLDEN / "manifest.json").read_text())
    worst = 0.0
    for case in man["cases"]:
        kw, gold = load(case["name"])
        out = mamba3_scan(
            kw["Q"], kw["K"], kw["V"], kw["ADT"], kw["DT"], kw["Trap"],
            kw["Q_bias"], kw["K_bias"], angles=kw["Angles"],
            D=kw.get("D"), z=kw.get("Z"))
        if out.shape != gold.shape:
            print(f"  {case['name']}: SHAPE {tuple(out.shape)} != "
                  f"{tuple(gold.shape)}")
            return False, 0.0
        scale = max(float(gold.abs().max()), 1e-30)
        ulps = float((bf16(out) - gold).abs().max()) / (scale * BF16_EPS)
        worst = max(worst, ulps)
        flag = "ok" if ulps <= MAX_ULPS else "FAIL"
        print(f"  {case['name']:<28} L={gold.shape[1]:<5} {ulps:7.2f} ULP  {flag}")
        if ulps > MAX_ULPS:
            return False, worst
        if not torch.isfinite(out).all() or float(out.abs().max()) == 0.0:
            print(f"  {case['name']}: non-finite or all-zero output")
            return False, worst
    return True, worst


def check_compile():
    """The fake kernel must let `torch.compile` trace straight through.

    A graph break here would not fail anything loudly — it would just make
    every benchmark against `torch.compile` quietly unfair, which is worse.
    """
    kw, _ = load("mamba3_siso_combined_06")
    cos, sin = angles_to_cos_sin(kw["Angles"], kw["DT"], kw["Q"].shape[-1] // 2)

    def f(v):
        return mamba3_scan(kw["Q"], kw["K"], v, kw["ADT"], kw["DT"],
                           kw["Trap"], kw["Q_bias"], kw["K_bias"],
                           D=kw.get("D"), z=kw.get("Z"), cos=cos, sin=sin)

    eager = f(kw["V"])
    compiled = torch.compile(f, fullgraph=True)(kw["V"])
    err = float((eager - compiled).abs().max())
    ok = err == 0.0
    print(f"  torch.compile(fullgraph=True): max_abs={err:.3e}  "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def check_reverse():
    """reverse == flip time, scan forward, flip the output back."""
    kw, _ = load("mamba3_siso_combined_06")
    cos, sin = angles_to_cos_sin(kw["Angles"], kw["DT"], kw["Q"].shape[-1] // 2)
    common = dict(q_bias=kw["Q_bias"], k_bias=kw["K_bias"], D=kw.get("D"),
                  z=kw.get("Z"), cos=cos, sin=sin)
    rev = mamba3_scan(kw["Q"], kw["K"], kw["V"], kw["ADT"], kw["DT"],
                      kw["Trap"], reverse=True, **common)

    fl = lambda t, d: torch.flip(t, dims=[d])  # noqa: E731
    cosf, sinf = fl(cos, 1), fl(sin, 1)
    fwd = mamba3_scan(
        fl(kw["Q"], 1), fl(kw["K"], 1), fl(kw["V"], 1), fl(kw["ADT"], 2),
        fl(kw["DT"], 2), fl(kw["Trap"], 2), q_bias=kw["Q_bias"],
        k_bias=kw["K_bias"], D=kw.get("D"),
        z=None if kw.get("Z") is None else fl(kw["Z"], 1),
        cos=cosf, sin=sinf, reverse=False)
    err = float((rev - fl(fwd, 1)).abs().max()) / max(
        float(rev.abs().max()), 1e-30)
    ok = err < 1e-5
    print(f"  reverse == flip/forward/flip: rel={err:.3e}  "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def check_pair():
    kw, _ = load("mamba3_siso_combined_06")
    args = (kw["Q"], kw["K"], kw["V"], kw["ADT"], kw["DT"], kw["Trap"],
            kw["Q_bias"], kw["K_bias"])
    common = dict(angles=kw["Angles"], D=kw.get("D"), z=kw.get("Z"))
    f, b = mamba3_scan_pair(*args, **common)
    f1 = mamba3_scan(*args, reverse=False, **common)
    b1 = mamba3_scan(*args, reverse=True, **common)
    ok = torch.equal(f, f1) and torch.equal(b, b1)
    print(f"  pair == two separate calls: {'ok' if ok else 'FAIL'}")
    return ok


def main():
    torch.manual_seed(0)
    print("Mamba-3 torch op vs captured official-kernel ground truth:")
    ok, worst = check_goldens()
    print(f"  worst: {worst:.2f} bf16 ULP (bound {MAX_ULPS})\n")
    ok = check_compile() and ok
    ok = check_reverse() and ok
    ok = check_pair() and ok
    print("\n" + ("MAMBA-3 OP CHECK: PASS" if ok else "MAMBA-3 OP CHECK: FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
