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

    # on the GPU box — NOTE: install from git, not PyPI (see below)
    MAMBA_SKIP_CUDA_BUILD=TRUE pip install --no-build-isolation \
        "git+https://github.com/state-spaces/mamba.git@main"
    python tools/capture_mamba3_goldens.py --out tests/golden/mamba3

    # then, on any CPU
    python tests/verify_golden_mamba3.py

Also captures full-model logits for a fixed prompt, so end-to-end parity can be
checked later without re-running anything on a GPU.

INSTALL FROM GIT, NOT PyPI — two reasons, both load-bearing
-----------------------------------------------------------
1. `mamba-ssm 2.3.2.post1` (PyPI, 2026-05-09) rejects the published Mamba-3
   checkpoints outright: `mixer_seq_simple.create_block` only accepts
   `["Mamba1", "Mamba2"]`, while every `mamba3-*` config declares
   `ssm_cfg.layer = "Mamba3"`. `main` added the `ssm_layer_map` that fixes it.
2. **More seriously**, upstream PR #997 (merged 2026-07-22, i.e. AFTER that
   release) fixes *silent forward-pass corruption* in `mamba3_siso_fwd_kernel`
   on Blackwell (SM100/103/120): `num_stages` of 2 or 3 in the Triton autotuner
   returns wrong outputs with no error. Capturing ground truth through that is
   the worst possible failure — every downstream Rust kernel would be validated
   against wrong numbers and look green. The released wheel does NOT contain
   the fix; `main` does. This script refuses to run without it on Blackwell —
   see `_has_blackwell_fix`, which inspects the source rather than the version
   string, because patched and unpatched installs both report 2.3.2.post1.

DTYPE, AND WHY IT DECIDES EVERY DOWNSTREAM TOLERANCE
----------------------------------------------------
The model runs in **fp32** by default, but the kernel is internally MIXED
precision and there is no flag to change it: `Q/K/V/Trap/Angles/Z` are cast to
bf16 on entry, `ADT/DT` stay fp32 "for stability", and `Q_bias/K_bias/D` stay
fp32 as model parameters. The output is bf16.

Two consequences, both load-bearing for every stage after this one:

1. We record inputs **post-cast** (see `_BF16_INPUTS`), so a CPU reference is
   fed exactly what the kernel was fed. Recording the pre-cast fp32 values
   would make the reference diverge in a way that compounds over the sequence
   and reads as a kernel bug.
2. The plan's Stage-1 gate — "reference reproduces every golden to < 1e-4 at
   f64" — **cannot hold against a bf16 output**, whose relative epsilon is
   ~0.4%. The honest gate is: round the f64 reference to bf16 and require
   agreement to ~1 ULP of bf16. That is a *tighter* test than a loose 1e-2
   absolute bound, not a weaker one.

The manifest records the true per-tensor dtype for every case, so no downstream
gate has to guess.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# The sweep seeds itself with the same stable, sha256-derived per-case seed the
# Mamba-1 goldens draw from, so a sweep case redraws identically on any box.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from golden_inputs import case_seed  # noqa: E402


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


def _has_blackwell_fix():
    """True if this mamba_ssm contains upstream PR #997.

    Checked by source inspection rather than version number, because the fix
    landed on `main` after the last PyPI release, so the version string cannot
    distinguish a patched install from an unpatched one — both report
    2.3.2.post1.
    """
    try:
        from mamba_ssm.ops.triton.mamba3 import mamba3_siso_fwd as m
        src = Path(m.__file__).read_text(encoding="utf-8")
        return "_prune_mamba3_siso_fwd_configs" in src
    except Exception:  # noqa: BLE001
        return False


def _is_blackwell():
    major, _ = torch.cuda.get_device_capability(0)
    return major in (10, 12)


# `mamba3_siso_combined` hard-casts these to bf16 the moment it is entered,
# whatever you hand it, and leaves ADT/DT fp32 ("for stability") and
# Q_bias/K_bias/D fp32 ("model parameters") — upstream's own comment.
#
# That matters enormously for a GOLDEN. If we record the fp32 tensors we passed
# in, we record inputs the kernel never saw: it computed on their bf16
# roundings. A CPU reference fed the fp32 values would then diverge from the
# recorded output by an amount that COMPOUNDS through the recurrence (2048
# steps in the long case), and the mismatch would look like a kernel bug rather
# than a capture artifact.
#
# So we apply the same cast ourselves BEFORE the call. Semantically a no-op —
# the kernel would do it anyway — but it makes "recorded input" and "consumed
# input" the same tensor, which is the entire point of ground truth.
_BF16_INPUTS = ("Q", "K", "V", "Trap", "Angles", "Z")


def _cast_list_still_matches():
    """Guard against upstream changing which tensors it downcasts.

    If this ever returns False the goldens are silently wrong again, so it is
    checked at capture time rather than trusted.
    """
    try:
        from mamba_ssm.ops.triton import mamba3 as _m3
        src = (Path(_m3.__file__).parent
               / "mamba3_siso_combined.py").read_text(encoding="utf-8")
        cast = {ln.split("=")[0].strip()
                for ln in src.splitlines() if ".to(torch.bfloat16)" in ln}
        return cast == set(_BF16_INPUTS), sorted(cast)
    except Exception as exc:  # noqa: BLE001
        return False, [f"unreadable: {exc}"]


class Capture:
    """Wrap a kernel entry point and record every call's tensors."""

    def __init__(self, fn, limit=64):
        self.fn, self.limit, self.calls = fn, limit, []

    def __call__(self, *args, **kwargs):
        for k in _BF16_INPUTS:
            if k in kwargs and torch.is_tensor(kwargs[k]):
                kwargs[k] = kwargs[k].to(torch.bfloat16)
        if args:
            print("  WARNING: positional args seen; the bf16 pre-cast only "
                  "covers keyword arguments, so those inputs may not match "
                  "what the kernel consumed.")
        out = self.fn(*args, **kwargs)
        if len(self.calls) < self.limit:
            # numpy has no bfloat16, so bf16 tensors must be widened to f32 to
            # be storable. Record the ORIGINAL dtype alongside: a bf16 golden
            # can only ever be reproduced to bf16 precision, and a downstream
            # gate that does not know this will either be vacuous or unpassable.
            dtypes = {}

            def snap(v, name):
                if torch.is_tensor(v):
                    dtypes[name] = str(v.dtype).replace("torch.", "")
                    return v.detach().float().cpu().numpy()
                return v
            self.calls.append({
                "args": [snap(a, f"arg{j}") for j, a in enumerate(args)],
                "kwargs": {k: snap(v, f"kw_{k}") for k, v in kwargs.items()},
                "out": (tuple(snap(o, f"out{j}") for j, o in enumerate(out))
                        if isinstance(out, tuple) else snap(out, "out")),
                "dtypes": dtypes,
            })
        return out


def install_block_hooks(model, limit=2):
    """Hook every Mamba3 BLOCK to record its input and output tensors.

    The scan-level goldens isolate the recurrence, which is what the *kernel*
    is validated against. This records one level out — the whole mixer, from
    hidden states in to hidden states out — which is what Path A's plain-PyTorch
    reimplementation is validated against.

    Why both, rather than just the end-to-end logits: a logits mismatch says
    "something is wrong somewhere in twelve layers". A layer-0 block mismatch
    says which of `in_proj`, the heavy-tail `A`, the softplus `dt`, the B/C
    norms, the angle pre-pass or `out_proj` is wrong. That is the difference
    between an afternoon and a week — the same lesson Stage 1 taught when the
    RoPE convention was wrong and only a boundary-level oracle localised it.

    Call before the forward pass; pass the result to `save_block_io`. Returns
    `(recorded, handles)`, and the caller MUST remove the handles.
    """
    try:
        from mamba_ssm.modules.mamba3 import Mamba3
    except Exception as exc:  # noqa: BLE001
        print(f"  (block hook unavailable: {exc})")
        return [], []

    recorded, handles = [], []

    def hook(mod, inp, out):
        if len(recorded) >= limit:
            return
        t_in = inp[0] if isinstance(inp, tuple) else inp
        t_out = out[0] if isinstance(out, tuple) else out
        if torch.is_tensor(t_in) and torch.is_tensor(t_out):
            recorded.append({
                "hidden_in": t_in.detach().float().cpu().numpy(),
                "hidden_out": t_out.detach().float().cpu().numpy(),
                "layer_idx": getattr(mod, "layer_idx", len(recorded)),
                "in_dtype": str(t_in.dtype),
                "out_dtype": str(t_out.dtype),
            })

    for m in model.modules():
        if isinstance(m, Mamba3):
            handles.append(m.register_forward_hook(hook))
    if not handles:
        print("  (no Mamba3 modules found to hook)")
    return recorded, handles


def _pack_param(p):
    """Store a parameter at its TRUE precision, not the dtype it was loaded at.

    The published checkpoints are bfloat16 on disk; we load them as fp32 so the
    capture runs in fp32. Writing those upcast fp32 arrays into the golden
    would double the bytes committed to git forever while adding exactly zero
    information, and would imply a precision the weights do not have.

    So round-trip through bf16 and keep that when it is exact — which it is for
    anything that started life as bf16. numpy has no bfloat16, so the bits ride
    as int16 and `_unpack_param` reverses it. Losslessness is CHECKED rather
    than assumed: a genuinely-fp32 parameter falls back to fp32.
    """
    f32 = p.detach().float().cpu()
    bf16 = f32.to(torch.bfloat16)
    if torch.equal(bf16.float(), f32):
        return bf16.view(torch.int16).numpy(), True
    return f32.numpy(), False


def save_block_io(recorded, model, outdir, manifest):
    """Write block-level goldens, each with the layer's own weights alongside.

    The weights are stored WITH the activations rather than left to be loaded
    from the checkpoint separately, so the gate is self-contained: a mismatch
    cannot be blamed on having loaded the wrong tensor into the wrong slot,
    which is the single most likely Path A bug. It also keeps the gate runnable
    in CI without a 357 MB checkpoint download.
    """
    if not recorded:
        return 0
    blocks = [m for m in model.modules()
              if type(m).__name__ == "Mamba3"]
    for i, rec in enumerate(recorded):
        arrays = {"hidden_in": rec["hidden_in"], "hidden_out": rec["hidden_out"]}
        bf16_params = []
        if i < len(blocks):
            for pname, p in blocks[i].named_parameters(recurse=True):
                packed, is_bf16 = _pack_param(p)
                arrays[f"param_{pname}"] = packed
                if is_bf16:
                    bf16_params.append(pname)
        name = f"block_io_{i:02d}"
        np.savez_compressed(outdir / f"{name}.npz", **arrays)
        manifest.setdefault("block_io", []).append({
            "name": name,
            "layer_idx": int(rec["layer_idx"]),
            "in_dtype": rec["in_dtype"],
            "out_dtype": rec["out_dtype"],
            # Which param_* arrays are int16-encoded bf16 rather than fp32.
            # The reader needs this; it cannot be inferred from the npz alone.
            "bf16_params": sorted(bf16_params),
            "arrays": {k: list(v.shape) for k, v in arrays.items()},
        })
        mb = (outdir / f"{name}.npz").stat().st_size / 1e6
        print(f"  {name}: layer {rec['layer_idx']}, "
              f"in{list(rec['hidden_in'].shape)} -> "
              f"out{list(rec['hidden_out'].shape)}, "
              f"{len(arrays) - 2} params "
              f"({len(bf16_params)} as bf16), {mb:.2f} MB")
    return len(recorded)


def record_reference_noise(logits, logits_repeat, previous, manifest):
    """How much does the official GPU model disagree with ITSELF?

    Two questions, and they have different answers — which is the whole point:

      WITHIN one process   two forward passes are BIT-IDENTICAL. So the scan's
                           reduction order is fixed once chosen, and there are
                           no non-deterministic atomics.
      ACROSS processes     they are NOT. Observed between two invocations:
                           5/256 argmax positions flipped and logits moved by
                           2.6e-3 relative.

    Taken together those localise the cause: the kernel is `triton.autotune`d,
    so the config is chosen by TIMING candidate variants at first call. A
    differently-loaded machine picks a different chunking, which changes the
    summation order, which moves the last bits. Deterministic within a run,
    not across them.

    Why this is recorded rather than merely noted: it bounds the accuracy claim
    upstream of everything else. A CPU reimplementation "disagreeing with
    ground truth on 4 tokens" reads as a defect right up until you learn that
    ground truth disagrees with ITSELF on 5. `check_mamba3_model.py` cites this
    so the gate is set against the achievable floor, not against a 100% that no
    implementation — including the reference — can reach.
    """
    same_proc = np.array_equal(logits, logits_repeat)
    entry = {
        "within_process_bit_identical": bool(same_proc),
        "note": "within-process: two forwards in ONE run. across-process: "
                "this run vs the previously captured model_forward.npz. The "
                "second is the real reproducibility floor; see the docstring "
                "for why they differ (triton.autotune picks by timing).",
    }
    print(f"  reference self-consistency (within process): "
          f"{'bit-identical' if same_proc else 'DIFFERS'}")

    if previous is not None:
        am_prev = previous["argmax"]
        am_now = logits.argmax(-1).astype(am_prev.dtype)
        n_diff = int((am_prev != am_now).sum())
        sub = logits[..., previous["vocab_subset"]]
        d = np.abs(sub - previous["logits_subset"])
        rng = float(np.ptp(previous["logits_subset"]))
        entry.update({
            "across_process_argmax_agreement": float((am_prev == am_now).mean()),
            "across_process_argmax_disagreements": n_diff,
            "positions": int(am_now.size),
            "across_process_max_abs_logit_delta": float(d.max()),
            "across_process_max_rel_logit_delta": float(d.max() / max(rng, 1e-30)),
            "logit_subset_range": rng,
        })
        print(f"  reference self-consistency (across processes): "
              f"{float((am_prev == am_now).mean()):.4%} argmax "
              f"({n_diff}/{am_now.size} differ), max rel logit delta "
              f"{d.max() / max(rng, 1e-30):.2e}")
    else:
        print("  (no previous model_forward.npz — across-process floor "
              "unmeasured this run; re-run to establish it)")
    manifest["reference_self_consistency"] = entry


def dump_model_shape(model, outdir, manifest):
    """Record config and every parameter's name/shape.

    Stage 6 has to rebuild this block in plain PyTorch and load the published
    weights into it. Guessing parameter names from a paper is miserable; this
    makes it mechanical.
    """
    params = {n: list(p.shape) for n, p in model.named_parameters()}
    buffers = {n: list(b.shape) for n, b in model.named_buffers()}
    cfg = {}
    for attr in ("config", "cfg"):
        c = getattr(model, attr, None)
        if c is not None:
            cfg = (c.__dict__ if hasattr(c, "__dict__")
                   else dict(c) if isinstance(c, dict) else str(c))
            break
    (Path(outdir) / "model_shape.json").write_text(json.dumps(
        {"config": cfg, "parameters": params, "buffers": buffers}, indent=2,
        default=str))
    manifest["n_parameters"] = len(params)
    print(f"  model_shape.json: {len(params)} params, {len(buffers)} buffers")


def save_model_forward(outdir, ids, logits, manifest, subset=512, seed=0):
    """Store an end-to-end parity artifact WITHOUT the full logit tensor.

    The full thing is (1, 256, 128256) f32 = 131 MB raw, 58 MB compressed —
    past GitHub's 50 MB warning threshold and ~20x every other golden in this
    repo combined, committed to history forever. It is also far more than the
    Stage-6 check needs.

    Three complementary views, ~0.5 MB total, and together they are a STRONGER
    test than a raw slice would be:

      argmax       - the discrete prediction. Catches any error big enough to
                     change the token, which is what "runs the real model"
                     actually means.
      logits_subset- exact values at a fixed pseudo-random set of vocab ids,
                     so continuous drift is measurable, not just rank order.
      logsumexp    - depends on ALL 128k logits, so an error confined to
                     vocab ids outside the subset still shows up here.
    """
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(logits.shape[-1], size=subset, replace=False))
    m = logits.max(axis=-1, keepdims=True)
    lse = (m + np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))).squeeze(-1)
    np.savez_compressed(
        outdir / "model_forward.npz",
        input_ids=ids.astype(np.int32),
        argmax=logits.argmax(axis=-1).astype(np.int32),
        vocab_subset=idx.astype(np.int32),
        logits_subset=logits[..., idx].astype(np.float32),
        logsumexp=lse.astype(np.float32),
    )
    manifest["model_forward"] = {
        "seq": int(ids.shape[-1]), "vocab": int(logits.shape[-1]),
        "subset_size": int(subset), "subset_seed": seed,
        "stored": ["input_ids", "argmax", "vocab_subset", "logits_subset",
                   "logsumexp"],
        "note": "full logits deliberately not stored; see save_model_forward",
    }
    size_mb = (outdir / "model_forward.npz").stat().st_size / 1e6
    print(f"  model_forward.npz: argmax + {subset}-vocab subset + logsumexp "
          f"({size_mb:.2f} MB)")


# (batch, seqlen, d_model, d_state, headdim) — chosen for EDGES, because the
# model-driven capture only ever produces one shape (its own), and a golden
# suite that covers a single shape cannot catch a tail/edge bug. Mirrors what
# tests/gen_golden.py does for Mamba-1 (edge_L1, state13_neon_tail, ...).
# Kept deliberately NARROW (small d_model / d_state / headdim): these exist to
# exercise edges, not to be big. A golden's cost is committed to git history
# forever, and the long case at the model's own width would be ~40 MB on its
# own — more than every other golden in this repo combined.
SWEEP = [
    ("edge_L1",        1,    1, 256,  64, 32),   # single timestep (decode edge)
    ("short_L63",      1,   63, 256,  64, 32),   # < chunk_size, not a multiple
    ("chunk_L64",      1,   64, 256,  64, 32),   # exactly one chunk
    ("odd_L255",       1,  255, 256, 128, 32),   # chunk tail: 3 chunks + 63
    ("batch2_L128",    2,  128, 256,  64, 32),   # batch > 1
    ("long_L2048",     1, 2048, 256,  64, 32),   # the PR#997 Blackwell regime
]


def sweep_shapes(caps, dtype, manifest):
    """Drive fresh Mamba3 blocks across edge shapes.

    Each case is independently guarded: a shape that upstream cannot handle
    (e.g. #985, seqlen=1) must not abort the whole capture — the GPU session is
    the expensive resource. Failures are recorded in the manifest rather than
    swallowed, so a missing shape is visible instead of merely absent.

    Seeding note: this used `abs(hash(name))`, and Python salts `str.__hash__`
    per process unless PYTHONHASHSEED is set — so every run drew different
    weights and activations, and re-running produced a numerically different
    (though self-consistent) golden set. The model-driven cases were unaffected
    because they load fixed checkpoint weights, which is why only the sweep
    cases moved. `case_seed` is sha256-based and stable across processes,
    interpreters and architectures.
    """
    from mamba_ssm.modules.mamba3 import Mamba3
    manifest["sweep"] = []
    for name, b, L, d_model, d_state, headdim in SWEEP:
        before = sum(len(c.calls) for _, _, c in caps)
        try:
            torch.manual_seed(case_seed(name))
            blk = Mamba3(d_model=d_model, d_state=d_state, headdim=headdim,
                         is_mimo=False).to(dtype).cuda().eval()
            with torch.no_grad():
                blk(torch.randn(b, L, d_model, device="cuda", dtype=dtype))
            gained = sum(len(c.calls) for _, _, c in caps) - before
            status = "ok" if gained else "no kernel call recorded"
            print(f"  sweep {name:14s} b{b} L{L:<5d} d{d_model} "
                  f"n{d_state} h{headdim}: {status}")
            manifest["sweep"].append(
                {"name": name, "batch": b, "seqlen": L, "d_model": d_model,
                 "d_state": d_state, "headdim": headdim, "status": status})
        except Exception as exc:  # noqa: BLE001
            print(f"  sweep {name:14s} FAILED: {type(exc).__name__}: "
                  f"{str(exc)[:70]}")
            manifest["sweep"].append(
                {"name": name, "batch": b, "seqlen": L, "d_model": d_model,
                 "d_state": d_state, "headdim": headdim,
                 "status": f"FAILED: {type(exc).__name__}: {str(exc)[:200]}"})
        finally:
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/golden/mamba3")
    ap.add_argument("--model", default="state-spaces/mamba3-siso-187m")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--max-calls", type=int, default=4,
                    help="MODEL layer calls to record (one per layer). Low by "
                         "default: every layer has the same shape, so extra "
                         "layers add megabytes but no coverage. Shape "
                         "diversity comes from the sweep instead")
    ap.add_argument("--skip-model", action="store_true",
                    help="capture from a freshly-built block only")
    ap.add_argument("--max-blocks", type=int, default=2,
                    help="MIXER-boundary goldens to record (see "
                         "install_block_hooks). Each carries that layer's "
                         "full parameter set, so these are the largest files "
                         "here; 2 is enough to catch a layer-dependent bug")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16", "float16"],
                    help="capture precision. fp32 keeps the downstream <1e-4 "
                         "gate meaningful; bf16 caps it at ~0.4%% relative")
    ap.add_argument("--no-sweep", action="store_true",
                    help="skip the edge-shape sweep (model layers only)")
    ap.add_argument("--vocab-subset", type=int, default=512,
                    help="vocab ids kept in model_forward.npz (see "
                         "save_model_forward for why not all of them)")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)

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

    ok, cast = _cast_list_still_matches()
    print(f"\nkernel downcasts to bf16: {cast}")
    if not ok:
        raise SystemExit(
            f"REFUSING TO CAPTURE. Upstream's bf16 downcast set is {cast}, but "
            f"this script mirrors {sorted(_BF16_INPUTS)}. Recorded inputs would "
            "not be the ones the kernel consumed, so the goldens would be "
            "quietly wrong. Update _BF16_INPUTS to match, then re-run.")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {"device": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "capture_dtype": args.dtype,
                "mamba_ssm": getattr(
                    __import__("mamba_ssm"), "__version__", "unknown"),
                "blackwell_fix_997": _has_blackwell_fix(),
                "cases": []}
    if _is_blackwell() and not manifest["blackwell_fix_997"]:
        raise SystemExit(
            "REFUSING TO CAPTURE. This mamba_ssm lacks upstream PR #997, which "
            "fixes SILENT forward-pass corruption in mamba3_siso_fwd_kernel on "
            "Blackwell (SM100/103/120). Ground truth captured without it can be "
            "wrong with no error raised, and every downstream kernel would then "
            "be validated against garbage.\n"
            "Fix: MAMBA_SKIP_CUDA_BUILD=TRUE pip install --no-build-isolation "
            "--force-reinstall --no-deps "
            "'git+https://github.com/state-spaces/mamba.git@main'")

    # --- capture at the kernel boundary --------------------------------
    import importlib
    caps = []
    for mod, attr, fn in found:
        m = importlib.import_module(mod)
        cap = Capture(fn, limit=args.max_calls)
        setattr(m, attr, cap)
        caps.append((mod, attr, cap))

    # --- drive it with the real model ----------------------------------
    if not args.skip_model:
        try:
            from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
            print(f"\nloading {args.model} (dtype {args.dtype}) ...")
            model = MambaLMHeadModel.from_pretrained(
                args.model, device="cuda", dtype=dtype)
            model.eval()
            torch.manual_seed(0)
            ids = torch.randint(0, 1000, (1, args.seq), device="cuda")
            # Block-level hooks must be installed BEFORE the forward. They
            # record the mixer boundary, which is the oracle Path A's
            # reimplementation is gated against — see install_block_hooks.
            recorded, handles = install_block_hooks(model, limit=args.max_blocks)
            try:
                with torch.no_grad():
                    logits = model(ids).logits.float().cpu().numpy()
            finally:
                for h in handles:
                    h.remove()
            manifest["model"] = args.model
            # Read the PREVIOUS capture before overwriting it: comparing this
            # run against it is the only way to measure the across-process
            # reproducibility floor, and it is the floor that bounds the
            # accuracy claim. See record_reference_noise.
            prev_path = outdir / "model_forward.npz"
            previous = dict(np.load(prev_path)) if prev_path.is_file() else None
            with torch.no_grad():
                logits_repeat = model(ids).logits.float().cpu().numpy()
            record_reference_noise(logits, logits_repeat, previous, manifest)
            save_model_forward(outdir, ids.cpu().numpy(), logits, manifest,
                               subset=args.vocab_subset)
            if recorded:
                print("\nblock-level goldens (mixer boundary, for Path A):")
                save_block_io(recorded, model, outdir, manifest)
            # Stage 6 has to rebuild this block in plain PyTorch and load the
            # published weights into it; without this file that is guesswork.
            dump_model_shape(model, outdir, manifest)
            del model
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            print(f"\nmodel load failed ({type(exc).__name__}: {exc})")
            print("continuing with block-level capture only")
            manifest["model_load_error"] = f"{type(exc).__name__}: {exc}"

    # --- edge shapes the model itself never exercises -------------------
    # The model contributes many cases but all at ONE shape (its own). A golden
    # suite covering a single shape cannot catch a chunk-tail or edge bug, so
    # sweep deliberately-awkward shapes too.
    if not args.no_sweep:
        # --max-calls bounds the MODEL layers; the sweep needs headroom beyond
        # it or the shapes it exists to capture would be silently dropped.
        for _, _, cap in caps:
            cap.limit = len(cap.calls) + len(SWEEP) + 2
        print("\nedge-shape sweep:")
        sweep_shapes(caps, dtype, manifest)

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
                # The true on-GPU dtype per tensor. Everything is STORED as f32
                # (numpy has no bfloat16); without this a downstream gate cannot
                # know whether 1e-4 or 1e-2 is the honest tolerance.
                "dtypes": call.get("dtypes", {}),
            })
            n += 1
            print(f"  {name}: " + ", ".join(
                f"{k}{list(v.shape)}" for k, v in arrays.items()))

    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total_mb = sum(f.stat().st_size for f in outdir.iterdir()) / 1e6
    print(f"\nwrote {n} golden case(s) + manifest.json to {outdir} "
          f"({total_mb:.1f} MB total)")
    if n == 0:
        raise SystemExit(
            "No kernel calls were captured. The scan may be dispatched under "
            "a different name; print the traceback inside Mamba3.forward to "
            "find the real entry point.")

    # --- exit gate, checked rather than assumed -------------------------
    # The plan's Stage-0 gate is ">=6 cases with inputs AND outputs, plus
    # model_forward.npz". A run that quietly misses it must not read as success:
    # everything downstream is validated against whatever this wrote.
    shapes = {tuple(c["arrays"].get("out", [])) for c in manifest["cases"]}
    problems = []
    if n < 6:
        problems.append(f"only {n} golden cases (gate: >= 6)")
    if len(shapes) < 2:
        problems.append(
            f"all {n} cases share one output shape — no edge coverage")
    for f in ("model_forward.npz", "model_shape.json"):
        if not (outdir / f).exists():
            problems.append(f"{f} missing")
    if problems:
        print("\n*** STAGE 0 EXIT GATE NOT MET ***")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print(f"\nSTAGE 0 EXIT GATE: PASS — {n} cases across {len(shapes)} "
          f"distinct output shapes, model_forward.npz + model_shape.json "
          f"present, dtype {args.dtype}, PR#997 fix "
          f"{'present' if manifest['blackwell_fix_997'] else 'N/A'}")
    print("\nCommit these files. The GPU is not needed again — the Rust "
          "kernel is built and validated on CPU against them.")


if __name__ == "__main__":
    main()
