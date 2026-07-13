"""End-to-end benchmark: HF Mamba generate() with and without the kernel.

Measures, for a real model on CPU:
  - prefill latency (time to first token — where the O(len) scan lives and
    the kernel engages),
  - decode throughput (tokens/s — single-token steps fall back to HF's
    own path by design, so this should be ~unchanged),
  - total generate() wall time.

Method: greedy decoding, median of --reps runs after one warmup, patched
vs unpatched in the same process (arm_scan.patch()/unpatch()), token-level
equality of the two outputs asserted per rep.

Usage:
    python bench/bench_e2e.py                       # mamba-130m defaults
    python bench/bench_e2e.py --prompt-tokens 512 --new-tokens 64 --reps 5
    python bench/bench_e2e.py --model state-spaces/mamba-370m-hf
"""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import arm_scan  # noqa: E402

BASE_TEXT = ("The Arm architecture powers most of the world's phones and an "
             "increasing share of its servers. State-space models such as "
             "Mamba promise linear-time sequence modeling on exactly that "
             "kind of hardware, provided the selective scan is fast. ")


def build_prompt(tok, n_tokens):
    text = BASE_TEXT * (n_tokens // 8 + 2)
    ids = tok(text, return_tensors="pt").input_ids[:, :n_tokens]
    assert ids.shape[1] == n_tokens, f"prompt too short: {ids.shape}"
    return {"input_ids": ids}


def timed_generate(model, inputs, new_tokens):
    """Returns (ids, prefill_s, total_s). Prefill is measured as the first
    forward pass with the full prompt (identical work to generate()'s own
    first step)."""
    with torch.no_grad():
        t0 = time.perf_counter()
        model(**inputs)
        prefill_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        ids = model.generate(
            **inputs, max_new_tokens=new_tokens, do_sample=False)
        total_s = time.perf_counter() - t0
    return ids, prefill_s, total_s


def run_side(model, inputs, new_tokens, reps):
    ids_ref = None
    prefill, total = [], []
    for i in range(reps + 1):  # +1 warmup
        ids, p, t = timed_generate(model, inputs, new_tokens)
        if i == 0:
            ids_ref = ids
            continue
        assert torch.equal(ids, ids_ref), "nondeterministic generation"
        prefill.append(p)
        total.append(t)
    return ids_ref, {
        "prefill_median_s": statistics.median(prefill),
        "total_median_s": statistics.median(total),
        "decode_tok_per_s": new_tokens / statistics.median(
            [t - p for p, t in zip(prefill, total)]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="state-spaces/mamba-130m-hf")
    ap.add_argument("--prompt-tokens", type=int, default=256)
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tag", type=str, default=platform.node(),
                    help="host label embedded in the JSON (e.g. ampere-a1)")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer, MambaForCausalLM

    print(f"model={args.model} prompt={args.prompt_tokens}tok "
          f"new={args.new_tokens}tok reps={args.reps}")
    print(f"host: {platform.platform()} / {platform.machine()}, "
          f"torch {torch.__version__}, "
          f"threads {torch.get_num_threads()}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = MambaForCausalLM.from_pretrained(args.model)
    model.eval()
    inputs = build_prompt(tok, args.prompt_tokens)

    print("\n--- unpatched (HF slow path) ---")
    ids_ref, base = run_side(model, inputs, args.new_tokens, args.reps)
    print(f"  prefill {base['prefill_median_s']*1e3:8.1f} ms   "
          f"decode {base['decode_tok_per_s']:6.2f} tok/s   "
          f"total {base['total_median_s']:.3f} s")

    print("\n--- patched (arm_scan kernel) ---")
    arm_scan.patch()
    ids_fast, fast = run_side(model, inputs, args.new_tokens, args.reps)
    stats = arm_scan.stats()
    arm_scan.unpatch()
    print(f"  prefill {fast['prefill_median_s']*1e3:8.1f} ms   "
          f"decode {fast['decode_tok_per_s']:6.2f} tok/s   "
          f"total {fast['total_median_s']:.3f} s")
    print(f"  engagement: {stats}")

    same = torch.equal(ids_ref, ids_fast)
    print(f"\ngreedy tokens identical patched vs unpatched: {same}")
    assert stats["fast_calls"] > 0, "kernel never engaged"

    speed_prefill = base["prefill_median_s"] / fast["prefill_median_s"]
    speed_total = base["total_median_s"] / fast["total_median_s"]
    print(f"speedup: prefill {speed_prefill:.2f}x, "
          f"end-to-end generate {speed_total:.2f}x "
          f"(prompt {args.prompt_tokens} + {args.new_tokens} new)")

    if args.json:
        import subprocess
        try:
            sha = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            sha = "unknown"
        out = {
            "kind": "e2e",
            "model": args.model,
            "prompt_tokens": args.prompt_tokens,
            "new_tokens": args.new_tokens,
            "unpatched": base,
            "patched": fast,
            "tokens_identical": same,
            "engagement": stats,
            "host": platform.platform(),
            "machine": platform.machine(),
            "tag": args.tag,
            "git_sha": sha,
            "timestamp_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"results written to {args.json}")


if __name__ == "__main__":
    main()
