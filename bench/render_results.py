"""Render every bench/results/*.json into bench/results/RESULTS.md.

No number in RESULTS.md is ever hand-copied: this script is the single
path from raw benchmark JSON to presentable tables. Re-run it after any
benchmark session; it regenerates the whole file from whatever JSONs are
present, grouped by host tag.

Usage: python bench/render_results.py
"""

import json
import time
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT = RESULTS_DIR / "RESULTS.md"


def fmt_ms(entry):
    if not entry or "median_s" not in entry:
        return "—"
    return f"{entry['median_s'] * 1e3:.2f}"


def render_op(doc, path):
    env = doc["env"]
    lines = [
        f"### op `{doc.get('suite', '?')}` — `{path.name}`",
        "",
        f"- host: {env['platform']} ({env['machine']}, "
        f"{env['cpu_count']} cpus), torch {env['torch']} "
        f"({env['torch_threads']} threads)",
        f"- git {env.get('git_sha', '?')}, {env.get('timestamp_utc', '?')}, "
        f"reps={doc['reps']}",
        "",
        "| shape B,D,L,N | eager ms | compile ms | kernel ms "
        "| ×eager | ×compile | max_abs_err |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in doc["shapes"]:
        b = row["baselines"]
        kern = b.get("kernel", {})
        eager = b.get("ref_eager", {})
        comp = b.get("ref_compile", {})
        k = kern.get("median_s")
        x_eager = (f"{eager['median_s'] / k:.2f}×"
                   if k and "median_s" in eager else "—")
        x_comp = (f"{comp['median_s'] / k:.2f}×"
                  if k and "median_s" in comp else "—")
        comp_ms = fmt_ms(comp)
        if "compile_s" in comp:
            comp_ms += f" (compile {comp['compile_s']:.0f}s)"
        if "error" in comp:
            comp_ms = "unavailable"
        lines.append(
            f"| {','.join(map(str, row['shape']))} | {fmt_ms(eager)} "
            f"| {comp_ms} | {fmt_ms(kern)} | {x_eager} | {x_comp} "
            f"| {row.get('kernel_vs_ref_max_abs', float('nan')):.2e} |")
    return lines + [""]


def render_e2e(doc, path):
    b, f = doc["unpatched"], doc["patched"]
    pre = b["prefill_median_s"] / f["prefill_median_s"]
    tot = b["total_median_s"] / f["total_median_s"]
    return [
        f"### e2e `{doc['model']}` — `{path.name}`",
        "",
        f"- host: {doc['host']} ({doc.get('machine', '?')}), torch "
        f"{doc['torch']} ({doc.get('torch_threads', '?')} threads), git "
        f"{doc.get('git_sha', '?')}, {doc.get('timestamp_utc', '?')}",
        f"- prompt {doc['prompt_tokens']} tok + {doc['new_tokens']} new, "
        f"greedy, tokens identical: **{doc['tokens_identical']}**",
        "",
        "| | prefill ms | decode tok/s | total s |",
        "|---|---|---|---|",
        f"| unpatched | {b['prefill_median_s'] * 1e3:.1f} "
        f"| {b['decode_tok_per_s']:.2f} | {b['total_median_s']:.3f} |",
        f"| patched | {f['prefill_median_s'] * 1e3:.1f} "
        f"| {f['decode_tok_per_s']:.2f} | {f['total_median_s']:.3f} |",
        f"| **speedup** | **{pre:.2f}×** | — | **{tot:.2f}×** |",
        "",
    ]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    docs = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            docs.append((path, json.loads(path.read_text())))
        except Exception as e:
            print(f"skipping {path.name}: {e}")

    by_tag = defaultdict(list)
    for path, doc in docs:
        tag = doc.get("tag") or doc.get("env", {}).get("tag") or "untagged"
        by_tag[tag].append((path, doc))

    lines = [
        "# Benchmark results",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by "
        "`bench/render_results.py` — do not edit numbers by hand.",
        "",
        "Surface tags per BASELINE_TEST_PLAN.md: dedicated Arm hardware is "
        "headline-grade; shared CI runners are provisional; x86 hosts "
        "exercise the scalar backend only.",
        "",
    ]
    for tag in sorted(by_tag):
        lines += [f"## host tag: `{tag}`", ""]
        for path, doc in by_tag[tag]:
            kind = doc.get("kind") or ("e2e" if "unpatched" in doc else "op")
            lines += (render_e2e(doc, path) if kind == "e2e"
                      else render_op(doc, path))

    criterion = sorted(RESULTS_DIR.glob("criterion_*.txt"))
    if criterion:
        lines += ["## raw criterion ladders", ""]
        lines += [f"- `{p.name}`" for p in criterion] + [""]

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT} ({len(docs)} result files, "
          f"{len(by_tag)} host tags)")


if __name__ == "__main__":
    main()
