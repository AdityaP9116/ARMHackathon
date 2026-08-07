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


def render_ss2d(doc, path):
    """2D cross-scan at the diffusion workload's real grids."""
    lines = [
        f"### ss2d (2D cross-scan) — `{path.name}`",
        "",
        f"- host: {doc.get('host', '?')} ({doc.get('machine', '?')}), torch "
        f"{doc.get('torch', '?')} ({doc.get('threads', '?')} threads), "
        f"reps={doc.get('reps', '?')}",
        "",
        "Traversal-pair path vs the legacy four-forward-scans formulation. "
        "**Same kernel on both sides**, so the ratio is attributable to the "
        "restructuring, not the backend.",
        "",
        "| case | pair ms | legacy ms | pair× | scan× | non-scan % | eager ms "
        "| compile ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in doc.get("cases", []):
        comp = "—"
        if c.get("ref_compile_total_s"):
            comp = f"{c['ref_compile_total_s'] * 1e3:.1f}"
        elif c.get("ref_compile_skipped"):
            comp = "skipped"
        elif c.get("ref_compile_error"):
            comp = "failed"
        eager = (f"{c['ref_eager_total_s'] * 1e3:.1f}"
                 if c.get("ref_eager_total_s")
                 else (f"{c['ref_total_s'] * 1e3:.1f}"
                       if c.get("ref_total_s") else "—"))
        # Results predating the traversal-pair rewrite have no legacy/pair
        # columns. Render them rather than failing — an old JSON is still
        # data, and a renderer should not be the thing that discards it.
        def _ms(key):
            return f"{c[key] * 1e3:.1f}" if key in c else "—"

        def _x(key, bold=False):
            if key not in c:
                return "—"
            return (f"**{c[key]:.2f}×**" if bold else f"{c[key]:.2f}×")

        lines.append(
            f"| `{c['case']}` | {_ms('arm_total_s')} "
            f"| {_ms('legacy_total_s')} "
            f"| {_x('pair_speedup_total', bold=True)} "
            f"| {_x('pair_speedup_scan')} "
            f"| {c.get('overhead_pct', float('nan')):.1f}% | {eager} "
            f"| {comp} |")

    if "pair_speedup_geomean" in doc:
        lines += [
            "",
            f"- traversal-pair rewrite: **{doc['pair_speedup_geomean']:.2f}× "
            f"geomean** on the production shapes (block total)",
        ]
    if "fused_kernel_justified" in doc:
        verdict = ("**JUSTIFIED**" if doc["fused_kernel_justified"]
                   else "**not justified**")
        lines.append(
            f"- fully fused `selective_scan_2d` (P1-7): {verdict} by the "
            f"15% non-scan-overhead rule")
    return lines + [""]


def render_bidi(doc, path):
    """Fused bidirectional scan: the exp-sharing win."""
    env = doc.get("env", {})
    lines = [
        f"### bidirectional `{doc.get('suite', '?')}` — `{path.name}`",
        "",
        f"- host: {env.get('platform', '?')} ({env.get('machine', '?')}, "
        f"{env.get('cpu_count', '?')} cpus), torch {env.get('torch', '?')}, "
        f"reps={doc.get('reps', '?')}",
        "",
        "| shape B,D,L,N | fused ms | two-call ms | eager ms | compile ms "
        "| ×eager | ×compile | exp-sharing |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in doc.get("shapes", []):
        t = row.get("timings", {})
        comp = t.get("ref_compile_bidi", {})
        comp_ms = fmt_ms(comp)
        if "compile_s" in comp:
            comp_ms += f" (compile {comp['compile_s']:.0f}s)"
        xc = (f"{row['speedup_vs_compile']:.2f}×"
              if "speedup_vs_compile" in row else "—")
        lines.append(
            f"| {','.join(map(str, row.get('shape', [])))} "
            f"| {fmt_ms(t.get('bidirectional'))} "
            f"| {fmt_ms(t.get('bidirectional_twocall'))} "
            f"| {fmt_ms(t.get('ref_eager_bidi'))} | {comp_ms} "
            f"| {row.get('speedup_vs_eager', float('nan')):.2f}× | {xc} "
            f"| **{row.get('exp_sharing_speedup', float('nan')):.2f}×** |")
    lines += [
        "",
        "`exp-sharing` is fused-vs-two-call: Pass A (discretize + `exp`) "
        "computed once instead of per direction.",
        "",
    ]
    return lines


def render_diffusion(doc, path):
    """Diffusion denoiser latency and cost per reconstruction."""
    lines = [
        f"### diffusion app — `{path.name}`",
        "",
        f"- host: {doc.get('host', '?')} ({doc.get('machine', '?')}), torch "
        f"{doc.get('torch', '?')} ({doc.get('threads', '?')} threads), "
        f"reps={doc.get('reps', '?')}",
        f"- prior: `{doc.get('prior', 'untrained (timing only)')}`",
        "",
        "| grid | params | per-NFE s | peak RSS MB | NFE=18 | NFE=69 "
        "| NFE=256 |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in doc.get("cases", []):
        proj = c.get("projected_s", {})
        # peak_rss_mb is explicitly null where the platform cannot report it
        # (no `resource` module on Windows), so `.get(k, 0)` is not enough —
        # the key exists and its value is None.
        rss = c.get("peak_rss_mb")
        rss_s = f"{rss:.0f}" if isinstance(rss, (int, float)) else "—"
        lines.append(
            f"| {c['res']} | {c.get('params', 0) / 1e6:.1f} M "
            f"| **{c['per_nfe_s']:.3f}** | {rss_s} "
            f"| {proj.get('18', 0):.1f} s | {proj.get('69', 0):.1f} s "
            f"| {proj.get('256', 0):.0f} s |")

    if doc.get("cost"):
        cost = doc["cost"]
        lines += [
            "",
            f"**Cost per reconstruction** at ${cost['usd_per_hour']:.4f}/h "
            f"(`{cost.get('instance', '?')}`):",
            "",
            "| grid | NFE=18 | NFE=69 | NFE=256 |",
            "|---|---|---|---|",
        ]
        for c in doc.get("cases", []):
            u = c.get("usd", {})
            lines.append(
                f"| {c['res']} | ${u.get('18', 0):.4f} | ${u.get('69', 0):.4f} "
                f"| ${u.get('256', 0):.4f} |")

    if doc.get("quality"):
        lines += [
            "",
            "| R | zero-filled PSNR | recon PSNR | gain | SSIM | NMSE |",
            "|---|---|---|---|---|---|",
        ]
        for q in doc["quality"]:
            lines.append(
                f"| {q['R']} | {q['zf_psnr']:.2f} dB | {q['psnr']:.2f} dB "
                f"| {q['psnr'] - q['zf_psnr']:+.2f} dB | {q['ssim']:.4f} "
                f"| {q['nmse']:.4f} |")
    else:
        lines += ["", "_Quality rows need a trained prior "
                      "(`--checkpoint`); timing above is prior-independent._"]
    return lines + [""]


def render_unknown(doc, path):
    """Never lose a session to an unrecognised file."""
    return [
        f"### unrecognised result — `{path.name}`",
        "",
        f"- `kind` = `{doc.get('kind', '(absent)')}`, top-level keys: "
        f"`{', '.join(sorted(doc)[:12])}`",
        "- No renderer matched. The JSON is intact; add a renderer in "
        "`render_results.py` rather than re-running the benchmark.",
        "",
    ]


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
    # Dispatch by `kind`. Everything is wrapped: a benchmark session is
    # expensive and must never be lost to a rendering bug, so a malformed or
    # unrecognised file degrades to a note instead of taking RESULTS.md down.
    renderers = {
        "e2e": render_e2e,
        "op": render_op,
        "ss2d": render_ss2d,
        "bidirectional-fused-exp-sharing": render_bidi,
        "diffusion": render_diffusion,
    }
    for tag in sorted(by_tag):
        lines += [f"## host tag: `{tag}`", ""]
        for path, doc in by_tag[tag]:
            kind = doc.get("kind") or ("e2e" if "unpatched" in doc else "op")
            fn = renderers.get(kind, render_unknown)
            try:
                lines += fn(doc, path)
            except Exception as exc:  # noqa: BLE001 - never lose the run
                lines += [
                    f"### FAILED TO RENDER — `{path.name}`",
                    "",
                    f"- `kind` = `{kind}`, renderer raised "
                    f"`{type(exc).__name__}: {exc}`",
                    "- The JSON is intact; fix the renderer, re-run this "
                    "script. Do not re-run the benchmark.",
                    "",
                ]
                print(f"  ! {path.name}: {type(exc).__name__}: {exc}")

    criterion = sorted(RESULTS_DIR.glob("criterion_*.txt"))
    if criterion:
        lines += ["## raw criterion ladders", ""]
        lines += [f"- `{p.name}`" for p in criterion] + [""]

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT} ({len(docs)} result files, "
          f"{len(by_tag)} host tags)")


if __name__ == "__main__":
    main()
