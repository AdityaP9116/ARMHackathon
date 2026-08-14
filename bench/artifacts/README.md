# Preserved benchmark artifacts

This directory stores immutable bundles captured from dedicated benchmark machines. Parsed,
reviewable results live in [`../results/`](../results/); these archives are retained so the raw
session can be audited without rerunning paid infrastructure.

| Artifact | Source | Contents | SHA-256 |
|---|---|---|---|
| `results-graviton.tgz` | AWS `c8g.16xlarge`, Graviton4 / Neoverse-V2, 64 vCPU | Raw benchmark JSON, logs, and Criterion output collected during the August 11, 2026 session | `32f4b2079e610b63ece7a4b6c36ca6c69446955555db1d71996fba0f1156e3a1` |

The archive contains no credentials, model weights, or dataset files.
