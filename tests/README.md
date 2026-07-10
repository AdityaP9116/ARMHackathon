# Phase 0 — Correctness Ground Truth

Everything the Rust kernel will be validated against lives here. See
`INTEGRATION_PLAN.md` (repo root) for how this fits the overall build.

## Layout

| Path | What it is |
|---|---|
| `reference/selective_scan_ref.py` | Vendored upstream reference scan from `state-spaces/mamba` (einops removed, `compute_dtype` knob added — deviations documented in the file header). **The** ground-truth function. |
| `gen_golden.py` | Deterministic golden-vector generator. 15 synthetic cases (shape grid + edge cases) with realistic Mamba value distributions. `--large` adds benchmark-shaped cases (not committed). |
| `golden/*.npz` | The golden vectors: f32 inputs, f64-computed ground-truth outputs (`out_f64`, `last_state_f64`), plus the upstream-identical f32 outputs (`out_f32`) that establish each case's tolerance floor. |
| `golden/manifest.json` | One metadata entry per case (shapes, flags, seed, observed f32 floor). |
| `verify_golden.py` | Independent verifier: recomputes every case with a pure-numpy, loop-based f64 implementation that shares no code with the generator; also checks determinism. |
| `check_hf_slow_path.py` | Proves HF `transformers` Mamba routes through `MambaMixer.slow_forward` on CPU (the Phase-4 patch target), shows the vendored reference reproduces the real mixer bit-exactly, and captures `golden/hf_mixer_layer0.npz` from a genuine mamba-130m forward pass. |

## Verified results (2026-07-10, torch 2.11, transformers 5.1)

- 16/16 golden cases: independent f64 implementations agree to ~1e-15
  (machine epsilon at these value scales).
- f32-vs-f64 floors: ~5e-8 … 8e-6 (worst is the deliberate `extreme_delta`
  underflow-stress case) — the kernel acceptance tolerance of
  `max_abs < 1e-4` has >10× headroom everywhere.
- HF mamba-130m on CPU: `slow_forward` called on 24/24 layers; vendored
  reference reproduces the layer-0 mixer output with max_abs error 0.0.
- Generator determinism: regenerated inputs are bit-identical.

## Kernel acceptance criteria (for Phase 1+)

For every `golden/*.npz`, a candidate kernel run on the f32 inputs must
satisfy `max_abs(out_kernel - out_f64) < 1e-4`, and should be compared
against the case's recorded `f32_max_abs_err` floor (a correct f32 kernel
lands within a small factor of it, not orders of magnitude above).

## Reproducing

```bash
python tests/gen_golden.py        # regenerate goldens (bit-identical)
python tests/verify_golden.py    # independent verification, exits nonzero on failure
python tests/check_hf_slow_path.py  # needs network on first run (~500MB model)
```
