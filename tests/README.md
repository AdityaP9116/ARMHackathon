# Phase 0 — Correctness Ground Truth

Everything the Rust kernel will be validated against lives here. See
`INTEGRATION_PLAN.md` (repo root) for how this fits the overall build.

## Layout

| Path | What it is |
|---|---|
| `reference/selective_scan_ref.py` | Vendored upstream reference scan from `state-spaces/mamba` (einops removed, `compute_dtype` knob added — deviations documented in the file header). **The** ground-truth function. |
| `golden_inputs.py` | The case tables and the input draws for **both** sets (1D and 2D), **numpy-only**. Draws from PCG64's raw stream with uniforms/normals derived in-file, so regeneration is independent of any library's RNG stream — see the module docstring for what broke when the draws went through `torch.Generator`. Imported by both generators (dev tier) and both verifiers (torch-free tier). |
| `gen_golden.py` | Deterministic golden-vector generator: draws via `golden_inputs`, computes ground truth with the vendored reference. 15 synthetic cases (shape grid + edge cases) with realistic Mamba value distributions. `--large` adds benchmark-shaped cases (not committed). |
| `golden/*.npz` | The golden vectors: f32 inputs, f64-computed ground-truth outputs (`out_f64`, `last_state_f64`), plus the upstream-identical f32 outputs (`out_f32`) that establish each case's tolerance floor. |
| `golden/manifest.json` | One metadata entry per case (shapes, flags, seed, observed f32 floor). |
| `verify_golden.py` | Independent verifier: recomputes every case with a pure-numpy, loop-based f64 implementation that shares no code with the generator; also redraws every core case's inputs and compares them bit-for-bit. Torch-free end to end, so the determinism check runs in CI's `test` job rather than skipping there. |
| `gen_golden_2d.py` | 2D cross-scan (SS2D) golden generator, `TOPOLOGY_IMPLEMENTATION_PLAN.md` §3.3. Draws via `golden_inputs` (`CORE_CASES_2D`, `draw_inputs_2d`), same as the 1D generator. Six grid cases — square, non-square, H/W not multiples of 4, degenerate height, and a `state=13` NEON-tail case. Stores the **four direction planes separately, before any merge**, so a kernel bug and a merge-strategy bug can't be confused. Writes to `golden/2d/`, kept apart from the 1D set because the schemas differ and `verify_golden.py` globs `golden/*.npz`. |
| `golden/2d/manifest.json` | Metadata + recorded f32 floor per 2D case, plus the `input_draw` spec the inputs were drawn by. |
| `verify_golden_2d.py` | Three checks: every case's inputs redrawn from its name and compared bit-for-bit, an independent numpy re-derivation of the 2D planes, **and** a replay of each case through `arm_scan.ss2d.ss2d_scan` on the real C ABI, reported as a multiple of that case's f32 floor (`--no-kernel` skips only the replay). The first two are torch-free. |
| `check_hf_slow_path.py` | Proves HF `transformers` Mamba routes through `MambaMixer.slow_forward` on CPU (the Phase-4 patch target), shows the vendored reference reproduces the real mixer bit-exactly, and captures `golden/hf_mixer_layer0.npz` from a genuine mamba-130m forward pass. |

## Verified results (regenerated 2026-08-06, torch 2.13, numpy 2.4)

The 15 synthetic 1D cases and all 6 2D cases were redrawn when the input draws
moved off `torch.Generator` (see `golden_inputs.py`); `hf_mixer_layer0` is
unchanged, being captured rather than drawn.

- 16/16 golden cases: independent f64 implementations agree to ≤3e-14
  (machine epsilon at these value scales; the 3e-14 is `extreme_delta`,
  everything else is ≤2e-15).
- f32-vs-f64 floors: ~4e-8 … 1.8e-5. The worst is the deliberate
  `extreme_delta` underflow-stress case, which leaves ~5.7× headroom under
  the `max_abs < 1e-4` kernel acceptance tolerance; every other case has
  >50×.
- All 16 replayed through the real C ABI (`check_ffi.py`) land at 0.9–1.1× of
  their recorded floor.
- HF mamba-130m on CPU: `slow_forward` called on 24/24 layers; vendored
  reference reproduces the layer-0 mixer output with max_abs error 0.0.
- 2D cross-scan (`verify_golden_2d.py`): 6/6 cases agree with the independent
  numpy re-derivation to ≤9e-16, and all 6 replay through the real C ABI at
  1.0–1.3× their recorded floor (2.3e-7 … 5.9e-7).
- Determinism: all 15 core 1D cases and all 6 2D cases redraw bit-identical,
  checked with numpy alone on every CI platform.

## Kernel acceptance criteria (for Phase 1+)

For every `golden/*.npz`, a candidate kernel run on the f32 inputs must
satisfy `max_abs(out_kernel - out_f64) < 1e-4`, and should be compared
against the case's recorded `f32_max_abs_err` floor (a correct f32 kernel
lands within a small factor of it, not orders of magnitude above).

## Reproducing

```bash
python tests/gen_golden.py        # regenerate goldens (inputs bit-identical on
                                  # any torch/numpy; needs torch for the outputs)
python tests/verify_golden.py    # independent verification, numpy-only,
                                  # exits nonzero on failure
python tests/gen_golden_2d.py     # same, for the SS2D grid cases
python tests/verify_golden_2d.py  # redraw + re-derive + replay through the C ABI
                                  # (--no-kernel drops the replay, leaving it
                                  # numpy-only)
python tests/check_hf_slow_path.py  # needs network on first run (~500MB model)
```
