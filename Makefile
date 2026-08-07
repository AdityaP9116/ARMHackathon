# Judge-facing entry points. `make validate` is the one command that proves
# the kernel is correct on this machine (README promise; keep it working).
#
# `validate` covers BOTH surfaces the submission claims: the 1D selective scan
# AND the SS2D cross-scan the diffusion app runs on. It needs no dataset, no
# credentials, and no external clone — target ~5 minutes on a laptop.
PY ?= python3

# Training-path defaults; override on the command line, e.g.
#   make train CACHE=data/knee_128.pt RES=128 PRIOR=data/prior_knee.pt
RAW   ?= data/raw
CACHE ?= data/knee_128.pt
PRIOR ?= data/prior.pt
RES   ?= 128

.PHONY: validate build test test-app test-app-slow test-mamba3 bench demo \
        goldens goldens-2d prepare-data calibrate train validate-prior \
        train-session

build:
	cd kernel && cargo build --release -p arm-scan-ffi

# Typecheck the aarch64-only code from an x86 dev box.
#
# `try_neon` and everything under `neon/` are `#[cfg(target_arch = "aarch64")]`,
# so `cargo build` on x86 does not compile them AT ALL. A field added to a
# shared struct can therefore pass every local check and break only on the Arm
# CI runners -- which is exactly what happened when `Mamba3Input` gained `mimo`.
# This catches it in about a second, and needs no Arm hardware: it is a
# typecheck, not a build (no linker involved).
#
#   rustup target add aarch64-unknown-linux-gnu   # one-off
#
# Skips (loudly) when the target is not installed, so `make test` still works on
# a fresh clone and on CI runners that do not have it. Skipping is safe here in
# a way that "CI silently stopped running" was not: the arm64 and macOS-arm64
# jobs compile this same code NATIVELY, so the coverage exists either way. This
# target only moves the discovery earlier for someone working on x86.
check-cross:
	@if rustup target list --installed 2>/dev/null | grep -q '^aarch64-unknown-linux-gnu$$'; then \
	  echo "check-cross: typechecking the aarch64-only paths"; \
	  cd kernel && cargo check --target aarch64-unknown-linux-gnu --all-targets; \
	else \
	  echo "check-cross: SKIPPED - aarch64-unknown-linux-gnu not installed."; \
	  echo "             Install it with: rustup target add aarch64-unknown-linux-gnu"; \
	  echo "             (the arm64 CI jobs compile these paths natively regardless)"; \
	fi

# Kernel correctness: Rust gates, then the goldens replayed through the real
# C ABI, then both golden sets re-derived by an independent implementation.
test: build check-cross
	cd kernel && cargo test --release
	$(PY) tests/check_ffi.py
	$(PY) tests/verify_golden.py
	$(PY) tests/verify_golden_2d.py

# App-level gates that are fast enough for the judge path (seconds each).
test-app: build
	$(PY) apps/mri_diffusion/tests/test_ss2d_pair_parity.py
	$(PY) apps/mri_diffusion/tests/test_phase_c_parity.py
	$(PY) apps/mri_diffusion/tests/test_phase_d_pipeline.py

# Mamba-3 (post-submission track, feature/mamba3 work): does the CPU reference
# still reproduce the official kernel's captured ground truth? Needs torch but
# NOT a GPU -- that is the whole point of Stage 0. Deliberately NOT part of
# `test` (which must stay torch-free; a previous change broke CI's torch-free
# tier that way) nor of `validate` (the judge path is the Mamba-1 submission).
test-mamba3:
	$(PY) tests/verify_golden_mamba3.py
	$(PY) tests/check_mamba3_op.py
	$(PY) tests/check_mamba3_block.py
	$(PY) tests/check_ss2d_mamba3.py
	$(PY) tests/verify_golden_mamba3_mimo.py
	$(PY) tests/check_mamba3_mimo_op.py

# Path A end to end: the published 187M checkpoint, on CPU, through our kernel.
# Kept OUT of `test-mamba3` because it downloads ~357 MB, which no CI job
# should do on every push. `check_mamba3_block` is the cheap proxy that runs
# there instead: it carries the real layer-0/1 weights inside the golden, so it
# catches the same plumbing bugs without the download.
test-mamba3-model:
	$(PY) tests/check_mamba3_model.py
	$(PY) tests/check_mamba3_model.py --dir tests/golden/mamba3_mimo

# The minutes-long ones: in-process prior training. CI and pre-release only.
test-app-slow: build
	$(PY) apps/mri_diffusion/tests/test_backbone_bringup.py
	$(PY) apps/mri_diffusion/tests/test_phase_d_quality.py

validate: test test-app
	$(PY) bench/bench_op.py --quick --no-compile

# The video's artifact: side-by-side reconstruction, phantom track.
demo: build
	$(PY) apps/mri_diffusion/demo.py --compare-reference

bench: build
	cd kernel && cargo bench

# ---- training (GPU box; needs no Rust — the kernel has no autograd) ----
# 1. prepare  2. calibrate the bar for THIS data  3. train  4. validate
prepare-data:
	$(PY) tools/prepare_fastmri.py --root $(RAW) --out $(CACHE) --res $(RES)

calibrate:
	$(PY) tools/calibrate_prior_bar.py --data fastmri --cache $(CACHE) \
		--res $(RES)

train:
	$(PY) tools/train_prior.py --cache $(CACHE) --res $(RES) --out $(PRIOR)

validate-prior:
	$(PY) tools/validate_prior.py --checkpoint $(PRIOR) --cache $(CACHE) \
		--json results/prior_validation.json

# The whole GPU session end to end.
train-session:
	bash tools/run_training_session.sh --cache $(CACHE)

goldens:
	$(PY) tests/gen_golden.py

goldens-2d:
	$(PY) tests/gen_golden_2d.py
