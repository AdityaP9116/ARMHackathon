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

.PHONY: validate build test test-app test-app-slow bench demo goldens \
        goldens-2d prepare-data calibrate train validate-prior train-session

build:
	cd kernel && cargo build --release -p arm-scan-ffi

# Kernel correctness: Rust gates, then the goldens replayed through the real
# C ABI, then both golden sets re-derived by an independent implementation.
test: build
	cd kernel && cargo test --release
	$(PY) tests/check_ffi.py
	$(PY) tests/verify_golden.py
	$(PY) tests/verify_golden_2d.py

# App-level gates that are fast enough for the judge path (seconds each).
test-app: build
	$(PY) apps/mri_diffusion/tests/test_ss2d_pair_parity.py
	$(PY) apps/mri_diffusion/tests/test_phase_c_parity.py
	$(PY) apps/mri_diffusion/tests/test_phase_d_pipeline.py

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
