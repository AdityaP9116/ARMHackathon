# Judge-facing entry points. `make validate` is the one command that proves
# the kernel is correct on this machine (README promise; keep it working).
#
# `validate` covers BOTH surfaces the submission claims: the 1D selective scan
# AND the SS2D cross-scan the diffusion app runs on. It needs no dataset, no
# credentials, and no external clone — target ~5 minutes on a laptop.
PY ?= python3

.PHONY: validate build test test-app test-app-slow bench demo goldens goldens-2d

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

# The minutes-long ones: in-process prior training. CI and pre-release only.
test-app-slow: build
	$(PY) apps/mri_diffusion/tests/test_backbone_bringup.py
	$(PY) apps/mri_diffusion/tests/test_phase_d_partial.py

validate: test test-app
	$(PY) bench/bench_op.py --quick --no-compile

# The video's artifact: side-by-side reconstruction, phantom track.
demo: build
	$(PY) apps/mri_diffusion/demo.py --compare-reference

bench: build
	cd kernel && cargo bench

goldens:
	$(PY) tests/gen_golden.py

goldens-2d:
	$(PY) tests/gen_golden_2d.py
