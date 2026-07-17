# Judge-facing entry points. `make validate` is the one command that proves
# the kernel is correct on this machine (README promise; keep it working).
PY ?= python3

.PHONY: validate build test bench
build:
	cd kernel && cargo build --release -p arm-scan-ffi

test: build
	cd kernel && cargo test --release
	$(PY) tests/check_ffi.py
	$(PY) tests/verify_golden.py

validate: test
	$(PY) bench/bench_op.py --quick --no-compile

bench: build
	cd kernel && cargo bench
