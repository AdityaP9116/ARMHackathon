#!/usr/bin/env bash
# The full reusable baseline suite (BASELINE_TEST_PLAN.md §4-5).
# Correctness gates first — a broken kernel is never benchmarked — then
# every performance tier, writing tagged JSONs to bench/results/ and
# regenerating RESULTS.md at the end.
#
# Usage:
#   bash bench/run_baseline.sh [tag]            # tag defaults to hostname
#
# Toggles (env vars):
#   DRY_RUN=1            print every command without executing
#   SKIP_CORRECTNESS=1   skip the correctness gate (never for baselines)
#   SKIP_CRITERION=1     skip the Rust criterion ladder
#   SKIP_OP=1            skip the op-level suites
#   SKIP_SCALING=1       skip the RAYON_NUM_THREADS scaling loop
#   SKIP_E2E=1           skip HF end-to-end (downloads mamba-130m once)
#   SKIP_SS2D=1          skip the 2D cross-scan + bidirectional topologies
#   THREADS_LIST="1 2 4" thread counts for the scaling loop
#                        (default: powers of 2 up to nproc)
#   PY=path              python to use (default: .venv/bin/python if
#                        present, else python3)
set -euo pipefail
cd "$(dirname "$0")/.."

TAG="${1:-$(hostname)}"
RES="bench/results"
mkdir -p "$RES"

PY="${PY:-}"
if [ -z "$PY" ]; then
    if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python3"; fi
fi

run() {
    echo "+ $*"
    if [ "${DRY_RUN:-0}" != 1 ]; then "$@"; fi
}

echo "=== baseline run: tag=$TAG  python=$PY  $(date -u +%FT%TZ) ==="
run uname -a
run nproc

echo "=== [0/6] build FFI cdylib (required by every python stage) ==="
(cd kernel && run cargo build --release -p arm-scan-ffi)

if [ "${SKIP_CORRECTNESS:-0}" != 1 ]; then
    echo "=== [1/6] correctness gate ==="
    (cd kernel && run cargo test --release)
    run "$PY" tests/check_ffi.py
    # 2D cross-scan: independent numpy re-derivation + the goldens replayed
    # through the real C ABI, reported against each case's f32 floor.
    run "$PY" tests/verify_golden_2d.py
    # The SS2D traversal-pair path vs the four-separate-scans oracle, on the
    # reference AND the kernel. This is the first time the pair path meets
    # NEON, so run it at several thread counts — rayon output must be
    # bit-identical to sequential regardless of count.
    for t in 1 2 8; do
        echo "+ RAYON_NUM_THREADS=$t $PY apps/mri_diffusion/tests/test_ss2d_pair_parity.py"
        if [ "${DRY_RUN:-0}" != 1 ]; then
            RAYON_NUM_THREADS="$t" "$PY" \
                apps/mri_diffusion/tests/test_ss2d_pair_parity.py
        fi
    done
fi

if [ "${SKIP_CRITERION:-0}" != 1 ]; then
    echo "=== [2/6] criterion ladder (scalar_seq / neon_seq / neon_par) ==="
    if [ "${DRY_RUN:-0}" != 1 ]; then
        (cd kernel && cargo bench) | tee "$RES/criterion_${TAG}.txt"
    else
        echo "+ (cd kernel && cargo bench) | tee $RES/criterion_${TAG}.txt"
    fi
fi

if [ "${SKIP_OP:-0}" != 1 ]; then
    echo "=== [3/6] op-level suites ==="
    run "$PY" bench/bench_op.py --suite basic --tag "$TAG" \
        --json "$RES/op_basic_${TAG}.json"
    run "$PY" bench/bench_op.py --suite sweep-len --tag "$TAG" \
        --compile-max-len "${COMPILE_MAX_LEN:-512}" \
        --json "$RES/op_sweep-len_${TAG}.json"
    run "$PY" bench/bench_op.py --suite sweep-dim --tag "$TAG" \
        --no-compile --json "$RES/op_sweep-dim_${TAG}.json"
    run "$PY" bench/bench_op.py --suite sweep-batch --tag "$TAG" \
        --no-compile --json "$RES/op_sweep-batch_${TAG}.json"
fi

if [ "${SKIP_SCALING:-0}" != 1 ]; then
    echo "=== [4/6] thread-scaling loop (one process per count) ==="
    if [ -z "${THREADS_LIST:-}" ]; then
        THREADS_LIST=""
        n=1
        max="$(nproc)"
        while [ "$n" -le "$max" ]; do
            THREADS_LIST="$THREADS_LIST $n"
            n=$((n * 2))
        done
    fi
    echo "thread counts:$THREADS_LIST"
    for t in $THREADS_LIST; do
        echo "+ RAYON_NUM_THREADS=$t $PY bench/bench_op.py --suite scaling-point --no-compile --reps 7 --tag ${TAG}-t${t} --json $RES/op_scaling_${TAG}_t${t}.json"
        if [ "${DRY_RUN:-0}" != 1 ]; then
            RAYON_NUM_THREADS="$t" "$PY" bench/bench_op.py \
                --suite scaling-point --no-compile --reps 7 \
                --tag "${TAG}-t${t}" \
                --json "$RES/op_scaling_${TAG}_t${t}.json"
        fi
    done
fi

if [ "${SKIP_E2E:-0}" != 1 ]; then
    echo "=== [5/6] end-to-end HF generate() sweeps ==="
    for p in 128 512 2048; do
        run "$PY" bench/bench_e2e.py --prompt-tokens "$p" \
            --new-tokens 32 --reps 5 --tag "$TAG" \
            --json "$RES/e2e_p${p}_${TAG}.json"
    done
fi

if [ "${SKIP_SS2D:-0}" != 1 ]; then
    echo "=== [6/6] topologies: 2D cross-scan + bidirectional ==="
    # SS2D at the diffusion workload's real grids. Reports the traversal-pair
    # path vs the legacy four-forward-scans formulation (same kernel both
    # sides, so the ratio is the restructuring alone) and the scan-vs-overhead
    # split that carries the P1-7 go/no-go. This re-takes on Arm a decision so
    # far made only on x86, where the worst shape sat at 13.8% against a 15%
    # bar — close enough that different cache behaviour could flip it.
    run "$PY" bench/bench_ss2d.py --reps "${SS2D_REPS:-5}" --tag "$TAG" \
        --compile-max-l "${SS2D_COMPILE_MAX_L:-2048}" \
        --json "$RES/ss2d_${TAG}.json"
    # The 1D bidirectional kernel SS2D is built on: exp-sharing across
    # directions, measured at 1.58-1.75x in CI. Confirm it holds on Arm.
    run "$PY" bench/bench_bidirectional.py --suite sweep-len \
        --compile-max-len "${COMPILE_MAX_LEN:-512}" --reps 5 --warmup 1 \
        --tag "$TAG" --json "$RES/bidi_${TAG}.json"
fi

echo "=== render ==="
run "$PY" bench/render_results.py
echo "=== done: results in $RES (copy back / commit deliberately) ==="
echo
echo "Reminder: nothing else should be running on this box. A contended"
echo "benchmark is void — an earlier x86 run under load reported a 0.50x"
echo "'regression' that the quiesced re-run put at 1.82x."
