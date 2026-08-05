#!/usr/bin/env bash
# The whole training session in one command: calibrate -> train -> validate.
#
# The counterpart to bench/run_baseline.sh (which owns the Arm CPU benchmark
# session). This one owns the GPU training session, and like that script it
# exists so the metered time is spent executing rather than deciding.
#
# Usage:
#   bash tools/run_training_session.sh --cache data/knee_128.pt
#   bash tools/run_training_session.sh                    # phantoms, no data
#
# Options (env vars):
#   RES=128            image resolution (must match the cache)
#   STEPS=20000        training step ceiling; early-stop usually ends sooner
#   BATCH=16
#   MODEL_CHANNELS=64  backbone width
#   BLOCKS=2           SS2D blocks per resolution level
#   DEVICE=cuda
#   AMP=1              bf16 autocast
#   OUT=data/prior.pt
#   SKIP_CALIBRATE=1   reuse an existing --nrmse-bar instead of deriving it
#   NRMSE_BAR=...      skip calibration and use this value
set -euo pipefail
cd "$(dirname "$0")/.."

CACHE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --cache) CACHE="$2"; shift 2 ;;
        *) echo "unknown argument: $1"; exit 2 ;;
    esac
done

RES="${RES:-128}"
STEPS="${STEPS:-20000}"
BATCH="${BATCH:-16}"
MODEL_CHANNELS="${MODEL_CHANNELS:-64}"
BLOCKS="${BLOCKS:-2}"
DEVICE="${DEVICE:-cuda}"
OUT="${OUT:-data/prior.pt}"
PY="${PY:-}"
if [ -z "$PY" ]; then
    if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python3"; fi
fi
AMP_FLAG=""
if [ "${AMP:-1}" = 1 ]; then AMP_FLAG="--amp"; fi
DATA_FLAG=""
if [ -n "$CACHE" ]; then DATA_FLAG="--cache $CACHE"; fi

echo "=== training session  $(date -u +%FT%TZ) ==="
echo "  data   : ${CACHE:-synthetic phantoms}"
echo "  model  : ${MODEL_CHANNELS}ch x ${BLOCKS} blocks/level at ${RES}px"
echo "  device : $DEVICE  (amp=${AMP:-1})"
echo "  out    : $OUT"
echo

# --- 1. calibrate the accuracy bar FOR THIS DATA ---------------------
# The bar encodes how much information the mask destroys, so it is a property
# of the dataset: ~0.62 on phantoms, ~0.10 on smooth data. Training against a
# bar derived from different images either stops far too early or never stops.
if [ -n "${NRMSE_BAR:-}" ]; then
    echo "=== [1/3] calibration skipped, using NRMSE_BAR=$NRMSE_BAR ==="
elif [ "${SKIP_CALIBRATE:-0}" = 1 ]; then
    NRMSE_BAR=0.62
    echo "=== [1/3] calibration skipped, using the phantom default 0.62 ==="
    echo "    (this is probably wrong for your data — see calibrate_prior_bar.py)"
else
    echo "=== [1/3] calibrating the accuracy bar for this data ==="
    CAL_ARGS="--res $RES --R 4,8"
    if [ -n "$CACHE" ]; then CAL_ARGS="$CAL_ARGS --data fastmri --cache $CACHE"; fi
    # shellcheck disable=SC2086
    "$PY" tools/calibrate_prior_bar.py $CAL_ARGS | tee /tmp/_calib.txt
    NRMSE_BAR="$(grep -oE 'NRMSE_BAR *= *[0-9.]+' /tmp/_calib.txt | tail -1 \
                 | grep -oE '[0-9.]+$' || true)"
    if [ -z "$NRMSE_BAR" ]; then
        echo "could not parse a bar from the calibration output; "
        echo "pass NRMSE_BAR=... explicitly."
        exit 1
    fi
    echo "    -> NRMSE_BAR=$NRMSE_BAR"
fi
echo

# --- 2. train --------------------------------------------------------
echo "=== [2/3] training (early-stops as soon as every rung clears $NRMSE_BAR) ==="
# shellcheck disable=SC2086
"$PY" tools/train_prior.py $DATA_FLAG --res "$RES" --steps "$STEPS" \
    --batch "$BATCH" --model-channels "$MODEL_CHANNELS" --blocks "$BLOCKS" \
    --device "$DEVICE" $AMP_FLAG --nrmse-bar "$NRMSE_BAR" --out "$OUT"
echo

# --- 3. validate -----------------------------------------------------
# Runs on CPU deliberately: the kernel-vs-reference parity check is the whole
# point of the third section, and arm_scan is a CPU kernel.
echo "=== [3/3] validating (accuracy + quality + kernel parity) ==="
# shellcheck disable=SC2086
"$PY" tools/validate_prior.py --checkpoint "$OUT" $DATA_FLAG \
    --res "$RES" --nrmse-bar "$NRMSE_BAR" \
    --json "results/prior_validation.json" \
    --png "demo_out/validation.png" || true

echo
echo "=== done ==="
echo "  prior      : $OUT"
echo "  validation : results/prior_validation.json, demo_out/validation.png"
echo
echo "Copy the prior off this box, then TERMINATE the instance:"
echo "  scp <key> <user>@<ip>:$(pwd)/$OUT ."
