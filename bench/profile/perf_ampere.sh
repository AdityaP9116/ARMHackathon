#!/usr/bin/env bash
# Tier 2: hardware-counter profiling on an Arm Linux box you control with root
# (Oracle Ampere A1 Always-Free, or a rented Graviton). Answers the question
# ablation timing can't: compute-bound vs memory-bound. See PROFILING.md.
#
# Requires: linux-perf (`sudo apt install linux-tools-generic linux-perf`),
# root/sudo, and a Rust toolchain. Free on Oracle A1.
#
# Usage:  sudo bash bench/profile/perf_ampere.sh [output_tag]
set -euo pipefail

tag="${1:-ampere-$(date +%Y%m%d-%H%M%S)}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
outdir="$here/out/$tag"
mkdir -p "$outdir"
cd "$here/../../kernel"

if ! command -v perf >/dev/null 2>&1; then
  echo "perf not found. Install: sudo apt install linux-tools-generic linux-perf"
  exit 1
fi

# Allow user-space access to hardware events for this session.
echo "-1" | sudo tee /proc/sys/kernel/perf_event_paranoid >/dev/null || true

echo "== build the single-threaded profiler binary =="
cargo build --release --example profile_phases --features profiling
bin="$(find target/release/examples -maxdepth 1 -name 'profile_phases-*' -type f ! -name '*.d' | head -n1)"
[[ -z "$bin" ]] && bin="target/release/examples/profile_phases"
echo "binary: $bin"

# Neoverse PMU event names vary by kernel; the generic aliases below are the
# most portable. If an event is unsupported, perf simply reports <not counted>.
events="cycles,instructions,stalled-cycles-backend,stalled-cycles-frontend,\
cache-references,cache-misses,L1-dcache-load-misses,LLC-load-misses,branch-misses"

echo "== perf stat (compute-bound vs memory-bound) =="
perf stat -e "$events" -- "$bin" 2>&1 | tee "$outdir/perf-stat.txt"

echo "== perf record + annotate (per-instruction hot spots) =="
perf record -g -o "$outdir/perf.data" -- "$bin" >/dev/null 2>&1 || true
{
  echo "===== top symbols ====="
  perf report -i "$outdir/perf.data" --stdio --sort overhead,symbol 2>/dev/null | head -40 || true
  echo
  echo "===== annotate vexpq_f32 ====="
  perf annotate -i "$outdir/perf.data" --stdio -M att 2>/dev/null \
    | sed -n '/vexpq_f32/,/Sorted summary/p' | head -80 || true
} | tee "$outdir/perf-annotate.txt"

echo
echo "Wrote results to: $outdir"
echo "Interpretation:"
echo "  high stalled-cycles-backend + high LLC/L1 misses -> MEMORY-bound -> §4.1/§4.2"
echo "  low stalls, cycles concentrated in vexpq_f32     -> COMPUTE-bound -> §3.1/§3.2"
echo "  cycles concentrated in the Pass-B FMA chain      -> LATENCY-bound -> §3.3"
