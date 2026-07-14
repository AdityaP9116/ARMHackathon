#!/usr/bin/env bash
# Tier 0/1 profiling on any Arm host that has a Rust toolchain (Apple Silicon,
# Oracle Ampere A1, a Graviton box). Produces the phase breakdown + the
# criterion ladder without needing perf or root. See PROFILING.md.
#
# Usage:  bash bench/profile/run_profile.sh [output_tag]
set -euo pipefail

tag="${1:-$(uname -m)-$(date +%Y%m%d-%H%M%S)}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
outdir="$here/out/$tag"
mkdir -p "$outdir"
cd "$here/../../kernel"

arch="$(uname -m)"
echo "== host: $arch | tag: $tag =="
if [[ "$arch" != "aarch64" && "$arch" != "arm64" ]]; then
  echo "WARNING: not an Arm host — the NEON path won't run and the phase"
  echo "profiler will print a stub. The criterion ladder still runs (scalar)."
fi

echo "== phase profiler =="
cargo run --release --example profile_phases --features profiling \
  2>&1 | tee "$outdir/profile-phases.txt" || true

echo "== criterion ladder (scalar -> NEON -> rayon) =="
cargo bench -p arm-scan-core 2>&1 | tee "$outdir/bench-scan.txt"

echo
echo "Wrote results to: $outdir"
echo "Read profile-phases.txt: the largest-% phase is your bottleneck."
