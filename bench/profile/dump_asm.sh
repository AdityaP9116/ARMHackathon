#!/usr/bin/env bash
# Tier 0 assembly audit (§3.7). Dumps the generated aarch64 asm for the hottest
# kernel functions so you can check for exp-constant re-materialization inside
# the A2 loop and h-register spills. Needs no Arm hardware — cross-compiles and
# reads what LLVM emits. See PROFILING.md.
#
# Usage:  bash bench/profile/dump_asm.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
outdir="$here/out/asm"
mkdir -p "$outdir"
cd "$here/../../kernel"

target="aarch64-unknown-linux-gnu"
rustup target add "$target" 2>/dev/null || true
cargo install cargo-show-asm --locked 2>/dev/null || true

for fn in \
  "arm_scan_core::neon::exp::vexpq_f32" \
  "arm_scan_core::neon::channel_n16" \
  "arm_scan_core::neon::math::vsoftplusq_f32" \
  "arm_scan_core::neon::math::vsiluq_f32"
do
  base="${fn##*::}"
  echo "== $fn =="
  cargo asm --release --target "$target" -p arm-scan-core "$fn" \
    2>&1 | tee "$outdir/$base.s" || true
  echo
done

echo "Wrote asm to: $outdir"
echo "Look for: repeated 'fmov'/'dup' of the same constant inside the exp loop"
echo "(missed hoisting), and 'str'/'ldr' of h0..h3 to [sp] (register spills)."
