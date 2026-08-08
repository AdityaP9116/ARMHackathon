#!/usr/bin/env bash
# Assemble a CUDA toolchain able to compile TileLang kernels for consumer
# Blackwell (RTX 50xx, sm_120). Needed ONLY to capture Mamba-3 MIMO ground
# truth; nothing downstream of the goldens requires it, and nothing on the Arm
# side requires CUDA at all.
#
# WHY THIS IS NEEDED
# ------------------
# Mamba-3 MIMO runs on TileLang, which shells out to `nvcc` to build a .cubin.
# SISO runs on Triton, which compiles PTX itself with a bundled ptxas and never
# invokes nvcc -- which is why SISO captured fine on a box where MIMO could not
# compile at all. Four separate things had to line up:
#
#   1. nvcc >= 12.8       sm_120a did not exist before it. The system nvcc here
#                         was 12.4, whose newest arch is compute_90 (Hopper).
#   2. CUDA 13 headers    CUDA 12.9's math headers collide with glibc >= 2.41,
#                         which added the C23 `cospi`/`sinpi`/`rsqrt` that CUDA
#                         also declares, with different exception specs. Six
#                         redefinition errors. CUDA 13.0 headers account for it.
#   3. libnvvm SEPARATELY In CUDA 13 NVIDIA split `cicc` (nvcc's internal
#                         compiler) out of cuda_nvcc into the libnvvm component.
#                         Installing cuda_nvcc alone yields "cicc: not found".
#   4. gcc <= 13          nvcc 13.0 refuses host gcc newer than 13; this box has
#                         gcc 15. Hence the shim below.
#
# The pip wheel `nvidia-cuda-nvcc-cu12` does NOT solve this: it ships ptxas and
# headers but not the nvcc driver binary.
#
# Usage:
#   bash tools/setup_cuda_toolchain.sh [install_dir]
#   eval "$(bash tools/setup_cuda_toolchain.sh --env-only [install_dir])"
#
# Then run the capture with CUDA_HOME/PATH as printed.

set -euo pipefail

ENV_ONLY=0
if [ "${1:-}" = "--env-only" ]; then ENV_ONLY=1; shift; fi
DEST="${1:-${HOME}/.cache/arm-scan/cuda13}"
BASE=https://developer.download.nvidia.com/compute/cuda/redist

# Pinned to CUDA 13.0.2's component versions. Do not float these: the whole
# point is that a specific combination compiles, and a newer nvcc paired with
# older headers is exactly the failure mode this script exists to avoid.
NVCC_V=13.0.88
CUDART_V=13.0.96
CCCL_V=13.0.85
NVVM_V=13.0.88

# sha256 from NVIDIA's own redistrib_13.0.2.json manifest. This script fetches
# COMPILERS and then runs them, so the download is verified rather than
# trusted: HTTPS authenticates the host, not the bytes you end up executing if
# a mirror or cache is ever wrong. Any mismatch aborts.
NVCC_SHA=48e35be3cfbf4b4fbc16828eaec8a7048ee789403049dc409f7b643d6259cf7a
CUDART_SHA=25b8071951baba827be1580b841d363464f6ee6c39f48d33a81646f90cc95ed1
CCCL_SHA=ed845eae8c1767706b6ee91e40c608a03f6f633551a849b63f7346d32d73ee60
NVVM_SHA=17ef1665b63670887eeba7d908da5669fa8c66bb73b5b4c1367f49929c086353

emit_env() {
  echo "export CUDA_HOME=${DEST}"
  echo "export PATH=${DEST}/shim:${DEST}/bin:\$PATH"
}

if [ "$ENV_ONLY" = "1" ]; then emit_env; exit 0; fi

if [ -x "${DEST}/bin/nvcc" ] && [ -x "${DEST}/nvvm/bin/cicc" ]; then
  echo "toolchain already present at ${DEST}"
  "${DEST}/bin/nvcc" --version | tail -2
  emit_env
  exit 0
fi

mkdir -p "${DEST}"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

fetch() {  # component, version, sha256
  local name="$1" ver="$2" want="$3"
  local tarball="${name}-linux-x86_64-${ver}-archive.tar.xz"
  echo "  fetching ${name} ${ver}"
  curl -fsSL -o "${tmp}/${tarball}" "${BASE}/${name}/linux-x86_64/${tarball}"
  local got
  got="$(sha256sum "${tmp}/${tarball}" | cut -d' ' -f1)"
  if [ "${got}" != "${want}" ]; then
    echo "CHECKSUM MISMATCH for ${tarball}" >&2
    echo "  expected ${want}" >&2
    echo "  got      ${got}" >&2
    echo "Refusing to extract. Do not work around this by deleting the check." >&2
    exit 1
  fi
  tar -xf "${tmp}/${tarball}" -C "${DEST}" --strip-components=1
}

echo "assembling CUDA ${NVCC_V%.*} toolchain in ${DEST}"
fetch cuda_nvcc   "${NVCC_V}"   "${NVCC_SHA}"
fetch cuda_cudart "${CUDART_V}" "${CUDART_SHA}"
fetch cuda_cccl   "${CCCL_V}"   "${CCCL_SHA}"
fetch libnvvm     "${NVVM_V}"   "${NVVM_SHA}"   # cicc lives here in CUDA 13

# nvcc 13 rejects host gcc > 13. Point it at an older one without touching the
# system default, which the rest of the box needs.
HOSTCC=""
for v in 13 12 11; do
  if command -v "gcc-${v}" >/dev/null 2>&1 && command -v "g++-${v}" >/dev/null 2>&1; then
    HOSTCC="${v}"; break
  fi
done
if [ -z "${HOSTCC}" ]; then
  echo "WARNING: no gcc-13 or older found. nvcc will refuse a newer host" >&2
  echo "         compiler. Install one, e.g.: sudo apt install gcc-13 g++-13" >&2
else
  mkdir -p "${DEST}/shim"
  ln -sf "$(command -v gcc-${HOSTCC})" "${DEST}/shim/gcc"
  ln -sf "$(command -v g++-${HOSTCC})" "${DEST}/shim/g++"
  ln -sf "$(command -v gcc-${HOSTCC})" "${DEST}/shim/cc"
  ln -sf "$(command -v g++-${HOSTCC})" "${DEST}/shim/c++"
  echo "  host compiler shim -> gcc-${HOSTCC}"
fi

echo
"${DEST}/bin/nvcc" --version | tail -2
echo "archs: $("${DEST}/bin/nvcc" --list-gpu-arch | tail -3 | tr '\n' ' ')"
echo
emit_env
