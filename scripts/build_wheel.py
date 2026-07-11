"""Build the arm-scan wheel: copy the prebuilt cdylib into the package,
run `pip wheel`, then remove the copy again (so repo imports keep loading
the fresh artifact from kernel/target instead of a stale snapshot).

Prerequisite:  cargo build --release -p arm-scan-ffi   (in kernel/)
Usage:         python scripts/build_wheel.py
Output:        python/dist/arm_scan-<ver>-py3-none-<plat>.whl
"""

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "python" / "arm_scan"
DIST = REPO / "python" / "dist"

LIB_NAMES = {
    "win32": "arm_scan_ffi.dll",
    "darwin": "libarm_scan_ffi.dylib",
}
LIB_NAME = LIB_NAMES.get(sys.platform, "libarm_scan_ffi.so")


def main():
    built = REPO / "kernel" / "target" / "release" / LIB_NAME
    if not built.is_file():
        sys.exit(f"{built} not found — run `cargo build --release -p "
                 f"arm-scan-ffi` in kernel/ first")

    # never ship a stale or foreign-platform library
    staged = []
    for pattern in ("*.so", "*.dylib", "*.dll"):
        for old in PKG.glob(pattern):
            old.unlink()
    target = PKG / LIB_NAME
    shutil.copy2(built, target)
    staged.append(target)

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", ".", "-w", "dist",
             "--no-deps"],
            cwd=REPO / "python",
            check=True,
        )
    finally:
        for f in staged:
            f.unlink(missing_ok=True)

    wheels = sorted(DIST.glob("arm_scan-*.whl"),
                    key=lambda p: p.stat().st_mtime)
    if not wheels:
        sys.exit("no wheel produced")
    print(f"\nwheel: {wheels[-1]}")


if __name__ == "__main__":
    main()
