"""Platform-tagged wheel build for arm-scan.

The package is pure Python (ctypes) plus one prebuilt cdylib, so a single
`py3-none-<platform>` wheel covers every Python version on that platform —
no per-interpreter builds. This shim only exists to override the wheel tag;
all metadata lives in pyproject.toml.

Tag selection (override with ARM_SCAN_PLAT_TAG):
  linux  -> manylinux_<glibc_major>_<glibc_minor>_<machine> of the build
            host (honest floor: the cdylib links that glibc)
  macOS  -> macosx_11_0_<machine> (build with MACOSX_DEPLOYMENT_TARGET=11.0)
  win    -> the default platform tag (e.g. win_amd64)
"""

import os
import platform
import sys

from setuptools import setup

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # older setuptools keeps it in the wheel package
    from wheel.bdist_wheel import bdist_wheel


def _plat_tag():
    env = os.environ.get("ARM_SCAN_PLAT_TAG")
    if env:
        return env
    machine = platform.machine().lower().replace("-", "_")
    if sys.platform == "darwin":
        return f"macosx_11_0_{machine}"
    if sys.platform.startswith("linux"):
        libc, ver = platform.libc_ver()
        if libc == "glibc" and ver:
            major, minor = ver.split(".")[:2]
            return f"manylinux_{major}_{minor}_{machine}"
    return None  # fall through to the default tag (windows etc.)


class PlatformWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, default_plat = super().get_tag()
        return "py3", "none", _plat_tag() or default_plat


setup(cmdclass={"bdist_wheel": PlatformWheel})
