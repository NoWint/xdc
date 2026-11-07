"""Build the optional C++ acceleration core.

The pure-Python package is fully functional without this extension.  A compiler
failure (common on locked-down Windows machines) downgrades gracefully instead of
breaking installation.
"""
from __future__ import annotations

import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

try:
    import numpy
    numpy_include = numpy.get_include()
except ImportError:  # pragma: no cover - build-system installs numpy
    numpy_include = ""

extra_compile_args = []
if sys.platform == "win32":
    extra_compile_args = ["/O2", "/std:c++17"]
else:
    extra_compile_args = ["-O3", "-std=c++17", "-fvisibility=hidden"]


class OptionalBuildExt(build_ext):
    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:  # noqa: BLE001
            print(f"[xdc] WARNING: native extension failed to build ({exc}); using Python fallback")
            # Leave the extension absent; importing xdc._native will fail safely.
            if os.path.exists(self.get_ext_fullpath(ext.name)):
                os.unlink(self.get_ext_fullpath(ext.name))


ext_modules = [
    Pybind11Extension(
        "xdc._native",
        ["native/xdc_native.cpp"],
        include_dirs=[numpy_include] if numpy_include else [],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
        define_macros=[("NOMINMAX", "1")] if sys.platform == "win32" else [],
    )
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": OptionalBuildExt})
