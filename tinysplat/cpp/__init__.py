"""Helpers for loading TinySplat C++ extensions."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional


@lru_cache(maxsize=1)
def load_cpu_extension() -> Optional[object]:
    """
    Try to build and import the TinySplat CPU extension.

    Returns None when extension loading is disabled or the build/import fails.
    Extension loading is enabled by default. Set ``TINYSPLAT_BUILD_EXTENSIONS=0``
    to disable the build attempt.
    """
    if os.environ.get("TINYSPLAT_BUILD_EXTENSIONS", "1") == "0":
        return None

    try:
        from torch.utils.cpp_extension import load
    except ImportError:
        return None

    source_path = Path(__file__).resolve().parent / "tinysplat_cpu.cpp"
    build_dir = Path(__file__).resolve().parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    try:
        return load(
            name="tinysplat_cpp",
            sources=[str(source_path)],
            build_directory=str(build_dir),
            extra_cflags=["-O3"],
            verbose=False,
        )
    except Exception:
        return None


__all__ = ["load_cpu_extension"]
