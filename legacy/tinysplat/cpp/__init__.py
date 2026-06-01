"""Helpers for loading TinySplat C++/CUDA extensions."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional


@lru_cache(maxsize=1)
def load_cpu_extension() -> Optional[object]:
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


@lru_cache(maxsize=1)
def load_cuda_extension() -> Optional[object]:
    if os.environ.get("TINYSPLAT_BUILD_EXTENSIONS", "1") == "0":
        return None
    if not os.path.exists("/usr/local/cuda"):
        return None
    try:
        from torch.utils.cpp_extension import load
    except ImportError:
        return None

    source_cpp = Path(__file__).resolve().parent / "tinysplat_cuda.cpp"
    source_cu = Path(__file__).resolve().parent / "tinysplat_cuda.cu"
    build_dir = Path(__file__).resolve().parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    if not source_cpp.exists() or not source_cu.exists():
        return None

    try:
        return load(
            name="tinysplat_cuda",
            sources=[str(source_cpp), str(source_cu)],
            build_directory=str(build_dir),
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--expt-relaxed-constexpr",
                "-use_fast_math",
                "-Xcompiler=-fPIC",
            ],
            verbose=False,
        )
    except Exception as e:
        print(f"CUDA extension build failed: {e}")
        return None


__all__ = ["load_cpu_extension", "load_cuda_extension"]
