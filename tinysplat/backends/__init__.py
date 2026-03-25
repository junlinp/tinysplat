"""Backend registry for TinySplat."""

import os

from .common import BackendOps
from .cpu import CPU_BACKEND
from .cuda import CUDA_BACKEND
from .mps import MPS_BACKEND


_BACKENDS = {
    "cpu": CPU_BACKEND,
    "cuda": CUDA_BACKEND,
    "mps": MPS_BACKEND,
}

# Optional Halide backend — loaded if TINYSPLAT_BACKEND=halide or if
# TINYSPLAT_HALIDE_LIB points to a valid compiled .so
_HALIDE_BACKEND = None


def _load_halide_backend():
    global _HALIDE_BACKEND
    if _HALIDE_BACKEND is not None:
        return
    try:
        from .. import halide_backend

        _HALIDE_BACKEND = halide_backend.HALIDE_BACKEND
    except ImportError:
        pass


def get_backend(device_type: str) -> BackendOps:
    """Return the registered backend for a torch device type."""
    # Honour TINYSPLAT_BACKEND=halide override (Halide targets any device)
    backend_override = os.environ.get("TINYSPLAT_BACKEND", "").lower()
    if backend_override == "halide":
        _load_halide_backend()
        if _HALIDE_BACKEND is not None:
            return _HALIDE_BACKEND
        raise RuntimeError(
            "TINYSPLAT_BACKEND=halide is set but Halide backend failed to load. "
            "Set TINYSPLAT_HALIDE_LIB and ensure Halide is installed."
        )

    if device_type not in _BACKENDS:
        raise ValueError(f"Unsupported backend device type: {device_type}")
    return _BACKENDS[device_type]


__all__ = ["BackendOps", "get_backend"]
