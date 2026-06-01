"""Backend registry for TinySplat."""

from .common import BackendOps
from .cpu import CPU_BACKEND
from .cuda import CUDA_BACKEND
from .mps import MPS_BACKEND


_BACKENDS = {
    "cpu": CPU_BACKEND,
    "cuda": CUDA_BACKEND,
    "mps": MPS_BACKEND,
}


def get_backend(device_type: str) -> BackendOps:
    """Return the registered backend for a torch device type."""
    if device_type not in _BACKENDS:
        raise ValueError(f"Unsupported backend device type: {device_type}")
    return _BACKENDS[device_type]


__all__ = ["BackendOps", "get_backend"]
