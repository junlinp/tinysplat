"""Backend registry for 3D TinySplat renderers."""

from .common import Backend3DOps
from .cpu import CPU_BACKEND_3D
from .cuda import CUDA_BACKEND_3D
from .mps import MPS_BACKEND_3D


_BACKENDS_3D = {
    "cpu": CPU_BACKEND_3D,
    "cuda": CUDA_BACKEND_3D,
    "mps": MPS_BACKEND_3D,
}


def get_backend_3d(device_type: str) -> Backend3DOps:
    """Return the registered 3D backend for a torch device type."""
    if device_type not in _BACKENDS_3D:
        raise ValueError(f"Unsupported 3D backend device type: {device_type}")
    return _BACKENDS_3D[device_type]


__all__ = ["Backend3DOps", "get_backend_3d"]
