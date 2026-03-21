"""CUDA backend for 3D Gaussian splatting."""

from .projected import make_projected_backend


CUDA_BACKEND_3D = make_projected_backend("cuda")
