"""
TinySplat: A lightweight 2D Gaussian splatting implementation for PyTorch.
"""

from .gaussian_splat_2d import gaussian_splat_2d, GaussianSplat2D
from .gaussian_splat_3d import gaussian_splat_3d, GaussianSplat3D, project_gaussians_3d_to_2d

__version__ = "0.1.0"
__all__ = [
    "gaussian_splat_2d",
    "GaussianSplat2D",
    "gaussian_splat_3d",
    "GaussianSplat3D",
    "project_gaussians_3d_to_2d",
]

