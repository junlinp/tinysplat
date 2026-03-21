"""Projected 3D fallback backends built on the 2D renderer."""

import torch

from ..gaussian_splat_2d import gaussian_splat_2d
from ..gaussian_splat_3d_core import prepare_projected_gaussians_3d
from .common import Backend3DOps


def render_projected_3d(
    means: torch.Tensor,
    covariances: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_to_world: torch.Tensor,
    height: int,
    width: int,
    near_plane: float,
    min_covariance: float,
    sigma_radius: float,
) -> torch.Tensor:
    """Project 3D Gaussians and rasterize them with the active 2D backend."""
    prepared = prepare_projected_gaussians_3d(
        means=means,
        covariances=covariances,
        colors=colors,
        opacities=opacities,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
        height=height,
        width=width,
        near_plane=near_plane,
        min_covariance=min_covariance,
        sigma_radius=sigma_radius,
    )
    if prepared is None:
        return torch.zeros(height, width, colors.shape[1], dtype=colors.dtype, device=means.device)

    projected_means, projected_covariances, projected_colors, projected_opacities, _ = prepared
    return gaussian_splat_2d(
        means=projected_means,
        covariances=projected_covariances,
        colors=projected_colors,
        opacities=projected_opacities,
        height=height,
        width=width,
        device=means.device.type,
    )


def make_projected_backend(name: str) -> Backend3DOps:
    """Create a 3D backend that projects into the existing 2D backend stack."""
    return Backend3DOps(
        name=name,
        render=render_projected_3d,
        is_compiled=False,
    )
