"""Shared 3D Gaussian projection helpers with device dispatch.

Provides a single API that routes to device-specific implementations
(MPS, CPU, CUDA) when available, falling back to PyTorch ops.
"""

from typing import Callable, Dict, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_intrinsics(intrinsics: torch.Tensor) -> None:
    if intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must have shape (3, 3)")


def validate_camera_to_world(camera_to_world: torch.Tensor) -> None:
    if camera_to_world.shape != (4, 4):
        raise ValueError("camera_to_world must have shape (4, 4)")


# ---------------------------------------------------------------------------
# PyTorch fallbacks
# ---------------------------------------------------------------------------


def _world_to_camera(camera_to_world: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    rotation_c2w = camera_to_world[:3, :3]
    translation_c2w = camera_to_world[:3, 3]
    rotation_w2c = rotation_c2w.transpose(0, 1)
    translation_w2c = -rotation_w2c @ translation_c2w
    return rotation_w2c, translation_w2c


def _project_gaussians_3d_to_2d_pytorch(
    means: torch.Tensor,
    covariances: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_to_world: torch.Tensor,
    near_plane: float = 1e-4,
    min_covariance: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rotation_w2c, translation_w2c = _world_to_camera(camera_to_world)
    means_camera = means @ rotation_w2c.transpose(0, 1) + translation_w2c
    covariances_camera = (
        rotation_w2c.unsqueeze(0) @ covariances @ rotation_w2c.transpose(0, 1).unsqueeze(0)
    )

    x = means_camera[:, 0]
    y = means_camera[:, 1]
    z = means_camera[:, 2]
    visible_mask = z > near_plane
    safe_z = torch.where(visible_mask, z, torch.ones_like(z))

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    projected_means = torch.stack(
        [fx * x / safe_z + cx, fy * y / safe_z + cy],
        dim=1,
    )

    jacobian = torch.zeros(means.shape[0], 2, 3, dtype=means.dtype, device=means.device)
    jacobian[:, 0, 0] = fx / safe_z
    jacobian[:, 0, 2] = -fx * x / (safe_z * safe_z)
    jacobian[:, 1, 1] = fy / safe_z
    jacobian[:, 1, 2] = -fy * y / (safe_z * safe_z)

    projected_covariances = jacobian @ covariances_camera @ jacobian.transpose(1, 2)
    projected_covariances = projected_covariances + (
        torch.eye(2, dtype=means.dtype, device=means.device).unsqueeze(0) * min_covariance
    )

    return projected_means, projected_covariances, z, visible_mask


def _prepare_projected_gaussians_3d_pytorch(
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
):
    projected_means, projected_covariances, depths, visible_mask = (
        _project_gaussians_3d_to_2d_pytorch(
            means=means,
            covariances=covariances,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            near_plane=near_plane,
            min_covariance=min_covariance,
        )
    )

    if not torch.any(visible_mask):
        return None

    visible_indices = torch.nonzero(visible_mask, as_tuple=False).squeeze(1)
    visible_means = projected_means[visible_indices]
    visible_covariances = projected_covariances[visible_indices]
    visible_depths = depths[visible_indices]
    visible_colors = colors[visible_indices]
    visible_opacities = opacities[visible_indices]

    cov_xx = visible_covariances[:, 0, 0]
    cov_xy = visible_covariances[:, 0, 1]
    cov_yy = visible_covariances[:, 1, 1]
    trace = cov_xx + cov_yy
    disc = torch.sqrt(torch.clamp((cov_xx - cov_yy) ** 2 + 4.0 * cov_xy * cov_xy, min=0.0))
    lambda_max = torch.clamp(0.5 * (trace + disc), min=min_covariance)
    support_radius = sigma_radius * torch.sqrt(lambda_max)

    min_x = torch.floor(visible_means[:, 0] - support_radius).to(torch.int64)
    max_x = torch.ceil(visible_means[:, 0] + support_radius).to(torch.int64)
    min_y = torch.floor(visible_means[:, 1] - support_radius).to(torch.int64)
    max_y = torch.ceil(visible_means[:, 1] + support_radius).to(torch.int64)

    overlap_mask = (max_x >= 0) & (min_x < width) & (max_y >= 0) & (min_y < height)
    if not torch.any(overlap_mask):
        return None

    visible_means = visible_means[overlap_mask]
    visible_covariances = visible_covariances[overlap_mask]
    visible_depths = visible_depths[overlap_mask]
    visible_colors = visible_colors[overlap_mask]
    visible_opacities = visible_opacities[overlap_mask]

    sort_indices = torch.argsort(visible_depths, descending=False)
    visible_means = visible_means[sort_indices]
    visible_covariances = visible_covariances[sort_indices]
    visible_colors = visible_colors[sort_indices]
    visible_opacities = visible_opacities[sort_indices]

    return (
        visible_means,
        visible_covariances,
        visible_colors,
        visible_opacities,
        visible_indices[overlap_mask][sort_indices],
    )


# ---------------------------------------------------------------------------
# Device dispatch
# ---------------------------------------------------------------------------

_PROJECT_REGISTRY: Dict[str, Callable] = {}
_PREPARE_REGISTRY: Dict[str, Callable] = {}


def register_project_fn(device: str, fn: Callable):
    """Register a device-specific project_gaussians_3d_to_2d implementation."""
    _PROJECT_REGISTRY[device] = fn


def register_prepare_fn(device: str, fn: Callable):
    """Register a device-specific prepare_projected_gaussians_3d implementation."""
    _PREPARE_REGISTRY[device] = fn


def project_gaussians_3d_to_2d(
    means: torch.Tensor,
    covariances: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_to_world: torch.Tensor,
    near_plane: float = 1e-4,
    min_covariance: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project 3D Gaussians to 2D screen space. Dispatches to device-specific impl."""
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("means must have shape (N, 3)")
    if covariances.ndim != 3 or covariances.shape[1:] != (3, 3):
        raise ValueError("covariances must have shape (N, 3, 3)")
    validate_intrinsics(intrinsics)
    validate_camera_to_world(camera_to_world)

    device = means.device.type
    fn = _PROJECT_REGISTRY.get(device, _project_gaussians_3d_to_2d_pytorch)
    return fn(means, covariances, intrinsics, camera_to_world, near_plane, min_covariance)


def prepare_projected_gaussians_3d(
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
):
    """Filter visible Gaussians and sort by depth. Dispatches to device-specific impl."""
    device = means.device.type
    fn = _PREPARE_REGISTRY.get(device, _prepare_projected_gaussians_3d_pytorch)
    return fn(
        means,
        covariances,
        colors,
        opacities,
        intrinsics,
        camera_to_world,
        height,
        width,
        near_plane,
        min_covariance,
        sigma_radius,
    )


# ---------------------------------------------------------------------------
# Auto-register device implementations
# ---------------------------------------------------------------------------


def _auto_register():
    """Try importing device-specific implementations."""
    try:
        from tinysplat.mps import register_mps_3d_core

        register_mps_3d_core()
    except Exception:
        pass

    try:
        from tinysplat.backends_3d.cpu import register_cpu_3d_core

        register_cpu_3d_core()
    except Exception:
        pass


_auto_register()
