"""3D Gaussian splatting front-end built on the 2D screen-space renderer."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .gaussian_splat_2d import gaussian_splat_2d


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _validate_intrinsics(intrinsics: torch.Tensor) -> None:
    if intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must have shape (3, 3)")


def _validate_camera_to_world(camera_to_world: torch.Tensor) -> None:
    if camera_to_world.shape != (4, 4):
        raise ValueError("camera_to_world must have shape (4, 4)")


def project_gaussians_3d_to_2d(
    means: torch.Tensor,
    covariances: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_to_world: torch.Tensor,
    near_plane: float = 1e-4,
    min_covariance: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project 3D Gaussians from world coordinates to screen-space 2D Gaussians.

    Args:
        means: Tensor of shape (N, 3) in world coordinates.
        covariances: Tensor of shape (N, 3, 3) in world coordinates.
        intrinsics: Camera intrinsics matrix of shape (3, 3).
        camera_to_world: Rigid camera pose of shape (4, 4), mapping camera to world.
        near_plane: Points with z <= near_plane in camera coordinates are discarded.
        min_covariance: Diagonal screen-space covariance regularizer.

    Returns:
        projected_means: Tensor of shape (N, 2)
        projected_covariances: Tensor of shape (N, 2, 2)
        depths: Tensor of shape (N,)
        visible_mask: Bool tensor of shape (N,)
    """
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("means must have shape (N, 3)")
    if covariances.ndim != 3 or covariances.shape[1:] != (3, 3):
        raise ValueError("covariances must have shape (N, 3, 3)")
    _validate_intrinsics(intrinsics)
    _validate_camera_to_world(camera_to_world)

    rotation_c2w = camera_to_world[:3, :3]
    translation_c2w = camera_to_world[:3, 3]

    rotation_w2c = rotation_c2w.transpose(0, 1)
    translation_w2c = -rotation_w2c @ translation_c2w

    means_camera = means @ rotation_w2c.transpose(0, 1) + translation_w2c
    covariances_camera = rotation_w2c.unsqueeze(0) @ covariances @ rotation_w2c.transpose(0, 1).unsqueeze(0)

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
        [
            fx * x / safe_z + cx,
            fy * y / safe_z + cy,
        ],
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


def gaussian_splat_3d(
    means: torch.Tensor,
    covariances: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_to_world: torch.Tensor,
    height: int,
    width: int,
    device: Optional[str] = None,
    near_plane: float = 1e-4,
    min_covariance: float = 1e-4,
) -> torch.Tensor:
    """
    Render 3D Gaussians onto an image plane using camera intrinsics and pose.

    The pose is defined from camera coordinates to world coordinates.
    """
    if device is None:
        device = _auto_device()
    device_obj = torch.device(device)

    means = means.to(device_obj)
    covariances = covariances.to(device_obj)
    colors = colors.to(device_obj)
    opacities = opacities.to(device_obj)
    intrinsics = intrinsics.to(device_obj)
    camera_to_world = camera_to_world.to(device_obj)

    projected_means, projected_covariances, depths, visible_mask = project_gaussians_3d_to_2d(
        means=means,
        covariances=covariances,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
        near_plane=near_plane,
        min_covariance=min_covariance,
    )

    in_frame_mask = (
        visible_mask
        & (projected_means[:, 0] >= 0)
        & (projected_means[:, 0] < width)
        & (projected_means[:, 1] >= 0)
        & (projected_means[:, 1] < height)
    )

    if not torch.any(in_frame_mask):
        return torch.zeros(height, width, colors.shape[1], dtype=colors.dtype, device=device_obj)

    depths_visible = depths[in_frame_mask]
    sort_indices = torch.argsort(depths_visible, descending=False)
    visible_indices = torch.nonzero(in_frame_mask, as_tuple=False).squeeze(1)[sort_indices]

    return gaussian_splat_2d(
        means=projected_means[visible_indices],
        covariances=projected_covariances[visible_indices],
        colors=colors[visible_indices],
        opacities=opacities[visible_indices],
        height=height,
        width=width,
        device=device_obj.type,
    )


class GaussianSplat3D(nn.Module):
    """PyTorch module for 3D Gaussian splatting with a fixed camera."""

    def __init__(
        self,
        intrinsics: torch.Tensor,
        camera_to_world: torch.Tensor,
        gaussians: Optional[dict] = None,
        num_gaussians: Optional[int] = None,
        num_channels: int = 3,
        height: int = 256,
        width: int = 256,
        device: Optional[str] = None,
        near_plane: float = 1e-4,
        min_covariance: float = 1e-4,
    ):
        super().__init__()

        if device is None:
            device = _auto_device()
        self.device_obj = torch.device(device)
        self.height = height
        self.width = width
        self.near_plane = near_plane
        self.min_covariance = min_covariance

        _validate_intrinsics(intrinsics)
        _validate_camera_to_world(camera_to_world)
        self.register_buffer("intrinsics", intrinsics.to(self.device_obj))
        self.register_buffer("camera_to_world", camera_to_world.to(self.device_obj))

        if gaussians is not None:
            self.num_gaussians = gaussians["means"].shape[0]
            self.num_channels = gaussians["colors"].shape[1]

            self.register_parameter("means", nn.Parameter(gaussians["means"].to(self.device_obj)))
            if "covariances" in gaussians:
                self.register_parameter(
                    "covariances",
                    nn.Parameter(gaussians["covariances"].to(self.device_obj)),
                )
                self.log_scales = None
            else:
                self.register_parameter(
                    "log_scales",
                    nn.Parameter(gaussians["log_scales"].to(self.device_obj)),
                )
                self.covariances = None
            self.register_parameter("colors", nn.Parameter(gaussians["colors"].to(self.device_obj)))
            self.register_parameter("opacities", nn.Parameter(gaussians["opacities"].to(self.device_obj)))
        else:
            if num_gaussians is None:
                raise ValueError("Either gaussians or num_gaussians must be provided")

            self.num_gaussians = num_gaussians
            self.num_channels = num_channels

            self.register_parameter(
                "means",
                nn.Parameter(
                    torch.randn(num_gaussians, 3, device=self.device_obj) * 0.5
                    + torch.tensor([0.0, 0.0, 3.0], device=self.device_obj)
                ),
            )
            self.register_parameter(
                "log_scales",
                nn.Parameter(torch.randn(num_gaussians, 3, device=self.device_obj) * 0.2 - 1.0),
            )
            self.register_parameter(
                "colors",
                nn.Parameter(torch.rand(num_gaussians, num_channels, device=self.device_obj)),
            )
            self.register_parameter(
                "opacities",
                nn.Parameter(torch.ones(num_gaussians, device=self.device_obj) * 0.5),
            )
            self.covariances = None

    def _build_covariance_matrix(self) -> torch.Tensor:
        if self.covariances is not None:
            return self.covariances
        scales = torch.exp(self.log_scales)
        covariance = torch.diag_embed(scales * scales)
        epsilon = torch.eye(3, device=self.device_obj).unsqueeze(0) * 1e-6
        return covariance + epsilon

    def forward(self) -> torch.Tensor:
        covariances = self._build_covariance_matrix()
        opacities = torch.sigmoid(self.opacities)
        colors = torch.clamp(self.colors, 0.0, 1.0)
        return gaussian_splat_3d(
            means=self.means,
            covariances=covariances,
            colors=colors,
            opacities=opacities,
            intrinsics=self.intrinsics,
            camera_to_world=self.camera_to_world,
            height=self.height,
            width=self.width,
            device=self.device_obj.type,
            near_plane=self.near_plane,
            min_covariance=self.min_covariance,
        )
