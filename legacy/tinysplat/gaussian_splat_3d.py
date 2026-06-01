"""Public interface for 3D Gaussian splatting."""

from typing import Optional

import torch
import torch.nn as nn

from .backends_3d import get_backend_3d
from .gaussian_splat_3d_core import (
    project_gaussians_3d_to_2d,
    validate_camera_to_world,
    validate_intrinsics,
)


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    sigma_radius: float = 3.0,
) -> torch.Tensor:
    """
    Render 3D Gaussians using camera intrinsics and a camera-to-world pose.
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

    backend = get_backend_3d(device_obj.type)
    return backend.render(
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
        sigma_radius: float = 3.0,
    ):
        super().__init__()

        if device is None:
            device = _auto_device()
        self.device_obj = torch.device(device)
        self.height = height
        self.width = width
        self.near_plane = near_plane
        self.min_covariance = min_covariance
        self.sigma_radius = sigma_radius

        validate_intrinsics(intrinsics)
        validate_camera_to_world(camera_to_world)
        self.register_buffer("intrinsics", intrinsics.to(self.device_obj))
        self.register_buffer("camera_to_world", camera_to_world.to(self.device_obj))

        if gaussians is not None:
            self.num_gaussians = gaussians["means"].shape[0]
            self.num_channels = gaussians["colors"].shape[1]
            self.register_parameter("means", nn.Parameter(gaussians["means"].to(self.device_obj)))
            if "covariances" in gaussians:
                self.register_parameter("covariances", nn.Parameter(gaussians["covariances"].to(self.device_obj)))
                self.log_scales = None
            else:
                self.register_parameter("log_scales", nn.Parameter(gaussians["log_scales"].to(self.device_obj)))
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
            sigma_radius=self.sigma_radius,
        )
