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
    covariances: Optional[torch.Tensor] = None,
    colors: torch.Tensor = None,
    opacities: torch.Tensor = None,
    intrinsics: torch.Tensor = None,
    camera_to_world: torch.Tensor = None,
    height: int = 0,
    width: int = 0,
    device: Optional[str] = None,
    near_plane: float = 1e-4,
    min_covariance: float = 1e-4,
    sigma_radius: float = 3.0,
    log_scales: Optional[torch.Tensor] = None,
    rotations: Optional[torch.Tensor] = None,
    sh_coeffs: Optional[torch.Tensor] = None,
    sh_degree: int = 0,
) -> torch.Tensor:
    """
    Render 3D Gaussians using camera intrinsics and a camera-to-world pose.

    On Metal, pass log_scales (N,3) and raw wxyz rotations (N,4) to fuse
    world covariance into the rasterizer (skips building (N,3,3) on MPS).
    Pass sh_coeffs (N,16,3) and sh_degree to evaluate view-dependent color on GPU.
    """
    if device is None:
        device = _auto_device()
    device_obj = torch.device(device)

    means = means.to(device_obj)
    opacities = opacities.to(device_obj)
    intrinsics = intrinsics.to(device_obj)
    camera_to_world = camera_to_world.to(device_obj)
    if colors is None:
        colors = torch.zeros(means.shape[0], 3, device=device_obj, dtype=means.dtype)
    else:
        colors = colors.to(device_obj)
    if covariances is not None:
        covariances = covariances.to(device_obj)
    if log_scales is not None:
        log_scales = log_scales.to(device_obj)
    if rotations is not None:
        rotations = rotations.to(device_obj)
    if sh_coeffs is not None:
        sh_coeffs = sh_coeffs.to(device_obj)

    backend = get_backend_3d(device_obj.type)
    render_kwargs = dict(
        means=means,
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
    if getattr(backend, "name", "") == "metal" and log_scales is not None and rotations is not None:
        render_kwargs["log_scales"] = log_scales
        render_kwargs["rotations"] = rotations
        if sh_coeffs is not None:
            render_kwargs["sh_coeffs"] = sh_coeffs
            render_kwargs["sh_degree"] = int(sh_degree)
        if covariances is not None:
            render_kwargs["covariances"] = covariances
    else:
        if covariances is None:
            raise ValueError("gaussian_splat_3d requires covariances unless Metal log_scales+rotations are provided")
        render_kwargs["covariances"] = covariances
    return backend.render(**render_kwargs)


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
