"""Native 3D Gaussian splatting with projected ellipses and alpha compositing."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .cpp import load_cpu_extension


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


def _world_to_camera(camera_to_world: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    rotation_c2w = camera_to_world[:3, :3]
    translation_c2w = camera_to_world[:3, 3]
    rotation_w2c = rotation_c2w.transpose(0, 1)
    translation_w2c = -rotation_w2c @ translation_c2w
    return rotation_w2c, translation_w2c


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
    """
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("means must have shape (N, 3)")
    if covariances.ndim != 3 or covariances.shape[1:] != (3, 3):
        raise ValueError("covariances must have shape (N, 3, 3)")
    _validate_intrinsics(intrinsics)
    _validate_camera_to_world(camera_to_world)

    rotation_w2c, translation_w2c = _world_to_camera(camera_to_world)
    means_camera = means @ rotation_w2c.transpose(0, 1) + translation_w2c
    covariances_camera = (
        rotation_w2c.unsqueeze(0)
        @ covariances
        @ rotation_w2c.transpose(0, 1).unsqueeze(0)
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


def _prepare_projected_gaussians_3d(
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
    projected_means, projected_covariances, depths, visible_mask = project_gaussians_3d_to_2d(
        means=means,
        covariances=covariances,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
        near_plane=near_plane,
        min_covariance=min_covariance,
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
    min_x = min_x[overlap_mask]
    max_x = max_x[overlap_mask]
    min_y = min_y[overlap_mask]
    max_y = max_y[overlap_mask]

    sort_indices = torch.argsort(visible_depths, descending=False)
    visible_means = visible_means[sort_indices]
    visible_covariances = visible_covariances[sort_indices]
    visible_colors = visible_colors[sort_indices]
    visible_opacities = visible_opacities[sort_indices]
    min_x = min_x[sort_indices]
    max_x = max_x[sort_indices]
    min_y = min_y[sort_indices]
    max_y = max_y[sort_indices]

    return (
        visible_means,
        visible_covariances,
        visible_colors,
        visible_opacities,
        visible_indices[overlap_mask][sort_indices],
    )


class _GaussianSplat3DCPUFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
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
        extension = load_cpu_extension()
        if extension is None or not hasattr(extension, "gaussian_splat_3d_forward_cpu"):
            raise RuntimeError("Compiled CPU 3D backend is not available.")

        ctx.save_for_backward(means, covariances, colors, opacities, intrinsics, camera_to_world)
        ctx.height = height
        ctx.width = width
        ctx.near_plane = near_plane
        ctx.min_covariance = min_covariance
        ctx.sigma_radius = sigma_radius
        prepared = _prepare_projected_gaussians_3d(
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
            ctx.visible_indices = None
            ctx.projected_count = 0
            return torch.zeros(height, width, colors.shape[1], dtype=colors.dtype, device=means.device)

        projected_means, projected_covariances, projected_colors, projected_opacities, visible_indices = prepared
        ctx.save_for_backward(
            means,
            covariances,
            colors,
            opacities,
            intrinsics,
            camera_to_world,
            projected_means,
            projected_covariances,
            projected_colors,
            projected_opacities,
            visible_indices,
        )
        ctx.visible_indices = visible_indices
        ctx.projected_count = projected_means.shape[0]

        return extension.gaussian_splat_3d_projected_forward_cpu(
            projected_means,
            projected_covariances,
            projected_colors,
            projected_opacities,
            height,
            width,
            min_covariance,
            sigma_radius,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.projected_count == 0:
            return (None, None, None, None, None, None, None, None, None, None, None)

        (
            means,
            covariances,
            colors,
            opacities,
            intrinsics,
            camera_to_world,
            projected_means,
            projected_covariances,
            projected_colors,
            projected_opacities,
            visible_indices,
        ) = ctx.saved_tensors

        extension = load_cpu_extension()
        if extension is None or not hasattr(extension, "gaussian_splat_3d_projected_backward_cpu"):
            raise RuntimeError("Compiled CPU 3D backward backend is not available.")

        grad_proj_means, grad_proj_covariances, grad_proj_colors, grad_proj_opacities = (
            extension.gaussian_splat_3d_projected_backward_cpu(
                grad_output,
                projected_means,
                projected_covariances,
                projected_colors,
                projected_opacities,
                ctx.height,
                ctx.width,
                ctx.min_covariance,
                ctx.sigma_radius,
            )
        )

        needs = ctx.needs_input_grad
        means_req = means.detach().clone().requires_grad_(needs[0])
        cov_req = covariances.detach().clone().requires_grad_(needs[1])
        intrinsics_req = intrinsics.detach().clone().requires_grad_(needs[4])
        pose_req = camera_to_world.detach().clone().requires_grad_(needs[5])

        with torch.enable_grad():
            reproj = project_gaussians_3d_to_2d(
                means=means_req,
                covariances=cov_req,
                intrinsics=intrinsics_req,
                camera_to_world=pose_req,
                near_plane=ctx.near_plane,
                min_covariance=ctx.min_covariance,
            )
            reproj_means = reproj[0].index_select(0, visible_indices)
            reproj_covariances = reproj[1].index_select(0, visible_indices)

            proj_inputs = []
            if needs[0]:
                proj_inputs.append(means_req)
            if needs[1]:
                proj_inputs.append(cov_req)
            if needs[4]:
                proj_inputs.append(intrinsics_req)
            if needs[5]:
                proj_inputs.append(pose_req)

            proj_grads = torch.autograd.grad(
                outputs=(reproj_means, reproj_covariances),
                inputs=proj_inputs,
                grad_outputs=(grad_proj_means, grad_proj_covariances),
                allow_unused=True,
            )

        grad_idx = 0
        grad_means = proj_grads[grad_idx] if needs[0] else None
        if needs[0]:
            grad_idx += 1
        grad_covariances = proj_grads[grad_idx] if needs[1] else None
        if needs[1]:
            grad_idx += 1

        grad_colors = None
        if needs[2]:
            grad_colors = torch.zeros_like(colors)
            grad_colors.index_add_(0, visible_indices, grad_proj_colors)

        grad_opacities = None
        if needs[3]:
            grad_opacities = torch.zeros_like(opacities)
            grad_opacities.index_add_(0, visible_indices, grad_proj_opacities)

        grad_intrinsics = proj_grads[grad_idx] if needs[4] else None
        if needs[4]:
            grad_idx += 1
        grad_pose = proj_grads[grad_idx] if needs[5] else None

        return (
            grad_means,
            grad_covariances,
            grad_colors,
            grad_opacities,
            grad_intrinsics,
            grad_pose,
            None,
            None,
            None,
            None,
            None,
        )


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

    if device_obj.type == "cpu":
        extension = load_cpu_extension()
        if extension is not None and hasattr(extension, "gaussian_splat_3d_projected_forward_cpu"):
            return _GaussianSplat3DCPUFunction.apply(
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
        raise RuntimeError("Compiled CPU 3D backend is required but unavailable.")

    raise NotImplementedError("Only the compiled CPU 3D backend is currently implemented.")


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

        _validate_intrinsics(intrinsics)
        _validate_camera_to_world(camera_to_world)
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
