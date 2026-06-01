"""CPU backend for 3D Gaussian splatting."""

import torch

from ..cpp import load_cpu_extension
from ..gaussian_splat_3d_core import prepare_projected_gaussians_3d, project_gaussians_3d_to_2d
from .common import Backend3DOps


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

        ctx.height = height
        ctx.width = width
        ctx.near_plane = near_plane
        ctx.min_covariance = min_covariance
        ctx.sigma_radius = sigma_radius
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


def render_cpu_3d(
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


CPU_BACKEND_3D = Backend3DOps(
    name="cpu",
    render=render_cpu_3d,
    is_compiled=True,
)
