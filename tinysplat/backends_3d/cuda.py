"""CUDA backend for 3D Gaussian splatting."""

import torch

from ..cpp import load_cuda_extension
from .common import Backend3DOps


class _GaussianSplat3DCUDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        projected_means: torch.Tensor,
        projected_covariances: torch.Tensor,
        projected_colors: torch.Tensor,
        projected_opacities: torch.Tensor,
        height: int,
        width: int,
        min_covariance: float,
        sigma_radius: float,
    ) -> torch.Tensor:
        extension = load_cuda_extension()
        if extension is None or not hasattr(extension, "gaussian_splat_3d_projected_forward_cuda"):
            raise RuntimeError("CUDA 3D backend is not available")

        ctx.height = height
        ctx.width = width
        ctx.min_covariance = min_covariance
        ctx.sigma_radius = sigma_radius
        ctx.save_for_backward(
            projected_means,
            projected_covariances,
            projected_colors,
            projected_opacities,
        )

        return extension.gaussian_splat_3d_projected_forward_cuda(
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
        projected_means, projected_covariances, projected_colors, projected_opacities = ctx.saved_tensors

        extension = load_cuda_extension()
        if extension is None or not hasattr(extension, "gaussian_splat_3d_projected_backward_cuda"):
            raise RuntimeError("CUDA 3D backward backend is not available")

        grads = extension.gaussian_splat_3d_projected_backward_cuda(
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

        return (
            grads[0],  # grad_projected_means
            grads[1],  # grad_projected_covariances
            grads[2],  # grad_projected_colors
            grads[3],  # grad_projected_opacities
            None,      # height
            None,      # width
            None,      # min_covariance
            None,      # sigma_radius
        )


def render_cuda_3d(
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
    # Project 3D Gaussians to 2D (on CPU/GPU as appropriate)
    from ..gaussian_splat_3d_core import prepare_projected_gaussians_3d

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

    # Move to CUDA
    projected_means = projected_means.to(torch.device("cuda"))
    projected_covariances = projected_covariances.to(torch.device("cuda"))
    projected_colors = projected_colors.to(torch.device("cuda"))
    projected_opacities = projected_opacities.to(torch.device("cuda"))

    return _GaussianSplat3DCUDAFunction.apply(
        projected_means,
        projected_covariances,
        projected_colors,
        projected_opacities,
        height,
        width,
        min_covariance,
        sigma_radius,
    )


CUDA_BACKEND_3D = Backend3DOps(
    name="cuda",
    render=render_cuda_3d,
    is_compiled=True,
)
