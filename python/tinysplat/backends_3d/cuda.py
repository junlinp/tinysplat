"""CUDA backend for 3D Gaussian splatting."""

import torch

from ..cpp import load_cuda_extension

# FastGS needs per-Gaussian statistics from the most recent render. The Metal
# backend keeps this inside its raster session; the CUDA path projects in
# PyTorch, so the equivalent state is cached here. Keys:
#   proj_means, proj_covs, proj_opacities  -- the projected splats
#   visible_indices                        -- projected -> original mapping
#   height, width, num_gaussians
#   grad_means2d_abs                       -- AbsGS sums, set by backward
_SESSION = {}

# The 2D mean gradients must NOT live in _SESSION: FastGS re-renders K views to
# build its VCD scores, every forward clears the session, and that wiped the
# gradients the training backward had just stored (absgrad read back as 0/0/0).
# They are keyed by Gaussian count so a stale set is never served after a
# densify step changes N.
_GRAD2D = {"signed": None, "abs": None, "n": 0}


def last_session():
    return _SESSION


def last_grad2d():
    return _GRAD2D

from .common import Backend3DOps


class _GaussianSplat3DCUDAFunction(torch.autograd.Function):
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
        from ..gaussian_splat_3d_core import prepare_projected_gaussians_3d

        extension = load_cuda_extension()
        if extension is None or not hasattr(extension, "gaussian_splat_3d_projected_forward_cuda"):
            raise RuntimeError("CUDA 3D backend is not available")

        # Project 3D Gaussians to 2D
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

        # The CUDA prepare returns depths as a 6th element: it skips the global
        # depth sort and the rasterizer orders each tile's bin by depth instead.
        if len(prepared) == 7:
            # CUDA path: uncompacted, plus depths (bin sort key) and a validity
            # mask (empty bbox for culled Gaussians).
            (projected_means, projected_covariances, projected_colors,
             projected_opacities, visible_indices, projected_depths,
             projected_valid) = prepared
        elif len(prepared) == 6:
            (projected_means, projected_covariances, projected_colors,
             projected_opacities, visible_indices, projected_depths) = prepared
            projected_valid = None
        else:
            (projected_means, projected_covariances, projected_colors,
             projected_opacities, visible_indices) = prepared
            projected_depths = None
            projected_valid = None

        ctx.height = height
        ctx.width = width
        ctx.near_plane = near_plane
        ctx.min_covariance = min_covariance
        ctx.sigma_radius = sigma_radius
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

        _SESSION.clear()
        _SESSION.update(
            proj_means=projected_means.detach(),
            proj_covs=projected_covariances.detach(),
            proj_opacities=projected_opacities.detach(),
            visible_indices=visible_indices.detach(),
            height=height,
            width=width,
            num_gaussians=int(means.shape[0]),
        )

        # Move to CUDA for rasterization
        projected_means = projected_means.to(torch.device("cuda"))
        projected_covariances = projected_covariances.to(torch.device("cuda"))
        projected_colors = projected_colors.to(torch.device("cuda"))
        projected_opacities = projected_opacities.to(torch.device("cuda"))

        if hasattr(extension, "gaussian_splat_3d_projected_forward_binned_cuda"):
            image, tile_starts, tile_bins = extension.gaussian_splat_3d_projected_forward_binned_cuda(
                projected_means,
                projected_covariances,
                projected_colors,
                projected_opacities,
                height,
                width,
                min_covariance,
                sigma_radius,
                projected_depths.to(torch.device("cuda")).contiguous()
                if projected_depths is not None else torch.empty(0, device="cuda"),
                projected_valid.to(torch.device("cuda")).contiguous()
                if projected_valid is not None else torch.empty(0, dtype=torch.bool, device="cuda"),
            )
            # Hand the bins to the backward; rebuilding them there repeated
            # count + scan + fill + sort for no reason (~13% of GPU time).
            ctx.tile_starts = tile_starts
            ctx.tile_bins = tile_bins
            ctx.valid = (projected_valid.to(torch.device("cuda")).contiguous()
                         if projected_valid is not None else None)
            return image

        ctx.tile_starts = None
        ctx.tile_bins = None
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

        if visible_indices.numel() == 0:
            return (None,) * 11

        extension = load_cuda_extension()
        if extension is None or not hasattr(extension, "gaussian_splat_3d_projected_backward_cuda"):
            raise RuntimeError("CUDA 3D backward backend is not available")

        # Ensure all inputs are on CUDA for the backward kernel
        proj_means_cuda = projected_means.to(torch.device("cuda"))
        proj_covs_cuda = projected_covariances.to(torch.device("cuda"))
        proj_colors_cuda = projected_colors.to(torch.device("cuda"))
        proj_opacities_cuda = projected_opacities.to(torch.device("cuda"))

        # 2D backward: get gradients for projected Gaussians
        (
            grad_proj_means,
            grad_proj_covs,
            grad_proj_colors,
            grad_proj_opacities,
            grad_proj_means_abs,
        ) = (
            extension.gaussian_splat_3d_projected_backward_cuda(
                grad_output,
                proj_means_cuda,
                proj_covs_cuda,
                proj_colors_cuda,
                proj_opacities_cuda,
                ctx.height,
                ctx.width,
                ctx.min_covariance,
                ctx.sigma_radius,
                ctx.tile_starts if ctx.tile_starts is not None else torch.empty(0),
                ctx.tile_bins if ctx.tile_bins is not None else torch.empty(0),
                ctx.valid if getattr(ctx, "valid", None) is not None
                else torch.empty(0, dtype=torch.bool),
            )
        )

        # Scatter to original Gaussian indices here, while visible_indices is in
        # scope, and keep it outside _SESSION so later forwards cannot clear it.
        n_orig = int(means.shape[0])
        idx = visible_indices.to(grad_proj_means.device).reshape(-1)
        k = min(idx.numel(), grad_proj_means.shape[0])
        signed_full = torch.zeros(n_orig, 2, device=grad_proj_means.device,
                                  dtype=grad_proj_means.dtype)
        abs_full = torch.zeros_like(signed_full)
        if k:
            signed_full.index_copy_(0, idx[:k], grad_proj_means[:k].detach())
            abs_full.index_copy_(0, idx[:k], grad_proj_means_abs[:k].detach())
        _GRAD2D["signed"] = signed_full
        _GRAD2D["abs"] = abs_full
        _GRAD2D["n"] = n_orig

        needs = ctx.needs_input_grad

        # Re-compute the projection with gradient tracking (like MPS backend)
        # This avoids the PyTorch advanced indexing backward bug
        means_req = means.detach().clone().requires_grad_(needs[0])
        cov_req = covariances.detach().clone().requires_grad_(needs[1])
        intrinsics_req = intrinsics.detach().clone().requires_grad_(needs[4])
        pose_req = camera_to_world.detach().clone().requires_grad_(needs[5])

        with torch.enable_grad():
            from ..gaussian_splat_3d_core import project_gaussians_3d_to_2d
            reproj_means, reproj_covs, _, _ = project_gaussians_3d_to_2d(
                means=means_req,
                covariances=cov_req,
                intrinsics=intrinsics_req,
                camera_to_world=pose_req,
                near_plane=ctx.near_plane,
                min_covariance=ctx.min_covariance,
            )
            # Filter to visible indices (same filtering as forward)
            reproj_means = reproj_means[visible_indices]
            reproj_covs = reproj_covs[visible_indices]

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
                outputs=(reproj_means, reproj_covs),
                inputs=proj_inputs,
                grad_outputs=(grad_proj_means, grad_proj_covs),
                allow_unused=True,
            )

        # Scatter gradient contributions back to full tensors
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
            if grad_proj_colors is not None:
                grad_colors.index_add_(0, visible_indices, grad_proj_colors)

        grad_opacities = None
        if needs[3]:
            grad_opacities = torch.zeros_like(opacities)
            if grad_proj_opacities is not None:
                grad_opacities.index_add_(0, visible_indices, grad_proj_opacities)

        return (
            grad_means,         # grad_means
            grad_covariances,   # grad_covariances
            grad_colors,        # grad_colors
            grad_opacities,     # grad_opacities
            None,               # intrinsics
            None,               # camera_to_world
            None,               # height
            None,               # width
            None,               # near_plane
            None,               # min_covariance
            None,               # sigma_radius
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
    return _GaussianSplat3DCUDAFunction.apply(
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


CUDA_BACKEND_3D = Backend3DOps(
    name="cuda",
    render=render_cuda_3d,
    is_compiled=True,
)
