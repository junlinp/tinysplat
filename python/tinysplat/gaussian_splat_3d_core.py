"""Shared 3D Gaussian projection helpers with device dispatch.

Provides a single API that routes to device-specific implementations
(MPS, CPU, CUDA) when available, falling back to PyTorch ops.
"""

from typing import Callable, Dict, Optional, Tuple

import os

import torch

# Escape hatch: TINYSPLAT_NO_FUSED_PROJECT=1 forces the reference PyTorch
# projection, for A/B-ing the fused CUDA kernel against it.
_DISABLE_FUSED_PROJECT = os.environ.get("TINYSPLAT_NO_FUSED_PROJECT", "") == "1"

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


class _FusedProjectCUDA(torch.autograd.Function):
    """Fused 3D->2D projection with a hand-written VJP.

    The pure-PyTorch chain below is mathematically identical but costs ~100 ms
    per iteration at N=136k, because autograd unwinds hundreds of small ops.
    The raster kernels are only 5-8 ms of a 112 ms step, so this projection --
    not rasterization -- is the training bottleneck.
    """

    @staticmethod
    def forward(ctx, means, covariances, intrinsics, camera_to_world, near_plane,
                min_covariance, height, width):
        from .cpp import load_cuda_extension

        ext = load_cuda_extension()
        pm, cov2d, depth, visible = ext.project_3d_forward_cuda(
            means.contiguous(), covariances.contiguous(),
            intrinsics, camera_to_world, float(near_plane), float(min_covariance),
            float(height or 0), float(width or 0),
        )
        ctx.save_for_backward(means, covariances, intrinsics, camera_to_world)
        ctx.near_plane = float(near_plane)
        ctx.min_covariance = float(min_covariance)
        ctx.hw = (float(height or 0), float(width or 0))
        return pm, cov2d.view(-1, 2, 2), depth, visible

    @staticmethod
    def backward(ctx, g_pm, g_cov2d, g_depth, _g_visible):
        from .cpp import load_cuda_extension

        means, covariances, intrinsics, camera_to_world = ctx.saved_tensors
        ext = load_cuda_extension()
        g_means, g_cov3 = ext.project_3d_backward_cuda(
            g_pm.contiguous(), g_cov2d.contiguous(), g_depth.contiguous(),
            means.contiguous(), covariances.contiguous(),
            intrinsics, camera_to_world, ctx.near_plane, ctx.min_covariance,
            ctx.hw[0], ctx.hw[1],
        )
        return g_means, g_cov3, None, None, None, None, None, None


def _project_gaussians_3d_to_2d_pytorch(
    means: torch.Tensor,
    covariances: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_to_world: torch.Tensor,
    near_plane: float = 1e-4,
    min_covariance: float = 1e-4,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Fused CUDA path; falls through to the reference implementation on any
    # other device or if the extension is unavailable.
    if means.is_cuda and not _DISABLE_FUSED_PROJECT:
        try:
            return _FusedProjectCUDA.apply(
                means, covariances, intrinsics, camera_to_world, near_plane,
                min_covariance, height, width
            )
        except Exception:  # pragma: no cover - fall back rather than fail training
            pass
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

    # Inria/FastGS EWA Jacobian clamp. Without it an off-axis or near-plane
    # Gaussian gets an unbounded d(screen)/d(camera) term, the 2D covariance
    # explodes, and a handful of splats dominate every gradient: measured
    # grad2d max was 706 on CUDA against 0.0037 on Metal, which has this clamp.
    # The projected mean itself stays unclamped -- only the covariance Jacobian
    # is limited.
    tan_fovx = (0.5 * float(width) / fx) if width else (cx / fx)
    tan_fovy = (0.5 * float(height) / fy) if height else (cy / fy)
    lim_x = 1.3 * tan_fovx
    lim_y = 1.3 * tan_fovy
    jx = torch.clamp(x / safe_z, -lim_x, lim_x) * safe_z
    jy = torch.clamp(y / safe_z, -lim_y, lim_y) * safe_z

    jacobian = torch.zeros(means.shape[0], 2, 3, dtype=means.dtype, device=means.device)
    jacobian[:, 0, 0] = fx / safe_z
    jacobian[:, 0, 2] = -fx * jx / (safe_z * safe_z)
    jacobian[:, 1, 1] = fy / safe_z
    jacobian[:, 1, 2] = -fy * jy / (safe_z * safe_z)

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
            height=height,
            width=width,
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


def register_cuda_3d_core():
    """CUDA-specific projection prep: keep the visibility/overlap filter, drop
    the global depth sort.

    The shared implementation sorts every visible Gaussian by depth so that
    array index order *is* depth order, which is what the CPU and MPS
    rasterizers rely on when they composite in array order. The CUDA
    rasterizer no longer needs that: it orders each tile's bin by depth
    directly, so the sort here is redundant work on every render.

    Returns the same tuple as the shared version plus the depths, which the
    CUDA backend forwards to the rasterizer as the bin sort key.
    """

    def _prepare_cuda(
        means, covariances, colors, opacities, intrinsics, camera_to_world,
        height, width, near_plane, min_covariance, sigma_radius,
    ):
        projected_means, projected_covariances, depths, visible_mask = (
            _project_gaussians_3d_to_2d_pytorch(
                means=means, covariances=covariances, intrinsics=intrinsics,
                camera_to_world=camera_to_world, near_plane=near_plane,
                min_covariance=min_covariance, height=height, width=width,
            )
        )
        if not torch.any(visible_mask):
            return None

        # Compaction is kept deliberately. Skipping it and marking culled
        # Gaussians with an empty bounding box is faster (~3%), but it feeds
        # invisible entries -- whose projected covariance is garbage, computed
        # with safe_z = 1 -- into FastGS's VCD and AbsGS statistics. Measured
        # over a 30k run that collapsed n_split from 13,710 to 3,985, held N at
        # ~195k instead of ~420k, and cost 0.54 dB PSNR. Revisit only with the
        # statistics masked by validity.
        idx = torch.nonzero(visible_mask, as_tuple=False).squeeze(1)
        vm = projected_means[idx]
        vc = projected_covariances[idx]

        cov_xx, cov_xy, cov_yy = vc[:, 0, 0], vc[:, 0, 1], vc[:, 1, 1]
        trace = cov_xx + cov_yy
        disc = torch.sqrt(torch.clamp((cov_xx - cov_yy) ** 2 + 4.0 * cov_xy * cov_xy, min=0.0))
        lambda_max = torch.clamp(0.5 * (trace + disc), min=min_covariance)
        radius = sigma_radius * torch.sqrt(lambda_max)

        overlap = (
            (torch.ceil(vm[:, 0] + radius) >= 0)
            & (torch.floor(vm[:, 0] - radius) < width)
            & (torch.ceil(vm[:, 1] + radius) >= 0)
            & (torch.floor(vm[:, 1] - radius) < height)
        )
        if not torch.any(overlap):
            return None

        keep = idx[overlap]
        # No argsort: the rasterizer orders each tile's bin by depth.
        return (
            projected_means[keep],
            projected_covariances[keep],
            colors[keep],
            opacities[keep],
            keep,
            depths[keep],
        )

    register_prepare_fn("cuda", _prepare_cuda)


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
        register_cuda_3d_core()
    except Exception:
        pass

    try:
        from tinysplat.backends_3d.cpu import register_cpu_3d_core

        register_cpu_3d_core()
    except Exception:
        pass


_auto_register()
