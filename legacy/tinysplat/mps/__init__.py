"""MPS-specific kernels for TinySplat.

These kernels execute entirely on MPS tensors and provide a stable import
surface for the MPS backends.
"""

import os
from functools import lru_cache
from pathlib import Path

import torch

from tinysplat.gaussian_splat_3d_core import (
    prepare_projected_gaussians_3d,
    project_gaussians_3d_to_2d,
)


MPS_GAUSSIAN_CHUNK_SIZE = 128
MPS_SHADER_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// Front-to-back alpha compositing.
// Gaussians MUST be sorted front-to-back (ascending depth).
kernel void gaussian_splat_2d_forward(
    device const float* dispatch_control [[buffer(0)]],
    device const float2* means [[buffer(1)]],
    device const float4* covariances [[buffer(2)]],
    device const float* colors [[buffer(3)]],
    device const float* opacities [[buffer(4)]],
    constant uint& num_gaussians [[buffer(5)]],
    constant uint& num_channels [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    constant uint& height [[buffer(8)]],
    device float* output [[buffer(9)]],
    uint idx [[thread_position_in_grid]]
) {
    (void)dispatch_control;
    if (idx >= width * height) return;

    const uint x = idx % width;
    const uint y = idx / width;

    float out0 = 0.0f;
    float out1 = 0.0f;
    float out2 = 0.0f;
    float out3 = 0.0f;
    float T = 1.0f; // transmittance

    for (uint i = 0; i < num_gaussians && T > 1e-4f; ++i) {
        const float2 mean = means[i];
        const float4 cov = covariances[i];
        const float cov_xx = cov.x;
        const float cov_xy = cov.y;
        const float cov_yx = cov.z;
        const float cov_yy = cov.w;

        const float det = cov_xx * cov_yy - cov_xy * cov_yx;
        if (det <= 1e-8f) continue;

        const float inv_det = 1.0f / det;
        const float inv_xx = cov_yy * inv_det;
        const float inv_xy = -cov_xy * inv_det;
        const float inv_yx = -cov_yx * inv_det;
        const float inv_yy = cov_xx * inv_det;

        const float dx = float(x) - mean.x;
        const float dy = float(y) - mean.y;
        const float quad =
            dx * (inv_xx * dx + inv_xy * dy) +
            dy * (inv_yx * dx + inv_yy * dy);

        const float gaussian = exp(-0.5f * quad);
        const float alpha = min(gaussian * opacities[i], 0.99f);
        if (alpha < 1e-4f) continue;

        const float w = T * alpha;
        const uint color_offset = i * num_channels;
        if (num_channels > 0) out0 += w * colors[color_offset];
        if (num_channels > 1) out1 += w * colors[color_offset + 1];
        if (num_channels > 2) out2 += w * colors[color_offset + 2];
        if (num_channels > 3) out3 += w * colors[color_offset + 3];
        T *= (1.0f - alpha);
    }

    const uint out_offset = (y * width + x) * num_channels;
    if (num_channels > 0) output[out_offset] = out0;
    if (num_channels > 1) output[out_offset + 1] = out1;
    if (num_channels > 2) output[out_offset + 2] = out2;
    if (num_channels > 3) output[out_offset + 3] = out3;
}

// Alpha compositing backward pass.
// Gaussians MUST be sorted front-to-back (ascending depth).
kernel void gaussian_splat_2d_backward(
    device const float* dispatch_control [[buffer(0)]],
    device const float2* means [[buffer(1)]],
    device const float4* covariances [[buffer(2)]],
    device const float* colors [[buffer(3)]],
    device const float* opacities [[buffer(4)]],
    device const float* grad_output [[buffer(5)]],
    constant uint& num_gaussians [[buffer(6)]],
    constant uint& num_channels [[buffer(7)]],
    constant uint& width [[buffer(8)]],
    constant uint& height [[buffer(9)]],
    device atomic_float* grad_means [[buffer(10)]],
    device atomic_float* grad_covariances [[buffer(11)]],
    device atomic_float* grad_colors [[buffer(12)]],
    device atomic_float* grad_opacities [[buffer(13)]],
    uint idx [[thread_position_in_grid]]
) {
    (void)dispatch_control;
    if (idx >= width * height) return;

    const uint x = idx % width;
    const uint y = idx / width;

    const uint grad_offset = (y * width + x) * num_channels;
    const float g0 = num_channels > 0 ? grad_output[grad_offset] : 0.0f;
    const float g1 = num_channels > 1 ? grad_output[grad_offset + 1] : 0.0f;
    const float g2 = num_channels > 2 ? grad_output[grad_offset + 2] : 0.0f;
    const float g3 = num_channels > 3 ? grad_output[grad_offset + 3] : 0.0f;

    float dL_dTn = 0.0f; // dL/d(transmittance after gaussian i)
    float T = 1.0f;

    for (uint i = 0; i < num_gaussians && T > 1e-4f; ++i) {
        const float2 mean = means[i];
        const float4 cov = covariances[i];
        const float cov_xx = cov.x;
        const float cov_xy = cov.y;
        const float cov_yx = cov.z;
        const float cov_yy = cov.w;

        const float det = cov_xx * cov_yy - cov_xy * cov_yx;
        if (det <= 1e-8f) continue;

        const float inv_det = 1.0f / det;
        const float inv_xx = cov_yy * inv_det;
        const float inv_xy = -cov_xy * inv_det;
        const float inv_yx = -cov_yx * inv_det;
        const float inv_yy = cov_xx * inv_det;

        const float dx = float(x) - mean.x;
        const float dy = float(y) - mean.y;
        const float quad =
            dx * (inv_xx * dx + inv_xy * dy) +
            dy * (inv_yx * dx + inv_yy * dy);

        const float gaussian = exp(-0.5f * quad);
        const float alpha = min(gaussian * opacities[i], 0.99f);
        if (alpha < 1e-4f) continue;

        const uint color_offset = i * num_channels;
        const float ci0 = num_channels > 0 ? colors[color_offset] : 0.0f;
        const float ci1 = num_channels > 1 ? colors[color_offset + 1] : 0.0f;
        const float ci2 = num_channels > 2 ? colors[color_offset + 2] : 0.0f;
        const float ci3 = num_channels > 3 ? colors[color_offset + 3] : 0.0f;

        // dL/d(gaussian_i) = T * (dL_dC . ci - dL_dTn) * opacity
        float dL_dC_dot_ci = 0.0f;
        if (num_channels > 0) dL_dC_dot_ci += g0 * ci0;
        if (num_channels > 1) dL_dC_dot_ci += g1 * ci1;
        if (num_channels > 2) dL_dC_dot_ci += g2 * ci2;
        if (num_channels > 3) dL_dC_dot_ci += g3 * ci3;

        const float dL_dgaussian = T * opacities[i] * (dL_dC_dot_ci - dL_dTn);

        // Update transmittance and dL_dTn before using them for next gaussian
        const float Tn = T * (1.0f - alpha);
        dL_dTn = dL_dTn * (1.0f - alpha);

        if (num_channels > 0) {
            dL_dTn += g0 * ci0 * alpha;
        }
        if (num_channels > 1) {
            dL_dTn += g1 * ci1 * alpha;
        }
        if (num_channels > 2) {
            dL_dTn += g2 * ci2 * alpha;
        }
        if (num_channels > 3) {
            dL_dTn += g3 * ci3 * alpha;
        }

        // dL/d(color_i) = dL/dC * T * alpha
        if (num_channels > 0) {
            atomic_fetch_add_explicit(&grad_colors[color_offset], g0 * T * alpha, memory_order_relaxed);
        }
        if (num_channels > 1) {
            atomic_fetch_add_explicit(&grad_colors[color_offset + 1], g1 * T * alpha, memory_order_relaxed);
        }
        if (num_channels > 2) {
            atomic_fetch_add_explicit(&grad_colors[color_offset + 2], g2 * T * alpha, memory_order_relaxed);
        }
        if (num_channels > 3) {
            atomic_fetch_add_explicit(&grad_colors[color_offset + 3], g3 * T * alpha, memory_order_relaxed);
        }

        // dL/d(opacity_i) = dL/d(gaussian_i) * gaussian
        atomic_fetch_add_explicit(&grad_opacities[i], dL_dgaussian * gaussian, memory_order_relaxed);

        // dL/d(mean_i) = dL/d(gaussian_i) * gaussian * (-0.5 * d(quad)/d(mean))
        const float ad_x = inv_xx * dx + inv_xy * dy;
        const float ad_y = inv_yx * dx + inv_yy * dy;
        const float common = dL_dgaussian * gaussian * 0.5f;

        atomic_fetch_add_explicit(&grad_means[i * 2], dL_dgaussian * gaussian * ad_x, memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_means[i * 2 + 1], dL_dgaussian * gaussian * ad_y, memory_order_relaxed);

        // dL/d(cov_i) = dL/d(gaussian_i) * gaussian * 0.5 * (ad * ad^T - inv)
        atomic_fetch_add_explicit(&grad_covariances[i * 4], common * (ad_x * ad_x - inv_xx), memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_covariances[i * 4 + 1], common * (ad_x * ad_y - inv_xy), memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_covariances[i * 4 + 2], common * (ad_y * ad_x - inv_yx), memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_covariances[i * 4 + 3], common * (ad_y * ad_y - inv_yy), memory_order_relaxed);

        T = Tn;
    }
}
"""


@lru_cache(maxsize=1)
def load_mps_shader_library():
    """Compile the direct Metal shader bridge exposed by torch.mps."""
    if not hasattr(torch, "mps") or not torch.backends.mps.is_available():
        return None

    try:
        return torch.mps.compile_shader(MPS_SHADER_SOURCE)
    except Exception:
        return None


@lru_cache(maxsize=1)
def load_mps_extension():
    """Try to build and import the TinySplat Metal extension."""
    if os.environ.get("TINYSPLAT_BUILD_EXTENSIONS", "1") == "0":
        return None

    try:
        from torch.utils.cpp_extension import load
    except ImportError:
        return None

    source_path = Path(__file__).resolve().parent / "tinysplat_mps.mm"
    build_dir = Path(__file__).resolve().parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    try:
        return load(
            name="tinysplat_mps_ext",
            sources=[str(source_path)],
            build_directory=str(build_dir),
            extra_cflags=["-O3", "-std=c++17", "-fobjc-arc"],
            extra_ldflags=["-framework", "Foundation", "-framework", "Metal"],
            verbose=False,
        )
    except Exception:
        return None


HAS_COMPILED_MPS_EXTENSION = (
    load_mps_shader_library() is not None or load_mps_extension() is not None
)


def _require_compiled_mps_shader(
    means: torch.Tensor,
    covariances: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    *,
    allow_four_channels: bool,
) -> object:
    """Validate that the strict compiled MPS path can be used."""
    shader_library = load_mps_shader_library()
    if shader_library is None:
        raise RuntimeError("Compiled MPS shader library is unavailable.")
    if means.device.type != "mps":
        raise RuntimeError("MPS backend requires `means` to be on the MPS device.")
    if covariances.device.type != "mps":
        raise RuntimeError("MPS backend requires `covariances` to be on the MPS device.")
    if colors.device.type != "mps":
        raise RuntimeError("MPS backend requires `colors` to be on the MPS device.")
    if opacities.device.type != "mps":
        raise RuntimeError("MPS backend requires `opacities` to be on the MPS device.")
    if means.dtype != torch.float32:
        raise NotImplementedError("Compiled MPS backend currently requires float32 means.")
    if covariances.dtype != torch.float32:
        raise NotImplementedError("Compiled MPS backend currently requires float32 covariances.")
    if colors.dtype != torch.float32:
        raise NotImplementedError("Compiled MPS backend currently requires float32 colors.")
    if opacities.dtype != torch.float32:
        raise NotImplementedError("Compiled MPS backend currently requires float32 opacities.")
    if covariances.shape[-2:] != (2, 2):
        raise NotImplementedError("Compiled MPS backend currently requires 2x2 covariances.")
    min_channels = 1
    max_channels = 4 if allow_four_channels else 3
    if not (min_channels <= colors.shape[1] <= max_channels):
        raise NotImplementedError(
            f"Compiled MPS backend currently supports {min_channels}-{max_channels} color channels."
        )
    return shader_library


def _gaussian_splat_2d_forward_mps_pytorch(
    means: torch.Tensor,
    covariances: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    height: int,
    width: int,
):
    """Render 2D Gaussians on MPS with front-to-back alpha compositing."""
    num_gaussians = means.shape[0]
    num_channels = colors.shape[1]
    device = means.device
    coord_dtype = means.dtype

    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, dtype=coord_dtype, device=device),
        torch.arange(width, dtype=coord_dtype, device=device),
        indexing="ij",
    )
    coords = torch.stack([x_coords, y_coords], dim=-1)

    image = torch.zeros(height, width, num_channels, dtype=colors.dtype, device=device)
    T = torch.ones(height, width, dtype=coord_dtype, device=device)

    for start_idx in range(0, num_gaussians, MPS_GAUSSIAN_CHUNK_SIZE):
        end_idx = min(start_idx + MPS_GAUSSIAN_CHUNK_SIZE, num_gaussians)

        means_chunk = means[start_idx:end_idx]
        covariances_chunk = covariances[start_idx:end_idx]
        colors_chunk = colors[start_idx:end_idx]
        opacities_chunk = opacities[start_idx:end_idx]

        diff = coords.unsqueeze(0) - means_chunk.unsqueeze(1).unsqueeze(1)
        inv_covariances = torch.linalg.inv(covariances_chunk)

        quad_form = (
            torch.matmul(
                torch.matmul(diff.unsqueeze(-2), inv_covariances.unsqueeze(1).unsqueeze(1)),
                diff.unsqueeze(-1),
            )
            .squeeze(-1)
            .squeeze(-1)
        )

        gaussian_values = torch.exp(-0.5 * quad_form)
        raw_alpha = gaussian_values * opacities_chunk.unsqueeze(1).unsqueeze(2)
        alpha = torch.clamp(raw_alpha, max=0.99)

        w = T.unsqueeze(-1) * alpha.unsqueeze(-1)
        image = image + (w * colors_chunk.unsqueeze(1).unsqueeze(1)).sum(dim=0)
        T = T * (1.0 - alpha).prod(dim=0)

    return image, []


def gaussian_splat_2d_forward_mps(
    means: torch.Tensor,
    covariances: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    height: int,
    width: int,
):
    """Render 2D Gaussians on MPS using the compiled Metal shader only."""
    shader_library = _require_compiled_mps_shader(
        means,
        covariances,
        colors,
        opacities,
        allow_four_channels=True,
    )
    output = torch.empty((height, width, colors.shape[1]), device=means.device, dtype=colors.dtype)
    flat_output = output.reshape(-1)
    dispatch_control = torch.empty((height * width,), device=means.device, dtype=means.dtype)
    flat_covariances = torch.stack(
        [
            covariances[:, 0, 0],
            covariances[:, 0, 1],
            covariances[:, 1, 0],
            covariances[:, 1, 1],
        ],
        dim=1,
    ).contiguous()
    shader_library.gaussian_splat_2d_forward(
        dispatch_control,
        means.contiguous(),
        flat_covariances,
        colors.contiguous().reshape(-1),
        opacities.contiguous(),
        means.shape[0],
        colors.shape[1],
        width,
        height,
        flat_output,
    )
    return output, []


def gaussian_splat_2d_backward_mps(
    grad_output: torch.Tensor,
    means: torch.Tensor,
    covariances: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    height: int,
    width: int,
    intermediates,
    needs_input_grad,
):
    """Differentiate 2D Gaussian splats on MPS with the compiled shader only."""
    del intermediates

    means_requires_grad = needs_input_grad[0] if len(needs_input_grad) > 0 else False
    cov_requires_grad = needs_input_grad[1] if len(needs_input_grad) > 1 else False
    colors_requires_grad = needs_input_grad[2] if len(needs_input_grad) > 2 else False
    opacities_requires_grad = needs_input_grad[3] if len(needs_input_grad) > 3 else False

    if grad_output.device.type != "mps":
        raise RuntimeError("MPS backward requires `grad_output` to be on the MPS device.")
    if grad_output.dtype != torch.float32:
        raise NotImplementedError("Compiled MPS backward currently requires float32 grad_output.")

    shader_library = _require_compiled_mps_shader(
        means,
        covariances,
        colors,
        opacities,
        allow_four_channels=True,
    )

    flat_covariances = torch.stack(
        [
            covariances[:, 0, 0],
            covariances[:, 0, 1],
            covariances[:, 1, 0],
            covariances[:, 1, 1],
        ],
        dim=1,
    ).contiguous()
    grad_means_flat = torch.zeros((means.shape[0] * 2,), device=means.device, dtype=means.dtype)
    grad_covariances_flat = torch.zeros(
        (means.shape[0] * 4,), device=means.device, dtype=means.dtype
    )
    grad_colors_flat = torch.zeros(
        (colors.shape[0] * colors.shape[1],), device=colors.device, dtype=colors.dtype
    )
    grad_opacities = torch.zeros_like(opacities)
    dispatch_control = torch.empty((height * width,), device=means.device, dtype=means.dtype)

    shader_library.gaussian_splat_2d_backward(
        dispatch_control,
        means.contiguous(),
        flat_covariances,
        colors.contiguous().reshape(-1),
        opacities.contiguous(),
        grad_output.contiguous().reshape(-1),
        means.shape[0],
        colors.shape[1],
        width,
        height,
        grad_means_flat,
        grad_covariances_flat,
        grad_colors_flat,
        grad_opacities,
    )

    grad_means = grad_means_flat.view_as(means) if means_requires_grad else None
    grad_cov = grad_covariances_flat.view(means.shape[0], 2, 2) if cov_requires_grad else None
    grad_colors = grad_colors_flat.view_as(colors) if colors_requires_grad else None
    grad_opacities = grad_opacities if opacities_requires_grad else None
    return grad_means, grad_cov, grad_colors, grad_opacities


def gaussian_splat_3d_forward_mps(
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
    """Project 3D Gaussians and rasterize them with the MPS 2D kernel."""
    return _GaussianSplat3DMPSFunction.apply(
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


class _GaussianSplat3DMPSFunction(torch.autograd.Function):
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
        from tinysplat.gaussian_splat_2d import gaussian_splat_2d

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
            return torch.zeros(
                height, width, colors.shape[1], dtype=colors.dtype, device=means.device
            )

        (
            projected_means,
            projected_covariances,
            projected_colors,
            projected_opacities,
            visible_indices,
        ) = prepared
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

        return gaussian_splat_2d(
            projected_means,
            projected_covariances,
            projected_colors,
            projected_opacities,
            height,
            width,
            device="mps",
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

        grad_proj_means, grad_proj_covs, grad_proj_colors, grad_proj_opacities = (
            gaussian_splat_2d_backward_mps(
                grad_output=grad_output,
                means=projected_means,
                covariances=projected_covariances,
                colors=projected_colors,
                opacities=projected_opacities,
                height=ctx.height,
                width=ctx.width,
                intermediates=[],
                needs_input_grad=[True, True, True, True],
            )
        )

        needs = ctx.needs_input_grad
        means_req = means.detach().clone().requires_grad_(needs[0])
        cov_req = covariances.detach().clone().requires_grad_(needs[1])
        intrinsics_req = intrinsics.detach().clone().requires_grad_(needs[4])
        pose_req = camera_to_world.detach().clone().requires_grad_(needs[5])

        with torch.enable_grad():
            reproj_means, reproj_covs, _, _ = project_gaussians_3d_to_2d(
                means=means_req,
                covariances=cov_req,
                intrinsics=intrinsics_req,
                camera_to_world=pose_req,
                near_plane=ctx.near_plane,
                min_covariance=ctx.min_covariance,
            )
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


__all__ = [
    "HAS_COMPILED_MPS_EXTENSION",
    "gaussian_splat_2d_forward_mps",
    "gaussian_splat_2d_backward_mps",
    "gaussian_splat_3d_forward_mps",
    "load_mps_extension",
    "load_mps_shader_library",
    "register_mps_3d_core",
]


def register_mps_3d_core():
    """Register MPS-specific 3D core projection functions."""
    from tinysplat.gaussian_splat_3d_core import register_project_fn, register_prepare_fn

    def _project_mps(
        means, covariances, intrinsics, camera_to_world, near_plane=1e-4, min_covariance=1e-4
    ):
        """MPS-optimized projection: compute on same device as inputs."""
        from tinysplat.gaussian_splat_3d_core import _project_gaussians_3d_to_2d_pytorch

        return _project_gaussians_3d_to_2d_pytorch(
            means, covariances, intrinsics, camera_to_world, near_plane, min_covariance
        )

    def _prepare_mps(
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
    ):
        """MPS-optimized preparation: batched filtering and sorting."""
        from tinysplat.gaussian_splat_3d_core import _prepare_projected_gaussians_3d_pytorch

        return _prepare_projected_gaussians_3d_pytorch(
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

    register_project_fn("mps", _project_mps)
    register_prepare_fn("mps", _prepare_mps)
