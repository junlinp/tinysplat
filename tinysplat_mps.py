"""MPS-specific kernels for TinySplat.

These kernels execute entirely on MPS tensors and provide a stable import
surface for the MPS backends.
"""

import os
from functools import lru_cache
from pathlib import Path

import torch

from tinysplat.gaussian_splat_3d_core import prepare_projected_gaussians_3d


MPS_GAUSSIAN_CHUNK_SIZE = 128
MPS_SHADER_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void gaussian_splat_2d_forward(
    device float* grid_driver [[buffer(0)]],
    device const float2* means [[buffer(1)]],
    device const float4* covariances [[buffer(2)]],
    device const float* colors [[buffer(3)]],
    device const float* opacities [[buffer(4)]],
    constant uint& num_gaussians [[buffer(5)]],
    constant uint& num_channels [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    device float* output [[buffer(8)]],
    uint idx [[thread_position_in_grid]]
) {
    (void)grid_driver;

    const uint pixel_index = idx;
    const uint x = pixel_index % width;
    const uint y = pixel_index / width;

    float numerator0 = 0.0f;
    float numerator1 = 0.0f;
    float numerator2 = 0.0f;
    float numerator3 = 0.0f;
    float total_weight = 0.0f;

    for (uint i = 0; i < num_gaussians; ++i) {
        const float2 mean = means[i];
        const float4 cov = covariances[i];
        const float cov_xx = cov.x;
        const float cov_xy = cov.y;
        const float cov_yx = cov.z;
        const float cov_yy = cov.w;

        const float det = cov_xx * cov_yy - cov_xy * cov_yx;
        if (det <= 1e-8f) {
            continue;
        }

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

        const float gaussian = exp(-0.5f * quad) / (2.0f * 3.14159265359f * sqrt(det + 1e-8f));
        const float weight = gaussian * opacities[i];

        total_weight += weight;
        const uint color_offset = i * num_channels;
        if (num_channels > 0) numerator0 += weight * colors[color_offset];
        if (num_channels > 1) numerator1 += weight * colors[color_offset + 1];
        if (num_channels > 2) numerator2 += weight * colors[color_offset + 2];
        if (num_channels > 3) numerator3 += weight * colors[color_offset + 3];
    }

    const float denom = max(total_weight, 1e-8f);
    const uint out_offset = pixel_index * num_channels;
    if (num_channels > 0) output[out_offset] = numerator0 / denom;
    if (num_channels > 1) output[out_offset + 1] = numerator1 / denom;
    if (num_channels > 2) output[out_offset + 2] = numerator2 / denom;
    if (num_channels > 3) output[out_offset + 3] = numerator3 / denom;

    if (num_channels == 4) {
        output[out_offset] *= output[out_offset + 3];
        output[out_offset + 1] *= output[out_offset + 3];
        output[out_offset + 2] *= output[out_offset + 3];
    }
}

kernel void gaussian_splat_2d_backward(
    device float* grid_driver [[buffer(0)]],
    device const float2* means [[buffer(1)]],
    device const float4* covariances [[buffer(2)]],
    device const float* colors [[buffer(3)]],
    device const float* opacities [[buffer(4)]],
    device const float* grad_output [[buffer(5)]],
    constant uint& num_gaussians [[buffer(6)]],
    constant uint& num_channels [[buffer(7)]],
    constant uint& width [[buffer(8)]],
    device atomic_float* grad_means [[buffer(9)]],
    device atomic_float* grad_covariances [[buffer(10)]],
    device atomic_float* grad_colors [[buffer(11)]],
    device atomic_float* grad_opacities [[buffer(12)]],
    uint idx [[thread_position_in_grid]]
) {
    (void)grid_driver;

    const uint pixel_index = idx;
    const uint x = pixel_index % width;
    const uint y = pixel_index / width;

    float numerator0 = 0.0f;
    float numerator1 = 0.0f;
    float numerator2 = 0.0f;
    float total_weight = 0.0f;

    for (uint i = 0; i < num_gaussians; ++i) {
        const float2 mean = means[i];
        const float4 cov = covariances[i];
        const float cov_xx = cov.x;
        const float cov_xy = cov.y;
        const float cov_yx = cov.z;
        const float cov_yy = cov.w;

        const float det = cov_xx * cov_yy - cov_xy * cov_yx;
        if (det <= 1e-8f) {
            continue;
        }

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

        const float gaussian = exp(-0.5f * quad) / (2.0f * 3.14159265359f * sqrt(det + 1e-8f));
        const float weight = gaussian * opacities[i];

        total_weight += weight;
        const uint color_offset = i * num_channels;
        if (num_channels > 0) numerator0 += weight * colors[color_offset];
        if (num_channels > 1) numerator1 += weight * colors[color_offset + 1];
        if (num_channels > 2) numerator2 += weight * colors[color_offset + 2];
    }

    const float denom = max(total_weight, 1e-8f);
    const uint grad_offset = pixel_index * num_channels;
    const float out0 = numerator0 / denom;
    const float out1 = num_channels > 1 ? numerator1 / denom : 0.0f;
    const float out2 = num_channels > 2 ? numerator2 / denom : 0.0f;
    const float grad0 = num_channels > 0 ? grad_output[grad_offset] : 0.0f;
    const float grad1 = num_channels > 1 ? grad_output[grad_offset + 1] : 0.0f;
    const float grad2 = num_channels > 2 ? grad_output[grad_offset + 2] : 0.0f;

    for (uint i = 0; i < num_gaussians; ++i) {
        const float2 mean = means[i];
        const float4 cov = covariances[i];
        const float cov_xx = cov.x;
        const float cov_xy = cov.y;
        const float cov_yx = cov.z;
        const float cov_yy = cov.w;

        const float det = cov_xx * cov_yy - cov_xy * cov_yx;
        if (det <= 1e-8f) {
            continue;
        }

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

        const float gaussian = exp(-0.5f * quad) / (2.0f * 3.14159265359f * sqrt(det + 1e-8f));
        const float opacity = opacities[i];
        const float weight = gaussian * opacity;
        const uint color_offset = i * num_channels;

        float dldw = 0.0f;
        if (num_channels > 0) {
            dldw += grad0 * (colors[color_offset] - out0);
        }
        if (num_channels > 1) {
            dldw += grad1 * (colors[color_offset + 1] - out1);
        }
        if (num_channels > 2) {
            dldw += grad2 * (colors[color_offset + 2] - out2);
        }
        dldw /= denom;

        const float ad_x = inv_xx * dx + inv_xy * dy;
        const float ad_y = inv_yx * dx + inv_yy * dy;
        const float common = dldw * weight * 0.5f;

        atomic_fetch_add_explicit(&grad_means[i * 2], dldw * weight * ad_x, memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_means[i * 2 + 1], dldw * weight * ad_y, memory_order_relaxed);

        atomic_fetch_add_explicit(&grad_covariances[i * 4], common * (ad_x * ad_x - inv_xx), memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_covariances[i * 4 + 1], common * (ad_x * ad_y - inv_xy), memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_covariances[i * 4 + 2], common * (ad_y * ad_x - inv_yx), memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_covariances[i * 4 + 3], common * (ad_y * ad_y - inv_yy), memory_order_relaxed);

        if (num_channels > 0) {
            atomic_fetch_add_explicit(&grad_colors[color_offset], grad0 * weight / denom, memory_order_relaxed);
        }
        if (num_channels > 1) {
            atomic_fetch_add_explicit(&grad_colors[color_offset + 1], grad1 * weight / denom, memory_order_relaxed);
        }
        if (num_channels > 2) {
            atomic_fetch_add_explicit(&grad_colors[color_offset + 2], grad2 * weight / denom, memory_order_relaxed);
        }

        atomic_fetch_add_explicit(&grad_opacities[i], dldw * gaussian, memory_order_relaxed);
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

    source_path = Path(__file__).resolve().parent / "tinysplat" / "mps" / "tinysplat_mps.mm"
    build_dir = Path(__file__).resolve().parent / "tinysplat" / "mps" / "build"
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
    """Render 2D Gaussians on MPS with chunked tensor math."""
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

    image_numerator = torch.zeros(height, width, num_channels, dtype=colors.dtype, device=device)
    total_weight = torch.zeros(height, width, dtype=coord_dtype, device=device)

    for start_idx in range(0, num_gaussians, MPS_GAUSSIAN_CHUNK_SIZE):
        end_idx = min(start_idx + MPS_GAUSSIAN_CHUNK_SIZE, num_gaussians)

        means_chunk = means[start_idx:end_idx]
        covariances_chunk = covariances[start_idx:end_idx]
        colors_chunk = colors[start_idx:end_idx]
        opacities_chunk = opacities[start_idx:end_idx]

        diff = coords.unsqueeze(0) - means_chunk.unsqueeze(1).unsqueeze(1)
        inv_covariances = torch.linalg.inv(covariances_chunk)

        quad_form = torch.matmul(
            torch.matmul(diff.unsqueeze(-2), inv_covariances.unsqueeze(1).unsqueeze(1)),
            diff.unsqueeze(-1),
        ).squeeze(-1).squeeze(-1)

        gaussian_values = torch.exp(-0.5 * quad_form)
        det_covariances = torch.linalg.det(covariances_chunk)
        normalization = 1.0 / (2 * torch.pi * torch.sqrt(det_covariances + 1e-8))
        weighted_gaussians = gaussian_values * normalization.unsqueeze(1).unsqueeze(2)
        weighted_gaussians = weighted_gaussians * opacities_chunk.unsqueeze(1).unsqueeze(2)

        total_weight = total_weight + weighted_gaussians.sum(dim=0)
        image_numerator = image_numerator + (
            weighted_gaussians.unsqueeze(-1) * colors_chunk.unsqueeze(1).unsqueeze(1)
        ).sum(dim=0)

    image = image_numerator / torch.clamp(total_weight.unsqueeze(-1), min=1e-8)
    if num_channels == 4:
        image[..., :3] = image[..., :3] * image[..., 3:4]
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
    grid_driver = torch.empty((height * width,), device=means.device, dtype=means.dtype)
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
        grid_driver,
        means.contiguous(),
        flat_covariances,
        colors.contiguous().reshape(-1),
        opacities.contiguous(),
        means.shape[0],
        colors.shape[1],
        width,
        output.reshape(-1),
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
        allow_four_channels=False,
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
    grad_covariances_flat = torch.zeros((means.shape[0] * 4,), device=means.device, dtype=means.dtype)
    grad_colors_flat = torch.zeros((colors.shape[0] * colors.shape[1],), device=colors.device, dtype=colors.dtype)
    grad_opacities = torch.zeros_like(opacities)
    grid_driver = torch.empty((height * width,), device=means.device, dtype=means.dtype)

    shader_library.gaussian_splat_2d_backward(
        grid_driver,
        means.contiguous(),
        flat_covariances,
        colors.contiguous().reshape(-1),
        opacities.contiguous(),
        grad_output.contiguous().reshape(-1),
        means.shape[0],
        colors.shape[1],
        width,
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
    from tinysplat.gaussian_splat_2d import gaussian_splat_2d

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
    return gaussian_splat_2d(
        projected_means,
        projected_covariances,
        projected_colors,
        projected_opacities,
        height,
        width,
        device="mps",
    )


__all__ = [
    "HAS_COMPILED_MPS_EXTENSION",
    "gaussian_splat_2d_forward_mps",
    "gaussian_splat_2d_backward_mps",
    "gaussian_splat_3d_forward_mps",
    "load_mps_extension",
    "load_mps_shader_library",
]
