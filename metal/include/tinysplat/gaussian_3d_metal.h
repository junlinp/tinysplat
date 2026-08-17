#pragma once

#include <cstdint>

namespace tinysplat {
namespace metal {

struct Splat3DMetalOptions {
  float near_plane = 1e-4f;
  float min_covariance = 1e-4f;
  float sigma_radius = 4.0f;
  /// Compact-box Mahalanobis scale (FastGS CB). Smaller = tighter footprint.
  float compact_box_beta = 3.0f;
  /// When true, use compact-box radius instead of plain 3-sigma AABB.
  bool use_compact_box = true;
  /// Force host backward (numeric tests).
  bool force_cpu = false;
  /// Active SH degree for fused eval (0..3). -1 means RGB colors are provided.
  int sh_degree = -1;
};

/// Returns true if a Metal device is available (macOS Apple Silicon / Metal GPU).
bool metal_available();

/// 3DGS-style tiled forward on Metal. Buffer layouts match the CUDA API:
/// means (N,3), covs row-major (N,9), colors (N,C) RGB or SH DC, opacities (N).
/// intrinsics: 3x3 row-major, camera_to_world: 4x4 row-major. output: (H,W,C).
bool gaussian_splat_3d_forward(const float* means, const float* covs, const float* colors,
                               const float* opacities, int num_gaussians, int num_channels,
                               const float* intrinsics, const float* camera_to_world, int height,
                               int width, float* output_host,
                               const Splat3DMetalOptions& opts = {});

/// Same raster as gaussian_splat_3d_forward, but builds world covariance from
/// log-scales (N,3) and raw wxyz quaternions (N,4) on GPU (matches GaussianData.covariance_matrices).
bool gaussian_splat_3d_forward_qs(const float* means, const float* log_scales, const float* quats,
                                  const float* colors, const float* opacities, int num_gaussians,
                                  int num_channels, const float* intrinsics,
                                  const float* camera_to_world, int height, int width,
                                  float* output_host, const Splat3DMetalOptions& opts = {});

/// qs forward that evaluates view-dependent RGB from SH (N,16,3) on GPU.
bool gaussian_splat_3d_forward_qs_sh(const float* means, const float* log_scales, const float* quats,
                                     const float* sh, const float* opacities, int num_gaussians,
                                     int num_channels, const float* intrinsics,
                                     const float* camera_to_world, int height, int width,
                                     float* output_host, const Splat3DMetalOptions& opts = {});

/// Backward that reuses GPU buffers from the last gaussian_splat_3d_forward
/// in this process (same N, C, H, W). Writes dL/dmean3d (N,3), dL/dcov3d (N,9),
/// plus color and opacity grads.
bool gaussian_splat_3d_session_backward(
    const float* grad_output, int num_gaussians, int num_channels, int height, int width,
    float* grad_means3d, float* grad_covs3d, float* grad_colors, float* grad_opacities,
    const Splat3DMetalOptions& opts = {});

/// Backward for a qs forward session. Writes dL/dmean3d (N,3), dL/dlog_scales (N,3),
/// dL/dquat_raw (N,4), plus color and opacity grads.
bool gaussian_splat_3d_session_backward_qs(
    const float* grad_output, int num_gaussians, int num_channels, int height, int width,
    float* grad_means3d, float* grad_log_scales, float* grad_quats, float* grad_colors,
    float* grad_opacities, const Splat3DMetalOptions& opts = {});

/// Backward for a qs+SH forward session. Writes dL/dmean3d, dL/dlog_scales,
/// dL/dquat_raw, dL/dsh (N,16,3 = N*48), and opacity grads.
bool gaussian_splat_3d_session_backward_qs_sh(
    const float* grad_output, int num_gaussians, int num_channels, int height, int width,
    float* grad_means3d, float* grad_log_scales, float* grad_quats, float* grad_sh,
    float* grad_opacities, const Splat3DMetalOptions& opts = {});
/// Backward in projected 2D space (tiled Metal; CPU fallback).
/// depths: optional camera-Z per Gaussian (N). When non-null, tile compositing
/// is sorted front-to-back by depth (matches 3D forward).
bool gaussian_splat_3d_projected_backward(
    const float* grad_output, const float* proj_means, const float* proj_covs,
    const float* colors, const float* opacities, int num_gaussians, int num_channels, int height,
    int width, float* grad_proj_means, float* grad_proj_covs, float* grad_colors,
    float* grad_opacities, const Splat3DMetalOptions& opts = {}, const float* depths = nullptr);

/// Count high-error pixels in each Gaussian footprint across one view (legacy AABB walk).
/// error_mask: H*W uint8 (1 = high error). Writes counts[N].
bool count_footprint_hits(const float* proj_means, const float* proj_covs, const float* opacities,
                          int num_gaussians, int height, int width, const uint8_t* error_mask,
                          int* counts, const Splat3DMetalOptions& opts = {});

/// FastGS metric pass: count Gaussians that composited into high-error pixels in the
/// last forward session (same tile list / compact boxes as the render). Does not
/// invalidate the session. error_mask: H*W uint8. Writes counts[N].
bool count_session_metric_hits(const uint8_t* error_mask, int* counts, int num_gaussians,
                               int height, int width);

/// Number of Gaussians in the last 2D mean-grad snapshot (0 if none).
int last_grad_means2d_count();

/// Copy last 2D mean grads (N,2) from the most recent session backward into out.
/// Returns N on success, 0 if none / max_n too small.
int copy_last_grad_means2d(float* out, int max_n);

/// AbsGS snapshot: per-pixel |dL/dmean2d| summed over the last backward (N,2).
int last_grad_means2d_abs_count();
int copy_last_grad_means2d_abs(float* out, int max_n);

/// Compact-box radii (pixels) from the last forward session (N).
int last_radii2d_count();
int copy_last_radii2d(float* out, int max_n);

}  // namespace metal
}  // namespace tinysplat
