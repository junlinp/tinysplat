#pragma once

namespace tinysplat {
namespace cuda {

struct Splat3DCudaOptions {
  float near_plane = 1e-4f;
  float min_covariance = 1e-4f;
  float sigma_radius = 4.0f;
};

/// 3DGS-style forward on GPU. means (N,3), covs row-major (N,9), colors (N,C), opacities (N).
/// intrinsics: 3x3 row-major, camera_to_world: 4x4 row-major. output: (H,W,C).
bool gaussian_splat_3d_forward(const float* means, const float* covs, const float* colors,
                               const float* opacities, int num_gaussians, int num_channels,
                               const float* intrinsics, const float* camera_to_world, int height,
                               int width, float* output_host,
                               const Splat3DCudaOptions& opts = {});

/// Backward in projected 2D space (grad w.r.t. projected means, covs, colors, opacities).
bool gaussian_splat_3d_projected_backward(
    const float* grad_output, const float* proj_means, const float* proj_covs,
    const float* colors, const float* opacities, int num_gaussians, int num_channels, int height,
    int width, float* grad_proj_means, float* grad_proj_covs, float* grad_colors,
    float* grad_opacities);

}  // namespace cuda
}  // namespace tinysplat
