#pragma once

#include <cstddef>

namespace tinysplat {
namespace cuda {

enum class CompositingMode {
  Weighted,
  Alpha,
};

/// Tiled 2D splat on GPU. Host buffers are contiguous float arrays.
/// means: (N,2), covs: (N,4) row-major [a,b,c,d], colors: (N,C), opacities: (N).
/// output: (H,W,C) row-major.
bool gaussian_splat_2d_forward(const float* means, const float* covs, const float* colors,
                               const float* opacities, int num_gaussians, int num_channels,
                               int height, int width, float* output_host,
                               CompositingMode mode = CompositingMode::Weighted);

/// Gradients w.r.t. means (N,2), covs (N,4), colors (N,C), opacities (N).
/// Uses weighted compositing backward (matches CPU 2D backward).
bool gaussian_splat_2d_backward(const float* grad_output, const float* means, const float* covs,
                                const float* colors, const float* opacities, int num_gaussians,
                                int num_channels, int height, int width, float* grad_means,
                                float* grad_covs, float* grad_colors, float* grad_opacities);

/// Alpha-compositing tiled forward (for projected 3D splats).
inline bool gaussian_splat_2d_forward_alpha(const float* means, const float* covs,
                                            const float* colors, const float* opacities,
                                            int num_gaussians, int num_channels, int height,
                                            int width, float* output_host) {
  return gaussian_splat_2d_forward(means, covs, colors, opacities, num_gaussians, num_channels,
                                    height, width, output_host, CompositingMode::Alpha);
}

bool cuda_available();

}  // namespace cuda
}  // namespace tinysplat
