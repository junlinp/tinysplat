#pragma once

#include <cuda_runtime.h>

namespace tinysplat {
namespace cuda {
namespace detail {

constexpr int kTileSize = 16;
constexpr float kEps = 1e-8f;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kSigmaRadius = 4.0f;

struct __align__(8) Gaussian2D {
  float mean_x;
  float mean_y;
  float inv_xx;
  float inv_xy;
  float inv_yx;
  float inv_yy;
  float normalization;
  float det_ratio;
  int min_x;
  int max_x;
  int min_y;
  int max_y;
};

__global__ void precompute_gaussians_kernel(const float* __restrict__ means,
                                            const float* __restrict__ covs,
                                            Gaussian2D* __restrict__ out, int n, int h,
                                            int w);

__global__ void count_tile_membership_kernel(const Gaussian2D* __restrict__ gaussians,
                                             int* __restrict__ tile_counts, int tiles_x,
                                             int tiles_y, int n);

__global__ void assign_tile_bins_kernel(const Gaussian2D* __restrict__ gaussians,
                                        int* __restrict__ tile_counts,
                                        const int* __restrict__ tile_starts,
                                        int* __restrict__ tile_bins, int tiles_x, int tiles_y,
                                        int n);

__global__ void rasterize_alpha_forward_kernel(const Gaussian2D* __restrict__ gaussians,
                                               const float* __restrict__ colors,
                                               const float* __restrict__ opacities,
                                               const int* __restrict__ tile_starts,
                                               const int* __restrict__ tile_bins,
                                               float* __restrict__ output, int h, int w, int c,
                                               int tiles_x);

__global__ void rasterize_weighted_forward_kernel(
    const Gaussian2D* __restrict__ gaussians, const float* __restrict__ colors,
    const float* __restrict__ opacities, const int* __restrict__ tile_starts,
    const int* __restrict__ tile_bins, float* __restrict__ output, int h, int w, int c,
    int tiles_x);

__global__ void rasterize_backward_kernel(
    const float* __restrict__ grad_output, const Gaussian2D* __restrict__ gaussians,
    const float* __restrict__ colors, const float* __restrict__ opacities,
    const int* __restrict__ tile_starts, const int* __restrict__ tile_bins,
    float* __restrict__ grad_means, float* __restrict__ grad_covs,
    float* __restrict__ grad_colors, float* __restrict__ grad_opacities, int n, int h, int w,
    int c, int tiles_x);

}  // namespace detail
}  // namespace cuda
}  // namespace tinysplat
