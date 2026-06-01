#include "tinysplat/gaussian_2d_cuda.h"

#include <tinysplat/cuda_common.cuh>
#include <tinysplat/cuda_gaussian2d.cuh>

#include <cuda_runtime.h>

#include <vector>

namespace tinysplat {
namespace cuda {
namespace {

using detail::Gaussian2D;
using detail::assign_tile_bins_kernel;
using detail::count_tile_membership_kernel;
using detail::precompute_gaussians_kernel;
using detail::rasterize_alpha_forward_kernel;
using detail::rasterize_backward_kernel;
using detail::rasterize_weighted_forward_kernel;

struct TileRasterState {
  int* d_tile_counts = nullptr;
  int* d_tile_bins = nullptr;
  int* d_tile_starts = nullptr;
  std::vector<int> tile_starts_host;
  int num_tiles = 0;
  int total_bins = 0;

  ~TileRasterState() {
    if (d_tile_counts) {
      cudaFree(d_tile_counts);
    }
    if (d_tile_bins) {
      cudaFree(d_tile_bins);
    }
    if (d_tile_starts) {
      cudaFree(d_tile_starts);
    }
  }
};

bool count_tile_bins(TileRasterState& state, Gaussian2D* d_gaussians, int n, int tiles_x,
                     int tiles_y) {
  const int num_tiles = tiles_x * tiles_y;
  state.num_tiles = num_tiles;

  if (!TINYSPLAT_CUDA_CHECK(cudaMemset(state.d_tile_counts, 0,
                                       static_cast<size_t>(num_tiles) * sizeof(int)))) {
    return false;
  }

  const int blocks = (n + 255) / 256;
  count_tile_membership_kernel<<<blocks, 256>>>(d_gaussians, state.d_tile_counts, tiles_x, tiles_y,
                                              n);
  if (!TINYSPLAT_CUDA_CHECK(cudaGetLastError())) {
    return false;
  }

  std::vector<int> counts(static_cast<size_t>(num_tiles));
  if (!TINYSPLAT_CUDA_CHECK(cudaMemcpy(counts.data(), state.d_tile_counts,
                                       static_cast<size_t>(num_tiles) * sizeof(int),
                                       cudaMemcpyDeviceToHost))) {
    return false;
  }

  state.tile_starts_host.assign(static_cast<size_t>(num_tiles) + 1, 0);
  for (int i = 0; i < num_tiles; ++i) {
    state.tile_starts_host[static_cast<size_t>(i) + 1] =
        state.tile_starts_host[static_cast<size_t>(i)] + counts[static_cast<size_t>(i)];
  }
  state.total_bins = state.tile_starts_host[static_cast<size_t>(num_tiles)];

  return TINYSPLAT_CUDA_CHECK(
      cudaMemcpy(state.d_tile_starts, state.tile_starts_host.data(),
                 static_cast<size_t>(num_tiles + 1) * sizeof(int), cudaMemcpyHostToDevice));
}

bool assign_tile_bins(TileRasterState& state, Gaussian2D* d_gaussians, int n, int tiles_x,
                      int tiles_y) {
  if (state.total_bins <= 0 || state.d_tile_bins == nullptr) {
    return true;
  }
  const int num_tiles = tiles_x * tiles_y;
  const int blocks = (n + 255) / 256;
  if (!TINYSPLAT_CUDA_CHECK(cudaMemset(state.d_tile_counts, 0,
                                       static_cast<size_t>(num_tiles) * sizeof(int)))) {
    return false;
  }
  assign_tile_bins_kernel<<<blocks, 256>>>(d_gaussians, state.d_tile_counts, state.d_tile_starts,
                                           state.d_tile_bins, tiles_x, tiles_y, n);
  return TINYSPLAT_CUDA_CHECK(cudaGetLastError());
}

bool run_forward(const float* means, const float* covs, const float* colors,
                 const float* opacities, int num_gaussians, int num_channels, int height,
                 int width, float* output_host, CompositingMode mode) {
  if (num_gaussians <= 0 || num_channels <= 0 || height <= 0 || width <= 0) {
    return false;
  }

  const int tiles_x = (width + detail::kTileSize - 1) / detail::kTileSize;
  const int tiles_y = (height + detail::kTileSize - 1) / detail::kTileSize;
  const int num_tiles = tiles_x * tiles_y;

  float *d_means = nullptr;
  float* d_covs = nullptr;
  float* d_colors = nullptr;
  float* d_opacities = nullptr;
  Gaussian2D* d_gaussians = nullptr;
  float* d_output = nullptr;

  const size_t means_bytes = static_cast<size_t>(num_gaussians) * 2 * sizeof(float);
  const size_t covs_bytes = static_cast<size_t>(num_gaussians) * 4 * sizeof(float);
  const size_t colors_bytes =
      static_cast<size_t>(num_gaussians) * num_channels * sizeof(float);
  const size_t opac_bytes = static_cast<size_t>(num_gaussians) * sizeof(float);
  const size_t out_bytes = static_cast<size_t>(height) * width * num_channels * sizeof(float);
  const size_t gauss_bytes = static_cast<size_t>(num_gaussians) * sizeof(Gaussian2D);

  TileRasterState tiles;
  if (!TINYSPLAT_CUDA_CHECK(cudaMalloc(&d_means, means_bytes))) {
    return false;
  }
  if (!TINYSPLAT_CUDA_CHECK(cudaMalloc(&d_covs, covs_bytes))) {
    cudaFree(d_means);
    return false;
  }
  if (!TINYSPLAT_CUDA_CHECK(cudaMalloc(&d_colors, colors_bytes))) {
    cudaFree(d_means);
    cudaFree(d_covs);
    return false;
  }
  if (!TINYSPLAT_CUDA_CHECK(cudaMalloc(&d_opacities, opac_bytes))) {
    cudaFree(d_means);
    cudaFree(d_covs);
    cudaFree(d_colors);
    return false;
  }
  if (!TINYSPLAT_CUDA_CHECK(cudaMalloc(&d_gaussians, gauss_bytes))) {
    cudaFree(d_means);
    cudaFree(d_covs);
    cudaFree(d_colors);
    cudaFree(d_opacities);
    return false;
  }
  if (!TINYSPLAT_CUDA_CHECK(cudaMalloc(&d_output, out_bytes))) {
    cudaFree(d_means);
    cudaFree(d_covs);
    cudaFree(d_colors);
    cudaFree(d_opacities);
    cudaFree(d_gaussians);
    return false;
  }
  if (!TINYSPLAT_CUDA_CHECK(cudaMalloc(&tiles.d_tile_counts,
                                       static_cast<size_t>(num_tiles) * sizeof(int)))) {
    cudaFree(d_means);
    cudaFree(d_covs);
    cudaFree(d_colors);
    cudaFree(d_opacities);
    cudaFree(d_gaussians);
    cudaFree(d_output);
    return false;
  }
  if (!TINYSPLAT_CUDA_CHECK(cudaMalloc(&tiles.d_tile_starts,
                                       static_cast<size_t>(num_tiles + 1) * sizeof(int)))) {
    cudaFree(d_means);
    cudaFree(d_covs);
    cudaFree(d_colors);
    cudaFree(d_opacities);
    cudaFree(d_gaussians);
    cudaFree(d_output);
    return false;
  }

  if (!TINYSPLAT_CUDA_CHECK(cudaMemcpy(d_means, means, means_bytes, cudaMemcpyHostToDevice)) ||
      !TINYSPLAT_CUDA_CHECK(cudaMemcpy(d_covs, covs, covs_bytes, cudaMemcpyHostToDevice)) ||
      !TINYSPLAT_CUDA_CHECK(cudaMemcpy(d_colors, colors, colors_bytes, cudaMemcpyHostToDevice)) ||
      !TINYSPLAT_CUDA_CHECK(
          cudaMemcpy(d_opacities, opacities, opac_bytes, cudaMemcpyHostToDevice))) {
    cudaFree(d_means);
    cudaFree(d_covs);
    cudaFree(d_colors);
    cudaFree(d_opacities);
    cudaFree(d_gaussians);
    cudaFree(d_output);
    return false;
  }

  const int blocks = (num_gaussians + 255) / 256;
  precompute_gaussians_kernel<<<blocks, 256>>>(d_means, d_covs, d_gaussians, num_gaussians,
                                               height, width);
  if (!TINYSPLAT_CUDA_CHECK(cudaGetLastError())) {
    cudaFree(d_means);
    cudaFree(d_covs);
    cudaFree(d_colors);
    cudaFree(d_opacities);
    cudaFree(d_gaussians);
    cudaFree(d_output);
    return false;
  }

  if (!count_tile_bins(tiles, d_gaussians, num_gaussians, tiles_x, tiles_y)) {
    cudaFree(d_means);
    cudaFree(d_covs);
    cudaFree(d_colors);
    cudaFree(d_opacities);
    cudaFree(d_gaussians);
    cudaFree(d_output);
    return false;
  }

  if (tiles.total_bins > 0 &&
      !TINYSPLAT_CUDA_CHECK(cudaMalloc(&tiles.d_tile_bins,
                                       static_cast<size_t>(tiles.total_bins) * sizeof(int)))) {
    cudaFree(d_means);
    cudaFree(d_covs);
    cudaFree(d_colors);
    cudaFree(d_opacities);
    cudaFree(d_gaussians);
    cudaFree(d_output);
    return false;
  }

  if (!assign_tile_bins(tiles, d_gaussians, num_gaussians, tiles_x, tiles_y)) {
    cudaFree(d_means);
    cudaFree(d_covs);
    cudaFree(d_colors);
    cudaFree(d_opacities);
    cudaFree(d_gaussians);
    cudaFree(d_output);
    return false;
  }

  if (!TINYSPLAT_CUDA_CHECK(cudaMemset(d_output, 0, out_bytes))) {
    cudaFree(d_means);
    cudaFree(d_covs);
    cudaFree(d_colors);
    cudaFree(d_opacities);
    cudaFree(d_gaussians);
    cudaFree(d_output);
    return false;
  }

  const dim3 block(16, 16);
  const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  if (mode == CompositingMode::Alpha) {
    rasterize_alpha_forward_kernel<<<grid, block>>>(
        d_gaussians, d_colors, d_opacities, tiles.d_tile_starts, tiles.d_tile_bins, d_output,
        height, width, num_channels, tiles_x);
  } else {
    rasterize_weighted_forward_kernel<<<grid, block>>>(
        d_gaussians, d_colors, d_opacities, tiles.d_tile_starts, tiles.d_tile_bins, d_output,
        height, width, num_channels, tiles_x);
  }
  if (!TINYSPLAT_CUDA_CHECK(cudaGetLastError()) ||
      !TINYSPLAT_CUDA_CHECK(cudaMemcpy(output_host, d_output, out_bytes, cudaMemcpyDeviceToHost)) ||
      !TINYSPLAT_CUDA_CHECK(cudaDeviceSynchronize())) {
    cudaFree(d_means);
    cudaFree(d_covs);
    cudaFree(d_colors);
    cudaFree(d_opacities);
    cudaFree(d_gaussians);
    cudaFree(d_output);
    return false;
  }

  cudaFree(d_means);
  cudaFree(d_covs);
  cudaFree(d_colors);
  cudaFree(d_opacities);
  cudaFree(d_gaussians);
  cudaFree(d_output);
  tiles.d_tile_bins = nullptr;
  tiles.d_tile_counts = nullptr;
  tiles.d_tile_starts = nullptr;
  return true;
}

}  // namespace

bool cuda_available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

bool gaussian_splat_2d_forward(const float* means, const float* covs, const float* colors,
                               const float* opacities, int num_gaussians, int num_channels,
                               int height, int width, float* output_host, CompositingMode mode) {
  return run_forward(means, covs, colors, opacities, num_gaussians, num_channels, height, width,
                     output_host, mode);
}

bool gaussian_splat_2d_backward(const float* grad_output, const float* means, const float* covs,
                                const float* colors, const float* opacities, int num_gaussians,
                                int num_channels, int height, int width, float* grad_means,
                                float* grad_covs, float* grad_colors, float* grad_opacities) {
  if (num_gaussians <= 0 || num_channels <= 0 || height <= 0 || width <= 0) {
    return false;
  }

  const int tiles_x = (width + detail::kTileSize - 1) / detail::kTileSize;
  const int tiles_y = (height + detail::kTileSize - 1) / detail::kTileSize;
  const int num_tiles = tiles_x * tiles_y;

  float *d_means = nullptr;
  float* d_covs = nullptr;
  float* d_colors = nullptr;
  float* d_opacities = nullptr;
  float* d_grad_output = nullptr;
  Gaussian2D* d_gaussians = nullptr;
  float* d_grad_means = nullptr;
  float* d_grad_covs = nullptr;
  float* d_grad_colors = nullptr;
  float* d_grad_opacities = nullptr;

  const size_t means_bytes = static_cast<size_t>(num_gaussians) * 2 * sizeof(float);
  const size_t covs_bytes = static_cast<size_t>(num_gaussians) * 4 * sizeof(float);
  const size_t colors_bytes =
      static_cast<size_t>(num_gaussians) * num_channels * sizeof(float);
  const size_t opac_bytes = static_cast<size_t>(num_gaussians) * sizeof(float);
  const size_t out_bytes = static_cast<size_t>(height) * width * num_channels * sizeof(float);
  const size_t gauss_bytes = static_cast<size_t>(num_gaussians) * sizeof(Gaussian2D);

  TileRasterState tiles;
  if (!TINYSPLAT_CUDA_CHECK(cudaMalloc(&d_means, means_bytes)) ||
      !TINYSPLAT_CUDA_CHECK(cudaMalloc(&d_covs, covs_bytes)) ||
      !TINYSPLAT_CUDA_CHECK(cudaMalloc(&d_colors, colors_bytes)) ||
      !TINYSPLAT_CUDA_CHECK(cudaMalloc(&d_opacities, opac_bytes)) ||
      !TINYSPLAT_CUDA_CHECK(cudaMalloc(&d_grad_output, out_bytes)) ||
      !TINYSPLAT_CUDA_CHECK(cudaMalloc(&d_gaussians, gauss_bytes)) ||
      !TINYSPLAT_CUDA_CHECK(cudaMalloc(&d_grad_means, means_bytes)) ||
      !TINYSPLAT_CUDA_CHECK(cudaMalloc(&d_grad_covs, covs_bytes)) ||
      !TINYSPLAT_CUDA_CHECK(cudaMalloc(&d_grad_colors, colors_bytes)) ||
      !TINYSPLAT_CUDA_CHECK(cudaMalloc(&d_grad_opacities, opac_bytes)) ||
      !TINYSPLAT_CUDA_CHECK(cudaMalloc(&tiles.d_tile_counts,
                                       static_cast<size_t>(num_tiles) * sizeof(int))) ||
      !TINYSPLAT_CUDA_CHECK(cudaMalloc(&tiles.d_tile_starts,
                                       static_cast<size_t>(num_tiles + 1) * sizeof(int)))) {
    cudaFree(d_means);
    cudaFree(d_covs);
    cudaFree(d_colors);
    cudaFree(d_opacities);
    cudaFree(d_grad_output);
    cudaFree(d_gaussians);
    cudaFree(d_grad_means);
    cudaFree(d_grad_covs);
    cudaFree(d_grad_colors);
    cudaFree(d_grad_opacities);
    return false;
  }
  if (!TINYSPLAT_CUDA_CHECK(cudaMemcpy(d_means, means, means_bytes, cudaMemcpyHostToDevice)) ||
      !TINYSPLAT_CUDA_CHECK(cudaMemcpy(d_covs, covs, covs_bytes, cudaMemcpyHostToDevice)) ||
      !TINYSPLAT_CUDA_CHECK(cudaMemcpy(d_colors, colors, colors_bytes, cudaMemcpyHostToDevice)) ||
      !TINYSPLAT_CUDA_CHECK(
          cudaMemcpy(d_opacities, opacities, opac_bytes, cudaMemcpyHostToDevice)) ||
      !TINYSPLAT_CUDA_CHECK(
          cudaMemcpy(d_grad_output, grad_output, out_bytes, cudaMemcpyHostToDevice)) ||
      !TINYSPLAT_CUDA_CHECK(cudaMemset(d_grad_means, 0, means_bytes)) ||
      !TINYSPLAT_CUDA_CHECK(cudaMemset(d_grad_covs, 0, covs_bytes)) ||
      !TINYSPLAT_CUDA_CHECK(cudaMemset(d_grad_colors, 0, colors_bytes)) ||
      !TINYSPLAT_CUDA_CHECK(cudaMemset(d_grad_opacities, 0, opac_bytes))) {
    cudaFree(d_means);
    cudaFree(d_covs);
    cudaFree(d_colors);
    cudaFree(d_opacities);
    cudaFree(d_grad_output);
    cudaFree(d_gaussians);
    cudaFree(d_grad_means);
    cudaFree(d_grad_covs);
    cudaFree(d_grad_colors);
    cudaFree(d_grad_opacities);
    return false;
  }

  const int blocks = (num_gaussians + 255) / 256;
  precompute_gaussians_kernel<<<blocks, 256>>>(d_means, d_covs, d_gaussians, num_gaussians,
                                               height, width);
  if (!TINYSPLAT_CUDA_CHECK(cudaGetLastError())) {
    cudaFree(d_means);
    cudaFree(d_covs);
    cudaFree(d_colors);
    cudaFree(d_opacities);
    cudaFree(d_grad_output);
    cudaFree(d_gaussians);
    cudaFree(d_grad_means);
    cudaFree(d_grad_covs);
    cudaFree(d_grad_colors);
    cudaFree(d_grad_opacities);
    return false;
  }

  if (!count_tile_bins(tiles, d_gaussians, num_gaussians, tiles_x, tiles_y)) {
    cudaFree(d_means);
    cudaFree(d_covs);
    cudaFree(d_colors);
    cudaFree(d_opacities);
    cudaFree(d_grad_output);
    cudaFree(d_gaussians);
    cudaFree(d_grad_means);
    cudaFree(d_grad_covs);
    cudaFree(d_grad_colors);
    cudaFree(d_grad_opacities);
    return false;
  }

  if (tiles.total_bins > 0 &&
      !TINYSPLAT_CUDA_CHECK(cudaMalloc(&tiles.d_tile_bins,
                                       static_cast<size_t>(tiles.total_bins) * sizeof(int)))) {
    cudaFree(d_means);
    cudaFree(d_covs);
    cudaFree(d_colors);
    cudaFree(d_opacities);
    cudaFree(d_grad_output);
    cudaFree(d_gaussians);
    cudaFree(d_grad_means);
    cudaFree(d_grad_covs);
    cudaFree(d_grad_colors);
    cudaFree(d_grad_opacities);
    return false;
  }

  if (!assign_tile_bins(tiles, d_gaussians, num_gaussians, tiles_x, tiles_y)) {
    cudaFree(d_means);
    cudaFree(d_covs);
    cudaFree(d_colors);
    cudaFree(d_opacities);
    cudaFree(d_grad_output);
    cudaFree(d_gaussians);
    cudaFree(d_grad_means);
    cudaFree(d_grad_covs);
    cudaFree(d_grad_colors);
    cudaFree(d_grad_opacities);
    return false;
  }

  rasterize_backward_kernel<<<blocks, 256>>>(
      d_grad_output, d_gaussians, d_colors, d_opacities, tiles.d_tile_starts, tiles.d_tile_bins,
      d_grad_means, d_grad_covs, d_grad_colors, d_grad_opacities, num_gaussians, height, width,
      num_channels, tiles_x);

  const bool ok =
      TINYSPLAT_CUDA_CHECK(cudaGetLastError()) &&
      TINYSPLAT_CUDA_CHECK(cudaMemcpy(grad_means, d_grad_means, means_bytes, cudaMemcpyDeviceToHost)) &&
      TINYSPLAT_CUDA_CHECK(cudaMemcpy(grad_covs, d_grad_covs, covs_bytes, cudaMemcpyDeviceToHost)) &&
      TINYSPLAT_CUDA_CHECK(
          cudaMemcpy(grad_colors, d_grad_colors, colors_bytes, cudaMemcpyDeviceToHost)) &&
      TINYSPLAT_CUDA_CHECK(
          cudaMemcpy(grad_opacities, d_grad_opacities, opac_bytes, cudaMemcpyDeviceToHost)) &&
      TINYSPLAT_CUDA_CHECK(cudaDeviceSynchronize());

  cudaFree(d_means);
  cudaFree(d_covs);
  cudaFree(d_colors);
  cudaFree(d_opacities);
  cudaFree(d_grad_output);
  cudaFree(d_gaussians);
  cudaFree(d_grad_means);
  cudaFree(d_grad_covs);
  cudaFree(d_grad_colors);
  cudaFree(d_grad_opacities);
  tiles.d_tile_bins = nullptr;
  tiles.d_tile_counts = nullptr;
  tiles.d_tile_starts = nullptr;
  return ok;
}

}  // namespace cuda
}  // namespace tinysplat
