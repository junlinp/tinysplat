#include "tinysplat/gaussian_2d_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

namespace tinysplat {
namespace cuda {
namespace {

constexpr int kTileSize = 16;
constexpr float kEps = 1e-8f;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kSigmaRadius = 4.0f;

struct Gaussian2D {
  float mean_x, mean_y;
  float inv_xx, inv_xy, inv_yx, inv_yy;
  float normalization;
  float pad;
  int min_x, max_x, min_y, max_y;
};

__global__ void precompute_kernel(const float* means, const float* covs, Gaussian2D* out, int n,
                                  int h, int w) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n) return;

  float a = covs[g * 4 + 0];
  float b = covs[g * 4 + 1];
  float c = covs[g * 4 + 2];
  float d = covs[g * 4 + 3];
  float det = a * d - b * c;
  if (det < kEps) det = kEps;
  float inv_det = 1.0f / det;
  float trace = a + d;
  float disc = sqrtf(fmaxf(0.0f, (a - d) * (a - d) + 4.0f * b * c));
  float lambda_max = fmaxf((trace + disc) * 0.5f, kEps);
  float radius = ceilf(kSigmaRadius * sqrtf(lambda_max));
  float mx = means[g * 2 + 0];
  float my = means[g * 2 + 1];
  int min_x = max(0, (int)floorf(mx - radius));
  int max_x = min(w - 1, (int)ceilf(mx + radius));
  int min_y = max(0, (int)floorf(my - radius));
  int max_y = min(h - 1, (int)ceilf(my + radius));
  out[g] = Gaussian2D{mx, my, d * inv_det, -b * inv_det, -c * inv_det, a * inv_det,
                      1.0f / (2.0f * kPi * sqrtf(det + kEps)), 0.0f,
                      min_x, max_x, min_y, max_y};
}

__global__ void splat_kernel(const Gaussian2D* gaussians, const float* colors,
                             const float* opacities, float* output, float* total_weight, int n,
                             int h, int w, int c) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;

  float accum[4] = {0, 0, 0, 0};
  float weight_sum = 0.0f;

  for (int g = 0; g < n; ++g) {
    const Gaussian2D& gp = gaussians[g];
    if (x < gp.min_x || x > gp.max_x || y < gp.min_y || y > gp.max_y) continue;
    float dx = (float)x - gp.mean_x;
    float dy = (float)y - gp.mean_y;
    float qx = gp.inv_xx * dx + gp.inv_xy * dy;
    float qy = gp.inv_yx * dx + gp.inv_yy * dy;
    float quad = dx * qx + dy * qy;
    float gaussian = expf(-0.5f * quad) * gp.normalization;
    float weight = gaussian * opacities[g];
    weight_sum += weight;
    for (int ch = 0; ch < c && ch < 4; ++ch) {
      accum[ch] += weight * colors[g * c + ch];
    }
  }

  float denom = fmaxf(weight_sum, kEps);
  total_weight[y * w + x] = denom;
  for (int ch = 0; ch < c; ++ch) {
    output[(y * w + x) * c + ch] = (ch < 4 ? accum[ch] : 0.0f) / denom;
  }
}

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t err = (call);                                                    \
    if (err != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,             \
              cudaGetErrorString(err));                                          \
      return false;                                                              \
    }                                                                            \
  } while (0)

}  // namespace

bool gaussian_splat_2d_forward(const float* means, const float* covs, const float* colors,
                               const float* opacities, int num_gaussians, int num_channels,
                               int height, int width, float* output_host) {
  if (num_gaussians <= 0 || num_channels <= 0) {
    return false;
  }

  float *d_means = nullptr, *d_covs = nullptr, *d_colors = nullptr, *d_opacities = nullptr;
  Gaussian2D* d_gaussians = nullptr;
  float *d_output = nullptr, *d_total = nullptr;

  const size_t means_bytes = static_cast<size_t>(num_gaussians) * 2 * sizeof(float);
  const size_t covs_bytes = static_cast<size_t>(num_gaussians) * 4 * sizeof(float);
  const size_t colors_bytes =
      static_cast<size_t>(num_gaussians) * num_channels * sizeof(float);
  const size_t opac_bytes = static_cast<size_t>(num_gaussians) * sizeof(float);
  const size_t out_bytes =
      static_cast<size_t>(height) * width * num_channels * sizeof(float);
  const size_t tw_bytes = static_cast<size_t>(height) * width * sizeof(float);

  CUDA_CHECK(cudaMalloc(&d_means, means_bytes));
  CUDA_CHECK(cudaMalloc(&d_covs, covs_bytes));
  CUDA_CHECK(cudaMalloc(&d_colors, colors_bytes));
  CUDA_CHECK(cudaMalloc(&d_opacities, opac_bytes));
  CUDA_CHECK(cudaMalloc(&d_gaussians, static_cast<size_t>(num_gaussians) * sizeof(Gaussian2D)));
  CUDA_CHECK(cudaMalloc(&d_output, out_bytes));
  CUDA_CHECK(cudaMalloc(&d_total, tw_bytes));

  CUDA_CHECK(cudaMemcpy(d_means, means, means_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_covs, covs, covs_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_colors, colors, colors_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_opacities, opacities, opac_bytes, cudaMemcpyHostToDevice));

  int blocks = (num_gaussians + 255) / 256;
  precompute_kernel<<<blocks, 256>>>(d_means, d_covs, d_gaussians, num_gaussians, height, width);
  CUDA_CHECK(cudaGetLastError());

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  splat_kernel<<<grid, block>>>(d_gaussians, d_colors, d_opacities, d_output, d_total,
                                num_gaussians, height, width, num_channels);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(output_host, d_output, out_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaFree(d_means);
  cudaFree(d_covs);
  cudaFree(d_colors);
  cudaFree(d_opacities);
  cudaFree(d_gaussians);
  cudaFree(d_output);
  cudaFree(d_total);
  return true;
}

}  // namespace cuda
}  // namespace tinysplat
