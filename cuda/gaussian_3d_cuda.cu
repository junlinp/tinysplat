#include "tinysplat/gaussian_3d_cuda.h"

#include <tinysplat/cuda_common.cuh>
#include <tinysplat/gaussian_2d_cuda.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

namespace tinysplat {
namespace cuda {
namespace {

struct ProjectedGaussianGPU {
  float mean_x;
  float mean_y;
  float depth;
  float inv_xx;
  float inv_xy;
  float inv_yx;
  float inv_yy;
  int min_x;
  int max_x;
  int min_y;
  int max_y;
  int source_index;
};

__global__ void project_gaussians_kernel(const float* means, const float* covs,
                                         ProjectedGaussianGPU* out, const float* intrinsics,
                                         const float* c2w, int n, int h, int w, float near_plane,
                                         float min_covariance, float sigma_radius) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n) {
    return;
  }

  const float fx = intrinsics[0];
  const float fy = intrinsics[4];
  const float cx = intrinsics[2];
  const float cy = intrinsics[5];

  const float r00 = c2w[0];
  const float r01 = c2w[1];
  const float r02 = c2w[2];
  const float r10 = c2w[4];
  const float r11 = c2w[5];
  const float r12 = c2w[6];
  const float r20 = c2w[8];
  const float r21 = c2w[9];
  const float r22 = c2w[10];
  const float tx = c2w[3];
  const float ty = c2w[7];
  const float tz = c2w[11];

  const float rwc00 = r00;
  const float rwc01 = r10;
  const float rwc02 = r20;
  const float rwc10 = r01;
  const float rwc11 = r11;
  const float rwc12 = r21;
  const float rwc20 = r02;
  const float rwc21 = r12;
  const float rwc22 = r22;

  const float twc0 = -(rwc00 * tx + rwc01 * ty + rwc02 * tz);
  const float twc1 = -(rwc10 * tx + rwc11 * ty + rwc12 * tz);
  const float twc2 = -(rwc20 * tx + rwc21 * ty + rwc22 * tz);

  const float mx = means[g * 3 + 0];
  const float my = means[g * 3 + 1];
  const float mz = means[g * 3 + 2];

  const float cam_x = rwc00 * mx + rwc01 * my + rwc02 * mz + twc0;
  const float cam_y = rwc10 * mx + rwc11 * my + rwc12 * mz + twc1;
  const float cam_z = rwc20 * mx + rwc21 * my + rwc22 * mz + twc2;
  if (cam_z <= near_plane) {
    out[g].min_x = 1;
    out[g].max_x = 0;
    return;
  }

  const float* wc = covs + g * 9;
  float t00 = rwc00 * wc[0] + rwc01 * wc[3] + rwc02 * wc[6];
  float t01 = rwc00 * wc[1] + rwc01 * wc[4] + rwc02 * wc[7];
  float t02 = rwc00 * wc[2] + rwc01 * wc[5] + rwc02 * wc[8];
  float t10 = rwc10 * wc[0] + rwc11 * wc[3] + rwc12 * wc[6];
  float t11 = rwc10 * wc[1] + rwc11 * wc[4] + rwc12 * wc[7];
  float t12 = rwc10 * wc[2] + rwc11 * wc[5] + rwc12 * wc[8];
  float t20 = rwc20 * wc[0] + rwc21 * wc[3] + rwc22 * wc[6];
  float t21 = rwc20 * wc[1] + rwc21 * wc[4] + rwc22 * wc[7];
  float t22 = rwc20 * wc[2] + rwc21 * wc[5] + rwc22 * wc[8];

  float ccov00 = t00 * rwc00 + t01 * rwc01 + t02 * rwc02;
  float ccov01 = t00 * rwc10 + t01 * rwc11 + t02 * rwc12;
  float ccov02 = t00 * rwc20 + t01 * rwc21 + t02 * rwc22;
  float ccov10 = t10 * rwc00 + t11 * rwc01 + t12 * rwc02;
  float ccov11 = t10 * rwc10 + t11 * rwc11 + t12 * rwc12;
  float ccov12 = t10 * rwc20 + t11 * rwc21 + t12 * rwc22;
  float ccov20 = t20 * rwc00 + t21 * rwc01 + t22 * rwc02;
  float ccov21 = t20 * rwc10 + t21 * rwc11 + t22 * rwc12;
  float ccov22 = t20 * rwc20 + t21 * rwc21 + t22 * rwc22;

  const float proj_x = fx * cam_x / cam_z + cx;
  const float proj_y = fy * cam_y / cam_z + cy;

  const float j00 = fx / cam_z;
  const float j02 = -fx * cam_x / (cam_z * cam_z);
  const float j11 = fy / cam_z;
  const float j12 = -fy * cam_y / (cam_z * cam_z);

  float s00 = j00 * (ccov00 * j00 + ccov02 * j02) + j02 * (ccov20 * j00 + ccov22 * j02);
  float s01 = j00 * (ccov01 * j11 + ccov02 * j12) + j02 * (ccov21 * j11 + ccov22 * j12);
  const float s10 = j11 * (ccov10 * j00 + ccov12 * j02) + j12 * (ccov21 * j00 + ccov22 * j02);
  float s11 = j11 * (ccov11 * j11 + ccov12 * j12) + j12 * (ccov21 * j11 + ccov22 * j12);

  s00 += min_covariance;
  s11 += min_covariance;
  float det = s00 * s11 - s01 * s10;
  if (det <= min_covariance) {
    det = min_covariance;
  }
  const float inv_det = 1.0f / det;
  const float inv_xx = s11 * inv_det;
  const float inv_xy = -s01 * inv_det;
  const float inv_yx = -s01 * inv_det;
  const float inv_yy = s00 * inv_det;

  const float trace = s00 + s11;
  const float disc = sqrtf(fmaxf(0.0f, (s00 - s11) * (s00 - s11) + 4.0f * s01 * s01));
  const float lambda_max = fmaxf((trace + disc) * 0.5f, min_covariance);
  const float radius = sigma_radius * sqrtf(lambda_max);

  const int min_x = static_cast<int>(floorf(proj_x - radius));
  const int max_x = static_cast<int>(ceilf(proj_x + radius));
  const int min_y = static_cast<int>(floorf(proj_y - radius));
  const int max_y = static_cast<int>(ceilf(proj_y + radius));
  if (max_x < 0 || min_x >= w || max_y < 0 || min_y >= h) {
    out[g].min_x = 1;
    out[g].max_x = 0;
    return;
  }

  out[g] = ProjectedGaussianGPU{proj_x,
                                proj_y,
                                cam_z,
                                inv_xx,
                                inv_xy,
                                inv_yx,
                                inv_yy,
                                min_x,
                                max_x,
                                min_y,
                                max_y,
                                g};
}

__global__ void splat_3d_per_pixel_kernel(const ProjectedGaussianGPU* sorted, int n_visible,
                                          const float* colors, const float* opacities, int c,
                                          float* output, int h, int w) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) {
    return;
  }

  float accum[4] = {0.f, 0.f, 0.f, 0.f};
  float transmittance = 1.0f;

  for (int i = 0; i < n_visible; ++i) {
    const ProjectedGaussianGPU& pg = sorted[i];
    if (x < pg.min_x || x > pg.max_x || y < pg.min_y || y > pg.max_y) {
      continue;
    }

    const float dx = static_cast<float>(x) - pg.mean_x;
    const float dy = static_cast<float>(y) - pg.mean_y;
    const float quad = dx * (pg.inv_xx * dx + pg.inv_xy * dy) + dy * (pg.inv_yx * dx + pg.inv_yy * dy);
    const float gaussian = expf(-0.5f * quad);
    float alpha = opacities[pg.source_index] * gaussian;
    if (alpha > 0.999f) {
      alpha = 0.999f;
    }
    if (alpha < 0.0f) {
      alpha = 0.0f;
    }

    const float weight = transmittance * alpha;
    for (int ch = 0; ch < c && ch < 4; ++ch) {
      accum[ch] += weight * colors[pg.source_index * c + ch];
    }
    transmittance *= (1.0f - alpha);
    if (transmittance < 1e-4f) {
      break;
    }
  }

  for (int ch = 0; ch < c; ++ch) {
    output[(y * w + x) * c + ch] = (ch < 4 ? accum[ch] : 0.0f);
  }
}

int compact_visible_host(ProjectedGaussianGPU* d_proj, int n) {
  std::vector<ProjectedGaussianGPU> host(static_cast<size_t>(n));
  if (cudaMemcpy(host.data(), d_proj, static_cast<size_t>(n) * sizeof(ProjectedGaussianGPU),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    return -1;
  }
  int count = 0;
  for (int i = 0; i < n; ++i) {
    if (host[static_cast<size_t>(i)].max_x >= host[static_cast<size_t>(i)].min_x &&
        host[static_cast<size_t>(i)].max_y >= host[static_cast<size_t>(i)].min_y) {
      host[static_cast<size_t>(count++)] = host[static_cast<size_t>(i)];
    }
  }
  if (count > 1) {
    std::sort(host.begin(), host.begin() + count,
              [](const ProjectedGaussianGPU& a, const ProjectedGaussianGPU& b) {
                return a.depth < b.depth;
              });
  }
  if (count > 0 &&
      cudaMemcpy(d_proj, host.data(), static_cast<size_t>(count) * sizeof(ProjectedGaussianGPU),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    return -1;
  }
  return count;
}

}  // namespace

bool gaussian_splat_3d_forward(const float* means, const float* covs, const float* colors,
                             const float* opacities, int num_gaussians, int num_channels,
                             const float* intrinsics, const float* camera_to_world, int height,
                             int width, float* output_host, const Splat3DCudaOptions& opts) {
  if (num_gaussians <= 0 || num_channels <= 0 || height <= 0 || width <= 0) {
    return false;
  }

  float *d_means = nullptr;
  float* d_covs = nullptr;
  float* d_colors = nullptr;
  float* d_opacities = nullptr;
  float* d_intrinsics = nullptr;
  float* d_c2w = nullptr;
  ProjectedGaussianGPU* d_projected = nullptr;
  float* d_output = nullptr;

  const size_t means_bytes = static_cast<size_t>(num_gaussians) * 3 * sizeof(float);
  const size_t covs_bytes = static_cast<size_t>(num_gaussians) * 9 * sizeof(float);
  const size_t colors_bytes =
      static_cast<size_t>(num_gaussians) * num_channels * sizeof(float);
  const size_t opac_bytes = static_cast<size_t>(num_gaussians) * sizeof(float);
  const size_t out_bytes = static_cast<size_t>(height) * width * num_channels * sizeof(float);
  const size_t proj_bytes = static_cast<size_t>(num_gaussians) * sizeof(ProjectedGaussianGPU);

  if (!TINYSPLAT_CUDA_CHECK(cudaMalloc(&d_means, means_bytes)) ||
      !TINYSPLAT_CUDA_CHECK(cudaMalloc(&d_covs, covs_bytes)) ||
      !TINYSPLAT_CUDA_CHECK(cudaMalloc(&d_colors, colors_bytes)) ||
      !TINYSPLAT_CUDA_CHECK(cudaMalloc(&d_opacities, opac_bytes)) ||
      !TINYSPLAT_CUDA_CHECK(cudaMalloc(&d_intrinsics, 9 * sizeof(float))) ||
      !TINYSPLAT_CUDA_CHECK(cudaMalloc(&d_c2w, 16 * sizeof(float))) ||
      !TINYSPLAT_CUDA_CHECK(cudaMalloc(&d_projected, proj_bytes)) ||
      !TINYSPLAT_CUDA_CHECK(cudaMalloc(&d_output, out_bytes))) {
    cudaFree(d_means);
    cudaFree(d_covs);
    cudaFree(d_colors);
    cudaFree(d_opacities);
    cudaFree(d_intrinsics);
    cudaFree(d_c2w);
    cudaFree(d_projected);
    cudaFree(d_output);
    return false;
  }

  if (!TINYSPLAT_CUDA_CHECK(cudaMemcpy(d_means, means, means_bytes, cudaMemcpyHostToDevice)) ||
      !TINYSPLAT_CUDA_CHECK(cudaMemcpy(d_covs, covs, covs_bytes, cudaMemcpyHostToDevice)) ||
      !TINYSPLAT_CUDA_CHECK(cudaMemcpy(d_colors, colors, colors_bytes, cudaMemcpyHostToDevice)) ||
      !TINYSPLAT_CUDA_CHECK(
          cudaMemcpy(d_opacities, opacities, opac_bytes, cudaMemcpyHostToDevice)) ||
      !TINYSPLAT_CUDA_CHECK(
          cudaMemcpy(d_intrinsics, intrinsics, 9 * sizeof(float), cudaMemcpyHostToDevice)) ||
      !TINYSPLAT_CUDA_CHECK(
          cudaMemcpy(d_c2w, camera_to_world, 16 * sizeof(float), cudaMemcpyHostToDevice))) {
    cudaFree(d_means);
    cudaFree(d_covs);
    cudaFree(d_colors);
    cudaFree(d_opacities);
    cudaFree(d_intrinsics);
    cudaFree(d_c2w);
    cudaFree(d_projected);
    cudaFree(d_output);
    return false;
  }

  const int blocks = (num_gaussians + 255) / 256;
  project_gaussians_kernel<<<blocks, 256>>>(d_means, d_covs, d_projected, d_intrinsics, d_c2w,
                                            num_gaussians, height, width, opts.near_plane,
                                            opts.min_covariance, opts.sigma_radius);
  if (!TINYSPLAT_CUDA_CHECK(cudaGetLastError())) {
    cudaFree(d_means);
    cudaFree(d_covs);
    cudaFree(d_colors);
    cudaFree(d_opacities);
    cudaFree(d_intrinsics);
    cudaFree(d_c2w);
    cudaFree(d_projected);
    cudaFree(d_output);
    return false;
  }

  const int n_visible = compact_visible_host(d_projected, num_gaussians);
  if (n_visible < 0) {
    cudaFree(d_means);
    cudaFree(d_covs);
    cudaFree(d_colors);
    cudaFree(d_opacities);
    cudaFree(d_intrinsics);
    cudaFree(d_c2w);
    cudaFree(d_projected);
    cudaFree(d_output);
    return false;
  }

  if (!TINYSPLAT_CUDA_CHECK(cudaMemset(d_output, 0, out_bytes))) {
    cudaFree(d_means);
    cudaFree(d_covs);
    cudaFree(d_colors);
    cudaFree(d_opacities);
    cudaFree(d_intrinsics);
    cudaFree(d_c2w);
    cudaFree(d_projected);
    cudaFree(d_output);
    return false;
  }

  const dim3 block(16, 16);
  const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  splat_3d_per_pixel_kernel<<<grid, block>>>(d_projected, n_visible, d_colors, d_opacities,
                                             num_channels, d_output, height, width);

  const bool ok =
      TINYSPLAT_CUDA_CHECK(cudaGetLastError()) &&
      TINYSPLAT_CUDA_CHECK(cudaMemcpy(output_host, d_output, out_bytes, cudaMemcpyDeviceToHost)) &&
      TINYSPLAT_CUDA_CHECK(cudaDeviceSynchronize());

  cudaFree(d_means);
  cudaFree(d_covs);
  cudaFree(d_colors);
  cudaFree(d_opacities);
  cudaFree(d_intrinsics);
  cudaFree(d_c2w);
  cudaFree(d_projected);
  cudaFree(d_output);
  return ok;
}

bool gaussian_splat_3d_projected_backward(const float* grad_output, const float* proj_means,
                                          const float* proj_covs, const float* colors,
                                          const float* opacities, int num_gaussians,
                                          int num_channels, int height, int width,
                                          float* grad_proj_means, float* grad_proj_covs,
                                          float* grad_colors, float* grad_opacities) {
  return gaussian_splat_2d_backward(grad_output, proj_means, proj_covs, colors, opacities,
                                    num_gaussians, num_channels, height, width, grad_proj_means,
                                    grad_proj_covs, grad_colors, grad_opacities);
}

}  // namespace cuda
}  // namespace tinysplat
