#include "tinysplat/cuda_gaussian2d.cuh"

namespace tinysplat {
namespace cuda {
namespace detail {

__global__ void precompute_gaussians_kernel(const float* __restrict__ means,
                                            const float* __restrict__ covs,
                                            Gaussian2D* __restrict__ out, int n, int h,
                                            int w) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n) {
    return;
  }

  const float a = covs[g * 4 + 0];
  const float b = covs[g * 4 + 1];
  const float c = covs[g * 4 + 2];
  const float d = covs[g * 4 + 3];

  float det = a * d - b * c;
  if (det < kEps) {
    det = kEps;
  }

  const float inv_det = 1.0f / det;
  const float trace = a + d;
  const float disc = sqrtf(fmaxf(0.0f, (a - d) * (a - d) + 4.0f * b * c));
  const float lambda_max = fmaxf((trace + disc) * 0.5f, kEps);
  const float radius = ceilf(kSigmaRadius * sqrtf(lambda_max));

  const float mx = means[g * 2 + 0];
  const float my = means[g * 2 + 1];

  const int min_x = max(0, static_cast<int>(floorf(mx - radius)));
  const int max_x = min(w - 1, static_cast<int>(ceilf(mx + radius)));
  const int min_y = max(0, static_cast<int>(floorf(my - radius)));
  const int max_y = min(h - 1, static_cast<int>(ceilf(my + radius)));

  out[g] = Gaussian2D{mx,
                      my,
                      d * inv_det,
                      -b * inv_det,
                      -c * inv_det,
                      a * inv_det,
                      1.0f / (2.0f * kPi * sqrtf(det + kEps)),
                      det / (det + kEps),
                      min_x,
                      max_x,
                      min_y,
                      max_y};
}

__global__ void count_tile_membership_kernel(const Gaussian2D* __restrict__ gaussians,
                                             int* __restrict__ tile_counts, int tiles_x,
                                             int tiles_y, int n) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n) {
    return;
  }

  const Gaussian2D& gk = gaussians[g];
  if (gk.max_x < gk.min_x || gk.max_y < gk.min_y) {
    return;
  }

  const int tile_min_x = gk.min_x / kTileSize;
  const int tile_max_x = gk.max_x / kTileSize;
  const int tile_min_y = gk.min_y / kTileSize;
  const int tile_max_y = gk.max_y / kTileSize;

  for (int ty = tile_min_y; ty <= tile_max_y; ++ty) {
    for (int tx = tile_min_x; tx <= tile_max_x; ++tx) {
      if (tx < 0 || tx >= tiles_x || ty < 0 || ty >= tiles_y) {
        continue;
      }
      atomicAdd(&tile_counts[ty * tiles_x + tx], 1);
    }
  }
}

__global__ void assign_tile_bins_kernel(const Gaussian2D* __restrict__ gaussians,
                                        int* __restrict__ tile_counts,
                                        const int* __restrict__ tile_starts,
                                        int* __restrict__ tile_bins, int tiles_x, int tiles_y,
                                        int n) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n) {
    return;
  }

  const Gaussian2D& gk = gaussians[g];
  if (gk.max_x < gk.min_x || gk.max_y < gk.min_y) {
    return;
  }

  const int tile_min_x = gk.min_x / kTileSize;
  const int tile_max_x = gk.max_x / kTileSize;
  const int tile_min_y = gk.min_y / kTileSize;
  const int tile_max_y = gk.max_y / kTileSize;

  for (int ty = tile_min_y; ty <= tile_max_y; ++ty) {
    for (int tx = tile_min_x; tx <= tile_max_x; ++tx) {
      if (tx < 0 || tx >= tiles_x || ty < 0 || ty >= tiles_y) {
        continue;
      }
      const int tile_idx = ty * tiles_x + tx;
      const int slot = tile_starts[tile_idx] + atomicAdd(&tile_counts[tile_idx], 1);
      tile_bins[slot] = g;
    }
  }
}

__global__ void rasterize_alpha_forward_kernel(const Gaussian2D* __restrict__ gaussians,
                                               const float* __restrict__ colors,
                                               const float* __restrict__ opacities,
                                               const int* __restrict__ tile_starts,
                                               const int* __restrict__ tile_bins,
                                               float* __restrict__ output, int h, int w, int c,
                                               int tiles_x) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) {
    return;
  }

  const int tile_x = x / kTileSize;
  const int tile_y = y / kTileSize;
  const int tile_idx = tile_y * tiles_x + tile_x;
  const int bin_start = tile_starts[tile_idx];
  const int bin_end = tile_starts[tile_idx + 1];

  float accum[4] = {0.f, 0.f, 0.f, 0.f};
  float transmittance = 1.0f;

  for (int idx = bin_start; idx < bin_end; ++idx) {
    const int g = tile_bins[idx];
    const Gaussian2D& gk = gaussians[g];
    if (x < gk.min_x || x > gk.max_x || y < gk.min_y || y > gk.max_y) {
      continue;
    }

    const float dx = static_cast<float>(x) - gk.mean_x;
    const float dy = static_cast<float>(y) - gk.mean_y;
    const float qx = gk.inv_xx * dx + gk.inv_xy * dy;
    const float qy = gk.inv_yx * dx + gk.inv_yy * dy;
    const float quad = dx * qx + dy * qy;
    const float gaussian = expf(-0.5f * quad) * gk.normalization;
    float alpha = opacities[g] * gaussian;
    if (alpha > 0.999f) {
      alpha = 0.999f;
    }
    if (alpha < 0.0f) {
      alpha = 0.0f;
    }

    const float weight = alpha * transmittance;
    for (int ch = 0; ch < c && ch < 4; ++ch) {
      accum[ch] += weight * colors[g * c + ch];
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

__global__ void rasterize_weighted_forward_kernel(
    const Gaussian2D* __restrict__ gaussians, const float* __restrict__ colors,
    const float* __restrict__ opacities, const int* __restrict__ tile_starts,
    const int* __restrict__ tile_bins, float* __restrict__ output, int h, int w, int c,
    int tiles_x) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) {
    return;
  }

  const int tile_x = x / kTileSize;
  const int tile_y = y / kTileSize;
  const int tile_idx = tile_y * tiles_x + tile_x;
  const int bin_start = tile_starts[tile_idx];
  const int bin_end = tile_starts[tile_idx + 1];

  float accum[4] = {0.f, 0.f, 0.f, 0.f};
  float weight_sum = 0.0f;

  for (int idx = bin_start; idx < bin_end; ++idx) {
    const int g = tile_bins[idx];
    const Gaussian2D& gk = gaussians[g];
    if (x < gk.min_x || x > gk.max_x || y < gk.min_y || y > gk.max_y) {
      continue;
    }

    const float dx = static_cast<float>(x) - gk.mean_x;
    const float dy = static_cast<float>(y) - gk.mean_y;
    const float qx = gk.inv_xx * dx + gk.inv_xy * dy;
    const float qy = gk.inv_yx * dx + gk.inv_yy * dy;
    const float quad = dx * qx + dy * qy;
    const float gaussian = expf(-0.5f * quad) * gk.normalization;
    const float weight = gaussian * opacities[g];
    weight_sum += weight;
    for (int ch = 0; ch < c && ch < 4; ++ch) {
      accum[ch] += weight * colors[g * c + ch];
    }
  }

  const float denom = fmaxf(weight_sum, kEps);
  for (int ch = 0; ch < c; ++ch) {
    output[(y * w + x) * c + ch] = (ch < 4 ? accum[ch] : 0.0f) / denom;
  }
}

__global__ void rasterize_backward_kernel(
    const float* __restrict__ grad_output, const Gaussian2D* __restrict__ gaussians,
    const float* __restrict__ colors, const float* __restrict__ opacities,
    const int* __restrict__ tile_starts, const int* __restrict__ tile_bins,
    float* __restrict__ grad_means, float* __restrict__ grad_covs,
    float* __restrict__ grad_colors, float* __restrict__ grad_opacities, int n, int h, int w,
    int c, int tiles_x) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n) {
    return;
  }

  const Gaussian2D& gk = gaussians[g];
  float gm_x = 0.0f;
  float gm_y = 0.0f;
  float gop = 0.0f;
  float gcv[4] = {0.f, 0.f, 0.f, 0.f};
  float gcl[4] = {0.f, 0.f, 0.f, 0.f};

  for (int y = gk.min_y; y <= gk.max_y; ++y) {
    for (int x = gk.min_x; x <= gk.max_x; ++x) {
      if (x < 0 || x >= w || y < 0 || y >= h) {
        continue;
      }

      const int tile_x = x / kTileSize;
      const int tile_y = y / kTileSize;
      const int tile_idx = tile_y * tiles_x + tile_x;
      const int bin_start = tile_starts[tile_idx];
      const int bin_end = tile_starts[tile_idx + 1];

      bool in_tile = false;
      for (int bi = bin_start; bi < bin_end; ++bi) {
        if (tile_bins[bi] == g) {
          in_tile = true;
          break;
        }
      }
      if (!in_tile) {
        continue;
      }

      const float dx = static_cast<float>(x) - gk.mean_x;
      const float dy = static_cast<float>(y) - gk.mean_y;
      const float v0 = gk.inv_xx * dx + gk.inv_yx * dy;
      const float v1 = gk.inv_xy * dx + gk.inv_yy * dy;
      const float quad = dx * v0 + dy * v1;
      const float gaussian = expf(-0.5f * quad) * gk.normalization;
      const float weight = gaussian * opacities[g];

      float dot_grad_color = 0.0f;
      for (int ch = 0; ch < c && ch < 4; ++ch) {
        const float grad_val = grad_output[(y * w + x) * c + ch];
        dot_grad_color += grad_val * colors[g * c + ch];
        gcl[ch] += grad_val * weight;
      }

      const float gamma = dot_grad_color;
      gop += gamma * gaussian;
      gm_x += gamma * weight * v0;
      gm_y += gamma * weight * v1;

      const float outer00 = v0 * v0;
      const float outer01 = v0 * v1;
      const float outer10 = v1 * v0;
      const float outer11 = v1 * v1;
      gcv[0] += gamma * weight * 0.5f * (outer00 - gk.det_ratio * gk.inv_xx);
      gcv[1] += gamma * weight * 0.5f * (outer01 - gk.det_ratio * gk.inv_yx);
      gcv[2] += gamma * weight * 0.5f * (outer10 - gk.det_ratio * gk.inv_xy);
      gcv[3] += gamma * weight * 0.5f * (outer11 - gk.det_ratio * gk.inv_yy);
    }
  }

  grad_means[g * 2 + 0] = gm_x;
  grad_means[g * 2 + 1] = gm_y;
  grad_opacities[g] = gop;
  for (int ch = 0; ch < c && ch < 4; ++ch) {
    grad_colors[g * c + ch] = gcl[ch];
  }
  for (int i = 0; i < 4; ++i) {
    grad_covs[g * 4 + i] = gcv[i];
  }
}

}  // namespace detail
}  // namespace cuda
}  // namespace tinysplat
