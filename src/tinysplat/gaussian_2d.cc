#include "tinysplat/gaussian_2d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <vector>

#include "tinysplat/parallel.h"

namespace tinysplat {
namespace {

constexpr int64_t kTileSize = 16;
constexpr float kSigmaRadius = 4.0f;
constexpr float kEps = 1e-8f;
constexpr float kPi = 3.14159265358979323846f;

struct GaussianPrecomputed {
  float inv_xx;
  float inv_xy;
  float inv_yx;
  float inv_yy;
  float normalization;
  float det_ratio;
  int64_t min_x;
  int64_t max_x;
  int64_t min_y;
  int64_t max_y;
};

std::vector<GaussianPrecomputed> precompute_gaussians(const Gaussians2D& gaussians, int height,
                                                      int width) {
  const int64_t n = static_cast<int64_t>(gaussians.means.size());
  std::vector<GaussianPrecomputed> params(static_cast<std::size_t>(n));

  parallel_for(0, n, [&](int64_t begin, int64_t end) {
    for (int64_t g = begin; g < end; ++g) {
      const float a = gaussians.covariances[g].m00;
      const float b = gaussians.covariances[g].m01;
      const float c = gaussians.covariances[g].m10;
      const float d = gaussians.covariances[g].m11;

      float det = a * d - b * c;
      if (det < kEps) {
        det = kEps;
      }

      const float inv_det = 1.0f / det;
      const float trace = a + d;
      const float disc = std::sqrt(std::max(0.0f, (a - d) * (a - d) + 4.0f * b * c));
      const float lambda_max = std::max((trace + disc) * 0.5f, kEps);
      const float radius = std::ceil(kSigmaRadius * std::sqrt(lambda_max));

      const float mx = gaussians.means[g].x;
      const float my = gaussians.means[g].y;

      const int64_t min_x =
          std::max<int64_t>(0, static_cast<int64_t>(std::floor(mx - radius)));
      const int64_t max_x =
          std::min<int64_t>(width - 1, static_cast<int64_t>(std::ceil(mx + radius)));
      const int64_t min_y =
          std::max<int64_t>(0, static_cast<int64_t>(std::floor(my - radius)));
      const int64_t max_y =
          std::min<int64_t>(height - 1, static_cast<int64_t>(std::ceil(my + radius)));

      params[static_cast<std::size_t>(g)] = GaussianPrecomputed{
          d * inv_det,
          -b * inv_det,
          -c * inv_det,
          a * inv_det,
          1.0f / (2.0f * kPi * std::sqrt(det + kEps)),
          det / (det + kEps),
          min_x,
          max_x,
          min_y,
          max_y,
      };
    }
  });

  return params;
}

std::vector<std::vector<int64_t>> build_tile_bins(const std::vector<GaussianPrecomputed>& params,
                                                  int64_t tiles_x, int64_t tiles_y) {
  const int64_t n = static_cast<int64_t>(params.size());
  std::vector<std::vector<int64_t>> tile_bins(static_cast<std::size_t>(tiles_x * tiles_y));
  std::vector<std::mutex> tile_mutexes(static_cast<std::size_t>(tiles_x * tiles_y));

  parallel_for(0, n, [&](int64_t begin, int64_t end) {
    for (int64_t g = begin; g < end; ++g) {
      const auto& gp = params[static_cast<std::size_t>(g)];
      if (gp.max_x < gp.min_x || gp.max_y < gp.min_y) {
        continue;
      }
      const int64_t tile_min_x = gp.min_x / kTileSize;
      const int64_t tile_max_x = gp.max_x / kTileSize;
      const int64_t tile_min_y = gp.min_y / kTileSize;
      const int64_t tile_max_y = gp.max_y / kTileSize;
      for (int64_t ty = tile_min_y; ty <= tile_max_y; ++ty) {
        for (int64_t tx = tile_min_x; tx <= tile_max_x; ++tx) {
          const std::size_t idx = static_cast<std::size_t>(ty * tiles_x + tx);
          std::lock_guard<std::mutex> lock(tile_mutexes[idx]);
          tile_bins[idx].push_back(g);
        }
      }
    }
  });

  return tile_bins;
}

void compute_forward(const Gaussians2D& gaussians,
                     const std::vector<GaussianPrecomputed>& gaussian_params,
                     const std::vector<std::vector<int64_t>>& tile_bins, Image& output,
                     std::vector<float>& total_weight, int64_t tiles_x, int64_t tiles_y,
                     int num_channels, int height, int width) {
  parallel_for(0, tiles_x * tiles_y, [&](int64_t begin, int64_t end) {
    std::vector<float> accum_dynamic;
    std::array<float, 4> accum_small{};

    for (int64_t tile_idx = begin; tile_idx < end; ++tile_idx) {
      const int64_t tile_y = tile_idx / tiles_x;
      const int64_t tile_x = tile_idx % tiles_x;
      const int64_t start_x = tile_x * kTileSize;
      const int64_t end_x = std::min(start_x + kTileSize, static_cast<int64_t>(width));
      const int64_t start_y = tile_y * kTileSize;
      const int64_t end_y = std::min(start_y + kTileSize, static_cast<int64_t>(height));
      const auto& gaussian_ids = tile_bins[static_cast<std::size_t>(tile_idx)];

      for (int64_t y = start_y; y < end_y; ++y) {
        for (int64_t x = start_x; x < end_x; ++x) {
          float* accum_ptr = nullptr;
          if (num_channels <= 4) {
            accum_small.fill(0.0f);
            accum_ptr = accum_small.data();
          } else {
            accum_dynamic.assign(static_cast<std::size_t>(num_channels), 0.0f);
            accum_ptr = accum_dynamic.data();
          }
          float weight_sum = 0.0f;

          for (const int64_t g : gaussian_ids) {
            const auto& gp = gaussian_params[static_cast<std::size_t>(g)];
            if (x < gp.min_x || x > gp.max_x || y < gp.min_y || y > gp.max_y) {
              continue;
            }
            const float dx = static_cast<float>(x) - gaussians.means[g].x;
            const float dy = static_cast<float>(y) - gaussians.means[g].y;
            const float qx = gp.inv_xx * dx + gp.inv_xy * dy;
            const float qy = gp.inv_yx * dx + gp.inv_yy * dy;
            const float quad = dx * qx + dy * qy;
            const float gaussian = std::exp(-0.5f * quad) * gp.normalization;
            const float weight = gaussian * gaussians.opacities[g];

            weight_sum += weight;
            for (int c = 0; c < num_channels; ++c) {
              accum_ptr[c] += weight * gaussians.colors[g][c];
            }
          }

          const float denom = std::max(weight_sum, kEps);
          total_weight[static_cast<std::size_t>(y * width + x)] = denom;
          for (int c = 0; c < num_channels; ++c) {
            output.at(static_cast<int>(y), static_cast<int>(x), c) = accum_ptr[c] / denom;
          }
        }
      }
    }
  });

  if (num_channels == 4) {
    parallel_for(0, static_cast<int64_t>(height) * width, [&](int64_t begin, int64_t end) {
      for (int64_t linear = begin; linear < end; ++linear) {
        const int y = static_cast<int>(linear / width);
        const int x = static_cast<int>(linear % width);
        const float alpha = output.at(y, x, 3);
        output.at(y, x, 0) *= alpha;
        output.at(y, x, 1) *= alpha;
        output.at(y, x, 2) *= alpha;
      }
    });
  }
}

}  // namespace

Image gaussian_splat_2d_forward(const Gaussians2D& gaussians, int height, int width) {
  if (gaussians.means.empty()) {
    return Image(height, width, 3);
  }
  const int num_channels = static_cast<int>(gaussians.colors[0].size());
  Image output(height, width, num_channels);
  std::vector<float> total_weight(static_cast<std::size_t>(height * width), 0.0f);

  const auto gaussian_params = precompute_gaussians(gaussians, height, width);
  const int64_t tiles_x = (width + kTileSize - 1) / kTileSize;
  const int64_t tiles_y = (height + kTileSize - 1) / kTileSize;
  const auto tile_bins = build_tile_bins(gaussian_params, tiles_x, tiles_y);

  compute_forward(gaussians, gaussian_params, tile_bins, output, total_weight, tiles_x, tiles_y,
                  num_channels, height, width);
  return output;
}

Gradients2D gaussian_splat_2d_backward(const Image& grad_output, const Gaussians2D& gaussians,
                                       int height, int width) {
  const int64_t n = static_cast<int64_t>(gaussians.means.size());
  const int num_channels = static_cast<int>(gaussians.colors[0].size());

  Gradients2D grads;
  grads.grad_means.assign(gaussians.means.size(), Vec2{});
  grads.grad_covariances.assign(gaussians.covariances.size(), Mat2{});
  grads.grad_colors = gaussians.colors;
  for (auto& row : grads.grad_colors) {
    std::fill(row.begin(), row.end(), 0.0f);
  }
  grads.grad_opacities.assign(gaussians.opacities.size(), 0.0f);

  Image output(height, width, num_channels);
  std::vector<float> total_weight(static_cast<std::size_t>(height * width), 0.0f);

  const auto gaussian_params = precompute_gaussians(gaussians, height, width);
  const int64_t tiles_x = (width + kTileSize - 1) / kTileSize;
  const int64_t tiles_y = (height + kTileSize - 1) / kTileSize;
  const auto tile_bins = build_tile_bins(gaussian_params, tiles_x, tiles_y);

  compute_forward(gaussians, gaussian_params, tile_bins, output, total_weight, tiles_x, tiles_y,
                  num_channels, height, width);

  std::vector<float> grad_image_dot_output(static_cast<std::size_t>(height * width), 0.0f);
  parallel_for(0, static_cast<int64_t>(height) * width, [&](int64_t begin, int64_t end) {
    for (int64_t linear = begin; linear < end; ++linear) {
      const int y = static_cast<int>(linear / width);
      const int x = static_cast<int>(linear % width);
      float value = 0.0f;
      for (int c = 0; c < num_channels; ++c) {
        value += grad_output.at(y, x, c) * output.at(y, x, c);
      }
      grad_image_dot_output[static_cast<std::size_t>(linear)] = value;
    }
  });

  parallel_for(0, n, [&](int64_t begin, int64_t end) {
    std::vector<float> grad_color_dynamic;
    std::array<float, 4> grad_color_small{};

    for (int64_t g = begin; g < end; ++g) {
      const auto& gp = gaussian_params[static_cast<std::size_t>(g)];
      float grad_mean_x = 0.0f;
      float grad_mean_y = 0.0f;
      float grad_opacity = 0.0f;
      float* grad_color_ptr = nullptr;
      if (num_channels <= 4) {
        grad_color_small.fill(0.0f);
        grad_color_ptr = grad_color_small.data();
      } else {
        grad_color_dynamic.assign(static_cast<std::size_t>(num_channels), 0.0f);
        grad_color_ptr = grad_color_dynamic.data();
      }
      float grad_cov_00 = 0.0f;
      float grad_cov_01 = 0.0f;
      float grad_cov_10 = 0.0f;
      float grad_cov_11 = 0.0f;

      for (int64_t y = gp.min_y; y <= gp.max_y; ++y) {
        for (int64_t x = gp.min_x; x <= gp.max_x; ++x) {
          const float dx = static_cast<float>(x) - gaussians.means[g].x;
          const float dy = static_cast<float>(y) - gaussians.means[g].y;
          const float v0 = gp.inv_xx * dx + gp.inv_yx * dy;
          const float v1 = gp.inv_xy * dx + gp.inv_yy * dy;
          const float quad = dx * v0 + dy * v1;
          const float gaussian = std::exp(-0.5f * quad) * gp.normalization;
          const float weight = gaussian * gaussians.opacities[g];
          const float denom =
              total_weight[static_cast<std::size_t>(y * width + x)];

          float dot_grad_color = 0.0f;
          for (int c = 0; c < num_channels; ++c) {
            const float grad_val = grad_output.at(static_cast<int>(y), static_cast<int>(x), c);
            dot_grad_color += grad_val * gaussians.colors[g][c];
            grad_color_ptr[c] += grad_val * weight / denom;
          }

          const float gamma =
              (dot_grad_color -
               grad_image_dot_output[static_cast<std::size_t>(y * width + x)]) /
              denom;
          grad_opacity += gamma * gaussian;
          grad_mean_x += gamma * weight * v0;
          grad_mean_y += gamma * weight * v1;

          const float outer00 = v0 * v0;
          const float outer01 = v0 * v1;
          const float outer10 = v1 * v0;
          const float outer11 = v1 * v1;

          grad_cov_00 += gamma * weight * 0.5f * (outer00 - gp.det_ratio * gp.inv_xx);
          grad_cov_01 += gamma * weight * 0.5f * (outer01 - gp.det_ratio * gp.inv_yx);
          grad_cov_10 += gamma * weight * 0.5f * (outer10 - gp.det_ratio * gp.inv_xy);
          grad_cov_11 += gamma * weight * 0.5f * (outer11 - gp.det_ratio * gp.inv_yy);
        }
      }

      grads.grad_means[g].x = grad_mean_x;
      grads.grad_means[g].y = grad_mean_y;
      grads.grad_opacities[g] = grad_opacity;
      for (int c = 0; c < num_channels; ++c) {
        grads.grad_colors[g][c] = grad_color_ptr[c];
      }
      grads.grad_covariances[g].m00 = grad_cov_00;
      grads.grad_covariances[g].m01 = grad_cov_01;
      grads.grad_covariances[g].m10 = grad_cov_10;
      grads.grad_covariances[g].m11 = grad_cov_11;
    }
  });

  return grads;
}

}  // namespace tinysplat
