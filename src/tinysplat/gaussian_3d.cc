#include "tinysplat/gaussian_3d.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <vector>

#include "tinysplat/parallel.h"

namespace tinysplat {
namespace {

struct ProjectedGaussian3D {
  float mean_x;
  float mean_y;
  float depth;
  float inv_xx;
  float inv_xy;
  float inv_yx;
  float inv_yy;
  int64_t min_x;
  int64_t max_x;
  int64_t min_y;
  int64_t max_y;
  int64_t source_index;
};

struct RasterGaussian2D {
  float mean_x;
  float mean_y;
  float inv_xx;
  float inv_xy;
  float inv_yx;
  float inv_yy;
  int64_t min_x;
  int64_t max_x;
  int64_t min_y;
  int64_t max_y;
};

std::vector<ProjectedGaussian3D> project_gaussians_3d_internal(
    const Gaussians3D& gaussians, const CameraIntrinsics& intrinsics,
    const Mat4& camera_to_world, int height, int width, const Splat3DOptions& opts) {
  const float fx = intrinsics.fx;
  const float fy = intrinsics.fy;
  const float cx = intrinsics.cx;
  const float cy = intrinsics.cy;

  const float r00 = camera_to_world.m[0][0];
  const float r01 = camera_to_world.m[0][1];
  const float r02 = camera_to_world.m[0][2];
  const float r10 = camera_to_world.m[1][0];
  const float r11 = camera_to_world.m[1][1];
  const float r12 = camera_to_world.m[1][2];
  const float r20 = camera_to_world.m[2][0];
  const float r21 = camera_to_world.m[2][1];
  const float r22 = camera_to_world.m[2][2];
  const float tx = camera_to_world.m[0][3];
  const float ty = camera_to_world.m[1][3];
  const float tz = camera_to_world.m[2][3];

  const float rwc00 = r00, rwc01 = r10, rwc02 = r20;
  const float rwc10 = r01, rwc11 = r11, rwc12 = r21;
  const float rwc20 = r02, rwc21 = r12, rwc22 = r22;
  const float twc0 = -(rwc00 * tx + rwc01 * ty + rwc02 * tz);
  const float twc1 = -(rwc10 * tx + rwc11 * ty + rwc12 * tz);
  const float twc2 = -(rwc20 * tx + rwc21 * ty + rwc22 * tz);

  std::vector<ProjectedGaussian3D> projected;
  std::mutex projected_mutex;
  const int64_t n = static_cast<int64_t>(gaussians.means.size());

  parallel_for(0, n, [&](int64_t begin, int64_t end) {
    std::vector<ProjectedGaussian3D> local;
    local.reserve(static_cast<std::size_t>(end - begin));

    for (int64_t g = begin; g < end; ++g) {
      const float mx = gaussians.means[g].x;
      const float my = gaussians.means[g].y;
      const float mz = gaussians.means[g].z;

      const float cam_x = rwc00 * mx + rwc01 * my + rwc02 * mz + twc0;
      const float cam_y = rwc10 * mx + rwc11 * my + rwc12 * mz + twc1;
      const float cam_z = rwc20 * mx + rwc21 * my + rwc22 * mz + twc2;
      if (cam_z <= opts.near_plane) {
        continue;
      }

      const Mat3& wc = gaussians.covariances[g];
      float t00 = rwc00 * wc.m[0][0] + rwc01 * wc.m[1][0] + rwc02 * wc.m[2][0];
      float t01 = rwc00 * wc.m[0][1] + rwc01 * wc.m[1][1] + rwc02 * wc.m[2][1];
      float t02 = rwc00 * wc.m[0][2] + rwc01 * wc.m[1][2] + rwc02 * wc.m[2][2];
      float t10 = rwc10 * wc.m[0][0] + rwc11 * wc.m[1][0] + rwc12 * wc.m[2][0];
      float t11 = rwc10 * wc.m[0][1] + rwc11 * wc.m[1][1] + rwc12 * wc.m[2][1];
      float t12 = rwc10 * wc.m[0][2] + rwc11 * wc.m[1][2] + rwc12 * wc.m[2][2];
      float t20 = rwc20 * wc.m[0][0] + rwc21 * wc.m[1][0] + rwc22 * wc.m[2][0];
      float t21 = rwc20 * wc.m[0][1] + rwc21 * wc.m[1][1] + rwc22 * wc.m[2][1];
      float t22 = rwc20 * wc.m[0][2] + rwc21 * wc.m[1][2] + rwc22 * wc.m[2][2];

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
      float s10 = j11 * (ccov01 * j00 + ccov02 * j02) + j12 * (ccov21 * j00 + ccov22 * j02);
      float s11 = j11 * (ccov01 * j11 + ccov02 * j12) + j12 * (ccov21 * j11 + ccov22 * j12);

      s00 += opts.min_covariance;
      s11 += opts.min_covariance;
      float det = s00 * s11 - s01 * s10;
      if (det <= opts.min_covariance) {
        det = opts.min_covariance;
      }
      const float inv_det = 1.0f / det;
      const float inv_xx = s11 * inv_det;
      const float inv_xy = -s01 * inv_det;
      const float inv_yx = -s10 * inv_det;
      const float inv_yy = s00 * inv_det;

      const float trace = s00 + s11;
      const float disc = std::sqrt(std::max(0.0f, (s00 - s11) * (s00 - s11) + 4.0f * s01 * s10));
      const float lambda_max = std::max((trace + disc) * 0.5f, opts.min_covariance);
      const float radius = opts.sigma_radius * std::sqrt(lambda_max);

      const int64_t min_x = static_cast<int64_t>(std::floor(proj_x - radius));
      const int64_t max_x = static_cast<int64_t>(std::ceil(proj_x + radius));
      const int64_t min_y = static_cast<int64_t>(std::floor(proj_y - radius));
      const int64_t max_y = static_cast<int64_t>(std::ceil(proj_y + radius));
      if (max_x < 0 || min_x >= width || max_y < 0 || min_y >= height) {
        continue;
      }

      local.push_back(ProjectedGaussian3D{proj_x, proj_y, cam_z, inv_xx, inv_xy, inv_yx, inv_yy,
                                          min_x, max_x, min_y, max_y, g});
    }

    std::lock_guard<std::mutex> lock(projected_mutex);
    projected.insert(projected.end(), local.begin(), local.end());
  });

  std::sort(projected.begin(), projected.end(),
            [](const ProjectedGaussian3D& a, const ProjectedGaussian3D& b) {
              return a.depth < b.depth;
            });
  return projected;
}

std::vector<RasterGaussian2D> precompute_raster_gaussians_2d(const ProjectedGaussians2D& gaussians,
                                                             int height, int width,
                                                             const Splat3DOptions& opts) {
  const int64_t n = static_cast<int64_t>(gaussians.means.size());
  std::vector<RasterGaussian2D> out(static_cast<std::size_t>(n));

  parallel_for(0, n, [&](int64_t begin, int64_t end) {
    for (int64_t g = begin; g < end; ++g) {
      float s00 = gaussians.covariances[g].m00 + opts.min_covariance;
      float s01 = gaussians.covariances[g].m01;
      float s10 = gaussians.covariances[g].m10;
      float s11 = gaussians.covariances[g].m11 + opts.min_covariance;
      float det = s00 * s11 - s01 * s10;
      if (det <= opts.min_covariance) {
        det = opts.min_covariance;
      }
      const float inv_det = 1.0f / det;
      const float inv_xx = s11 * inv_det;
      const float inv_xy = -s01 * inv_det;
      const float inv_yx = -s10 * inv_det;
      const float inv_yy = s00 * inv_det;

      const float trace = s00 + s11;
      const float disc =
          std::sqrt(std::max(0.0f, (s00 - s11) * (s00 - s11) + 4.0f * s01 * s10));
      const float lambda_max = std::max((trace + disc) * 0.5f, opts.min_covariance);
      const float radius = opts.sigma_radius * std::sqrt(lambda_max);

      int64_t min_x = static_cast<int64_t>(std::floor(gaussians.means[g].x - radius));
      int64_t max_x = static_cast<int64_t>(std::ceil(gaussians.means[g].x + radius));
      int64_t min_y = static_cast<int64_t>(std::floor(gaussians.means[g].y - radius));
      int64_t max_y = static_cast<int64_t>(std::ceil(gaussians.means[g].y + radius));
      if (max_x < 0 || min_x >= width || max_y < 0 || min_y >= height) {
        min_x = 1;
        max_x = 0;
        min_y = 1;
        max_y = 0;
      }

      out[static_cast<std::size_t>(g)] = RasterGaussian2D{
          gaussians.means[g].x,
          gaussians.means[g].y,
          inv_xx,
          inv_xy,
          inv_yx,
          inv_yy,
          min_x,
          max_x,
          min_y,
          max_y,
      };
    }
  });

  return out;
}

}  // namespace

Image gaussian_splat_3d_forward(const Gaussians3D& gaussians, const CameraIntrinsics& intrinsics,
                                const Mat4& camera_to_world, int height, int width,
                                const Splat3DOptions& opts) {
  const int num_channels =
      gaussians.colors.empty() ? 3 : static_cast<int>(gaussians.colors[0].size());
  Image image(height, width, num_channels);
  std::vector<float> transmittance(static_cast<std::size_t>(height * width), 1.0f);

  const auto projected =
      project_gaussians_3d_internal(gaussians, intrinsics, camera_to_world, height, width, opts);

  for (const auto& pg : projected) {
    const int64_t x0 = std::max<int64_t>(0, pg.min_x);
    const int64_t x1 = std::min<int64_t>(width - 1, pg.max_x);
    const int64_t y0 = std::max<int64_t>(0, pg.min_y);
    const int64_t y1 = std::min<int64_t>(height - 1, pg.max_y);
    const int64_t src = pg.source_index;

    for (int64_t y = y0; y <= y1; ++y) {
      for (int64_t x = x0; x <= x1; ++x) {
        const float dx = static_cast<float>(x) - pg.mean_x;
        const float dy = static_cast<float>(y) - pg.mean_y;
        const float quad = dx * (pg.inv_xx * dx + pg.inv_xy * dy) +
                           dy * (pg.inv_yx * dx + pg.inv_yy * dy);
        const float gaussian = std::exp(-0.5f * quad);
        float alpha = gaussians.opacities[src] * gaussian;
        alpha = std::clamp(alpha, 0.0f, 0.999f);

        const std::size_t idx = static_cast<std::size_t>(y * width + x);
        const float t = transmittance[idx];
        for (int c = 0; c < num_channels; ++c) {
          image.at(static_cast<int>(y), static_cast<int>(x), c) +=
              t * alpha * gaussians.colors[src][c];
        }
        transmittance[idx] = t * (1.0f - alpha);
      }
    }
  }

  return image;
}

Image gaussian_splat_3d_projected_forward(const ProjectedGaussians2D& gaussians, int height,
                                          int width, const Splat3DOptions& opts) {
  const int num_channels =
      gaussians.colors.empty() ? 3 : static_cast<int>(gaussians.colors[0].size());
  Image image(height, width, num_channels);
  std::vector<float> transmittance(static_cast<std::size_t>(height * width), 1.0f);

  const auto raster = precompute_raster_gaussians_2d(gaussians, height, width, opts);
  const int64_t n = static_cast<int64_t>(raster.size());

  for (int64_t g = 0; g < n; ++g) {
    const auto& rg = raster[static_cast<std::size_t>(g)];
    if (rg.max_x < rg.min_x || rg.max_y < rg.min_y) {
      continue;
    }
    const int64_t x0 = std::max<int64_t>(0, rg.min_x);
    const int64_t x1 = std::min<int64_t>(width - 1, rg.max_x);
    const int64_t y0 = std::max<int64_t>(0, rg.min_y);
    const int64_t y1 = std::min<int64_t>(height - 1, rg.max_y);

    for (int64_t y = y0; y <= y1; ++y) {
      for (int64_t x = x0; x <= x1; ++x) {
        const float dx = static_cast<float>(x) - rg.mean_x;
        const float dy = static_cast<float>(y) - rg.mean_y;
        const float quad = dx * (rg.inv_xx * dx + rg.inv_xy * dy) +
                           dy * (rg.inv_yx * dx + rg.inv_yy * dy);
        const float gaussian = std::exp(-0.5f * quad);
        float alpha = gaussians.opacities[g] * gaussian;
        alpha = std::clamp(alpha, 0.0f, 0.999f);

        const std::size_t idx = static_cast<std::size_t>(y * width + x);
        const float t = transmittance[idx];
        for (int c = 0; c < num_channels; ++c) {
          image.at(static_cast<int>(y), static_cast<int>(x), c) +=
              t * alpha * gaussians.colors[g][c];
        }
        transmittance[idx] = t * (1.0f - alpha);
      }
    }
  }

  return image;
}

ProjectedGaussians2D project_gaussians_3d_to_2d(const Gaussians3D& gaussians,
                                                const CameraIntrinsics& intrinsics,
                                                const Mat4& camera_to_world, int height,
                                                int width, const Splat3DOptions& opts) {
  ProjectedGaussians2D out;
  const auto projected =
      project_gaussians_3d_internal(gaussians, intrinsics, camera_to_world, height, width, opts);

  out.means.reserve(projected.size());
  out.covariances.reserve(projected.size());
  out.colors.reserve(projected.size());
  out.opacities.reserve(projected.size());

  for (const auto& pg : projected) {
    out.means.push_back({pg.mean_x, pg.mean_y});
    const float inv_xx = pg.inv_xx;
    const float inv_xy = pg.inv_xy;
    const float inv_yx = pg.inv_yx;
    const float inv_yy = pg.inv_yy;
    const float det = inv_xx * inv_yy - inv_xy * inv_yx;
    const float inv_det = 1.0f / std::max(det, opts.min_covariance);
    Mat2 cov;
    cov.m00 = inv_yy * inv_det;
    cov.m01 = -inv_xy * inv_det;
    cov.m10 = -inv_yx * inv_det;
    cov.m11 = inv_xx * inv_det;
    out.covariances.push_back(cov);
    out.colors.push_back(gaussians.colors[pg.source_index]);
    out.opacities.push_back(gaussians.opacities[pg.source_index]);
  }

  return out;
}

}  // namespace tinysplat
