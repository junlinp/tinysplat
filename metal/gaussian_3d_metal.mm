#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "tinysplat/gaussian_3d_metal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace tinysplat {
namespace metal {
namespace {

constexpr int kTileSize = 16;
constexpr int kMaxChannels = 3;

struct ProjectedGaussian {
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

struct RasterParams {
  uint32_t height;
  uint32_t width;
  uint32_t num_channels;
  uint32_t tiles_x;
  uint32_t tile_size;
  uint32_t num_projected;
  uint32_t tiles_y;
};

struct CameraParams {
  float fx;
  float fy;
  float cx;
  float cy;
  float rwc00;
  float rwc01;
  float rwc02;
  float twc0;
  float rwc10;
  float rwc11;
  float rwc12;
  float twc1;
  float rwc20;
  float rwc21;
  float rwc22;
  float twc2;
  float near_plane;
  float min_covariance;
  float sigma_radius;
  float compact_box_beta;
  uint32_t use_compact_box;
  uint32_t height;
  uint32_t width;
  uint32_t n;
  uint32_t has_depths;
};

// Embedded Metal shaders: GPU project/tiles + tiled forward/backward with cooperative load.
constexpr const char* kMetalShaders = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct ProjectedGaussian {
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

struct RasterParams {
  uint height;
  uint width;
  uint num_channels;
  uint tiles_x;
  uint tile_size;
  uint num_projected;
  uint tiles_y;
};

struct CameraParams {
  float fx;
  float fy;
  float cx;
  float cy;
  float rwc00;
  float rwc01;
  float rwc02;
  float twc0;
  float rwc10;
  float rwc11;
  float rwc12;
  float twc1;
  float rwc20;
  float rwc21;
  float rwc22;
  float twc2;
  float near_plane;
  float min_covariance;
  float sigma_radius;
  float compact_box_beta;
  uint use_compact_box;
  uint height;
  uint width;
  uint n;
  uint has_depths;
};

inline float compact_radius_m(float s00, float s01, float s11, float min_covariance,
                              float sigma_radius, float beta, bool use_compact_box) {
  const float trace = s00 + s11;
  const float disc = sqrt(max(0.0f, (s00 - s11) * (s00 - s11) + 4.0f * s01 * s01));
  const float lambda_max = max((trace + disc) * 0.5f, min_covariance);
  const float sigma = sqrt(lambda_max);
  if (use_compact_box) {
    return min(sigma_radius, beta) * sigma;
  }
  return sigma_radius * sigma;
}

inline void world_cov_from_quat_logscale(float qw, float qx, float qy, float qz, float lx, float ly,
                                         float lz, thread float* wc) {
  float n = sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
  n = max(n, 1e-8f);
  qw /= n;
  qx /= n;
  qy /= n;
  qz /= n;
  const float xx = qx * qx, yy = qy * qy, zz = qz * qz;
  const float xy = qx * qy, xz = qx * qz, yz = qy * qz;
  const float wx = qw * qx, wy = qw * qy, wz = qw * qz;
  const float r00 = 1.0f - 2.0f * (yy + zz);
  const float r01 = 2.0f * (xy - wz);
  const float r02 = 2.0f * (xz + wy);
  const float r10 = 2.0f * (xy + wz);
  const float r11 = 1.0f - 2.0f * (xx + zz);
  const float r12 = 2.0f * (yz - wx);
  const float r20 = 2.0f * (xz - wy);
  const float r21 = 2.0f * (yz + wx);
  const float r22 = 1.0f - 2.0f * (xx + yy);
  const float sx = exp(lx);
  const float sy = exp(ly);
  const float sz = exp(lz);
  const float d0 = sx * sx;
  const float d1 = sy * sy;
  const float d2 = sz * sz;
  const float m00 = r00 * d0, m01 = r01 * d1, m02 = r02 * d2;
  const float m10 = r10 * d0, m11 = r11 * d1, m12 = r12 * d2;
  const float m20 = r20 * d0, m21 = r21 * d1, m22 = r22 * d2;
  wc[0] = m00 * r00 + m01 * r01 + m02 * r02;
  wc[1] = m00 * r10 + m01 * r11 + m02 * r12;
  wc[2] = m00 * r20 + m01 * r21 + m02 * r22;
  wc[3] = m10 * r00 + m11 * r01 + m12 * r02;
  wc[4] = m10 * r10 + m11 * r11 + m12 * r12;
  wc[5] = m10 * r20 + m11 * r21 + m12 * r22;
  wc[6] = m20 * r00 + m21 * r01 + m22 * r02;
  wc[7] = m20 * r10 + m21 * r11 + m22 * r12;
  wc[8] = m20 * r20 + m21 * r21 + m22 * r22;
  wc[0] += 1e-6f;
  wc[4] += 1e-6f;
  wc[8] += 1e-6f;
}

inline void dsigma_to_quat_logscale(float qw, float qx, float qy, float qz, float lx, float ly,
                                    float lz, float g00, float g01, float g02, float g10, float g11,
                                    float g12, float g20, float g21, float g22,
                                    thread float* dlog, thread float* dquat) {
  float n = sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
  n = max(n, 1e-8f);
  const float inv_n = 1.0f / n;
  qw *= inv_n;
  qx *= inv_n;
  qy *= inv_n;
  qz *= inv_n;
  const float xx = qx * qx, yy = qy * qy, zz = qz * qz;
  const float xy = qx * qy, xz = qx * qz, yz = qy * qz;
  const float wx = qw * qx, wy = qw * qy, wz = qw * qz;
  const float r00 = 1.0f - 2.0f * (yy + zz);
  const float r01 = 2.0f * (xy - wz);
  const float r02 = 2.0f * (xz + wy);
  const float r10 = 2.0f * (xy + wz);
  const float r11 = 1.0f - 2.0f * (xx + zz);
  const float r12 = 2.0f * (yz - wx);
  const float r20 = 2.0f * (xz - wy);
  const float r21 = 2.0f * (yz + wx);
  const float r22 = 1.0f - 2.0f * (xx + yy);
  const float sx = exp(lx);
  const float sy = exp(ly);
  const float sz = exp(lz);
  const float d0 = sx * sx;
  const float d1 = sy * sy;
  const float d2 = sz * sz;

  const float gs00 = g00 + g00;
  const float gs01 = g01 + g10;
  const float gs02 = g02 + g20;
  const float gs10 = gs01;
  const float gs11 = g11 + g11;
  const float gs12 = g12 + g21;
  const float gs20 = gs02;
  const float gs21 = gs12;
  const float gs22 = g22 + g22;

  const float gr00 = gs00 * r00 + gs01 * r10 + gs02 * r20;
  const float gr01 = gs00 * r01 + gs01 * r11 + gs02 * r21;
  const float gr02 = gs00 * r02 + gs01 * r12 + gs02 * r22;
  const float gr10 = gs10 * r00 + gs11 * r10 + gs12 * r20;
  const float gr11 = gs10 * r01 + gs11 * r11 + gs12 * r21;
  const float gr12 = gs10 * r02 + gs11 * r12 + gs12 * r22;
  const float gr20 = gs20 * r00 + gs21 * r10 + gs22 * r20;
  const float gr21 = gs20 * r01 + gs21 * r11 + gs22 * r21;
  const float gr22 = gs20 * r02 + gs21 * r12 + gs22 * r22;
  const float dR00 = gr00 * d0;
  const float dR01 = gr01 * d1;
  const float dR02 = gr02 * d2;
  const float dR10 = gr10 * d0;
  const float dR11 = gr11 * d1;
  const float dR12 = gr12 * d2;
  const float dR20 = gr20 * d0;
  const float dR21 = gr21 * d1;
  const float dR22 = gr22 * d2;

  const float rtg00 = r00 * g00 + r10 * g10 + r20 * g20;
  const float rtg01 = r00 * g01 + r10 * g11 + r20 * g21;
  const float rtg02 = r00 * g02 + r10 * g12 + r20 * g22;
  const float rtg10 = r01 * g00 + r11 * g10 + r21 * g20;
  const float rtg11 = r01 * g01 + r11 * g11 + r21 * g21;
  const float rtg12 = r01 * g02 + r11 * g12 + r21 * g22;
  const float rtg20 = r02 * g00 + r12 * g10 + r22 * g20;
  const float rtg21 = r02 * g01 + r12 * g11 + r22 * g21;
  const float rtg22 = r02 * g02 + r12 * g12 + r22 * g22;
  dlog[0] = 2.0f * d0 * (rtg00 * r00 + rtg01 * r10 + rtg02 * r20);
  dlog[1] = 2.0f * d1 * (rtg10 * r01 + rtg11 * r11 + rtg12 * r21);
  dlog[2] = 2.0f * d2 * (rtg20 * r02 + rtg21 * r12 + rtg22 * r22);

  float dw = 0.0f, dx = 0.0f, dy = 0.0f, dz = 0.0f;
  dy += dR00 * (-4.0f * qy);
  dz += dR00 * (-4.0f * qz);
  dx += dR01 * (2.0f * qy);
  dy += dR01 * (2.0f * qx);
  dw += dR01 * (-2.0f * qz);
  dz += dR01 * (-2.0f * qw);
  dx += dR02 * (2.0f * qz);
  dz += dR02 * (2.0f * qx);
  dw += dR02 * (2.0f * qy);
  dy += dR02 * (2.0f * qw);
  dx += dR10 * (2.0f * qy);
  dy += dR10 * (2.0f * qx);
  dw += dR10 * (2.0f * qz);
  dz += dR10 * (2.0f * qw);
  dx += dR11 * (-4.0f * qx);
  dz += dR11 * (-4.0f * qz);
  dy += dR12 * (2.0f * qz);
  dz += dR12 * (2.0f * qy);
  dw += dR12 * (-2.0f * qx);
  dx += dR12 * (-2.0f * qw);
  dx += dR20 * (2.0f * qz);
  dz += dR20 * (2.0f * qx);
  dw += dR20 * (-2.0f * qy);
  dy += dR20 * (-2.0f * qw);
  dy += dR21 * (2.0f * qz);
  dz += dR21 * (2.0f * qy);
  dw += dR21 * (2.0f * qx);
  dx += dR21 * (2.0f * qw);
  dx += dR22 * (-4.0f * qx);
  dy += dR22 * (-4.0f * qy);
  const float qdot = dw * qw + dx * qx + dy * qy + dz * qz;
  dquat[0] = (dw - qw * qdot) * inv_n;
  dquat[1] = (dx - qx * qdot) * inv_n;
  dquat[2] = (dy - qy * qdot) * inv_n;
  dquat[3] = (dz - qz * qdot) * inv_n;
}

inline ProjectedGaussian invalid_projected(uint gid) {
  ProjectedGaussian pg;
  pg.mean_x = 0.0f;
  pg.mean_y = 0.0f;
  pg.depth = 1e30f;
  pg.inv_xx = 1.0f;
  pg.inv_xy = 0.0f;
  pg.inv_yx = 0.0f;
  pg.inv_yy = 1.0f;
  pg.min_x = 1;
  pg.max_x = 0;
  pg.min_y = 1;
  pg.max_y = 0;
  pg.source_index = int(gid);
  return pg;
}

kernel void project_gaussians_3d(
    device const float* means [[buffer(0)]],
    device const float* covs [[buffer(1)]],
    constant CameraParams& cam [[buffer(2)]],
    device ProjectedGaussian* out [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
  if (gid >= cam.n) return;
  const float mx = means[gid * 3 + 0];
  const float my = means[gid * 3 + 1];
  const float mz = means[gid * 3 + 2];
  const float cam_x = cam.rwc00 * mx + cam.rwc01 * my + cam.rwc02 * mz + cam.twc0;
  const float cam_y = cam.rwc10 * mx + cam.rwc11 * my + cam.rwc12 * mz + cam.twc1;
  const float cam_z = cam.rwc20 * mx + cam.rwc21 * my + cam.rwc22 * mz + cam.twc2;
  if (cam_z <= cam.near_plane) {
    out[gid] = invalid_projected(gid);
    return;
  }

  device const float* wc = covs + gid * 9;
  float t00 = cam.rwc00 * wc[0] + cam.rwc01 * wc[3] + cam.rwc02 * wc[6];
  float t01 = cam.rwc00 * wc[1] + cam.rwc01 * wc[4] + cam.rwc02 * wc[7];
  float t02 = cam.rwc00 * wc[2] + cam.rwc01 * wc[5] + cam.rwc02 * wc[8];
  float t10 = cam.rwc10 * wc[0] + cam.rwc11 * wc[3] + cam.rwc12 * wc[6];
  float t11 = cam.rwc10 * wc[1] + cam.rwc11 * wc[4] + cam.rwc12 * wc[7];
  float t12 = cam.rwc10 * wc[2] + cam.rwc11 * wc[5] + cam.rwc12 * wc[8];
  float t20 = cam.rwc20 * wc[0] + cam.rwc21 * wc[3] + cam.rwc22 * wc[6];
  float t21 = cam.rwc20 * wc[1] + cam.rwc21 * wc[4] + cam.rwc22 * wc[7];
  float t22 = cam.rwc20 * wc[2] + cam.rwc21 * wc[5] + cam.rwc22 * wc[8];

  const float ccov00 = t00 * cam.rwc00 + t01 * cam.rwc01 + t02 * cam.rwc02;
  const float ccov01 = t00 * cam.rwc10 + t01 * cam.rwc11 + t02 * cam.rwc12;
  const float ccov02 = t00 * cam.rwc20 + t01 * cam.rwc21 + t02 * cam.rwc22;
  const float ccov10 = t10 * cam.rwc00 + t11 * cam.rwc01 + t12 * cam.rwc02;
  const float ccov11 = t10 * cam.rwc10 + t11 * cam.rwc11 + t12 * cam.rwc12;
  const float ccov12 = t10 * cam.rwc20 + t11 * cam.rwc21 + t12 * cam.rwc22;
  const float ccov20 = t20 * cam.rwc00 + t21 * cam.rwc01 + t22 * cam.rwc02;
  const float ccov21 = t20 * cam.rwc10 + t21 * cam.rwc11 + t22 * cam.rwc12;
  const float ccov22 = t20 * cam.rwc20 + t21 * cam.rwc21 + t22 * cam.rwc22;

  const float proj_x = cam.fx * cam_x / cam_z + cam.cx;
  const float proj_y = cam.fy * cam_y / cam_z + cam.cy;
  const float j00 = cam.fx / cam_z;
  const float j02 = -cam.fx * cam_x / (cam_z * cam_z);
  const float j11 = cam.fy / cam_z;
  const float j12 = -cam.fy * cam_y / (cam_z * cam_z);

  float s00 = j00 * (ccov00 * j00 + ccov02 * j02) + j02 * (ccov20 * j00 + ccov22 * j02);
  float s01 = j00 * (ccov01 * j11 + ccov02 * j12) + j02 * (ccov21 * j11 + ccov22 * j12);
  float s10 = j11 * (ccov10 * j00 + ccov12 * j02) + j12 * (ccov20 * j00 + ccov22 * j02);
  float s11 = j11 * (ccov11 * j11 + ccov12 * j12) + j12 * (ccov21 * j11 + ccov22 * j12);
  s00 += cam.min_covariance;
  s11 += cam.min_covariance;
  float det = s00 * s11 - s01 * s10;
  if (det <= cam.min_covariance) {
    det = cam.min_covariance;
  }
  const float inv_det = 1.0f / det;
  const float radius =
      compact_radius_m(s00, s01, s11, cam.min_covariance, cam.sigma_radius, cam.compact_box_beta,
                       cam.use_compact_box != 0);
  const int min_x = int(floor(proj_x - radius));
  const int max_x = int(ceil(proj_x + radius));
  const int min_y = int(floor(proj_y - radius));
  const int max_y = int(ceil(proj_y + radius));
  if (max_x < 0 || min_x >= int(cam.width) || max_y < 0 || min_y >= int(cam.height)) {
    out[gid] = invalid_projected(gid);
    return;
  }

  ProjectedGaussian pg;
  pg.mean_x = proj_x;
  pg.mean_y = proj_y;
  pg.depth = cam_z;
  pg.inv_xx = s11 * inv_det;
  pg.inv_xy = -s01 * inv_det;
  pg.inv_yx = -s10 * inv_det;
  pg.inv_yy = s00 * inv_det;
  pg.min_x = min_x;
  pg.max_x = max_x;
  pg.min_y = min_y;
  pg.max_y = max_y;
  pg.source_index = int(gid);
  out[gid] = pg;
}

kernel void project_gaussians_3d_qs(
    device const float* means [[buffer(0)]],
    device const float* log_scales [[buffer(1)]],
    device const float* quats [[buffer(2)]],
    constant CameraParams& cam [[buffer(3)]],
    device ProjectedGaussian* out [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
  if (gid >= cam.n) return;
  const float mx = means[gid * 3 + 0];
  const float my = means[gid * 3 + 1];
  const float mz = means[gid * 3 + 2];
  const float cam_x = cam.rwc00 * mx + cam.rwc01 * my + cam.rwc02 * mz + cam.twc0;
  const float cam_y = cam.rwc10 * mx + cam.rwc11 * my + cam.rwc12 * mz + cam.twc1;
  const float cam_z = cam.rwc20 * mx + cam.rwc21 * my + cam.rwc22 * mz + cam.twc2;
  if (cam_z <= cam.near_plane) {
    out[gid] = invalid_projected(gid);
    return;
  }

  float wc[9];
  world_cov_from_quat_logscale(quats[gid * 4 + 0], quats[gid * 4 + 1], quats[gid * 4 + 2],
                               quats[gid * 4 + 3], log_scales[gid * 3 + 0], log_scales[gid * 3 + 1],
                               log_scales[gid * 3 + 2], wc);
  float t00 = cam.rwc00 * wc[0] + cam.rwc01 * wc[3] + cam.rwc02 * wc[6];
  float t01 = cam.rwc00 * wc[1] + cam.rwc01 * wc[4] + cam.rwc02 * wc[7];
  float t02 = cam.rwc00 * wc[2] + cam.rwc01 * wc[5] + cam.rwc02 * wc[8];
  float t10 = cam.rwc10 * wc[0] + cam.rwc11 * wc[3] + cam.rwc12 * wc[6];
  float t11 = cam.rwc10 * wc[1] + cam.rwc11 * wc[4] + cam.rwc12 * wc[7];
  float t12 = cam.rwc10 * wc[2] + cam.rwc11 * wc[5] + cam.rwc12 * wc[8];
  float t20 = cam.rwc20 * wc[0] + cam.rwc21 * wc[3] + cam.rwc22 * wc[6];
  float t21 = cam.rwc20 * wc[1] + cam.rwc21 * wc[4] + cam.rwc22 * wc[7];
  float t22 = cam.rwc20 * wc[2] + cam.rwc21 * wc[5] + cam.rwc22 * wc[8];

  const float ccov00 = t00 * cam.rwc00 + t01 * cam.rwc01 + t02 * cam.rwc02;
  const float ccov01 = t00 * cam.rwc10 + t01 * cam.rwc11 + t02 * cam.rwc12;
  const float ccov02 = t00 * cam.rwc20 + t01 * cam.rwc21 + t02 * cam.rwc22;
  const float ccov10 = t10 * cam.rwc00 + t11 * cam.rwc01 + t12 * cam.rwc02;
  const float ccov11 = t10 * cam.rwc10 + t11 * cam.rwc11 + t12 * cam.rwc12;
  const float ccov12 = t10 * cam.rwc20 + t11 * cam.rwc21 + t12 * cam.rwc22;
  const float ccov20 = t20 * cam.rwc00 + t21 * cam.rwc01 + t22 * cam.rwc02;
  const float ccov21 = t20 * cam.rwc10 + t21 * cam.rwc11 + t22 * cam.rwc12;
  const float ccov22 = t20 * cam.rwc20 + t21 * cam.rwc21 + t22 * cam.rwc22;

  const float proj_x = cam.fx * cam_x / cam_z + cam.cx;
  const float proj_y = cam.fy * cam_y / cam_z + cam.cy;
  const float j00 = cam.fx / cam_z;
  const float j02 = -cam.fx * cam_x / (cam_z * cam_z);
  const float j11 = cam.fy / cam_z;
  const float j12 = -cam.fy * cam_y / (cam_z * cam_z);

  float s00 = j00 * (ccov00 * j00 + ccov02 * j02) + j02 * (ccov20 * j00 + ccov22 * j02);
  float s01 = j00 * (ccov01 * j11 + ccov02 * j12) + j02 * (ccov21 * j11 + ccov22 * j12);
  float s10 = j11 * (ccov10 * j00 + ccov12 * j02) + j12 * (ccov20 * j00 + ccov22 * j02);
  float s11 = j11 * (ccov11 * j11 + ccov12 * j12) + j12 * (ccov21 * j11 + ccov22 * j12);
  s00 += cam.min_covariance;
  s11 += cam.min_covariance;
  float det = s00 * s11 - s01 * s10;
  if (det <= cam.min_covariance) {
    det = cam.min_covariance;
  }
  const float inv_det = 1.0f / det;
  const float radius =
      compact_radius_m(s00, s01, s11, cam.min_covariance, cam.sigma_radius, cam.compact_box_beta,
                       cam.use_compact_box != 0);
  const int min_x = int(floor(proj_x - radius));
  const int max_x = int(ceil(proj_x + radius));
  const int min_y = int(floor(proj_y - radius));
  const int max_y = int(ceil(proj_y + radius));
  if (max_x < 0 || min_x >= int(cam.width) || max_y < 0 || min_y >= int(cam.height)) {
    out[gid] = invalid_projected(gid);
    return;
  }

  ProjectedGaussian pg;
  pg.mean_x = proj_x;
  pg.mean_y = proj_y;
  pg.depth = cam_z;
  pg.inv_xx = s11 * inv_det;
  pg.inv_xy = -s01 * inv_det;
  pg.inv_yx = -s10 * inv_det;
  pg.inv_yy = s00 * inv_det;
  pg.min_x = min_x;
  pg.max_x = max_x;
  pg.min_y = min_y;
  pg.max_y = max_y;
  pg.source_index = int(gid);
  out[gid] = pg;
}

kernel void project_gaussians_2d(
    device const float* proj_means [[buffer(0)]],
    device const float* proj_covs [[buffer(1)]],
    device const float* depths [[buffer(2)]],
    constant CameraParams& cam [[buffer(3)]],
    device ProjectedGaussian* out [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
  if (gid >= cam.n) return;
  float s00 = proj_covs[gid * 4 + 0] + cam.min_covariance;
  float s01 = proj_covs[gid * 4 + 1];
  float s10 = proj_covs[gid * 4 + 2];
  float s11 = proj_covs[gid * 4 + 3] + cam.min_covariance;
  float det = s00 * s11 - s01 * s10;
  if (det <= cam.min_covariance) {
    det = cam.min_covariance;
  }
  const float inv_det = 1.0f / det;
  const float radius =
      compact_radius_m(s00, s01, s11, cam.min_covariance, cam.sigma_radius, cam.compact_box_beta,
                       cam.use_compact_box != 0);
  const float mx = proj_means[gid * 2 + 0];
  const float my = proj_means[gid * 2 + 1];
  const int min_x = int(floor(mx - radius));
  const int max_x = int(ceil(mx + radius));
  const int min_y = int(floor(my - radius));
  const int max_y = int(ceil(my + radius));
  if (max_x < 0 || min_x >= int(cam.width) || max_y < 0 || min_y >= int(cam.height)) {
    out[gid] = invalid_projected(gid);
    return;
  }
  ProjectedGaussian pg;
  pg.mean_x = mx;
  pg.mean_y = my;
  pg.depth = cam.has_depths != 0 ? depths[gid] : float(gid);
  pg.inv_xx = s11 * inv_det;
  pg.inv_xy = -s01 * inv_det;
  pg.inv_yx = -s10 * inv_det;
  pg.inv_yy = s00 * inv_det;
  pg.min_x = min_x;
  pg.max_x = max_x;
  pg.min_y = min_y;
  pg.max_y = max_y;
  pg.source_index = int(gid);
  out[gid] = pg;
}

kernel void tile_histogram(
    device const ProjectedGaussian* projected [[buffer(0)]],
    constant RasterParams& params [[buffer(1)]],
    device atomic_int* counts [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
  if (gid >= params.num_projected) return;
  const ProjectedGaussian pg = projected[gid];
  if (pg.max_x < pg.min_x || pg.max_y < pg.min_y) return;
  const int tiles_x = int(params.tiles_x);
  const int tiles_y = int(params.tiles_y);
  const int ts = int(params.tile_size);
  const int tx0 = max(0, pg.min_x / ts);
  const int tx1 = min(tiles_x - 1, pg.max_x / ts);
  const int ty0 = max(0, pg.min_y / ts);
  const int ty1 = min(tiles_y - 1, pg.max_y / ts);
  if (tx1 < tx0 || ty1 < ty0) return;
  for (int ty = ty0; ty <= ty1; ++ty) {
    for (int tx = tx0; tx <= tx1; ++tx) {
      atomic_fetch_add_explicit(&counts[ty * tiles_x + tx], 1, memory_order_relaxed);
    }
  }
}

kernel void tile_scatter(
    device const ProjectedGaussian* projected [[buffer(0)]],
    constant RasterParams& params [[buffer(1)]],
    device atomic_int* write_heads [[buffer(2)]],
    device int* ids [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
  if (gid >= params.num_projected) return;
  const ProjectedGaussian pg = projected[gid];
  if (pg.max_x < pg.min_x || pg.max_y < pg.min_y) return;
  const int tiles_x = int(params.tiles_x);
  const int tiles_y = int(params.tiles_y);
  const int ts = int(params.tile_size);
  const int tx0 = max(0, pg.min_x / ts);
  const int tx1 = min(tiles_x - 1, pg.max_x / ts);
  const int ty0 = max(0, pg.min_y / ts);
  const int ty1 = min(tiles_y - 1, pg.max_y / ts);
  if (tx1 < tx0 || ty1 < ty0) return;
  for (int ty = ty0; ty <= ty1; ++ty) {
    for (int tx = tx0; tx <= tx1; ++tx) {
      const int pos = atomic_fetch_add_explicit(&write_heads[ty * tiles_x + tx], 1,
                                                memory_order_relaxed);
      ids[pos] = int(gid);
    }
  }
}

kernel void sort_tile_ids(
    device const ProjectedGaussian* projected [[buffer(0)]],
    device const int* offsets [[buffer(1)]],
    device int* ids [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]]
) {
  const int start = offsets[tgid];
  const int n = offsets[tgid + 1] - start;
  if (n <= 1) return;

    if (n > 1024) {
      return;
    }

  threadgroup int tg_id[1024];
  threadgroup float tg_d[1024];
  for (uint i = lid; i < 1024; i += ntg) {
    if (i < uint(n)) {
      const int id = ids[start + int(i)];
      tg_id[i] = id;
      tg_d[i] = projected[id].depth;
    } else {
      tg_id[i] = 0x7fffffff;
      tg_d[i] = 1e30f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint k = 2; k <= 1024; k <<= 1) {
    for (uint j = k >> 1; j > 0; j >>= 1) {
      for (uint i = lid; i < 1024; i += ntg) {
        const uint ixj = i ^ j;
        if (ixj > i) {
          const bool ascending = (i & k) == 0;
          const bool left_gt = (tg_d[i] > tg_d[ixj]) || (tg_d[i] == tg_d[ixj] && tg_id[i] > tg_id[ixj]);
          if (left_gt == ascending) {
            const float td = tg_d[i];
            tg_d[i] = tg_d[ixj];
            tg_d[ixj] = td;
            const int ti = tg_id[i];
            tg_id[i] = tg_id[ixj];
            tg_id[ixj] = ti;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  for (uint i = lid; i < uint(n); i += ntg) {
    ids[start + int(i)] = tg_id[i];
  }
}

kernel void tiled_alpha_forward(
    device const ProjectedGaussian* projected [[buffer(0)]],
    device const float* colors [[buffer(1)]],
    device const float* opacities [[buffer(2)]],
    device const int* tile_offsets [[buffer(3)]],
    device const int* tile_ids [[buffer(4)]],
    constant RasterParams& params [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
  const uint x = tgid.x * params.tile_size + lid.x;
  const uint y = tgid.y * params.tile_size + lid.y;
  const bool inside = x < params.width && y < params.height;
  const uint linear = lid.y * params.tile_size + lid.x;
  const uint tile_idx = tgid.y * params.tiles_x + tgid.x;
  const int start = tile_offsets[tile_idx];
  const int end = tile_offsets[tile_idx + 1];

  threadgroup ProjectedGaussian tg_pg[64];
  threadgroup float tg_opa[64];
  threadgroup float tg_c0[64];
  threadgroup float tg_c1[64];
  threadgroup float tg_c2[64];
  threadgroup float tg_tmax[8];

  float T = 1.0f;
  float accum0 = 0.0f;
  float accum1 = 0.0f;
  float accum2 = 0.0f;

  for (int base = start; base < end; base += 64) {
    const int nload = min(64, end - base);
    if (linear < uint(nload)) {
      const int idx = tile_ids[base + int(linear)];
      const ProjectedGaussian pg = projected[idx];
      tg_pg[linear] = pg;
      const int src = pg.source_index;
      tg_opa[linear] = opacities[src];
      const int cbase = src * int(params.num_channels);
      tg_c0[linear] = params.num_channels > 0 ? colors[cbase] : 0.0f;
      tg_c1[linear] = params.num_channels > 1 ? colors[cbase + 1] : 0.0f;
      tg_c2[linear] = params.num_channels > 2 ? colors[cbase + 2] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (inside && T > 1e-4f) {
      for (int j = 0; j < nload && T > 1e-4f; ++j) {
        const ProjectedGaussian pg = tg_pg[j];
        if ((int)x < pg.min_x || (int)x > pg.max_x || (int)y < pg.min_y || (int)y > pg.max_y) {
          continue;
        }
        const float dx = float(x) - pg.mean_x;
        const float dy = float(y) - pg.mean_y;
        const float quad =
            dx * (pg.inv_xx * dx + pg.inv_xy * dy) + dy * (pg.inv_yx * dx + pg.inv_yy * dy);
        const float gaussian = exp(-0.5f * quad);
        float alpha = tg_opa[j] * gaussian;
        alpha = clamp(alpha, 0.0f, 0.999f);
        const float w = T * alpha;
        if (params.num_channels > 0) accum0 += w * tg_c0[j];
        if (params.num_channels > 1) accum1 += w * tg_c1[j];
        if (params.num_channels > 2) accum2 += w * tg_c2[j];
        T *= (1.0f - alpha);
      }
    }

    float tmax = inside ? T : 0.0f;
    tmax = simd_max(tmax);
    if (simd_is_first()) {
      tg_tmax[linear / 32] = tmax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (linear == 0) {
      float m = 0.0f;
      for (uint i = 0; i < 8; ++i) m = max(m, tg_tmax[i]);
      tg_tmax[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tg_tmax[0] <= 1e-4f) break;
  }

  if (inside) {
    const uint out = (y * params.width + x) * params.num_channels;
    if (params.num_channels > 0) output[out] = accum0;
    if (params.num_channels > 1) output[out + 1] = accum1;
    if (params.num_channels > 2) output[out + 2] = accum2;
  }
}

kernel void footprint_hit_count(
    device const ProjectedGaussian* projected [[buffer(0)]],
    device const float* opacities [[buffer(1)]],
    device const uchar* error_mask [[buffer(2)]],
    constant RasterParams& params [[buffer(3)]],
    device atomic_int* counts [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
  const uint n = params.num_projected;
  if (gid >= n) return;
  const ProjectedGaussian pg = projected[gid];
  if (pg.max_x < pg.min_x || pg.max_y < pg.min_y) return;

  int hits = 0;
  const int x0 = max(pg.min_x, 0);
  const int x1 = min(pg.max_x, int(params.width) - 1);
  const int y0 = max(pg.min_y, 0);
  const int y1 = min(pg.max_y, int(params.height) - 1);
  for (int y = y0; y <= y1; ++y) {
    for (int x = x0; x <= x1; ++x) {
      if (error_mask[y * params.width + x] == 0) continue;
      const float dx = float(x) - pg.mean_x;
      const float dy = float(y) - pg.mean_y;
      const float quad =
          dx * (pg.inv_xx * dx + pg.inv_xy * dy) + dy * (pg.inv_yx * dx + pg.inv_yy * dy);
      const float gaussian = exp(-0.5f * quad);
      float alpha = opacities[pg.source_index] * gaussian;
      if (alpha < 1e-4f) continue;
      hits += 1;
    }
  }
  atomic_fetch_add_explicit(&counts[pg.source_index], hits, memory_order_relaxed);
}

#ifdef USE_ATOMIC_FLOAT
using atomic_grad_t = atomic_float;
inline void atomic_add_float(device atomic_grad_t* addr, float value) {
  atomic_fetch_add_explicit(addr, value, memory_order_relaxed);
}
#else
using atomic_grad_t = atomic_uint;
inline void atomic_add_float(device atomic_grad_t* addr, float value) {
  uint current = atomic_load_explicit(addr, memory_order_relaxed);
  uint new_value;
  do {
    new_value = as_type<uint>(as_type<float>(current) + value);
  } while (!atomic_compare_exchange_weak_explicit(addr, &current, new_value, memory_order_relaxed,
                                                  memory_order_relaxed));
}
#endif

kernel void tiled_alpha_backward(
    device const ProjectedGaussian* projected [[buffer(0)]],
    device const float* colors [[buffer(1)]],
    device const float* opacities [[buffer(2)]],
    device const int* tile_offsets [[buffer(3)]],
    device const int* tile_ids [[buffer(4)]],
    constant RasterParams& params [[buffer(5)]],
    device const float* grad_output [[buffer(6)]],
    device atomic_grad_t* grad_means [[buffer(7)]],
    device atomic_grad_t* grad_covs [[buffer(8)]],
    device atomic_grad_t* grad_colors [[buffer(9)]],
    device atomic_grad_t* grad_opacities [[buffer(10)]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
  const uint x = tgid.x * params.tile_size + lid.x;
  const uint y = tgid.y * params.tile_size + lid.y;
  const bool inside = x < params.width && y < params.height;
  const uint linear = lid.y * params.tile_size + lid.x;
  const uint tile_idx = tgid.y * params.tiles_x + tgid.x;
  const int start = tile_offsets[tile_idx];
  const int end = tile_offsets[tile_idx + 1];

  threadgroup atomic_int tg_last;

  float T = 1.0f;
  int last = -1;
  if (inside) {
    for (int i = start; i < end && T > 1e-4f; ++i) {
      const ProjectedGaussian pg = projected[tile_ids[i]];
      if ((int)x < pg.min_x || (int)x > pg.max_x || (int)y < pg.min_y || (int)y > pg.max_y) {
        continue;
      }
      const float dx = float(x) - pg.mean_x;
      const float dy = float(y) - pg.mean_y;
      const float quad =
          dx * (pg.inv_xx * dx + pg.inv_xy * dy) + dy * (pg.inv_yx * dx + pg.inv_yy * dy);
      const float gaussian = exp(-0.5f * quad);
      float alpha = opacities[pg.source_index] * gaussian;
      alpha = clamp(alpha, 0.0f, 0.999f);
      if (alpha < 1e-4f) continue;
      T *= (1.0f - alpha);
      last = i;
    }
  }

  if (linear == 0) {
    atomic_store_explicit(&tg_last, -1, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  atomic_fetch_max_explicit(&tg_last, last, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int i_end = atomic_load_explicit(&tg_last, memory_order_relaxed);
  if (i_end < start) return;

  float go0 = 0.0f;
  float go1 = 0.0f;
  float go2 = 0.0f;
  if (inside) {
    const uint go_off = (y * params.width + x) * params.num_channels;
    if (params.num_channels > 0) go0 = grad_output[go_off];
    if (params.num_channels > 1) go1 = grad_output[go_off + 1];
    if (params.num_channels > 2) go2 = grad_output[go_off + 2];
  }

  float T_out = T;
  float dL_dTnext = 0.0f;
  for (int i = i_end; i >= start; --i) {
    const ProjectedGaussian pg = projected[tile_ids[i]];
    const int src = pg.source_index;
    const int cbase = src * int(params.num_channels);
    bool hit = inside && i <= last;
    if (hit && ((int)x < pg.min_x || (int)x > pg.max_x || (int)y < pg.min_y || (int)y > pg.max_y)) {
      hit = false;
    }
    float dx = 0.0f;
    float dy = 0.0f;
    float gaussian = 0.0f;
    float alpha = 0.0f;
    if (hit) {
      dx = float(x) - pg.mean_x;
      dy = float(y) - pg.mean_y;
      const float quad =
          dx * (pg.inv_xx * dx + pg.inv_xy * dy) + dy * (pg.inv_yx * dx + pg.inv_yy * dy);
      gaussian = exp(-0.5f * quad);
      alpha = clamp(opacities[src] * gaussian, 0.0f, 0.999f);
      if (alpha < 1e-4f) hit = false;
    }

    float g_mx = 0.0f, g_my = 0.0f;
    float g_c00 = 0.0f, g_c01 = 0.0f, g_c10 = 0.0f, g_c11 = 0.0f;
    float g_c0 = 0.0f, g_c1 = 0.0f, g_c2 = 0.0f, g_o = 0.0f;
    if (hit) {
      const float one_m_a = 1.0f - alpha;
      const float T_in = T_out / max(one_m_a, 1e-8f);
      float dL_dC_dot_c = 0.0f;
      if (params.num_channels > 0) {
        dL_dC_dot_c += go0 * colors[cbase];
        g_c0 = go0 * T_in * alpha;
      }
      if (params.num_channels > 1) {
        dL_dC_dot_c += go1 * colors[cbase + 1];
        g_c1 = go1 * T_in * alpha;
      }
      if (params.num_channels > 2) {
        dL_dC_dot_c += go2 * colors[cbase + 2];
        g_c2 = go2 * T_in * alpha;
      }
      const float dL_dalpha = T_in * (dL_dC_dot_c - dL_dTnext);
      g_o = dL_dalpha * gaussian;
      const float dL_dgaussian = dL_dalpha * opacities[src];
      const float ad_x = pg.inv_xx * dx + pg.inv_xy * dy;
      const float ad_y = pg.inv_yx * dx + pg.inv_yy * dy;
      const float common = dL_dgaussian * gaussian;
      g_mx = common * ad_x;
      g_my = common * ad_y;
      const float half_common = 0.5f * common;
      g_c00 = half_common * (ad_x * ad_x);
      g_c01 = half_common * (ad_x * ad_y);
      g_c10 = half_common * (ad_y * ad_x);
      g_c11 = half_common * (ad_y * ad_y);
      dL_dTnext = dL_dTnext * one_m_a + dL_dC_dot_c * alpha;
      T_out = T_in;
    }
    g_mx = simd_sum(g_mx);
    g_my = simd_sum(g_my);
    g_c00 = simd_sum(g_c00);
    g_c01 = simd_sum(g_c01);
    g_c10 = simd_sum(g_c10);
    g_c11 = simd_sum(g_c11);
    g_c0 = simd_sum(g_c0);
    g_c1 = simd_sum(g_c1);
    g_c2 = simd_sum(g_c2);
    g_o = simd_sum(g_o);
    if (simd_is_first()) {
      atomic_add_float(&grad_means[src * 2 + 0], g_mx);
      atomic_add_float(&grad_means[src * 2 + 1], g_my);
      atomic_add_float(&grad_covs[src * 4 + 0], g_c00);
      atomic_add_float(&grad_covs[src * 4 + 1], g_c01);
      atomic_add_float(&grad_covs[src * 4 + 2], g_c10);
      atomic_add_float(&grad_covs[src * 4 + 3], g_c11);
      if (params.num_channels > 0) atomic_add_float(&grad_colors[cbase], g_c0);
      if (params.num_channels > 1) atomic_add_float(&grad_colors[cbase + 1], g_c1);
      if (params.num_channels > 2) atomic_add_float(&grad_colors[cbase + 2], g_c2);
      atomic_add_float(&grad_opacities[src], g_o);
    }
  }
}

kernel void project_3d_vjp(
    device const float* means [[buffer(0)]],
    device const float* covs [[buffer(1)]],
    constant CameraParams& cam [[buffer(2)]],
    device const float* dmean2d [[buffer(3)]],
    device const float* dcov2d [[buffer(4)]],
    device float* dmean3d [[buffer(5)]],
    device float* dcov3d [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
  if (gid >= cam.n) return;
  dmean3d[gid * 3 + 0] = 0.0f;
  dmean3d[gid * 3 + 1] = 0.0f;
  dmean3d[gid * 3 + 2] = 0.0f;
  for (int i = 0; i < 9; ++i) dcov3d[gid * 9 + i] = 0.0f;

  const float mx = means[gid * 3 + 0];
  const float my = means[gid * 3 + 1];
  const float mz = means[gid * 3 + 2];
  const float cam_x = cam.rwc00 * mx + cam.rwc01 * my + cam.rwc02 * mz + cam.twc0;
  const float cam_y = cam.rwc10 * mx + cam.rwc11 * my + cam.rwc12 * mz + cam.twc1;
  const float cam_z = cam.rwc20 * mx + cam.rwc21 * my + cam.rwc22 * mz + cam.twc2;
  if (cam_z <= cam.near_plane) return;

  device const float* wc = covs + gid * 9;
  float t00 = cam.rwc00 * wc[0] + cam.rwc01 * wc[3] + cam.rwc02 * wc[6];
  float t01 = cam.rwc00 * wc[1] + cam.rwc01 * wc[4] + cam.rwc02 * wc[7];
  float t02 = cam.rwc00 * wc[2] + cam.rwc01 * wc[5] + cam.rwc02 * wc[8];
  float t10 = cam.rwc10 * wc[0] + cam.rwc11 * wc[3] + cam.rwc12 * wc[6];
  float t11 = cam.rwc10 * wc[1] + cam.rwc11 * wc[4] + cam.rwc12 * wc[7];
  float t12 = cam.rwc10 * wc[2] + cam.rwc11 * wc[5] + cam.rwc12 * wc[8];
  float t20 = cam.rwc20 * wc[0] + cam.rwc21 * wc[3] + cam.rwc22 * wc[6];
  float t21 = cam.rwc20 * wc[1] + cam.rwc21 * wc[4] + cam.rwc22 * wc[7];
  float t22 = cam.rwc20 * wc[2] + cam.rwc21 * wc[5] + cam.rwc22 * wc[8];
  const float cc00 = t00 * cam.rwc00 + t01 * cam.rwc01 + t02 * cam.rwc02;
  const float cc01 = t00 * cam.rwc10 + t01 * cam.rwc11 + t02 * cam.rwc12;
  const float cc02 = t00 * cam.rwc20 + t01 * cam.rwc21 + t02 * cam.rwc22;
  const float cc10 = t10 * cam.rwc00 + t11 * cam.rwc01 + t12 * cam.rwc02;
  const float cc11 = t10 * cam.rwc10 + t11 * cam.rwc11 + t12 * cam.rwc12;
  const float cc12 = t10 * cam.rwc20 + t11 * cam.rwc21 + t12 * cam.rwc22;
  const float cc20 = t20 * cam.rwc00 + t21 * cam.rwc01 + t22 * cam.rwc02;
  const float cc21 = t20 * cam.rwc10 + t21 * cam.rwc11 + t22 * cam.rwc12;
  const float cc22 = t20 * cam.rwc20 + t21 * cam.rwc21 + t22 * cam.rwc22;

  const float z = cam_z;
  const float z2 = z * z;
  const float z3 = z2 * z;
  const float j00 = cam.fx / z;
  const float j02 = -cam.fx * cam_x / z2;
  const float j11 = cam.fy / z;
  const float j12 = -cam.fy * cam_y / z2;

  const float gpx = dmean2d[gid * 2 + 0];
  const float gpy = dmean2d[gid * 2 + 1];
  const float gs00 = dcov2d[gid * 4 + 0];
  const float gs01 = dcov2d[gid * 4 + 1];
  const float gs10 = dcov2d[gid * 4 + 2];
  const float gs11 = dcov2d[gid * 4 + 3];

  // J^T is 3x2: [[j00, 0], [0, j11], [j02, j12]]; dL/dΣ_c = J^T G J
  const float a00 = j00 * gs00;
  const float a01 = j00 * gs01;
  const float a10 = j11 * gs10;
  const float a11 = j11 * gs11;
  const float a20 = j02 * gs00 + j12 * gs10;
  const float a21 = j02 * gs01 + j12 * gs11;
  const float dc00 = a00 * j00;
  const float dc01 = a01 * j11;
  const float dc02 = a00 * j02 + a01 * j12;
  const float dc10 = a10 * j00;
  const float dc11 = a11 * j11;
  const float dc12 = a10 * j02 + a11 * j12;
  const float dc20 = a20 * j00;
  const float dc21 = a21 * j11;
  const float dc22 = a20 * j02 + a21 * j12;

  // JS = J Σ_c (2x3). dL/dJ = G (J Σ^T) + G^T (J Σ); Σ_c is symmetric.
  const float js00 = j00 * cc00 + j02 * cc20;
  const float js01 = j00 * cc01 + j02 * cc21;
  const float js02 = j00 * cc02 + j02 * cc22;
  const float js10 = j11 * cc10 + j12 * cc20;
  const float js11 = j11 * cc11 + j12 * cc21;
  const float js12 = j11 * cc12 + j12 * cc22;
  const float dJ00 = gs00 * js00 + gs01 * js10 + gs00 * js00 + gs10 * js10;
  const float dJ02 = gs00 * js02 + gs01 * js12 + gs00 * js02 + gs10 * js12;
  const float dJ11 = gs10 * js01 + gs11 * js11 + gs01 * js01 + gs11 * js11;
  const float dJ12 = gs10 * js02 + gs11 * js12 + gs01 * js02 + gs11 * js12;

  float dmx = gpx * j00;
  float dmy = gpy * j11;
  float dmz = gpx * j02 + gpy * j12;
  dmx += dJ02 * (-cam.fx / z2);
  dmy += dJ12 * (-cam.fy / z2);
  dmz += dJ00 * (-cam.fx / z2) + dJ02 * (2.0f * cam.fx * cam_x / z3) +
         dJ11 * (-cam.fy / z2) + dJ12 * (2.0f * cam.fy * cam_y / z3);

  dmean3d[gid * 3 + 0] = cam.rwc00 * dmx + cam.rwc10 * dmy + cam.rwc20 * dmz;
  dmean3d[gid * 3 + 1] = cam.rwc01 * dmx + cam.rwc11 * dmy + cam.rwc21 * dmz;
  dmean3d[gid * 3 + 2] = cam.rwc02 * dmx + cam.rwc12 * dmy + cam.rwc22 * dmz;

  // dL/dΣ_w = R^T dL/dΣ_c R
  float u00 = cam.rwc00 * dc00 + cam.rwc10 * dc10 + cam.rwc20 * dc20;
  float u01 = cam.rwc00 * dc01 + cam.rwc10 * dc11 + cam.rwc20 * dc21;
  float u02 = cam.rwc00 * dc02 + cam.rwc10 * dc12 + cam.rwc20 * dc22;
  float u10 = cam.rwc01 * dc00 + cam.rwc11 * dc10 + cam.rwc21 * dc20;
  float u11 = cam.rwc01 * dc01 + cam.rwc11 * dc11 + cam.rwc21 * dc21;
  float u12 = cam.rwc01 * dc02 + cam.rwc11 * dc12 + cam.rwc21 * dc22;
  float u20 = cam.rwc02 * dc00 + cam.rwc12 * dc10 + cam.rwc22 * dc20;
  float u21 = cam.rwc02 * dc01 + cam.rwc12 * dc11 + cam.rwc22 * dc21;
  float u22 = cam.rwc02 * dc02 + cam.rwc12 * dc12 + cam.rwc22 * dc22;
  dcov3d[gid * 9 + 0] = u00 * cam.rwc00 + u01 * cam.rwc10 + u02 * cam.rwc20;
  dcov3d[gid * 9 + 1] = u00 * cam.rwc01 + u01 * cam.rwc11 + u02 * cam.rwc21;
  dcov3d[gid * 9 + 2] = u00 * cam.rwc02 + u01 * cam.rwc12 + u02 * cam.rwc22;
  dcov3d[gid * 9 + 3] = u10 * cam.rwc00 + u11 * cam.rwc10 + u12 * cam.rwc20;
  dcov3d[gid * 9 + 4] = u10 * cam.rwc01 + u11 * cam.rwc11 + u12 * cam.rwc21;
  dcov3d[gid * 9 + 5] = u10 * cam.rwc02 + u11 * cam.rwc12 + u12 * cam.rwc22;
  dcov3d[gid * 9 + 6] = u20 * cam.rwc00 + u21 * cam.rwc10 + u22 * cam.rwc20;
  dcov3d[gid * 9 + 7] = u20 * cam.rwc01 + u21 * cam.rwc11 + u22 * cam.rwc21;
  dcov3d[gid * 9 + 8] = u20 * cam.rwc02 + u21 * cam.rwc12 + u22 * cam.rwc22;
}

kernel void project_3d_vjp_qs(
    device const float* means [[buffer(0)]],
    device const float* log_scales [[buffer(1)]],
    device const float* quats [[buffer(2)]],
    constant CameraParams& cam [[buffer(3)]],
    device const float* dmean2d [[buffer(4)]],
    device const float* dcov2d [[buffer(5)]],
    device float* dmean3d [[buffer(6)]],
    device float* dlog_scales [[buffer(7)]],
    device float* dquats [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
  if (gid >= cam.n) return;
  dmean3d[gid * 3 + 0] = 0.0f;
  dmean3d[gid * 3 + 1] = 0.0f;
  dmean3d[gid * 3 + 2] = 0.0f;
  dlog_scales[gid * 3 + 0] = 0.0f;
  dlog_scales[gid * 3 + 1] = 0.0f;
  dlog_scales[gid * 3 + 2] = 0.0f;
  dquats[gid * 4 + 0] = 0.0f;
  dquats[gid * 4 + 1] = 0.0f;
  dquats[gid * 4 + 2] = 0.0f;
  dquats[gid * 4 + 3] = 0.0f;

  const float mx = means[gid * 3 + 0];
  const float my = means[gid * 3 + 1];
  const float mz = means[gid * 3 + 2];
  const float cam_x = cam.rwc00 * mx + cam.rwc01 * my + cam.rwc02 * mz + cam.twc0;
  const float cam_y = cam.rwc10 * mx + cam.rwc11 * my + cam.rwc12 * mz + cam.twc1;
  const float cam_z = cam.rwc20 * mx + cam.rwc21 * my + cam.rwc22 * mz + cam.twc2;
  if (cam_z <= cam.near_plane) return;

  float wc[9];
  world_cov_from_quat_logscale(quats[gid * 4 + 0], quats[gid * 4 + 1], quats[gid * 4 + 2],
                               quats[gid * 4 + 3], log_scales[gid * 3 + 0], log_scales[gid * 3 + 1],
                               log_scales[gid * 3 + 2], wc);
  float t00 = cam.rwc00 * wc[0] + cam.rwc01 * wc[3] + cam.rwc02 * wc[6];
  float t01 = cam.rwc00 * wc[1] + cam.rwc01 * wc[4] + cam.rwc02 * wc[7];
  float t02 = cam.rwc00 * wc[2] + cam.rwc01 * wc[5] + cam.rwc02 * wc[8];
  float t10 = cam.rwc10 * wc[0] + cam.rwc11 * wc[3] + cam.rwc12 * wc[6];
  float t11 = cam.rwc10 * wc[1] + cam.rwc11 * wc[4] + cam.rwc12 * wc[7];
  float t12 = cam.rwc10 * wc[2] + cam.rwc11 * wc[5] + cam.rwc12 * wc[8];
  float t20 = cam.rwc20 * wc[0] + cam.rwc21 * wc[3] + cam.rwc22 * wc[6];
  float t21 = cam.rwc20 * wc[1] + cam.rwc21 * wc[4] + cam.rwc22 * wc[7];
  float t22 = cam.rwc20 * wc[2] + cam.rwc21 * wc[5] + cam.rwc22 * wc[8];
  const float cc00 = t00 * cam.rwc00 + t01 * cam.rwc01 + t02 * cam.rwc02;
  const float cc01 = t00 * cam.rwc10 + t01 * cam.rwc11 + t02 * cam.rwc12;
  const float cc02 = t00 * cam.rwc20 + t01 * cam.rwc21 + t02 * cam.rwc22;
  const float cc10 = t10 * cam.rwc00 + t11 * cam.rwc01 + t12 * cam.rwc02;
  const float cc11 = t10 * cam.rwc10 + t11 * cam.rwc11 + t12 * cam.rwc12;
  const float cc12 = t10 * cam.rwc20 + t11 * cam.rwc21 + t12 * cam.rwc22;
  const float cc20 = t20 * cam.rwc00 + t21 * cam.rwc01 + t22 * cam.rwc02;
  const float cc21 = t20 * cam.rwc10 + t21 * cam.rwc11 + t22 * cam.rwc12;
  const float cc22 = t20 * cam.rwc20 + t21 * cam.rwc21 + t22 * cam.rwc22;

  const float z = cam_z;
  const float z2 = z * z;
  const float z3 = z2 * z;
  const float j00 = cam.fx / z;
  const float j02 = -cam.fx * cam_x / z2;
  const float j11 = cam.fy / z;
  const float j12 = -cam.fy * cam_y / z2;

  const float gpx = dmean2d[gid * 2 + 0];
  const float gpy = dmean2d[gid * 2 + 1];
  const float gs00 = dcov2d[gid * 4 + 0];
  const float gs01 = dcov2d[gid * 4 + 1];
  const float gs10 = dcov2d[gid * 4 + 2];
  const float gs11 = dcov2d[gid * 4 + 3];

  const float a00 = j00 * gs00;
  const float a01 = j00 * gs01;
  const float a10 = j11 * gs10;
  const float a11 = j11 * gs11;
  const float a20 = j02 * gs00 + j12 * gs10;
  const float a21 = j02 * gs01 + j12 * gs11;
  const float dc00 = a00 * j00;
  const float dc01 = a01 * j11;
  const float dc02 = a00 * j02 + a01 * j12;
  const float dc10 = a10 * j00;
  const float dc11 = a11 * j11;
  const float dc12 = a10 * j02 + a11 * j12;
  const float dc20 = a20 * j00;
  const float dc21 = a21 * j11;
  const float dc22 = a20 * j02 + a21 * j12;

  const float js00 = j00 * cc00 + j02 * cc20;
  const float js01 = j00 * cc01 + j02 * cc21;
  const float js02 = j00 * cc02 + j02 * cc22;
  const float js10 = j11 * cc10 + j12 * cc20;
  const float js11 = j11 * cc11 + j12 * cc21;
  const float js12 = j11 * cc12 + j12 * cc22;
  const float dJ00 = gs00 * js00 + gs01 * js10 + gs00 * js00 + gs10 * js10;
  const float dJ02 = gs00 * js02 + gs01 * js12 + gs00 * js02 + gs10 * js12;
  const float dJ11 = gs10 * js01 + gs11 * js11 + gs01 * js01 + gs11 * js11;
  const float dJ12 = gs10 * js02 + gs11 * js12 + gs01 * js02 + gs11 * js12;

  float dmx = gpx * j00;
  float dmy = gpy * j11;
  float dmz = gpx * j02 + gpy * j12;
  dmx += dJ02 * (-cam.fx / z2);
  dmy += dJ12 * (-cam.fy / z2);
  dmz += dJ00 * (-cam.fx / z2) + dJ02 * (2.0f * cam.fx * cam_x / z3) +
         dJ11 * (-cam.fy / z2) + dJ12 * (2.0f * cam.fy * cam_y / z3);

  dmean3d[gid * 3 + 0] = cam.rwc00 * dmx + cam.rwc10 * dmy + cam.rwc20 * dmz;
  dmean3d[gid * 3 + 1] = cam.rwc01 * dmx + cam.rwc11 * dmy + cam.rwc21 * dmz;
  dmean3d[gid * 3 + 2] = cam.rwc02 * dmx + cam.rwc12 * dmy + cam.rwc22 * dmz;

  float u00 = cam.rwc00 * dc00 + cam.rwc10 * dc10 + cam.rwc20 * dc20;
  float u01 = cam.rwc00 * dc01 + cam.rwc10 * dc11 + cam.rwc20 * dc21;
  float u02 = cam.rwc00 * dc02 + cam.rwc10 * dc12 + cam.rwc20 * dc22;
  float u10 = cam.rwc01 * dc00 + cam.rwc11 * dc10 + cam.rwc21 * dc20;
  float u11 = cam.rwc01 * dc01 + cam.rwc11 * dc11 + cam.rwc21 * dc21;
  float u12 = cam.rwc01 * dc02 + cam.rwc11 * dc12 + cam.rwc21 * dc22;
  float u20 = cam.rwc02 * dc00 + cam.rwc12 * dc10 + cam.rwc22 * dc20;
  float u21 = cam.rwc02 * dc01 + cam.rwc12 * dc11 + cam.rwc22 * dc21;
  float u22 = cam.rwc02 * dc02 + cam.rwc12 * dc12 + cam.rwc22 * dc22;
  const float gw00 = u00 * cam.rwc00 + u01 * cam.rwc10 + u02 * cam.rwc20;
  const float gw01 = u00 * cam.rwc01 + u01 * cam.rwc11 + u02 * cam.rwc21;
  const float gw02 = u00 * cam.rwc02 + u01 * cam.rwc12 + u02 * cam.rwc22;
  const float gw10 = u10 * cam.rwc00 + u11 * cam.rwc10 + u12 * cam.rwc20;
  const float gw11 = u10 * cam.rwc01 + u11 * cam.rwc11 + u12 * cam.rwc21;
  const float gw12 = u10 * cam.rwc02 + u11 * cam.rwc12 + u12 * cam.rwc22;
  const float gw20 = u20 * cam.rwc00 + u21 * cam.rwc10 + u22 * cam.rwc20;
  const float gw21 = u20 * cam.rwc01 + u21 * cam.rwc11 + u22 * cam.rwc21;
  const float gw22 = u20 * cam.rwc02 + u21 * cam.rwc12 + u22 * cam.rwc22;

  float dls[3];
  float dq[4];
  dsigma_to_quat_logscale(quats[gid * 4 + 0], quats[gid * 4 + 1], quats[gid * 4 + 2],
                          quats[gid * 4 + 3], log_scales[gid * 3 + 0], log_scales[gid * 3 + 1],
                          log_scales[gid * 3 + 2], gw00, gw01, gw02, gw10, gw11, gw12, gw20, gw21,
                          gw22, dls, dq);
  dlog_scales[gid * 3 + 0] = dls[0];
  dlog_scales[gid * 3 + 1] = dls[1];
  dlog_scales[gid * 3 + 2] = dls[2];
  dquats[gid * 4 + 0] = dq[0];
  dquats[gid * 4 + 1] = dq[1];
  dquats[gid * 4 + 2] = dq[2];
  dquats[gid * 4 + 3] = dq[3];
}
)METAL";

struct GrowBuf {
  id<MTLBuffer> buf = nil;
  NSUInteger cap = 0;
};

class MetalContext {
 public:
  static MetalContext& instance() {
    static MetalContext ctx;
    return ctx;
  }

  bool ok() const { return device_ != nil && forward_pipeline_ != nil; }
  bool prep_ok() const { return project_3d_pipeline_ != nil; }

  std::mutex& mutex() { return mutex_; }
  id<MTLDevice> device() const { return device_; }
  id<MTLCommandQueue> queue() const { return queue_; }
  id<MTLComputePipelineState> forward_pipeline() const { return forward_pipeline_; }
  id<MTLComputePipelineState> footprint_pipeline() const { return footprint_pipeline_; }
  id<MTLComputePipelineState> backward_pipeline() const { return backward_pipeline_; }
  id<MTLComputePipelineState> project_3d_pipeline() const { return project_3d_pipeline_; }
  id<MTLComputePipelineState> project_3d_qs_pipeline() const { return project_3d_qs_pipeline_; }
  id<MTLComputePipelineState> project_2d_pipeline() const { return project_2d_pipeline_; }
  id<MTLComputePipelineState> vjp_pipeline() const { return vjp_pipeline_; }
  id<MTLComputePipelineState> vjp_qs_pipeline() const { return vjp_qs_pipeline_; }
  bool qs_ok() const { return project_3d_qs_pipeline_ != nil && vjp_qs_pipeline_ != nil; }

  bool session_valid = false;
  bool session_qs = false;
  int session_n = 0;
  int session_c = 0;
  int session_h = 0;
  int session_w = 0;

  id<MTLBuffer> acquire(GrowBuf& slot, NSUInteger bytes) {
    if (bytes == 0) {
      bytes = 4;
    }
    if (slot.buf == nil || slot.cap < bytes) {
      NSUInteger next = slot.cap == 0 ? bytes : std::max(bytes, slot.cap + slot.cap / 2);
      slot.cap = next;
      slot.buf = [device_ newBufferWithLength:next options:MTLResourceStorageModeShared];
    }
    return slot.buf;
  }

  GrowBuf means, covs, scales, quats, colors, opa, proj, offsets, ids, output, go;
  GrowBuf gm, gcov, gcol, gopa, gm3, gcov3, gls, gq, camera, params;
  GrowBuf mask, hitcounts, proj_means, proj_covs, depths;

 private:
  MetalContext() {
    device_ = MTLCreateSystemDefaultDevice();
    if (device_ == nil) {
      std::fprintf(stderr, "[tinysplat-metal] MTLCreateSystemDefaultDevice failed\n");
      return;
    }
    queue_ = [device_ newCommandQueue];
    NSString* source = [NSString stringWithUTF8String:kMetalShaders];
    auto compile = [&](MTLCompileOptions* opts, NSError** err) -> id<MTLLibrary> {
      return [device_ newLibraryWithSource:source options:opts error:err];
    };
    NSError* error = nil;
    const char* compile_mode = "atomic_float";
    MTLCompileOptions* metal3_atomic = [MTLCompileOptions new];
    metal3_atomic.languageVersion = MTLLanguageVersion3_0;
    metal3_atomic.preprocessorMacros = @{@"USE_ATOMIC_FLOAT" : @"1"};
    id<MTLLibrary> library = compile(metal3_atomic, &error);
    if (library == nil) {
      const char* msg = error ? [[error localizedDescription] UTF8String] : "unknown";
      std::fprintf(stderr, "[tinysplat-metal] atomic_float compile failed, using CAS: %s\n", msg);
      error = nil;
      compile_mode = "cas";
      MTLCompileOptions* metal3 = [MTLCompileOptions new];
      metal3.languageVersion = MTLLanguageVersion3_0;
      library = compile(metal3, &error);
    }
    if (library == nil) {
      error = nil;
      compile_mode = "default";
      library = compile(nil, &error);
    }
    if (library == nil) {
      const char* msg = error ? [[error localizedDescription] UTF8String] : "unknown";
      std::fprintf(stderr, "[tinysplat-metal] newLibraryWithSource failed: %s\n", msg);
      return;
    }
    std::fprintf(stderr, "[tinysplat-metal] shader compile mode=%s\n", compile_mode);

    auto make_pipe = [&](NSString* name) -> id<MTLComputePipelineState> {
      id<MTLFunction> fn = [library newFunctionWithName:name];
      if (fn == nil) {
        std::fprintf(stderr, "[tinysplat-metal] missing shader %s\n", [name UTF8String]);
        return nil;
      }
      NSError* err = nil;
      id<MTLComputePipelineState> p = [device_ newComputePipelineStateWithFunction:fn error:&err];
      if (p == nil) {
        const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown";
        std::fprintf(stderr, "[tinysplat-metal] pipeline %s failed: %s\n", [name UTF8String], msg);
      }
      return p;
    };

    forward_pipeline_ = make_pipe(@"tiled_alpha_forward");
    footprint_pipeline_ = make_pipe(@"footprint_hit_count");
    backward_pipeline_ = make_pipe(@"tiled_alpha_backward");
    project_3d_pipeline_ = make_pipe(@"project_gaussians_3d");
    project_3d_qs_pipeline_ = make_pipe(@"project_gaussians_3d_qs");
    project_2d_pipeline_ = make_pipe(@"project_gaussians_2d");
    vjp_pipeline_ = make_pipe(@"project_3d_vjp");
    vjp_qs_pipeline_ = make_pipe(@"project_3d_vjp_qs");
  }

  std::mutex mutex_;
  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  id<MTLComputePipelineState> forward_pipeline_ = nil;
  id<MTLComputePipelineState> footprint_pipeline_ = nil;
  id<MTLComputePipelineState> backward_pipeline_ = nil;
  id<MTLComputePipelineState> project_3d_pipeline_ = nil;
  id<MTLComputePipelineState> project_3d_qs_pipeline_ = nil;
  id<MTLComputePipelineState> project_2d_pipeline_ = nil;
  id<MTLComputePipelineState> vjp_pipeline_ = nil;
  id<MTLComputePipelineState> vjp_qs_pipeline_ = nil;
};

void blit_in(id<MTLBuffer> buf, const void* src, size_t n) {
  if (src != nullptr && n > 0) {
    std::memcpy([buf contents], src, n);
  }
}

void dispatch_1d(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipe, NSUInteger n) {
  if (n == 0 || pipe == nil) {
    return;
  }
  [enc setComputePipelineState:pipe];
  NSUInteger tw = pipe.threadExecutionWidth;
  if (tw == 0) {
    tw = 32;
  }
  NSUInteger tgs = tw;
  const NSUInteger cap = std::min<NSUInteger>(pipe.maxTotalThreadsPerThreadgroup, 256);
  while (tgs * 2 <= cap) {
    tgs *= 2;
  }
  tgs = std::min(tgs, n);
  [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];
}

CameraParams make_camera(const float* intrinsics, const float* c2w, int n, int h, int w,
                         const Splat3DMetalOptions& opts, bool has_depths) {
  CameraParams cam{};
  if (intrinsics != nullptr) {
    cam.fx = intrinsics[0];
    cam.fy = intrinsics[4];
    cam.cx = intrinsics[2];
    cam.cy = intrinsics[5];
  }
  if (c2w != nullptr) {
    const float r00 = c2w[0], r01 = c2w[1], r02 = c2w[2];
    const float r10 = c2w[4], r11 = c2w[5], r12 = c2w[6];
    const float r20 = c2w[8], r21 = c2w[9], r22 = c2w[10];
    const float tx = c2w[3], ty = c2w[7], tz = c2w[11];
    cam.rwc00 = r00;
    cam.rwc01 = r10;
    cam.rwc02 = r20;
    cam.rwc10 = r01;
    cam.rwc11 = r11;
    cam.rwc12 = r21;
    cam.rwc20 = r02;
    cam.rwc21 = r12;
    cam.rwc22 = r22;
    cam.twc0 = -(cam.rwc00 * tx + cam.rwc01 * ty + cam.rwc02 * tz);
    cam.twc1 = -(cam.rwc10 * tx + cam.rwc11 * ty + cam.rwc12 * tz);
    cam.twc2 = -(cam.rwc20 * tx + cam.rwc21 * ty + cam.rwc22 * tz);
  }
  cam.near_plane = opts.near_plane;
  cam.min_covariance = opts.min_covariance;
  cam.sigma_radius = opts.sigma_radius;
  cam.compact_box_beta = opts.compact_box_beta;
  cam.use_compact_box = opts.use_compact_box ? 1u : 0u;
  cam.height = static_cast<uint32_t>(h);
  cam.width = static_cast<uint32_t>(w);
  cam.n = static_cast<uint32_t>(n);
  cam.has_depths = has_depths ? 1u : 0u;
  return cam;
}

RasterParams make_raster_params(int h, int w, int c, int n_proj) {
  RasterParams params{};
  params.height = static_cast<uint32_t>(h);
  params.width = static_cast<uint32_t>(w);
  params.num_channels = static_cast<uint32_t>(c);
  params.tiles_x = static_cast<uint32_t>((w + kTileSize - 1) / kTileSize);
  params.tile_size = static_cast<uint32_t>(kTileSize);
  params.num_projected = static_cast<uint32_t>(n_proj);
  params.tiles_y = static_cast<uint32_t>((h + kTileSize - 1) / kTileSize);
  return params;
}

float compact_radius(float s00, float s01, float s11, float min_covariance, float sigma_radius,
                     float beta, bool use_compact_box) {
  const float trace = s00 + s11;
  const float disc = std::sqrt(std::max(0.0f, (s00 - s11) * (s00 - s11) + 4.0f * s01 * s01));
  const float lambda_max = std::max((trace + disc) * 0.5f, min_covariance);
  const float sigma = std::sqrt(lambda_max);
  if (use_compact_box) {
    return std::min(sigma_radius, beta) * sigma;
  }
  return sigma_radius * sigma;
}

std::vector<ProjectedGaussian> project_gaussians(const float* means, const float* covs, int n,
                                                 const float* intrinsics, const float* c2w, int h,
                                                 int w, const Splat3DMetalOptions& opts) {
  const CameraParams cam = make_camera(intrinsics, c2w, n, h, w, opts, false);
  std::vector<ProjectedGaussian> projected;
  projected.reserve(static_cast<size_t>(n));

  for (int g = 0; g < n; ++g) {
    const float mx = means[g * 3 + 0];
    const float my = means[g * 3 + 1];
    const float mz = means[g * 3 + 2];
    const float cam_x = cam.rwc00 * mx + cam.rwc01 * my + cam.rwc02 * mz + cam.twc0;
    const float cam_y = cam.rwc10 * mx + cam.rwc11 * my + cam.rwc12 * mz + cam.twc1;
    const float cam_z = cam.rwc20 * mx + cam.rwc21 * my + cam.rwc22 * mz + cam.twc2;
    if (cam_z <= opts.near_plane) {
      continue;
    }

    const float* wc = covs + g * 9;
    float t00 = cam.rwc00 * wc[0] + cam.rwc01 * wc[3] + cam.rwc02 * wc[6];
    float t01 = cam.rwc00 * wc[1] + cam.rwc01 * wc[4] + cam.rwc02 * wc[7];
    float t02 = cam.rwc00 * wc[2] + cam.rwc01 * wc[5] + cam.rwc02 * wc[8];
    float t10 = cam.rwc10 * wc[0] + cam.rwc11 * wc[3] + cam.rwc12 * wc[6];
    float t11 = cam.rwc10 * wc[1] + cam.rwc11 * wc[4] + cam.rwc12 * wc[7];
    float t12 = cam.rwc10 * wc[2] + cam.rwc11 * wc[5] + cam.rwc12 * wc[8];
    float t20 = cam.rwc20 * wc[0] + cam.rwc21 * wc[3] + cam.rwc22 * wc[6];
    float t21 = cam.rwc20 * wc[1] + cam.rwc21 * wc[4] + cam.rwc22 * wc[7];
    float t22 = cam.rwc20 * wc[2] + cam.rwc21 * wc[5] + cam.rwc22 * wc[8];

    const float ccov00 = t00 * cam.rwc00 + t01 * cam.rwc01 + t02 * cam.rwc02;
    const float ccov01 = t00 * cam.rwc10 + t01 * cam.rwc11 + t02 * cam.rwc12;
    const float ccov02 = t00 * cam.rwc20 + t01 * cam.rwc21 + t02 * cam.rwc22;
    const float ccov10 = t10 * cam.rwc00 + t11 * cam.rwc01 + t12 * cam.rwc02;
    const float ccov11 = t10 * cam.rwc10 + t11 * cam.rwc11 + t12 * cam.rwc12;
    const float ccov12 = t10 * cam.rwc20 + t11 * cam.rwc21 + t12 * cam.rwc22;
    const float ccov20 = t20 * cam.rwc00 + t21 * cam.rwc01 + t22 * cam.rwc02;
    const float ccov21 = t20 * cam.rwc10 + t21 * cam.rwc11 + t22 * cam.rwc12;
    const float ccov22 = t20 * cam.rwc20 + t21 * cam.rwc21 + t22 * cam.rwc22;

    const float proj_x = cam.fx * cam_x / cam_z + cam.cx;
    const float proj_y = cam.fy * cam_y / cam_z + cam.cy;
    const float j00 = cam.fx / cam_z;
    const float j02 = -cam.fx * cam_x / (cam_z * cam_z);
    const float j11 = cam.fy / cam_z;
    const float j12 = -cam.fy * cam_y / (cam_z * cam_z);

    float s00 = j00 * (ccov00 * j00 + ccov02 * j02) + j02 * (ccov20 * j00 + ccov22 * j02);
    float s01 = j00 * (ccov01 * j11 + ccov02 * j12) + j02 * (ccov21 * j11 + ccov22 * j12);
    float s10 = j11 * (ccov10 * j00 + ccov12 * j02) + j12 * (ccov20 * j00 + ccov22 * j02);
    float s11 = j11 * (ccov11 * j11 + ccov12 * j12) + j12 * (ccov21 * j11 + ccov22 * j12);
    s00 += opts.min_covariance;
    s11 += opts.min_covariance;
    float det = s00 * s11 - s01 * s10;
    if (det <= opts.min_covariance) {
      det = opts.min_covariance;
    }
    const float inv_det = 1.0f / det;
    const float radius =
        compact_radius(s00, s01, s11, opts.min_covariance, opts.sigma_radius,
                       opts.compact_box_beta, opts.use_compact_box);
    const int min_x = static_cast<int>(std::floor(proj_x - radius));
    const int max_x = static_cast<int>(std::ceil(proj_x + radius));
    const int min_y = static_cast<int>(std::floor(proj_y - radius));
    const int max_y = static_cast<int>(std::ceil(proj_y + radius));
    if (max_x < 0 || min_x >= w || max_y < 0 || min_y >= h) {
      continue;
    }

    projected.push_back(ProjectedGaussian{proj_x, proj_y, cam_z, s11 * inv_det, -s01 * inv_det,
                                          -s10 * inv_det, s00 * inv_det, min_x, max_x, min_y, max_y,
                                          g});
  }

  std::sort(projected.begin(), projected.end(),
            [](const ProjectedGaussian& a, const ProjectedGaussian& b) {
              return a.depth < b.depth;
            });
  return projected;
}

void build_tile_lists(const std::vector<ProjectedGaussian>& projected, int h, int w,
                      std::vector<int>& offsets, std::vector<int>& ids) {
  const int tiles_x = (w + kTileSize - 1) / kTileSize;
  const int tiles_y = (h + kTileSize - 1) / kTileSize;
  const int num_tiles = tiles_x * tiles_y;
  std::vector<int> counts(static_cast<size_t>(num_tiles), 0);

  for (int i = 0; i < static_cast<int>(projected.size()); ++i) {
    const auto& pg = projected[static_cast<size_t>(i)];
    if (pg.max_x < pg.min_x || pg.max_y < pg.min_y) {
      continue;
    }
    const int tx0 = std::max(0, pg.min_x / kTileSize);
    const int tx1 = std::min(tiles_x - 1, pg.max_x / kTileSize);
    const int ty0 = std::max(0, pg.min_y / kTileSize);
    const int ty1 = std::min(tiles_y - 1, pg.max_y / kTileSize);
    if (tx1 < tx0 || ty1 < ty0) {
      continue;
    }
    for (int ty = ty0; ty <= ty1; ++ty) {
      for (int tx = tx0; tx <= tx1; ++tx) {
        counts[static_cast<size_t>(ty * tiles_x + tx)]++;
      }
    }
  }

  offsets.assign(static_cast<size_t>(num_tiles + 1), 0);
  for (int t = 0; t < num_tiles; ++t) {
    offsets[static_cast<size_t>(t + 1)] = offsets[static_cast<size_t>(t)] + counts[static_cast<size_t>(t)];
  }
  ids.assign(static_cast<size_t>(offsets[static_cast<size_t>(num_tiles)]), 0);
  std::vector<int> cursor(offsets.begin(), offsets.begin() + num_tiles);
  for (int i = 0; i < static_cast<int>(projected.size()); ++i) {
    const auto& pg = projected[static_cast<size_t>(i)];
    if (pg.max_x < pg.min_x || pg.max_y < pg.min_y) {
      continue;
    }
    const int tx0 = std::max(0, pg.min_x / kTileSize);
    const int tx1 = std::min(tiles_x - 1, pg.max_x / kTileSize);
    const int ty0 = std::max(0, pg.min_y / kTileSize);
    const int ty1 = std::min(tiles_y - 1, pg.max_y / kTileSize);
    if (tx1 < tx0 || ty1 < ty0) {
      continue;
    }
    for (int ty = ty0; ty <= ty1; ++ty) {
      for (int tx = tx0; tx <= tx1; ++tx) {
        const int tile = ty * tiles_x + tx;
        ids[static_cast<size_t>(cursor[static_cast<size_t>(tile)]++)] = i;
      }
    }
  }
}

void build_tile_lists_from_ptr(const ProjectedGaussian* projected, int n, int h, int w,
                               std::vector<int>& offsets, std::vector<int>& ids) {
  const int tiles_x = (w + kTileSize - 1) / kTileSize;
  const int tiles_y = (h + kTileSize - 1) / kTileSize;
  const int num_tiles = tiles_x * tiles_y;
  std::vector<int> counts(static_cast<size_t>(num_tiles), 0);

  for (int i = 0; i < n; ++i) {
    const auto& pg = projected[i];
    if (pg.max_x < pg.min_x || pg.max_y < pg.min_y) {
      continue;
    }
    const int tx0 = std::max(0, pg.min_x / kTileSize);
    const int tx1 = std::min(tiles_x - 1, pg.max_x / kTileSize);
    const int ty0 = std::max(0, pg.min_y / kTileSize);
    const int ty1 = std::min(tiles_y - 1, pg.max_y / kTileSize);
    if (tx1 < tx0 || ty1 < ty0) {
      continue;
    }
    for (int ty = ty0; ty <= ty1; ++ty) {
      for (int tx = tx0; tx <= tx1; ++tx) {
        counts[static_cast<size_t>(ty * tiles_x + tx)]++;
      }
    }
  }

  offsets.assign(static_cast<size_t>(num_tiles + 1), 0);
  for (int t = 0; t < num_tiles; ++t) {
    offsets[static_cast<size_t>(t + 1)] = offsets[static_cast<size_t>(t)] + counts[static_cast<size_t>(t)];
  }
  ids.assign(static_cast<size_t>(offsets[static_cast<size_t>(num_tiles)]), 0);
  std::vector<int> cursor(offsets.begin(), offsets.begin() + num_tiles);
  for (int i = 0; i < n; ++i) {
    const auto& pg = projected[i];
    if (pg.max_x < pg.min_x || pg.max_y < pg.min_y) {
      continue;
    }
    const int tx0 = std::max(0, pg.min_x / kTileSize);
    const int tx1 = std::min(tiles_x - 1, pg.max_x / kTileSize);
    const int ty0 = std::max(0, pg.min_y / kTileSize);
    const int ty1 = std::min(tiles_y - 1, pg.max_y / kTileSize);
    if (tx1 < tx0 || ty1 < ty0) {
      continue;
    }
    for (int ty = ty0; ty <= ty1; ++ty) {
      for (int tx = tx0; tx <= tx1; ++tx) {
        const int tile = ty * tiles_x + tx;
        ids[static_cast<size_t>(cursor[static_cast<size_t>(tile)]++)] = i;
      }
    }
  }
}

void cpu_tiled_forward(const std::vector<ProjectedGaussian>& projected,
                       const std::vector<int>& offsets, const std::vector<int>& ids,
                       const float* colors, const float* opacities, int c, int h, int w,
                       float* output) {
  const int tiles_x = (w + kTileSize - 1) / kTileSize;
  std::fill(output, output + static_cast<size_t>(h) * w * c, 0.0f);

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int tile_x = x / kTileSize;
      const int tile_y = y / kTileSize;
      const int tile_idx = tile_y * tiles_x + tile_x;
      const int start = offsets[static_cast<size_t>(tile_idx)];
      const int end = offsets[static_cast<size_t>(tile_idx + 1)];
      float T = 1.0f;
      float accum[kMaxChannels] = {0, 0, 0};
      for (int i = start; i < end && T > 1e-4f; ++i) {
        const auto& pg = projected[static_cast<size_t>(ids[static_cast<size_t>(i)])];
        if (x < pg.min_x || x > pg.max_x || y < pg.min_y || y > pg.max_y) {
          continue;
        }
        const float dx = static_cast<float>(x) - pg.mean_x;
        const float dy = static_cast<float>(y) - pg.mean_y;
        const float quad =
            dx * (pg.inv_xx * dx + pg.inv_xy * dy) + dy * (pg.inv_yx * dx + pg.inv_yy * dy);
        const float gaussian = std::exp(-0.5f * quad);
        float alpha = opacities[pg.source_index] * gaussian;
        alpha = std::clamp(alpha, 0.0f, 0.999f);
        const float weight = T * alpha;
        for (int ch = 0; ch < c && ch < kMaxChannels; ++ch) {
          accum[ch] += weight * colors[pg.source_index * c + ch];
        }
        T *= (1.0f - alpha);
      }
      for (int ch = 0; ch < c; ++ch) {
        output[(y * w + x) * c + ch] = (ch < kMaxChannels ? accum[ch] : 0.0f);
      }
    }
  }
}

void bind_tile_buffers(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipe,
                       id<MTLBuffer> proj_buf, id<MTLBuffer> color_buf, id<MTLBuffer> opa_buf,
                       id<MTLBuffer> off_buf, id<MTLBuffer> id_buf, id<MTLBuffer> param_buf) {
  [enc setComputePipelineState:pipe];
  [enc setBuffer:proj_buf offset:0 atIndex:0];
  [enc setBuffer:color_buf offset:0 atIndex:1];
  [enc setBuffer:opa_buf offset:0 atIndex:2];
  [enc setBuffer:off_buf offset:0 atIndex:3];
  [enc setBuffer:id_buf offset:0 atIndex:4];
  [enc setBuffer:param_buf offset:0 atIndex:5];
}

void dispatch_tiles(id<MTLComputeCommandEncoder> enc, int tiles_x, int tiles_y) {
  MTLSize tgs = MTLSizeMake(static_cast<NSUInteger>(kTileSize), static_cast<NSUInteger>(kTileSize),
                            1);
  MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(tiles_x), static_cast<NSUInteger>(tiles_y), 1);
  [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgs];
}

bool metal_tiled_forward(const std::vector<ProjectedGaussian>& projected,
                         const std::vector<int>& offsets, const std::vector<int>& ids,
                         const float* colors, const float* opacities, int n, int c, int h, int w,
                         float* output) {
  auto& ctx = MetalContext::instance();
  if (!ctx.ok() || projected.empty()) {
    return false;
  }

  @autoreleasepool {
    const RasterParams params = make_raster_params(h, w, c, static_cast<int>(projected.size()));
    id<MTLBuffer> proj_buf =
        ctx.acquire(ctx.proj, projected.size() * sizeof(ProjectedGaussian));
    id<MTLBuffer> color_buf =
        ctx.acquire(ctx.colors, static_cast<NSUInteger>(n) * c * sizeof(float));
    id<MTLBuffer> opa_buf = ctx.acquire(ctx.opa, static_cast<NSUInteger>(n) * sizeof(float));
    id<MTLBuffer> off_buf = ctx.acquire(ctx.offsets, offsets.size() * sizeof(int));
    id<MTLBuffer> id_buf =
        ctx.acquire(ctx.ids, std::max<size_t>(ids.size(), 1) * sizeof(int));
    id<MTLBuffer> out_buf =
        ctx.acquire(ctx.output, static_cast<NSUInteger>(h) * w * c * sizeof(float));
    id<MTLBuffer> param_buf = ctx.acquire(ctx.params, sizeof(RasterParams));

    blit_in(proj_buf, projected.data(), projected.size() * sizeof(ProjectedGaussian));
    blit_in(color_buf, colors, static_cast<size_t>(n) * c * sizeof(float));
    blit_in(opa_buf, opacities, static_cast<size_t>(n) * sizeof(float));
    blit_in(off_buf, offsets.data(), offsets.size() * sizeof(int));
    if (!ids.empty()) {
      blit_in(id_buf, ids.data(), ids.size() * sizeof(int));
    }
    blit_in(param_buf, &params, sizeof(RasterParams));

    id<MTLCommandBuffer> cmd = [ctx.queue() commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    bind_tile_buffers(enc, ctx.forward_pipeline(), proj_buf, color_buf, opa_buf, off_buf, id_buf,
                      param_buf);
    [enc setBuffer:out_buf offset:0 atIndex:6];
    dispatch_tiles(enc, static_cast<int>(params.tiles_x), static_cast<int>(params.tiles_y));
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    std::memcpy(output, [out_buf contents], static_cast<size_t>(h) * w * c * sizeof(float));
  }
  return true;
}

std::vector<ProjectedGaussian> project_from_2d(const float* proj_means, const float* proj_covs,
                                               int n, int h, int w,
                                               const Splat3DMetalOptions& opts,
                                               const float* depths) {
  std::vector<ProjectedGaussian> projected;
  projected.reserve(static_cast<size_t>(n));
  for (int g = 0; g < n; ++g) {
    float s00 = proj_covs[g * 4 + 0] + opts.min_covariance;
    float s01 = proj_covs[g * 4 + 1];
    float s10 = proj_covs[g * 4 + 2];
    float s11 = proj_covs[g * 4 + 3] + opts.min_covariance;
    float det = s00 * s11 - s01 * s10;
    if (det <= opts.min_covariance) {
      det = opts.min_covariance;
    }
    const float inv_det = 1.0f / det;
    const float radius =
        compact_radius(s00, s01, s11, opts.min_covariance, opts.sigma_radius,
                       opts.compact_box_beta, opts.use_compact_box);
    const float mx = proj_means[g * 2 + 0];
    const float my = proj_means[g * 2 + 1];
    const int min_x = static_cast<int>(std::floor(mx - radius));
    const int max_x = static_cast<int>(std::ceil(mx + radius));
    const int min_y = static_cast<int>(std::floor(my - radius));
    const int max_y = static_cast<int>(std::ceil(my + radius));
    if (max_x < 0 || min_x >= w || max_y < 0 || min_y >= h) {
      continue;
    }
    const float depth = depths != nullptr ? depths[g] : static_cast<float>(g);
    projected.push_back(ProjectedGaussian{mx, my, depth, s11 * inv_det, -s01 * inv_det,
                                          -s10 * inv_det, s00 * inv_det, min_x, max_x, min_y, max_y,
                                          g});
  }
  std::sort(projected.begin(), projected.end(),
            [](const ProjectedGaussian& a, const ProjectedGaussian& b) {
              return a.depth < b.depth;
            });
  return projected;
}

void cpu_projected_backward(const float* grad_output, const std::vector<ProjectedGaussian>& projected,
                            const std::vector<int>& offsets, const std::vector<int>& ids,
                            const float* colors, const float* opacities, int num_gaussians,
                            int num_channels, int height, int width, float* grad_proj_means,
                            float* grad_proj_covs, float* grad_colors, float* grad_opacities) {
  const int tiles_x = (width + kTileSize - 1) / kTileSize;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int tile_idx = (y / kTileSize) * tiles_x + (x / kTileSize);
      const int start = offsets[static_cast<size_t>(tile_idx)];
      const int end = offsets[static_cast<size_t>(tile_idx + 1)];

      struct Hit {
        int idx;
        float alpha;
        float T;
        float gaussian;
      };
      std::vector<Hit> hits;
      hits.reserve(64);
      float T = 1.0f;
      for (int i = start; i < end && T > 1e-4f; ++i) {
        const auto& pg = projected[static_cast<size_t>(ids[static_cast<size_t>(i)])];
        if (x < pg.min_x || x > pg.max_x || y < pg.min_y || y > pg.max_y) {
          continue;
        }
        const float dx = static_cast<float>(x) - pg.mean_x;
        const float dy = static_cast<float>(y) - pg.mean_y;
        const float quad =
            dx * (pg.inv_xx * dx + pg.inv_xy * dy) + dy * (pg.inv_yx * dx + pg.inv_yy * dy);
        const float gaussian = std::exp(-0.5f * quad);
        float alpha = opacities[pg.source_index] * gaussian;
        alpha = std::clamp(alpha, 0.0f, 0.999f);
        if (alpha < 1e-4f) {
          continue;
        }
        hits.push_back(Hit{static_cast<int>(ids[static_cast<size_t>(i)]), alpha, T, gaussian});
        T *= (1.0f - alpha);
      }
      if (hits.empty()) {
        continue;
      }

      const float* go = grad_output + (y * width + x) * num_channels;
      float dL_dTnext = 0.0f;
      for (int hi = static_cast<int>(hits.size()) - 1; hi >= 0; --hi) {
        const Hit& hit = hits[static_cast<size_t>(hi)];
        const auto& pg = projected[static_cast<size_t>(hit.idx)];
        const int src = pg.source_index;
        const float T_in = hit.T;
        const float alpha = hit.alpha;
        const float gaussian = hit.gaussian;

        float dL_dC_dot_c = 0.0f;
        for (int ch = 0; ch < num_channels; ++ch) {
          const float col = colors[src * num_channels + ch];
          dL_dC_dot_c += go[ch] * col;
          grad_colors[src * num_channels + ch] += go[ch] * T_in * alpha;
        }

        const float dL_dalpha = T_in * (dL_dC_dot_c - dL_dTnext);
        grad_opacities[src] += dL_dalpha * gaussian;

        const float dL_dgaussian = dL_dalpha * opacities[src];
        const float dx = static_cast<float>(x) - pg.mean_x;
        const float dy = static_cast<float>(y) - pg.mean_y;
        const float ad_x = pg.inv_xx * dx + pg.inv_xy * dy;
        const float ad_y = pg.inv_yx * dx + pg.inv_yy * dy;
        const float common = dL_dgaussian * gaussian;

        grad_proj_means[src * 2 + 0] += common * ad_x;
        grad_proj_means[src * 2 + 1] += common * ad_y;
        const float half = 0.5f * common;
        grad_proj_covs[src * 4 + 0] += half * (ad_x * ad_x);
        grad_proj_covs[src * 4 + 1] += half * (ad_x * ad_y);
        grad_proj_covs[src * 4 + 2] += half * (ad_y * ad_x);
        grad_proj_covs[src * 4 + 3] += half * (ad_y * ad_y);

        dL_dTnext = dL_dTnext * (1.0f - alpha) + dL_dC_dot_c * alpha;
      }
    }
  }
  (void)num_gaussians;
}

bool metal_tiled_backward(const std::vector<ProjectedGaussian>& projected,
                          const std::vector<int>& offsets, const std::vector<int>& ids,
                          const float* colors, const float* opacities, const float* grad_output,
                          int n, int c, int h, int w, float* grad_proj_means, float* grad_proj_covs,
                          float* grad_colors, float* grad_opacities) {
  auto& ctx = MetalContext::instance();
  if (ctx.backward_pipeline() == nil || projected.empty()) {
    return false;
  }

  @autoreleasepool {
    const RasterParams params = make_raster_params(h, w, c, static_cast<int>(projected.size()));
    id<MTLBuffer> proj_buf =
        ctx.acquire(ctx.proj, projected.size() * sizeof(ProjectedGaussian));
    id<MTLBuffer> color_buf =
        ctx.acquire(ctx.colors, static_cast<NSUInteger>(n) * c * sizeof(float));
    id<MTLBuffer> opa_buf = ctx.acquire(ctx.opa, static_cast<NSUInteger>(n) * sizeof(float));
    id<MTLBuffer> off_buf = ctx.acquire(ctx.offsets, offsets.size() * sizeof(int));
    id<MTLBuffer> id_buf =
        ctx.acquire(ctx.ids, std::max<size_t>(ids.size(), 1) * sizeof(int));
    id<MTLBuffer> go_buf =
        ctx.acquire(ctx.go, static_cast<NSUInteger>(h) * w * c * sizeof(float));
    const NSUInteger gm_bytes = static_cast<NSUInteger>(n) * 2 * sizeof(float);
    const NSUInteger gc_bytes = static_cast<NSUInteger>(n) * 4 * sizeof(float);
    const NSUInteger gcol_bytes = static_cast<NSUInteger>(n) * c * sizeof(float);
    const NSUInteger gopa_bytes = static_cast<NSUInteger>(n) * sizeof(float);
    id<MTLBuffer> gm_buf = ctx.acquire(ctx.gm, gm_bytes);
    id<MTLBuffer> gcov_buf = ctx.acquire(ctx.gcov, gc_bytes);
    id<MTLBuffer> gcol_buf = ctx.acquire(ctx.gcol, gcol_bytes);
    id<MTLBuffer> gopa_buf = ctx.acquire(ctx.gopa, gopa_bytes);
    id<MTLBuffer> param_buf = ctx.acquire(ctx.params, sizeof(RasterParams));

    blit_in(proj_buf, projected.data(), projected.size() * sizeof(ProjectedGaussian));
    blit_in(color_buf, colors, static_cast<size_t>(n) * c * sizeof(float));
    blit_in(opa_buf, opacities, static_cast<size_t>(n) * sizeof(float));
    blit_in(off_buf, offsets.data(), offsets.size() * sizeof(int));
    if (!ids.empty()) {
      blit_in(id_buf, ids.data(), ids.size() * sizeof(int));
    }
    blit_in(go_buf, grad_output, static_cast<size_t>(h) * w * c * sizeof(float));
    std::memset([gm_buf contents], 0, gm_bytes);
    std::memset([gcov_buf contents], 0, gc_bytes);
    std::memset([gcol_buf contents], 0, gcol_bytes);
    std::memset([gopa_buf contents], 0, gopa_bytes);
    blit_in(param_buf, &params, sizeof(RasterParams));

    id<MTLCommandBuffer> cmd = [ctx.queue() commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    bind_tile_buffers(enc, ctx.backward_pipeline(), proj_buf, color_buf, opa_buf, off_buf, id_buf,
                      param_buf);
    [enc setBuffer:go_buf offset:0 atIndex:6];
    [enc setBuffer:gm_buf offset:0 atIndex:7];
    [enc setBuffer:gcov_buf offset:0 atIndex:8];
    [enc setBuffer:gcol_buf offset:0 atIndex:9];
    [enc setBuffer:gopa_buf offset:0 atIndex:10];
    dispatch_tiles(enc, static_cast<int>(params.tiles_x), static_cast<int>(params.tiles_y));
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    std::memcpy(grad_proj_means, [gm_buf contents], gm_bytes);
    std::memcpy(grad_proj_covs, [gcov_buf contents], gc_bytes);
    std::memcpy(grad_colors, [gcol_buf contents], gcol_bytes);
    std::memcpy(grad_opacities, [gopa_buf contents], gopa_bytes);
  }
  return true;
}

void sort_projected_by_depth(id<MTLBuffer> proj_buf, int n) {
  auto* projected = static_cast<ProjectedGaussian*>([proj_buf contents]);
  std::sort(projected, projected + n, [](const ProjectedGaussian& a, const ProjectedGaussian& b) {
    if (a.depth != b.depth) {
      return a.depth < b.depth;
    }
    return a.source_index < b.source_index;
  });
}

struct TileBufs {
  id<MTLBuffer> offsets = nil;
  id<MTLBuffer> ids = nil;
};

TileBufs fast_cpu_tiles(MetalContext& ctx, id<MTLBuffer> proj_buf, int n, int h, int w) {
  std::vector<int> offsets;
  std::vector<int> ids;
  build_tile_lists_from_ptr(static_cast<const ProjectedGaussian*>([proj_buf contents]), n, h, w,
                            offsets, ids);
  TileBufs out;
  out.offsets = ctx.acquire(ctx.offsets, offsets.size() * sizeof(int));
  out.ids = ctx.acquire(ctx.ids, std::max<size_t>(ids.size(), 1) * sizeof(int));
  blit_in(out.offsets, offsets.data(), offsets.size() * sizeof(int));
  if (!ids.empty()) {
    blit_in(out.ids, ids.data(), ids.size() * sizeof(int));
  }
  return out;
}

bool gpu_forward(const float* means, const float* covs, const float* colors, const float* opacities,
                 int n, int c, const float* intrinsics, const float* c2w, int h, int w,
                 float* output, const Splat3DMetalOptions& opts) {
  auto& ctx = MetalContext::instance();
  ctx.session_valid = false;
  ctx.session_qs = false;
  if (!ctx.ok() || !ctx.prep_ok()) {
    return false;
  }
  const bool profile = std::getenv("TINYSPLAT_METAL_PROFILE") != nullptr;
  const auto t0 = std::chrono::steady_clock::now();
  @autoreleasepool {
    CameraParams cam = make_camera(intrinsics, c2w, n, h, w, opts, false);
    id<MTLBuffer> means_buf = ctx.acquire(ctx.means, static_cast<NSUInteger>(n) * 3 * sizeof(float));
    id<MTLBuffer> covs_buf = ctx.acquire(ctx.covs, static_cast<NSUInteger>(n) * 9 * sizeof(float));
    id<MTLBuffer> color_buf =
        ctx.acquire(ctx.colors, static_cast<NSUInteger>(n) * c * sizeof(float));
    id<MTLBuffer> opa_buf = ctx.acquire(ctx.opa, static_cast<NSUInteger>(n) * sizeof(float));
    id<MTLBuffer> cam_buf = ctx.acquire(ctx.camera, sizeof(CameraParams));
    id<MTLBuffer> proj_buf =
        ctx.acquire(ctx.proj, static_cast<NSUInteger>(n) * sizeof(ProjectedGaussian));
    blit_in(means_buf, means, static_cast<size_t>(n) * 3 * sizeof(float));
    blit_in(covs_buf, covs, static_cast<size_t>(n) * 9 * sizeof(float));
    blit_in(color_buf, colors, static_cast<size_t>(n) * c * sizeof(float));
    blit_in(opa_buf, opacities, static_cast<size_t>(n) * sizeof(float));
    blit_in(cam_buf, &cam, sizeof(CameraParams));
    const auto t1 = std::chrono::steady_clock::now();

    id<MTLCommandBuffer> cmdp = [ctx.queue() commandBuffer];
    id<MTLComputeCommandEncoder> encp = [cmdp computeCommandEncoder];
    [encp setBuffer:means_buf offset:0 atIndex:0];
    [encp setBuffer:covs_buf offset:0 atIndex:1];
    [encp setBuffer:cam_buf offset:0 atIndex:2];
    [encp setBuffer:proj_buf offset:0 atIndex:3];
    dispatch_1d(encp, ctx.project_3d_pipeline(), static_cast<NSUInteger>(n));
    [encp endEncoding];
    [cmdp commit];
    [cmdp waitUntilCompleted];
    const auto t2 = std::chrono::steady_clock::now();

    sort_projected_by_depth(proj_buf, n);
    TileBufs tiles = fast_cpu_tiles(ctx, proj_buf, n, h, w);
    const RasterParams params = make_raster_params(h, w, c, n);
    id<MTLBuffer> param_buf = ctx.acquire(ctx.params, sizeof(RasterParams));
    blit_in(param_buf, &params, sizeof(RasterParams));
    id<MTLBuffer> out_buf =
        ctx.acquire(ctx.output, static_cast<NSUInteger>(h) * w * c * sizeof(float));
    const auto t3 = std::chrono::steady_clock::now();

    id<MTLCommandBuffer> cmdr = [ctx.queue() commandBuffer];
    id<MTLComputeCommandEncoder> encr = [cmdr computeCommandEncoder];
    bind_tile_buffers(encr, ctx.forward_pipeline(), proj_buf, color_buf, opa_buf, tiles.offsets,
                      tiles.ids, param_buf);
    [encr setBuffer:out_buf offset:0 atIndex:6];
    dispatch_tiles(encr, static_cast<int>(params.tiles_x), static_cast<int>(params.tiles_y));
    [encr endEncoding];
    [cmdr commit];
    [cmdr waitUntilCompleted];
    std::memcpy(output, [out_buf contents], static_cast<size_t>(h) * w * c * sizeof(float));
    const auto t4 = std::chrono::steady_clock::now();

    ctx.session_valid = true;
    ctx.session_qs = false;
    ctx.session_n = n;
    ctx.session_c = c;
    ctx.session_h = h;
    ctx.session_w = w;
    if (profile) {
      auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
      std::fprintf(stderr,
                   "[tinysplat-metal] fwd N=%d %dx%d blit=%.1fms project=%.1fms tiles=%.1fms "
                   "kernel=%.1fms total=%.1fms\n",
                   n, w, h, ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4), ms(t0, t4));
    }
  }
  return true;
}

bool gpu_forward_qs(const float* means, const float* log_scales, const float* quats,
                    const float* colors, const float* opacities, int n, int c,
                    const float* intrinsics, const float* c2w, int h, int w, float* output,
                    const Splat3DMetalOptions& opts) {
  auto& ctx = MetalContext::instance();
  ctx.session_valid = false;
  ctx.session_qs = false;
  if (!ctx.ok() || !ctx.qs_ok()) {
    return false;
  }
  const bool profile = std::getenv("TINYSPLAT_METAL_PROFILE") != nullptr;
  const auto t0 = std::chrono::steady_clock::now();
  @autoreleasepool {
    CameraParams cam = make_camera(intrinsics, c2w, n, h, w, opts, false);
    id<MTLBuffer> means_buf = ctx.acquire(ctx.means, static_cast<NSUInteger>(n) * 3 * sizeof(float));
    id<MTLBuffer> scales_buf =
        ctx.acquire(ctx.scales, static_cast<NSUInteger>(n) * 3 * sizeof(float));
    id<MTLBuffer> quats_buf = ctx.acquire(ctx.quats, static_cast<NSUInteger>(n) * 4 * sizeof(float));
    id<MTLBuffer> color_buf =
        ctx.acquire(ctx.colors, static_cast<NSUInteger>(n) * c * sizeof(float));
    id<MTLBuffer> opa_buf = ctx.acquire(ctx.opa, static_cast<NSUInteger>(n) * sizeof(float));
    id<MTLBuffer> cam_buf = ctx.acquire(ctx.camera, sizeof(CameraParams));
    id<MTLBuffer> proj_buf =
        ctx.acquire(ctx.proj, static_cast<NSUInteger>(n) * sizeof(ProjectedGaussian));
    blit_in(means_buf, means, static_cast<size_t>(n) * 3 * sizeof(float));
    blit_in(scales_buf, log_scales, static_cast<size_t>(n) * 3 * sizeof(float));
    blit_in(quats_buf, quats, static_cast<size_t>(n) * 4 * sizeof(float));
    blit_in(color_buf, colors, static_cast<size_t>(n) * c * sizeof(float));
    blit_in(opa_buf, opacities, static_cast<size_t>(n) * sizeof(float));
    blit_in(cam_buf, &cam, sizeof(CameraParams));
    const auto t1 = std::chrono::steady_clock::now();

    id<MTLCommandBuffer> cmdp = [ctx.queue() commandBuffer];
    id<MTLComputeCommandEncoder> encp = [cmdp computeCommandEncoder];
    [encp setBuffer:means_buf offset:0 atIndex:0];
    [encp setBuffer:scales_buf offset:0 atIndex:1];
    [encp setBuffer:quats_buf offset:0 atIndex:2];
    [encp setBuffer:cam_buf offset:0 atIndex:3];
    [encp setBuffer:proj_buf offset:0 atIndex:4];
    dispatch_1d(encp, ctx.project_3d_qs_pipeline(), static_cast<NSUInteger>(n));
    [encp endEncoding];
    [cmdp commit];
    [cmdp waitUntilCompleted];
    const auto t2 = std::chrono::steady_clock::now();

    sort_projected_by_depth(proj_buf, n);
    TileBufs tiles = fast_cpu_tiles(ctx, proj_buf, n, h, w);
    const RasterParams params = make_raster_params(h, w, c, n);
    id<MTLBuffer> param_buf = ctx.acquire(ctx.params, sizeof(RasterParams));
    blit_in(param_buf, &params, sizeof(RasterParams));
    id<MTLBuffer> out_buf =
        ctx.acquire(ctx.output, static_cast<NSUInteger>(h) * w * c * sizeof(float));
    const auto t3 = std::chrono::steady_clock::now();

    id<MTLCommandBuffer> cmdr = [ctx.queue() commandBuffer];
    id<MTLComputeCommandEncoder> encr = [cmdr computeCommandEncoder];
    bind_tile_buffers(encr, ctx.forward_pipeline(), proj_buf, color_buf, opa_buf, tiles.offsets,
                      tiles.ids, param_buf);
    [encr setBuffer:out_buf offset:0 atIndex:6];
    dispatch_tiles(encr, static_cast<int>(params.tiles_x), static_cast<int>(params.tiles_y));
    [encr endEncoding];
    [cmdr commit];
    [cmdr waitUntilCompleted];
    std::memcpy(output, [out_buf contents], static_cast<size_t>(h) * w * c * sizeof(float));
    const auto t4 = std::chrono::steady_clock::now();

    ctx.session_valid = true;
    ctx.session_qs = true;
    ctx.session_n = n;
    ctx.session_c = c;
    ctx.session_h = h;
    ctx.session_w = w;
    if (profile) {
      auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
      };
      std::fprintf(stderr,
                   "[tinysplat-metal] fwd_qs N=%d %dx%d blit=%.1fms project=%.1fms tiles=%.1fms "
                   "kernel=%.1fms total=%.1fms\n",
                   n, w, h, ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4), ms(t0, t4));
    }
  }
  return true;
}

bool gpu_backward(const float* grad_output, const float* proj_means, const float* proj_covs,
                  const float* colors, const float* opacities, int n, int c, int h, int w,
                  float* grad_proj_means, float* grad_proj_covs, float* grad_colors,
                  float* grad_opacities, const Splat3DMetalOptions& opts, const float* depths) {
  auto& ctx = MetalContext::instance();
  ctx.session_valid = false;
  ctx.session_qs = false;
  if (ctx.backward_pipeline() == nil || ctx.project_2d_pipeline() == nil) {
    return false;
  }
  @autoreleasepool {
    CameraParams cam = make_camera(nullptr, nullptr, n, h, w, opts, depths != nullptr);
    id<MTLBuffer> pm_buf =
        ctx.acquire(ctx.proj_means, static_cast<NSUInteger>(n) * 2 * sizeof(float));
    id<MTLBuffer> pc_buf =
        ctx.acquire(ctx.proj_covs, static_cast<NSUInteger>(n) * 4 * sizeof(float));
    id<MTLBuffer> depth_buf = ctx.acquire(ctx.depths, static_cast<NSUInteger>(n) * sizeof(float));
    id<MTLBuffer> color_buf =
        ctx.acquire(ctx.colors, static_cast<NSUInteger>(n) * c * sizeof(float));
    id<MTLBuffer> opa_buf = ctx.acquire(ctx.opa, static_cast<NSUInteger>(n) * sizeof(float));
    id<MTLBuffer> cam_buf = ctx.acquire(ctx.camera, sizeof(CameraParams));
    id<MTLBuffer> proj_buf =
        ctx.acquire(ctx.proj, static_cast<NSUInteger>(n) * sizeof(ProjectedGaussian));
    blit_in(pm_buf, proj_means, static_cast<size_t>(n) * 2 * sizeof(float));
    blit_in(pc_buf, proj_covs, static_cast<size_t>(n) * 4 * sizeof(float));
    if (depths != nullptr) {
      blit_in(depth_buf, depths, static_cast<size_t>(n) * sizeof(float));
    }
    blit_in(color_buf, colors, static_cast<size_t>(n) * c * sizeof(float));
    blit_in(opa_buf, opacities, static_cast<size_t>(n) * sizeof(float));
    blit_in(cam_buf, &cam, sizeof(CameraParams));

    id<MTLCommandBuffer> cmdp = [ctx.queue() commandBuffer];
    id<MTLComputeCommandEncoder> encp = [cmdp computeCommandEncoder];
    [encp setBuffer:pm_buf offset:0 atIndex:0];
    [encp setBuffer:pc_buf offset:0 atIndex:1];
    [encp setBuffer:depth_buf offset:0 atIndex:2];
    [encp setBuffer:cam_buf offset:0 atIndex:3];
    [encp setBuffer:proj_buf offset:0 atIndex:4];
    dispatch_1d(encp, ctx.project_2d_pipeline(), static_cast<NSUInteger>(n));
    [encp endEncoding];
    [cmdp commit];
    [cmdp waitUntilCompleted];
    sort_projected_by_depth(proj_buf, n);

    TileBufs tiles = fast_cpu_tiles(ctx, proj_buf, n, h, w);

    const RasterParams params = make_raster_params(h, w, c, n);
    id<MTLBuffer> param_buf = ctx.acquire(ctx.params, sizeof(RasterParams));
    blit_in(param_buf, &params, sizeof(RasterParams));
    id<MTLBuffer> go_buf =
        ctx.acquire(ctx.go, static_cast<NSUInteger>(h) * w * c * sizeof(float));
    const NSUInteger gm_bytes = static_cast<NSUInteger>(n) * 2 * sizeof(float);
    const NSUInteger gc_bytes = static_cast<NSUInteger>(n) * 4 * sizeof(float);
    const NSUInteger gcol_bytes = static_cast<NSUInteger>(n) * c * sizeof(float);
    const NSUInteger gopa_bytes = static_cast<NSUInteger>(n) * sizeof(float);
    id<MTLBuffer> gm_buf = ctx.acquire(ctx.gm, gm_bytes);
    id<MTLBuffer> gcov_buf = ctx.acquire(ctx.gcov, gc_bytes);
    id<MTLBuffer> gcol_buf = ctx.acquire(ctx.gcol, gcol_bytes);
    id<MTLBuffer> gopa_buf = ctx.acquire(ctx.gopa, gopa_bytes);
    blit_in(go_buf, grad_output, static_cast<size_t>(h) * w * c * sizeof(float));
    std::memset([gm_buf contents], 0, gm_bytes);
    std::memset([gcov_buf contents], 0, gc_bytes);
    std::memset([gcol_buf contents], 0, gcol_bytes);
    std::memset([gopa_buf contents], 0, gopa_bytes);

    id<MTLCommandBuffer> cmdr = [ctx.queue() commandBuffer];
    id<MTLComputeCommandEncoder> encr = [cmdr computeCommandEncoder];
    bind_tile_buffers(encr, ctx.backward_pipeline(), proj_buf, color_buf, opa_buf, tiles.offsets,
                      tiles.ids, param_buf);
    [encr setBuffer:go_buf offset:0 atIndex:6];
    [encr setBuffer:gm_buf offset:0 atIndex:7];
    [encr setBuffer:gcov_buf offset:0 atIndex:8];
    [encr setBuffer:gcol_buf offset:0 atIndex:9];
    [encr setBuffer:gopa_buf offset:0 atIndex:10];
    dispatch_tiles(encr, static_cast<int>(params.tiles_x), static_cast<int>(params.tiles_y));
    [encr endEncoding];
    [cmdr commit];
    [cmdr waitUntilCompleted];

    std::memcpy(grad_proj_means, [gm_buf contents], gm_bytes);
    std::memcpy(grad_proj_covs, [gcov_buf contents], gc_bytes);
    std::memcpy(grad_colors, [gcol_buf contents], gcol_bytes);
    std::memcpy(grad_opacities, [gopa_buf contents], gopa_bytes);
  }
  return true;
}

bool gpu_session_backward(const float* grad_output, int n, int c, int h, int w,
                          float* grad_means3d, float* grad_covs3d, float* grad_colors,
                          float* grad_opacities, const Splat3DMetalOptions& opts) {
  auto& ctx = MetalContext::instance();
  if (opts.force_cpu || !ctx.session_valid || ctx.session_qs || ctx.session_n != n ||
      ctx.session_c != c || ctx.session_h != h || ctx.session_w != w ||
      ctx.backward_pipeline() == nil || ctx.vjp_pipeline() == nil) {
    return false;
  }
  const bool profile = std::getenv("TINYSPLAT_METAL_PROFILE") != nullptr;
  const auto t0 = std::chrono::steady_clock::now();
  @autoreleasepool {
    const RasterParams params = make_raster_params(h, w, c, n);
    id<MTLBuffer> param_buf = ctx.acquire(ctx.params, sizeof(RasterParams));
    blit_in(param_buf, &params, sizeof(RasterParams));
    id<MTLBuffer> go_buf =
        ctx.acquire(ctx.go, static_cast<NSUInteger>(h) * w * c * sizeof(float));
    const NSUInteger gm_bytes = static_cast<NSUInteger>(n) * 2 * sizeof(float);
    const NSUInteger gc_bytes = static_cast<NSUInteger>(n) * 4 * sizeof(float);
    const NSUInteger gcol_bytes = static_cast<NSUInteger>(n) * c * sizeof(float);
    const NSUInteger gopa_bytes = static_cast<NSUInteger>(n) * sizeof(float);
    const NSUInteger gm3_bytes = static_cast<NSUInteger>(n) * 3 * sizeof(float);
    const NSUInteger gc3_bytes = static_cast<NSUInteger>(n) * 9 * sizeof(float);
    id<MTLBuffer> gm_buf = ctx.acquire(ctx.gm, gm_bytes);
    id<MTLBuffer> gcov_buf = ctx.acquire(ctx.gcov, gc_bytes);
    id<MTLBuffer> gcol_buf = ctx.acquire(ctx.gcol, gcol_bytes);
    id<MTLBuffer> gopa_buf = ctx.acquire(ctx.gopa, gopa_bytes);
    id<MTLBuffer> gm3_buf = ctx.acquire(ctx.gm3, gm3_bytes);
    id<MTLBuffer> gcov3_buf = ctx.acquire(ctx.gcov3, gc3_bytes);
    blit_in(go_buf, grad_output, static_cast<size_t>(h) * w * c * sizeof(float));
    std::memset([gm_buf contents], 0, gm_bytes);
    std::memset([gcov_buf contents], 0, gc_bytes);
    std::memset([gcol_buf contents], 0, gcol_bytes);
    std::memset([gopa_buf contents], 0, gopa_bytes);
    const auto t1 = std::chrono::steady_clock::now();

    id<MTLCommandBuffer> cmd = [ctx.queue() commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    bind_tile_buffers(enc, ctx.backward_pipeline(), ctx.proj.buf, ctx.colors.buf, ctx.opa.buf,
                      ctx.offsets.buf, ctx.ids.buf, param_buf);
    [enc setBuffer:go_buf offset:0 atIndex:6];
    [enc setBuffer:gm_buf offset:0 atIndex:7];
    [enc setBuffer:gcov_buf offset:0 atIndex:8];
    [enc setBuffer:gcol_buf offset:0 atIndex:9];
    [enc setBuffer:gopa_buf offset:0 atIndex:10];
    dispatch_tiles(enc, static_cast<int>(params.tiles_x), static_cast<int>(params.tiles_y));
    [enc endEncoding];

    id<MTLComputeCommandEncoder> encv = [cmd computeCommandEncoder];
    [encv setBuffer:ctx.means.buf offset:0 atIndex:0];
    [encv setBuffer:ctx.covs.buf offset:0 atIndex:1];
    [encv setBuffer:ctx.camera.buf offset:0 atIndex:2];
    [encv setBuffer:gm_buf offset:0 atIndex:3];
    [encv setBuffer:gcov_buf offset:0 atIndex:4];
    [encv setBuffer:gm3_buf offset:0 atIndex:5];
    [encv setBuffer:gcov3_buf offset:0 atIndex:6];
    dispatch_1d(encv, ctx.vjp_pipeline(), static_cast<NSUInteger>(n));
    [encv endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    const auto t2 = std::chrono::steady_clock::now();

    std::memcpy(grad_means3d, [gm3_buf contents], gm3_bytes);
    std::memcpy(grad_covs3d, [gcov3_buf contents], gc3_bytes);
    std::memcpy(grad_colors, [gcol_buf contents], gcol_bytes);
    std::memcpy(grad_opacities, [gopa_buf contents], gopa_bytes);
    ctx.session_valid = false;
    ctx.session_qs = false;
    if (profile) {
      auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
      std::fprintf(stderr,
                   "[tinysplat-metal] bwd_session N=%d %dx%d blit=%.1fms kernel+vjp=%.1fms "
                   "total=%.1fms\n",
                   n, w, h, ms(t0, t1), ms(t1, t2), ms(t0, t2));
    }
  }
  return true;
}

bool gpu_session_backward_qs(const float* grad_output, int n, int c, int h, int w,
                             float* grad_means3d, float* grad_log_scales, float* grad_quats,
                             float* grad_colors, float* grad_opacities,
                             const Splat3DMetalOptions& opts) {
  auto& ctx = MetalContext::instance();
  if (opts.force_cpu || !ctx.session_valid || !ctx.session_qs || ctx.session_n != n ||
      ctx.session_c != c || ctx.session_h != h || ctx.session_w != w ||
      ctx.backward_pipeline() == nil || ctx.vjp_qs_pipeline() == nil) {
    return false;
  }
  const bool profile = std::getenv("TINYSPLAT_METAL_PROFILE") != nullptr;
  const auto t0 = std::chrono::steady_clock::now();
  @autoreleasepool {
    const RasterParams params = make_raster_params(h, w, c, n);
    id<MTLBuffer> param_buf = ctx.acquire(ctx.params, sizeof(RasterParams));
    blit_in(param_buf, &params, sizeof(RasterParams));
    id<MTLBuffer> go_buf =
        ctx.acquire(ctx.go, static_cast<NSUInteger>(h) * w * c * sizeof(float));
    const NSUInteger gm_bytes = static_cast<NSUInteger>(n) * 2 * sizeof(float);
    const NSUInteger gc_bytes = static_cast<NSUInteger>(n) * 4 * sizeof(float);
    const NSUInteger gcol_bytes = static_cast<NSUInteger>(n) * c * sizeof(float);
    const NSUInteger gopa_bytes = static_cast<NSUInteger>(n) * sizeof(float);
    const NSUInteger gm3_bytes = static_cast<NSUInteger>(n) * 3 * sizeof(float);
    const NSUInteger gls_bytes = static_cast<NSUInteger>(n) * 3 * sizeof(float);
    const NSUInteger gq_bytes = static_cast<NSUInteger>(n) * 4 * sizeof(float);
    id<MTLBuffer> gm_buf = ctx.acquire(ctx.gm, gm_bytes);
    id<MTLBuffer> gcov_buf = ctx.acquire(ctx.gcov, gc_bytes);
    id<MTLBuffer> gcol_buf = ctx.acquire(ctx.gcol, gcol_bytes);
    id<MTLBuffer> gopa_buf = ctx.acquire(ctx.gopa, gopa_bytes);
    id<MTLBuffer> gm3_buf = ctx.acquire(ctx.gm3, gm3_bytes);
    id<MTLBuffer> gls_buf = ctx.acquire(ctx.gls, gls_bytes);
    id<MTLBuffer> gq_buf = ctx.acquire(ctx.gq, gq_bytes);
    blit_in(go_buf, grad_output, static_cast<size_t>(h) * w * c * sizeof(float));
    std::memset([gm_buf contents], 0, gm_bytes);
    std::memset([gcov_buf contents], 0, gc_bytes);
    std::memset([gcol_buf contents], 0, gcol_bytes);
    std::memset([gopa_buf contents], 0, gopa_bytes);
    const auto t1 = std::chrono::steady_clock::now();

    id<MTLCommandBuffer> cmd = [ctx.queue() commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    bind_tile_buffers(enc, ctx.backward_pipeline(), ctx.proj.buf, ctx.colors.buf, ctx.opa.buf,
                      ctx.offsets.buf, ctx.ids.buf, param_buf);
    [enc setBuffer:go_buf offset:0 atIndex:6];
    [enc setBuffer:gm_buf offset:0 atIndex:7];
    [enc setBuffer:gcov_buf offset:0 atIndex:8];
    [enc setBuffer:gcol_buf offset:0 atIndex:9];
    [enc setBuffer:gopa_buf offset:0 atIndex:10];
    dispatch_tiles(enc, static_cast<int>(params.tiles_x), static_cast<int>(params.tiles_y));
    [enc endEncoding];

    id<MTLComputeCommandEncoder> encv = [cmd computeCommandEncoder];
    [encv setBuffer:ctx.means.buf offset:0 atIndex:0];
    [encv setBuffer:ctx.scales.buf offset:0 atIndex:1];
    [encv setBuffer:ctx.quats.buf offset:0 atIndex:2];
    [encv setBuffer:ctx.camera.buf offset:0 atIndex:3];
    [encv setBuffer:gm_buf offset:0 atIndex:4];
    [encv setBuffer:gcov_buf offset:0 atIndex:5];
    [encv setBuffer:gm3_buf offset:0 atIndex:6];
    [encv setBuffer:gls_buf offset:0 atIndex:7];
    [encv setBuffer:gq_buf offset:0 atIndex:8];
    dispatch_1d(encv, ctx.vjp_qs_pipeline(), static_cast<NSUInteger>(n));
    [encv endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    const auto t2 = std::chrono::steady_clock::now();

    std::memcpy(grad_means3d, [gm3_buf contents], gm3_bytes);
    std::memcpy(grad_log_scales, [gls_buf contents], gls_bytes);
    std::memcpy(grad_quats, [gq_buf contents], gq_bytes);
    std::memcpy(grad_colors, [gcol_buf contents], gcol_bytes);
    std::memcpy(grad_opacities, [gopa_buf contents], gopa_bytes);
    ctx.session_valid = false;
    ctx.session_qs = false;
    if (profile) {
      auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
      };
      std::fprintf(stderr,
                   "[tinysplat-metal] bwd_session_qs N=%d %dx%d blit=%.1fms kernel+vjp=%.1fms "
                   "total=%.1fms\n",
                   n, w, h, ms(t0, t1), ms(t1, t2), ms(t0, t2));
    }
  }
  return true;
}

}  // namespace

bool metal_available() {
#if defined(__APPLE__)
  return MetalContext::instance().ok();
#else
  return false;
#endif
}

bool gaussian_splat_3d_forward(const float* means, const float* covs, const float* colors,
                               const float* opacities, int num_gaussians, int num_channels,
                               const float* intrinsics, const float* camera_to_world, int height,
                               int width, float* output_host, const Splat3DMetalOptions& opts) {
  if (num_gaussians <= 0 || height <= 0 || width <= 0 || num_channels <= 0 ||
      output_host == nullptr) {
    return false;
  }
  auto& ctx = MetalContext::instance();
  if (metal_available()) {
    std::lock_guard<std::mutex> lock(ctx.mutex());
    if (gpu_forward(means, covs, colors, opacities, num_gaussians, num_channels, intrinsics,
                    camera_to_world, height, width, output_host, opts)) {
      return true;
    }
  }
  auto projected =
      project_gaussians(means, covs, num_gaussians, intrinsics, camera_to_world, height, width, opts);
  std::vector<int> offsets;
  std::vector<int> ids;
  build_tile_lists(projected, height, width, offsets, ids);

  if (metal_available()) {
    std::lock_guard<std::mutex> lock(ctx.mutex());
    if (metal_tiled_forward(projected, offsets, ids, colors, opacities, num_gaussians, num_channels,
                            height, width, output_host)) {
      return true;
    }
  }
  cpu_tiled_forward(projected, offsets, ids, colors, opacities, num_channels, height, width,
                    output_host);
  return true;
}

bool gaussian_splat_3d_forward_qs(const float* means, const float* log_scales, const float* quats,
                                  const float* colors, const float* opacities, int num_gaussians,
                                  int num_channels, const float* intrinsics,
                                  const float* camera_to_world, int height, int width,
                                  float* output_host, const Splat3DMetalOptions& opts) {
  if (num_gaussians <= 0 || height <= 0 || width <= 0 || num_channels <= 0 ||
      output_host == nullptr || means == nullptr || log_scales == nullptr || quats == nullptr) {
    return false;
  }
  auto& ctx = MetalContext::instance();
  if (metal_available()) {
    std::lock_guard<std::mutex> lock(ctx.mutex());
    if (gpu_forward_qs(means, log_scales, quats, colors, opacities, num_gaussians, num_channels,
                       intrinsics, camera_to_world, height, width, output_host, opts)) {
      return true;
    }
  }
  return false;
}

bool gaussian_splat_3d_projected_backward(const float* grad_output, const float* proj_means,
                                          const float* proj_covs, const float* colors,
                                          const float* opacities, int num_gaussians,
                                          int num_channels, int height, int width,
                                          float* grad_proj_means, float* grad_proj_covs,
                                          float* grad_colors, float* grad_opacities,
                                          const Splat3DMetalOptions& opts, const float* depths) {
  if (num_gaussians <= 0) {
    return false;
  }
  std::fill(grad_proj_means, grad_proj_means + static_cast<size_t>(num_gaussians) * 2, 0.0f);
  std::fill(grad_proj_covs, grad_proj_covs + static_cast<size_t>(num_gaussians) * 4, 0.0f);
  std::fill(grad_colors, grad_colors + static_cast<size_t>(num_gaussians) * num_channels, 0.0f);
  std::fill(grad_opacities, grad_opacities + static_cast<size_t>(num_gaussians), 0.0f);

  const bool profile = std::getenv("TINYSPLAT_METAL_PROFILE") != nullptr;
  const auto t0 = std::chrono::steady_clock::now();
  auto& ctx = MetalContext::instance();
  if (!opts.force_cpu && ctx.backward_pipeline() != nil) {
    std::lock_guard<std::mutex> lock(ctx.mutex());
    if (gpu_backward(grad_output, proj_means, proj_covs, colors, opacities, num_gaussians,
                     num_channels, height, width, grad_proj_means, grad_proj_covs, grad_colors,
                     grad_opacities, opts, depths)) {
      if (profile) {
        const auto t1 = std::chrono::steady_clock::now();
        std::fprintf(stderr, "[tinysplat-metal] bwd N=%d %dx%d gpu_prep+kernel=%.1fms\n",
                     num_gaussians, width, height,
                     std::chrono::duration<double, std::milli>(t1 - t0).count());
      }
      return true;
    }
  }

  auto projected =
      project_from_2d(proj_means, proj_covs, num_gaussians, height, width, opts, depths);
  const auto t1 = std::chrono::steady_clock::now();
  std::vector<int> offsets;
  std::vector<int> ids;
  build_tile_lists(projected, height, width, offsets, ids);
  const auto t2 = std::chrono::steady_clock::now();

  bool used_metal = false;
  if (!opts.force_cpu && ctx.backward_pipeline() != nil) {
    std::lock_guard<std::mutex> lock(ctx.mutex());
    if (metal_tiled_backward(projected, offsets, ids, colors, opacities, grad_output, num_gaussians,
                             num_channels, height, width, grad_proj_means, grad_proj_covs,
                             grad_colors, grad_opacities)) {
      used_metal = true;
    }
  }
  if (!used_metal) {
    cpu_projected_backward(grad_output, projected, offsets, ids, colors, opacities, num_gaussians,
                           num_channels, height, width, grad_proj_means, grad_proj_covs,
                           grad_colors, grad_opacities);
  }
  if (profile) {
    const auto t3 = std::chrono::steady_clock::now();
    auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::fprintf(stderr,
                 "[tinysplat-metal] bwd N=%d %dx%d projected=%zu ids=%zu metal=%d "
                 "project=%.1fms tiles=%.1fms kernel=%.1fms\n",
                 num_gaussians, width, height, projected.size(), ids.size(), used_metal ? 1 : 0,
                 ms(t0, t1), ms(t1, t2), ms(t2, t3));
  }
  return true;
}

bool gaussian_splat_3d_session_backward(const float* grad_output, int num_gaussians,
                                        int num_channels, int height, int width,
                                        float* grad_means3d, float* grad_covs3d,
                                        float* grad_colors, float* grad_opacities,
                                        const Splat3DMetalOptions& opts) {
  if (num_gaussians <= 0 || height <= 0 || width <= 0 || num_channels <= 0 ||
      grad_output == nullptr || grad_means3d == nullptr || grad_covs3d == nullptr ||
      grad_colors == nullptr || grad_opacities == nullptr) {
    return false;
  }
  auto& ctx = MetalContext::instance();
  std::lock_guard<std::mutex> lock(ctx.mutex());
  return gpu_session_backward(grad_output, num_gaussians, num_channels, height, width,
                              grad_means3d, grad_covs3d, grad_colors, grad_opacities, opts);
}

bool gaussian_splat_3d_session_backward_qs(const float* grad_output, int num_gaussians,
                                           int num_channels, int height, int width,
                                           float* grad_means3d, float* grad_log_scales,
                                           float* grad_quats, float* grad_colors,
                                           float* grad_opacities, const Splat3DMetalOptions& opts) {
  if (num_gaussians <= 0 || height <= 0 || width <= 0 || num_channels <= 0 ||
      grad_output == nullptr || grad_means3d == nullptr || grad_log_scales == nullptr ||
      grad_quats == nullptr || grad_colors == nullptr || grad_opacities == nullptr) {
    return false;
  }
  auto& ctx = MetalContext::instance();
  std::lock_guard<std::mutex> lock(ctx.mutex());
  return gpu_session_backward_qs(grad_output, num_gaussians, num_channels, height, width,
                                 grad_means3d, grad_log_scales, grad_quats, grad_colors,
                                 grad_opacities, opts);
}

bool count_footprint_hits(const float* proj_means, const float* proj_covs, const float* opacities,
                          int num_gaussians, int height, int width, const uint8_t* error_mask,
                          int* counts, const Splat3DMetalOptions& opts) {
  if (num_gaussians <= 0) {
    return false;
  }
  std::fill(counts, counts + num_gaussians, 0);
  auto& ctx = MetalContext::instance();
  if (ctx.ok() && ctx.footprint_pipeline() != nil && ctx.project_2d_pipeline() != nil) {
    std::lock_guard<std::mutex> lock(ctx.mutex());
    ctx.session_valid = false;
    ctx.session_qs = false;
    @autoreleasepool {
      CameraParams cam =
          make_camera(nullptr, nullptr, num_gaussians, height, width, opts, false);
      id<MTLBuffer> pm_buf =
          ctx.acquire(ctx.proj_means, static_cast<NSUInteger>(num_gaussians) * 2 * sizeof(float));
      id<MTLBuffer> pc_buf =
          ctx.acquire(ctx.proj_covs, static_cast<NSUInteger>(num_gaussians) * 4 * sizeof(float));
      id<MTLBuffer> depth_buf =
          ctx.acquire(ctx.depths, static_cast<NSUInteger>(num_gaussians) * sizeof(float));
      id<MTLBuffer> cam_buf = ctx.acquire(ctx.camera, sizeof(CameraParams));
      id<MTLBuffer> proj_buf = ctx.acquire(
          ctx.proj, static_cast<NSUInteger>(num_gaussians) * sizeof(ProjectedGaussian));
      blit_in(pm_buf, proj_means, static_cast<size_t>(num_gaussians) * 2 * sizeof(float));
      blit_in(pc_buf, proj_covs, static_cast<size_t>(num_gaussians) * 4 * sizeof(float));
      blit_in(cam_buf, &cam, sizeof(CameraParams));

      id<MTLCommandBuffer> cmdp = [ctx.queue() commandBuffer];
      id<MTLComputeCommandEncoder> encp = [cmdp computeCommandEncoder];
      [encp setBuffer:pm_buf offset:0 atIndex:0];
      [encp setBuffer:pc_buf offset:0 atIndex:1];
      [encp setBuffer:depth_buf offset:0 atIndex:2];
      [encp setBuffer:cam_buf offset:0 atIndex:3];
      [encp setBuffer:proj_buf offset:0 atIndex:4];
      dispatch_1d(encp, ctx.project_2d_pipeline(), static_cast<NSUInteger>(num_gaussians));
      [encp endEncoding];
      [cmdp commit];
      [cmdp waitUntilCompleted];

      id<MTLBuffer> opa_buf =
          ctx.acquire(ctx.opa, static_cast<NSUInteger>(num_gaussians) * sizeof(float));
      id<MTLBuffer> mask_buf =
          ctx.acquire(ctx.mask, static_cast<NSUInteger>(height) * width);
      id<MTLBuffer> count_buf =
          ctx.acquire(ctx.hitcounts, static_cast<NSUInteger>(num_gaussians) * sizeof(int));
      RasterParams params = make_raster_params(height, width, 1, num_gaussians);
      id<MTLBuffer> param_buf = ctx.acquire(ctx.params, sizeof(RasterParams));
      blit_in(opa_buf, opacities, static_cast<size_t>(num_gaussians) * sizeof(float));
      blit_in(mask_buf, error_mask, static_cast<size_t>(height) * width);
      std::memset([count_buf contents], 0, static_cast<size_t>(num_gaussians) * sizeof(int));
      blit_in(param_buf, &params, sizeof(RasterParams));

      id<MTLCommandBuffer> cmd = [ctx.queue() commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      [enc setComputePipelineState:ctx.footprint_pipeline()];
      [enc setBuffer:proj_buf offset:0 atIndex:0];
      [enc setBuffer:opa_buf offset:0 atIndex:1];
      [enc setBuffer:mask_buf offset:0 atIndex:2];
      [enc setBuffer:param_buf offset:0 atIndex:3];
      [enc setBuffer:count_buf offset:0 atIndex:4];
      dispatch_1d(enc, ctx.footprint_pipeline(), static_cast<NSUInteger>(num_gaussians));
      [enc endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
      std::memcpy(counts, [count_buf contents], static_cast<size_t>(num_gaussians) * sizeof(int));
      return true;
    }
  }

  auto projected =
      project_from_2d(proj_means, proj_covs, num_gaussians, height, width, opts, nullptr);
  for (const auto& pg : projected) {
    const int x0 = std::max(pg.min_x, 0);
    const int x1 = std::min(pg.max_x, width - 1);
    const int y0 = std::max(pg.min_y, 0);
    const int y1 = std::min(pg.max_y, height - 1);
    int hits = 0;
    for (int y = y0; y <= y1; ++y) {
      for (int x = x0; x <= x1; ++x) {
        if (error_mask[y * width + x] == 0) {
          continue;
        }
        const float dx = static_cast<float>(x) - pg.mean_x;
        const float dy = static_cast<float>(y) - pg.mean_y;
        const float quad =
            dx * (pg.inv_xx * dx + pg.inv_xy * dy) + dy * (pg.inv_yx * dx + pg.inv_yy * dy);
        const float gaussian = std::exp(-0.5f * quad);
        if (opacities[pg.source_index] * gaussian < 1e-4f) {
          continue;
        }
        ++hits;
      }
    }
    counts[pg.source_index] += hits;
  }
  return true;
}

}  // namespace metal
}  // namespace tinysplat
