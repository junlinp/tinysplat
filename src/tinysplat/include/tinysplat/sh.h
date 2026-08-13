#pragma once

#include <cmath>
#include <vector>

#include "tinysplat/types.h"

namespace tinysplat {

/// Evaluate degree-0..3 SH RGB color for a view direction (unit vector in world space).
/// sh_coeffs layout: 16 coeffs per channel interleaved as [c0_r,c0_g,c0_b, c1_r,...] (48 floats)
/// or planar [16 R][16 G][16 B]. We use interleaved DC-first 3DGS convention:
/// indices 0..2 = DC RGB; remaining up to 16*3.
inline Vec3 eval_sh_rgb(const std::vector<float>& sh, int degree, float dir_x, float dir_y,
                        float dir_z) {
  // 3DGS SH constants
  constexpr float C0 = 0.28209479177387814f;
  constexpr float C1 = 0.4886025119029199f;
  constexpr float C2[5] = {1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
                           -1.0925484305920792f, 0.5462742152960396f};
  constexpr float C3[7] = {-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f,
                           0.3731763325901154f, -0.4570457994644658f, 1.445305721320277f,
                           -0.5900435899266435f};

  auto coeff = [&](int i, int ch) -> float {
    const size_t idx = static_cast<size_t>(i * 3 + ch);
    return idx < sh.size() ? sh[idx] : 0.0f;
  };

  Vec3 result;
  for (int ch = 0; ch < 3; ++ch) {
    float c = C0 * coeff(0, ch);
    if (degree >= 1) {
      c += -C1 * dir_y * coeff(1, ch) + C1 * dir_z * coeff(2, ch) - C1 * dir_x * coeff(3, ch);
    }
    if (degree >= 2) {
      const float xx = dir_x * dir_x, yy = dir_y * dir_y, zz = dir_z * dir_z;
      const float xy = dir_x * dir_y, yz = dir_y * dir_z, xz = dir_x * dir_z;
      c += C2[0] * xy * coeff(4, ch) + C2[1] * yz * coeff(5, ch) +
           C2[2] * (2.0f * zz - xx - yy) * coeff(6, ch) + C2[3] * xz * coeff(7, ch) +
           C2[4] * (xx - yy) * coeff(8, ch);
    }
    if (degree >= 3) {
      c += C3[0] * dir_y * (3.0f * dir_x * dir_x - dir_y * dir_y) * coeff(9, ch) +
           C3[1] * dir_x * dir_y * dir_z * coeff(10, ch) +
           C3[2] * dir_y * (4.0f * dir_z * dir_z - dir_x * dir_x - dir_y * dir_y) * coeff(11, ch) +
           C3[3] * dir_z * (2.0f * dir_z * dir_z - 3.0f * dir_x * dir_x - 3.0f * dir_y * dir_y) *
               coeff(12, ch) +
           C3[4] * dir_x * (4.0f * dir_z * dir_z - dir_x * dir_x - dir_y * dir_y) * coeff(13, ch) +
           C3[5] * dir_z * (dir_x * dir_x - dir_y * dir_y) * coeff(14, ch) +
           C3[6] * dir_x * (dir_x * dir_x - 3.0f * dir_y * dir_y) * coeff(15, ch);
    }
    // Map SH to RGB like 3DGS: color = sigmoid-ish via 0.5 + c for DC-only paths is applied
    // by callers for DC; full SH already includes DC. Clamp later.
    if (ch == 0) result.x = c + 0.5f;
    if (ch == 1) result.y = c + 0.5f;
    if (ch == 2) result.z = c + 0.5f;
  }
  return result;
}

/// Fill RGB colors from SH for a camera looking from camera center toward each mean.
inline void fill_colors_from_sh(Gaussians3D& g, const Mat4& camera_to_world) {
  if (g.sh_coeffs.empty()) {
    return;
  }
  const float cam_x = camera_to_world.m[0][3];
  const float cam_y = camera_to_world.m[1][3];
  const float cam_z = camera_to_world.m[2][3];
  const int n = static_cast<int>(g.means.size());
  g.colors.resize(static_cast<size_t>(n), std::vector<float>(3));
  for (int i = 0; i < n; ++i) {
    float dx = g.means[static_cast<size_t>(i)].x - cam_x;
    float dy = g.means[static_cast<size_t>(i)].y - cam_y;
    float dz = g.means[static_cast<size_t>(i)].z - cam_z;
    const float inv = 1.0f / std::sqrt(std::max(1e-12f, dx * dx + dy * dy + dz * dz));
    dx *= inv;
    dy *= inv;
    dz *= inv;
    const Vec3 rgb =
        eval_sh_rgb(g.sh_coeffs[static_cast<size_t>(i)], g.sh_degree, dx, dy, dz);
    g.colors[static_cast<size_t>(i)] = {
        std::max(0.0f, std::min(1.0f, rgb.x)),
        std::max(0.0f, std::min(1.0f, rgb.y)),
        std::max(0.0f, std::min(1.0f, rgb.z)),
    };
  }
}

}  // namespace tinysplat
