#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace tinysplat {

struct Vec2 {
  float x = 0.0f;
  float y = 0.0f;
};

struct Vec3 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

struct Mat2 {
  float m00 = 1.0f;
  float m01 = 0.0f;
  float m10 = 0.0f;
  float m11 = 1.0f;
};

struct Mat3 {
  float m[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
};

struct Mat4 {
  float m[4][4] = {};
};

struct Gaussians2D {
  std::vector<Vec2> means;
  std::vector<Mat2> covariances;
  std::vector<std::vector<float>> colors;
  std::vector<float> opacities;
};

struct Gradients2D {
  std::vector<Vec2> grad_means;
  std::vector<Mat2> grad_covariances;
  std::vector<std::vector<float>> grad_colors;
  std::vector<float> grad_opacities;
};

struct Gaussians3D {
  std::vector<Vec3> means;
  std::vector<Mat3> covariances;
  std::vector<std::vector<float>> colors;
  std::vector<float> opacities;
  /// Optional anisotropic scale (N,3) and unit quaternion wxyz (N,4).
  std::vector<Vec3> scales;
  std::vector<std::array<float, 4>> rotations;
  /// Optional SH coeffs: N vectors of size 16*3 (degree-3 RGB interleaved as
  /// [dc_r,dc_g,dc_b, rest...]). When non-empty, Metal/CUDA may evaluate view-dependent color.
  std::vector<std::vector<float>> sh_coeffs;
  int sh_degree = 0;
};

struct CameraIntrinsics {
  float fx = 1.0f;
  float fy = 1.0f;
  float cx = 0.0f;
  float cy = 0.0f;
};

struct ProjectedGaussians2D {
  std::vector<Vec2> means;
  std::vector<Mat2> covariances;
  std::vector<std::vector<float>> colors;
  std::vector<float> opacities;
};

}  // namespace tinysplat
