#pragma once

#include "tinysplat/image.h"
#include "tinysplat/types.h"

namespace tinysplat {

struct Splat3DOptions {
  float near_plane = 1e-4f;
  float min_covariance = 1e-4f;
  float sigma_radius = 4.0f;
  /// When true and CUDA is available, use the GPU rasterizer.
  bool use_cuda = false;
  /// When true and Metal is available, use the Metal tiled rasterizer (preferred on macOS).
  bool use_metal = false;
  /// FastGS compact-box Mahalanobis scale (smaller = tighter tile footprint).
  float compact_box_beta = 3.0f;
  bool use_compact_box = true;
};

struct GradientsProjected2D {
  std::vector<Vec2> grad_means;
  std::vector<Mat2> grad_covariances;
  std::vector<std::vector<float>> grad_colors;
  std::vector<float> grad_opacities;
};

/// Project 3D Gaussians and alpha-composite front-to-back (3DGS-style).
Image gaussian_splat_3d_forward(const Gaussians3D& gaussians, const CameraIntrinsics& intrinsics,
                                const Mat4& camera_to_world, int height, int width,
                                const Splat3DOptions& opts = {});

/// Render already-projected 2D Gaussians with alpha compositing.
Image gaussian_splat_3d_projected_forward(const ProjectedGaussians2D& gaussians, int height,
                                          int width, const Splat3DOptions& opts = {});

/// Project 3D means/covariances to 2D (visibility + overlap culling).
ProjectedGaussians2D project_gaussians_3d_to_2d(const Gaussians3D& gaussians,
                                                const CameraIntrinsics& intrinsics,
                                                const Mat4& camera_to_world, int height,
                                                int width, const Splat3DOptions& opts = {});

GradientsProjected2D gaussian_splat_3d_projected_backward(
    const Image& grad_output, const ProjectedGaussians2D& gaussians, int height, int width,
    const Splat3DOptions& opts = {});

/// Returns true if a CUDA device is available for tinysplat CUDA kernels.
bool cuda_raster_available();

/// Returns true if a Metal device is available for tinysplat Metal kernels.
bool metal_raster_available();

}  // namespace tinysplat
