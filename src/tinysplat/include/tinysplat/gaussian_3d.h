#pragma once

#include "tinysplat/image.h"
#include "tinysplat/types.h"

namespace tinysplat {

struct Splat3DOptions {
  float near_plane = 1e-4f;
  float min_covariance = 1e-4f;
  float sigma_radius = 4.0f;
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

}  // namespace tinysplat
