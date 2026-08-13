#pragma once

#include "tinysplat/gaussian_3d.h"
#include "tinysplat/image.h"
#include "tinysplat/types.h"

namespace tinysplat {

#ifdef TINYSPLAT_METAL
Image gaussian_splat_3d_forward_metal_impl(const Gaussians3D& gaussians,
                                           const CameraIntrinsics& intrinsics,
                                           const Mat4& camera_to_world, int height, int width,
                                           const Splat3DOptions& opts);

GradientsProjected2D gaussian_splat_3d_projected_backward_metal_impl(
    const Image& grad_output, const ProjectedGaussians2D& gaussians, int height, int width,
    const Splat3DOptions& opts);

bool metal_device_available();
#endif

}  // namespace tinysplat
