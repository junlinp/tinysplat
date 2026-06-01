#pragma once

#include "tinysplat/gaussian_3d.h"

namespace tinysplat {

#ifdef TINYSPLAT_CUDA
bool cuda_device_available();

Image gaussian_splat_3d_forward_cuda_impl(const Gaussians3D& gaussians,
                                          const CameraIntrinsics& intrinsics,
                                          const Mat4& camera_to_world, int height, int width,
                                          const Splat3DOptions& opts);

GradientsProjected2D gaussian_splat_3d_projected_backward_cuda_impl(
    const Image& grad_output, const ProjectedGaussians2D& gaussians, int height, int width);
#endif

}  // namespace tinysplat
