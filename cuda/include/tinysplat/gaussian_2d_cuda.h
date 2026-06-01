#pragma once

#include <cstddef>

namespace tinysplat {
namespace cuda {

/// Weight-normalized 2D splat on GPU. Host buffers must be contiguous float arrays.
/// means: (N,2), covs: (N,4) row-major [a,b,c,d], colors: (N,C), opacities: (N).
/// output: (H,W,C) row-major. Returns false on CUDA error.
bool gaussian_splat_2d_forward(
    const float* means,
    const float* covs,
    const float* colors,
    const float* opacities,
    int num_gaussians,
    int num_channels,
    int height,
    int width,
    float* output_host);

}  // namespace cuda
}  // namespace tinysplat
