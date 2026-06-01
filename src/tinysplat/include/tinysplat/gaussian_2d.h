#pragma once

#include "tinysplat/image.h"
#include "tinysplat/types.h"

namespace tinysplat {

/// Weight-normalized 2D Gaussian splat (same compositing as the original TinySplat CPU path).
Image gaussian_splat_2d_forward(const Gaussians2D& gaussians, int height, int width);

Gradients2D gaussian_splat_2d_backward(const Image& grad_output, const Gaussians2D& gaussians,
                                       int height, int width);

}  // namespace tinysplat
