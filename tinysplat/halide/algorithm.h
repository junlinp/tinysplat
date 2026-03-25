#ifndef TINYSPLAT_HALIDE_ALGORITHM_H
#define TINYSPLAT_HALIDE_ALGORITHM_H

/**
 * TinySplat Halide Algorithm Definitions
 * ======================================
 * Forward pipeline for 2D Gaussian splatting.
 * Backward (Phase 2) and backward (Phase 1 stub) live in algorithm.cpp.
 *
 * Build:  HL_PATH=$HOME/halide make
 */

#include <Halide.h>
#include <vector>

using Halide::Buffer;
using Halide::Expr;
using Halide::Func;

namespace tinysplat_halide {

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

/**
 * Build the forward Gaussian splatting pipeline.
 *
 * @param means_var        (N, 2)    float32
 * @param covariances_var  (N, 2, 2) float32
 * @param colors_var       (N, C)    float32
 * @param opacities_var    (N,)      float32
 * @param height           canvas height
 * @param width            canvas width
 * @param num_channels     3 or 4
 * @return {output_func, intermediates}
 */
std::pair<Func, std::vector<Buffer<float>>>
build_forward_pipeline(const Buffer<float>& means_var,
                       const Buffer<float>& covariances_var,
                       const Buffer<float>& colors_var,
                       const Buffer<float>& opacities_var,
                       int height, int width, int num_channels);

// ---------------------------------------------------------------------------
// Backward pass (Phase 1: stub — PyTorch fallback)
// ---------------------------------------------------------------------------

/**
 * Build the backward gradient pipeline.
 * Phase 1 returns empty vector; Python falls back to PyTorch.
 * Phase 2 will return {grad_means, grad_cov, grad_colors, grad_opacities}.
 */
std::vector<Buffer<float>>
build_backward_pipeline(const Buffer<float>& grad_output_var,
                         const Buffer<float>& means_var,
                         const Buffer<float>& covariances_var,
                         const Buffer<float>& colors_var,
                         const Buffer<float>& opacities_var,
                         int height, int width, int num_channels);

}  // namespace tinysplat_halide

#endif  // TINYSPLAT_HALIDE_ALGORITHM_H
