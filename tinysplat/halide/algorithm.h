#ifndef TINYSPLAT_HALIDE_ALGORITHM_H
#define TINYSPLAT_HALIDE_ALGORITHM_H

/**
 * TinySplat Halide Algorithm Definitions
 * ======================================
 * Forward + Backward pipelines for 2D Gaussian splatting.
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
// Backward pass — analytical gradients
// ---------------------------------------------------------------------------

/**
 * Gradient Funcs from backward pipeline.
 * Each Func can be realized into a Buffer of the appropriate shape.
 */
struct GradientFuncs {
    Func grad_means;      // (N, 2)
    Func grad_cov;        // (N, 2, 2) — stub (zero)
    Func grad_colors;     // (N, C)
    Func grad_opacities;  // (N,)
};

/**
 * Build the backward gradient pipeline.
 *
 * Computes analytical gradients:
 *   grad_colors    = Σ_{x,y} grad_out[x,y,c] × weight[n,x,y] / total_weight[x,y]
 *   grad_opacities = Σ_{x,y,c} grad_out × norm × exp(-0.5×mahal)
 *                     × (colors - output) / total_weight
 *   grad_means     = Σ_{x,y,c} grad_out × weight × (colors - output) / total_weight
 *                     × (-inv_cov × (pixel - mean))
 *   grad_cov       = stub (zero); Phase 3 via Halide generate_adjoints
 *
 * @return GradientFuncs struct with Funcs ready to realize
 */
GradientFuncs
build_backward_pipeline(const Buffer<float>& grad_output_var,
                         const Buffer<float>& means_var,
                         const Buffer<float>& covariances_var,
                         const Buffer<float>& colors_var,
                         const Buffer<float>& opacities_var,
                         int height, int width, int num_channels);

}  // namespace tinysplat_halide

#endif  // TINYSPLAT_HALIDE_ALGORITHM_H
