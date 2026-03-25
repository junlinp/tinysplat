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
// Forward pass — returns Funcs for scheduling before realize
// ---------------------------------------------------------------------------

/**
 * Forward pipeline Funcs.
 * All Funcs are built but NOT scheduled — caller applies schedule
 * via apply_forward_schedule(Target) before realize().
 */
struct ForwardPipeline {
    Func output;           // (y, x, c) — final composited image
    Func accum_color;      // (y, x, c) — numerator sum
    Func accum_weight;     // (y, x)    — denominator sum
    Func weight;           // (y, x, n) — per-Gaussian per-pixel weight
    Func total_weight_pix; // (y, x)    — alias for accum_weight
    Func total_color_pix;  // (y, x, c) — alias for accum_color
    Func output_norm;      // (y, x, c) — normalized output
};

/**
 * Build the forward pipeline.
 * Returns ForwardPipeline with all Funcs defined.
 * Caller applies schedule, then realizes output.
 */
ForwardPipeline
build_forward_pipeline(const Buffer<float>& means_var,
                      const Buffer<float>& covariances_var,
                      const Buffer<float>& colors_var,
                      const Buffer<float>& opacities_var,
                      int height, int width, int num_channels);

/**
 * Apply CPU schedule to a ForwardPipeline.
 */
void apply_cpu_schedule_forward(ForwardPipeline& p, int height, int width, int num_channels);

/**
 * Apply CUDA schedule to a ForwardPipeline.
 */
void apply_cuda_schedule_forward(ForwardPipeline& p, int height, int width, int num_channels);

/**
 * Apply Metal schedule to a ForwardPipeline.
 */
void apply_metal_schedule_forward(ForwardPipeline& p, int height, int width, int num_channels);


// ---------------------------------------------------------------------------
// Backward pass — analytical gradients
// ---------------------------------------------------------------------------

/**
 * Gradient Funcs from backward pipeline.
 */
struct GradientFuncs {
    Func grad_means;      // (n, c) — c=0 is x, c=1 is y
    Func grad_cov;        // (n, c, r) — stub (zero) until Phase 3
    Func grad_colors;     // (n, c)
    Func grad_opacities;  // (n,)
};

/**
 * Build the backward gradient pipeline.
 * Returns GradientFuncs ready to schedule and realize.
 */
GradientFuncs
build_backward_pipeline(const Buffer<float>& grad_output_var,
                        const Buffer<float>& means_var,
                        const Buffer<float>& covariances_var,
                        const Buffer<float>& colors_var,
                        const Buffer<float>& opacities_var,
                        int height, int width, int num_channels);

/**
 * Apply CPU schedule to GradientFuncs.
 */
void apply_cpu_schedule_backward(GradientFuncs& g, int height, int width, int num_channels, int N);

/**
 * Apply CUDA schedule to GradientFuncs.
 */
void apply_cuda_schedule_backward(GradientFuncs& g, int height, int width, int num_channels, int N);

/**
 * Apply Metal schedule to GradientFuncs.
 */
void apply_metal_schedule_backward(GradientFuncs& g, int height, int width, int num_channels, int N);

}  // namespace tinysplat_halide

#endif  // TINYSPLAT_HALIDE_ALGORITHM_H
