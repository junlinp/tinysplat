#ifndef TINYSPLAT_HALIDE_SCHEDULE_CUDA_H
#define TINYSPLAT_HALIDE_SCHEDULE_CUDA_H

/**
 * CUDA Schedule for TinySplat Halide Pipeline
 * =========================================
 * Target: CUDA GPU (sm_60+ / Pascal and later)
 *
 * Forward schedule:
 *   - Tile canvas 16×16 per block.
 *   - Thread-level: each thread computes its pixel's contribution.
 *   - Cooperative groups for block-level synchronization.
 */

#include <Halide.h>

namespace tinysplat_halide {
namespace schedule {

using namespace Halide;

/**
 * Apply CUDA schedule to forward pipeline Funcs.
 * Block: 16×16 threads covering a 16×16 pixel tile.
 * Grid: ceil(width/16) × ceil(height/16) blocks.
 */
inline void apply_cuda_schedule(Func& output,
                                Func& accum_color,
                                Func& accum_weight,
                                int height,
                                int width,
                                int num_channels) {
    Var x("x"), y("y"), c("c"), xi("xi"), yi("yi"), xo("xo"), yo("yo");
    (void)num_channels;

    // ---- output: split canvas into 16×16 tiles ----
    output
        .tile(x, y, xo, yo, xi, yi, 16, 16)
        .reorder(xi, yi, c, xo, yo)
        .gpu_blocks(xo, yo)
        .gpu_threads(xi, yi);

    // ---- accum_weight: per-tile ----
    accum_weight
        .compute_at(output, xo)
        .tile(x, y, xi, yi, 16, 16)
        .gpu_threads(xi, yi);

    // ---- accum_color ----
    accum_color
        .compute_at(output, xo)
        .tile(x, y, xi, yi, 16, 16)
        .reorder(xi, yi, c)
        .gpu_threads(xi, yi);
}

/**
 * Apply CUDA schedule to backward pipeline Funcs.
 * Uses gpu_blocks/gpu_threads without complex tiling for reduction Funcs.
 */
inline void apply_cuda_schedule_backward(Func& grad_means,
                                         Func& grad_cov,
                                         Func& grad_colors,
                                         Func& grad_opacities,
                                         int height,
                                         int width,
                                         int num_channels,
                                         int N) {
    Var n("n"), c("c"), r("r");
    (void)height; (void)width; (void)num_channels; (void)N;

    // grad_colors: (n, c) — map n to blocks, c to threads
    grad_colors
        .split(n, n, c, 256)
        .gpu_blocks(n)
        .gpu_threads(c);

    // grad_opacities: (n,) — just parallelize over n
    grad_opacities
        .gpu_blocks(n)
        .gpu_threads(n);

    // grad_means: (n, c=2) — map n to blocks, c to threads
    grad_means
        .split(n, n, c, 256)
        .gpu_blocks(n)
        .gpu_threads(c);
}

}  // namespace schedule
}  // namespace tinysplat_halide

#endif  // TINYSPLAT_HALIDE_SCHEDULE_CUDA_H
