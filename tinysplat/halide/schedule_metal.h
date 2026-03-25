#ifndef TINYSPLAT_HALIDE_SCHEDULE_METAL_H
#define TINYSPLAT_HALIDE_SCHEDULE_METAL_H

/**
 * Metal / MPS Schedule for TinySplat Halide Pipeline
 * ==================================================
 * Target: Apple Metal GPU (MPS backend on macOS)
 *
 * Forward schedule:
 *   - Threadgroup: 16×16 threads covering a pixel tile.
 *   - Threadgrid: full canvas split into tiles.
 */

#include <Halide.h>

namespace tinysplat_halide {
namespace schedule {

using namespace Halide;

/**
 * Apply Metal schedule to forward pipeline Funcs.
 * Metal gpu_tile maps to threadgroup scheduling.
 */
inline void apply_metal_schedule(Func& output,
                                Func& accum_color,
                                Func& accum_weight,
                                int height,
                                int width,
                                int num_channels) {
    Var x("x"), y("y"), c("c"), xi("xi"), yi("yi");
    (void)num_channels;

    // ---- Forward output ----
    output
        .tile(x, y, xi, yi, 16, 16)
        .reorder(xi, yi, c, x, y)
        .gpu_blocks(x, y)
        .gpu_threads(xi, yi);

    // ---- Accumulation ----
    accum_weight
        .tile(x, y, xi, yi, 16, 16)
        .gpu_blocks(x, y)
        .gpu_threads(xi, yi);

    accum_color
        .tile(x, y, xi, yi, 16, 16)
        .reorder(xi, yi, c, x, y)
        .gpu_blocks(x, y)
        .gpu_threads(xi, yi);
}

/**
 * Apply Metal schedule to backward pipeline Funcs.
 */
inline void apply_metal_schedule_backward(Func& grad_means,
                                        Func& grad_cov,
                                        Func& grad_colors,
                                        Func& grad_opacities,
                                        int height,
                                        int width,
                                        int num_channels,
                                        int N) {
    Var n("n"), c("c");
    (void)height; (void)width; (void)num_channels; (void)N;

    // grad_colors: (n, c)
    grad_colors
        .gpu_blocks(n)
        .gpu_threads(n);

    // grad_opacities: (n,)
    grad_opacities
        .gpu_blocks(n)
        .gpu_threads(n);

    // grad_means: (n, 2)
    grad_means
        .gpu_blocks(n)
        .gpu_threads(n);
}

}  // namespace schedule
}  // namespace tinysplat_halide

#endif  // TINYSPLAT_HALIDE_SCHEDULE_METAL_H
