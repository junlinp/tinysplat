#ifndef TINYSPLAT_HALIDE_SCHEDULE_METAL_H
#define TINYSPLAT_HALIDE_SCHEDULE_METAL_H

#include <Halide.h>

namespace tinysplat_halide {
namespace schedule {

using namespace Halide;

/**
 * Apply Metal schedule to forward pipeline.
 * Pipeline uses (y, x, c) convention.
 */
inline void apply_metal_schedule(Func& output,
                                 Func& accum_color,
                                 Func& accum_weight,
                                 int height,
                                 int width,
                                 int num_channels) {
    Var x("x"), y("y"), c("c"), xi("xi"), yi("yi"), xo("xo"), yo("yo");
    (void)height; (void)width; (void)num_channels;

    output
        .tile(y, x, yo, xo, yi, xi, 16, 16, TailStrategy::RoundUp)
        .gpu_blocks(yo, xo)
        .gpu_threads(yi, xi);

    accum_weight
        .compute_at(output, xo)
        .gpu_threads(x);

    accum_color
        .compute_at(output, xo)
        .gpu_threads(x, c);
}

/**
 * Apply Metal schedule to backward pipeline.
 */
inline void apply_metal_schedule_backward(Func& grad_means,
                                          Func& grad_cov,
                                          Func& grad_colors,
                                          Func& grad_opacities,
                                          int height, int width,
                                          int num_channels, int N) {
    (void)height; (void)width; (void)num_channels; (void)N;

    grad_means.compute_root();
    grad_cov.compute_root();
    grad_colors.compute_root();
    grad_opacities.compute_root();
}

}  // namespace schedule
}  // namespace tinysplat_halide

#endif  // TINYSPLAT_HALIDE_SCHEDULE_METAL_H
