#ifndef TINYSPLAT_HALIDE_SCHEDULE_CUDA_H
#define TINYSPLAT_HALIDE_SCHEDULE_CUDA_H

#include <Halide.h>

namespace tinysplat_halide {
namespace schedule {

using namespace Halide;

/**
 * Apply CUDA schedule to forward pipeline.
 * Pipeline uses (y, x, c) convention.
 */
inline void apply_cuda_schedule(Func& output,
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
 * Apply CUDA schedule to backward pipeline.
 */
inline void apply_cuda_schedule_backward(Func& grad_means,
                                         Func& grad_cov,
                                         Func& grad_colors,
                                         Func& grad_opacities,
                                         int height, int width,
                                         int num_channels, int N) {
    (void)height; (void)width; (void)num_channels;

    grad_means.compute_root().gpu_blocks(N > 0 ? Var("n") : Var("x"));
    grad_cov.compute_root().gpu_blocks(N > 0 ? Var("n") : Var("x"));
    grad_colors.compute_root().gpu_blocks(N > 0 ? Var("n") : Var("x"));
    grad_opacities.compute_root().gpu_blocks(N > 0 ? Var("n") : Var("x"));
}

}  // namespace schedule
}  // namespace tinysplat_halide

#endif  // TINYSPLAT_HALIDE_SCHEDULE_CUDA_H
