#ifndef TINYSPLAT_HALIDE_SCHEDULE_METAL_H
#define TINYSPLAT_HALIDE_SCHEDULE_METAL_H

#include "algorithm.h"
#include <Halide.h>

namespace tinysplat_halide {

using namespace Halide;

inline void apply_metal_schedule_forward(ForwardPipeline& p,
                                         int height, int width,
                                         int num_channels) {
    Var x("x"), y("y"), c("c");
    Var xi("xi"), yi("yi"), xo("xo"), yo("yo");
    (void)height; (void)width; (void)num_channels;

    p.det.compute_root();
    p.norm_factor.compute_root();
    p.inv_cov.compute_root();

    p.output
        .tile(y, x, yo, xo, yi, xi, 16, 16, TailStrategy::RoundUp)
        .gpu_blocks(yo, xo)
        .gpu_threads(yi, xi);

    p.accum_weight
        .compute_at(p.output, xo)
        .gpu_threads(x);

    p.accum_color
        .compute_at(p.output, xo)
        .gpu_threads(x, c);
}

inline void apply_metal_schedule_backward(GradientPipeline& g,
                                          int height, int width,
                                          int num_channels) {
    (void)height; (void)width; (void)num_channels;

    g.det.compute_root();
    g.norm_factor.compute_root();
    g.inv_cov.compute_root();

    g.total_weight_pix.compute_root();
    g.total_color_pix.compute_root();
    g.output_norm.compute_root();

    g.grad_means.compute_root();
    g.grad_cov.compute_root();
    g.grad_colors.compute_root();
    g.grad_opacities.compute_root();
}

}  // namespace tinysplat_halide

#endif  // TINYSPLAT_HALIDE_SCHEDULE_METAL_H
