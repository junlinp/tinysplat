#ifndef TINYSPLAT_HALIDE_SCHEDULE_CPU_H
#define TINYSPLAT_HALIDE_SCHEDULE_CPU_H

#include "algorithm.h"
#include <Halide.h>

namespace tinysplat_halide {

using namespace Halide;

// -------------------------------------------------------------------------
// Forward CPU schedule
// -------------------------------------------------------------------------

inline void apply_cpu_schedule_forward(ForwardPipeline& p,
                                       int height,
                                       int width,
                                       int num_channels) {
    Var x("x"), y("y"), c("c"), n("n");
    Var xi("xi"), yi("yi"), xo("xo"), yo("yo");
    (void)height; (void)width; (void)num_channels;

    const int vec = 8;  // SIMD width for AVX2

    // Per-Gaussian invariants: compute ONCE for all Gaussians.
    // Without this they are inlined and recomputed per pixel*Gaussian.
    p.det.compute_root().vectorize(n, vec, TailStrategy::GuardWithIf);
    p.norm_factor.compute_root().vectorize(n, vec, TailStrategy::GuardWithIf);
    p.inv_cov.compute_root().vectorize(n, vec, TailStrategy::GuardWithIf);
    p.inv_cov.update(0).unscheduled();
    p.inv_cov.update(1).unscheduled();
    p.inv_cov.update(2).unscheduled();
    p.inv_cov.update(3).unscheduled();

    // Output: tile (y,x), vectorize inner x, parallelize outer y tiles
    p.output
        .tile(y, x, yo, xo, yi, xi, 32, 32, TailStrategy::GuardWithIf)
        .vectorize(xi, vec, TailStrategy::GuardWithIf)
        .parallel(yo);

    // Accumulations: compute within each yo tile strip for locality
    p.accum_weight
        .compute_at(p.output, yo)
        .vectorize(x, vec, TailStrategy::GuardWithIf);
    p.accum_weight.update(0)
        .vectorize(x, vec, TailStrategy::GuardWithIf);

    p.accum_color
        .compute_at(p.output, yo)
        .vectorize(x, vec, TailStrategy::GuardWithIf);
    p.accum_color.update(0)
        .vectorize(x, vec, TailStrategy::GuardWithIf);
}

// -------------------------------------------------------------------------
// Backward CPU schedule
// -------------------------------------------------------------------------

inline void apply_cpu_schedule_backward(GradientPipeline& g,
                                        int height,
                                        int width,
                                        int num_channels) {
    Var x("x"), y("y"), c("c"), n("n");
    (void)height; (void)width; (void)num_channels;

    const int vec = 8;  // SIMD width for AVX2

    // Per-Gaussian invariants
    g.det.compute_root().vectorize(n, vec, TailStrategy::GuardWithIf);
    g.norm_factor.compute_root().vectorize(n, vec, TailStrategy::GuardWithIf);
    g.inv_cov.compute_root().vectorize(n, vec, TailStrategy::GuardWithIf);
    g.inv_cov.update(0).unscheduled();
    g.inv_cov.update(1).unscheduled();
    g.inv_cov.update(2).unscheduled();
    g.inv_cov.update(3).unscheduled();

    // Forward-recomputed pixel totals: needed by all gradient outputs.
    // compute_root so they are not redundantly recomputed per gradient output.
    g.total_weight_pix.compute_root()
        .vectorize(x, vec, TailStrategy::GuardWithIf)
        .parallel(y);
    g.total_weight_pix.update(0)
        .vectorize(x, vec, TailStrategy::GuardWithIf)
        .parallel(y);

    g.total_color_pix.compute_root()
        .vectorize(x, vec, TailStrategy::GuardWithIf)
        .parallel(y);
    g.total_color_pix.update(0)
        .vectorize(x, vec, TailStrategy::GuardWithIf)
        .parallel(y);

    g.output_norm.compute_root()
        .vectorize(x, vec, TailStrategy::GuardWithIf)
        .parallel(y);

    // Gradient outputs: parallelize over n
    g.grad_colors.compute_root()
        .vectorize(c, std::min(vec, num_channels), TailStrategy::GuardWithIf)
        .parallel(n);
    g.grad_colors.update(0).unscheduled();

    g.grad_opacities.compute_root()
        .parallel(n);
    g.grad_opacities.update(0).unscheduled();

    g.grad_means.compute_root()
        .parallel(n);

    g.grad_cov.compute_root()
        .parallel(n);
    g.grad_cov.update(0).unscheduled();
    g.grad_cov.update(1).unscheduled();
    g.grad_cov.update(2).unscheduled();
    g.grad_cov.update(3).unscheduled();
}

}  // namespace tinysplat_halide

#endif  // TINYSPLAT_HALIDE_SCHEDULE_CPU_H
