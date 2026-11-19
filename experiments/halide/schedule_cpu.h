#ifndef TINYSPLAT_HALIDE_SCHEDULE_CPU_H
#define TINYSPLAT_HALIDE_SCHEDULE_CPU_H

#include <Halide.h>

namespace tinysplat_halide {
namespace schedule {

using namespace Halide;

/**
 * Apply CPU schedule to forward pipeline.
 * Pipeline uses (y, x, c) convention:
 *   y = row (height), x = column (width), c = channel
 */
inline void apply_cpu_schedule(Func& output,
                               Func& accum_color,
                               Func& accum_weight,
                               int height,
                               int width,
                               int num_channels) {
    Var x("x"), y("y"), c("c"), xi("xi"), yi("yi"), xo("xo"), yo("yo");
    (void)height; (void)width; (void)num_channels;

    // output(y, x, c): tile y,x → vectorize x → parallel yo
    output
        .tile(y, x, yo, xo, yi, xi, 32, 32, TailStrategy::RoundUp)
        .vectorize(xi, 8, TailStrategy::RoundUp)
        .parallel(yo);

    // accum_weight(y, x): compute at output's yi
    accum_weight
        .compute_at(output, yi)
        .vectorize(x, 8);

    // accum_color(y, x, c): compute at output's yi
    accum_color
        .compute_at(output, yi)
        .vectorize(x, 8);
}

}  // namespace schedule
}  // namespace tinysplat_halide

#endif  // TINYSPLAT_HALIDE_SCHEDULE_CPU_H
