#ifndef TINYSPLAT_HALIDE_SCHEDULE_CPU_H
#define TINYSPLAT_HALIDE_SCHEDULE_CPU_H

/**
 * CPU Schedule for TinySplat Halide Pipeline
 * ==========================================
 * Target: x86-64 Linux with AVX2 + FMA + SSE41.
 *
 * Key decisions:
 *   - Tile the (x,y) canvas into 32×32 tiles.
 *   - Vectorise the inner x dimension (8-wide float32 SIMD).
 *   - Parallelise over tiles (Halide auto-parallelizes on CPU).
 *   - Inner Gaussian loop is fused into the tile computation.
 *
 * Schedule per Func:
 *   output:       root → tile(x,16,y,16) → vectorize(xi,8) → parallel(yo)
 *   accum_color:  computed at output tile → vectorize → unroll yi
 *   accum_weight: computed at output tile → vectorize
 */

#include <Halide.h>

namespace tinysplat_halide {
namespace schedule {

using namespace Halide;

/**
 * Apply CPU schedule to forward pipeline Funcs.
 */
inline void apply_cpu_schedule(Func& output,
                               Func& accum_color,
                               Func& accum_weight,
                               int height,
                               int width,
                               int num_channels) {
    Var x("x"), y("y"), c("c"), xi("xi"), yi("yi"), xo("xo"), yo("yo");

    (void)num_channels;

    // ---- output: tile → vectorize → parallel ----
    output
        .tile(x, y, xo, yo, xi, yi, 32, 32, TailStrategy::RoundUp)
        .vectorize(xi, 8, TailStrategy::RoundUp)
        .parallel(yo);

    // ---- accum_weight: same tiling, fused at output ----
    accum_weight
        .compute_at(output, xo)
        .tile(x, y, xi, yi, 8, 8)
        .vectorize(xi, 8)
        .unroll(yi);

    // ---- accum_color: same tiling with channel dim ----
    accum_color
        .compute_at(output, xo)
        .tile(x, y, xi, yi, 8, 4)
        .vectorize(xi, 8)
        .unroll(yi);
}

}  // namespace schedule
}  // namespace tinysplat_halide

#endif  // TINYSPLAT_HALIDE_SCHEDULE_CPU_H
