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
 *   - Parallelise over outer tile y.
 *   - Accumulation (accum_weight, accum_color): compute at innermost xi level
 *     so each thread computes its tile's accumulation fully.
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

    // ---- accum_weight: compute at innermost xi to fuse into tile ----
    // accum_weight(x, y) computes Σ_n weight(x, y, n)
    // by computing at xi, each thread's accum is computed within its tile
    accum_weight
        .compute_at(output, xi)
        .vectorize(x, 8);

    // ---- accum_color: same, but with channel dim ----
    accum_color
        .compute_at(output, xi)
        .vectorize(x, 8);
}

}  // namespace schedule
}  // namespace tinysplat_halide

#endif  // TINYSPLAT_HALIDE_SCHEDULE_CPU_H
