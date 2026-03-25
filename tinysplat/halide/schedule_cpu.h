#ifndef TINYSPLAT_HALIDE_SCHEDULE_CPU_H
#define TINYSPLAT_HALIDE_SCHEDULE_CPU_H

/**
 * CPU Schedule for TinySplat Halide Pipeline
 * ==========================================
 * Target: x86-64 Linux with AVX2 + FMA + SSE41.
 *
 * Key decisions:
 *   - Tile the (x,y) canvas into 32×32 tiles, vectorised on inner dim.
 *   - Parallelise over tiles using ThreadPool (Halide's default).
 *   - Inner Gaussian loop is unrolled manually in 64-Gaussian chunks
 *     (avoids large RDom overhead; keeps registers tight).
 *   - Use fma for the dot products.
 */

#include <Halide.h>

namespace tinysplat_halide {
namespace schedule {

using namespace Halide;

inline void apply_cpu_schedule(Func& forward,    Func& accum_color,
                               Func& accum_weight,
                               int height, int width) {
    Var x("x"), y("y"), c("c"), n("n"), xi("xi"), yi("yi"), xo("xo"), yo("yo");

    // Tile the canvas 32×32, vectorise the inner x dimension (8× float32 SIMD)
    forward.compute_root()
        .tile(x, y, xo, yo, xi, yi, 32, 32, TailStrategy::RoundUp)
        .vectorize(xi, 8, TailStrategy::RoundUp)
        .parallel(yo);

    // accum_color and accum_weight: same tiling, fused with forward
    accum_color
        .compute_at(forward, xo)
        .tile(x, y, xi, yi, 8, 4)
        .vectorize(xi, 8)
        .unroll(yi);

    accum_weight
        .compute_at(forward, xo)
        .tile(x, y, xi, yi, 8, 4)
        .vectorize(xi, 8)
        .unroll(yi);

    // Outer tile parallel, inner vectorised
    // The RDom over Gaussians is already inside compute_at(forward, xo)
    // so it tiles across canvas naturally.
}

}  // namespace schedule
}  // namespace tinysplat_halide

#endif  // TINYSPLAT_HALIDE_SCHEDULE_CPU_H
