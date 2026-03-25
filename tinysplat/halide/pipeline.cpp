/**
 * pipeline.cpp
 *
 * JIT-compiled forward + backward pipelines for 2D Gaussian splatting.
 * Exposes a C API callable from Python via ctypes.
 *
 * Build:
 *   HL_PATH=/path/to/halide
 *   mkdir build && cd build
 *   cmake -DHL_PATH=$HL_PATH ..
 *   make
 *
 *   Produces: libtinysplat_halide_pipeline.so
 */

#include <Halide.h>
#include <HalideRuntime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "algorithm.h"
#include "schedule_cpu.h"

using namespace Halide;

extern "C" {

// ---------------------------------------------------------------------------
// Forward pass — C API
// ---------------------------------------------------------------------------

/**
 * gaussian_splat_forward(
 *     float* means,           // (N, 2)
 *     float* covariances,     // (N, 2, 2)
 *     float* colors,          // (N, C)
 *     float* opacities,       // (N,)
 *     int N, int height, int width, int C,
 *     float* output           // (height, width, C) — output
 * )
 *
 * Returns 0 on success, non-zero on error.
 */
int gaussian_splat_forward(float* means, float* covariances, float* colors,
                           float* opacities, int N, int height, int width,
                           int C, float* output) {
    try {
        // Wrap raw pointers as Halide buffers (host-side, not device)
        Halide::Buffer<float> means_buf(means, {N, 2});
        Halide::Buffer<float> cov_buf(covariances, {N, 2, 2});
        Halide::Buffer<float> colors_buf(colors, {N, C});
        Halide::Buffer<float> opacities_buf(opacities, {N});
        Halide::Buffer<float> output_buf(output, {height, width, C});

        means_buf.set_host_dirty(true);
        cov_buf.set_host_dirty(true);
        colors_buf.set_host_dirty(true);
        opacities_buf.set_host_dirty(true);

        // Build and JIT-compile
        auto [output_func, intermediates] =
            tinysplat_halide::build_forward_pipeline(
                means_buf, cov_buf, colors_buf, opacities_buf,
                height, width, C);

        // Apply CPU schedule
        Var x("x"), y("y"), c("c");
        Func accum_color, accum_weight;
        // Note: accum_color and accum_weight are internal to algorithm.cpp
        // We need to expose them or apply schedule differently.
        // For now, just bound the output and let Halide auto-schedule.
        output_func.bound(x, 0, width);
        output_func.bound(y, 0, height);
        output_func.bound(c, 0, C);

        // Realize to output buffer
        output_func.realize(output_buf);

        output_buf.copy_to_host();
        return 0;
    } catch (const Halide::Error& e) {
        fprintf(stderr, "Halide error: %s\n", e.what());
        return -1;
    } catch (const std::exception& e) {
        fprintf(stderr, "std error: %s\n", e.what());
        return -1;
    } catch (...) {
        fprintf(stderr, "Unknown error\n");
        return -1;
    }
}

// ---------------------------------------------------------------------------
// Backward pass — C API
// ---------------------------------------------------------------------------

/**
 * gaussian_splat_backward(
 *     float* grad_output,      // (height, width, C)
 *     float* means,           // (N, 2)
 *     float* covariances,     // (N, 2, 2)
 *     float* colors,          // (N, C)
 *     float* opacities,       // (N,)
 *     int N, int height, int width, int C,
 *     float* grad_means,      // (N, 2)      — output
 *     float* grad_cov,        // (N, 2, 2)   — output
 *     float* grad_colors,     // (N, C)      — output
 *     float* grad_opacities  // (N,)        — output
 * )
 *
 * Returns 0 on success, non-zero on error.
 */
int gaussian_splat_backward(float* grad_output, float* means,
                            float* covariances, float* colors,
                            float* opacities, int N, int height,
                            int width, int C,
                            float* grad_means, float* grad_cov,
                            float* grad_colors, float* grad_opacities) {
    // Phase 2 implementation — for now use PyTorch fallback.
    // Gradient computation not yet wired in Halide.
    fprintf(stderr, "Backward not yet implemented in Halide backend — "
                    "falling back to PyTorch.\n");
    return -1;
}

}  // extern "C"
