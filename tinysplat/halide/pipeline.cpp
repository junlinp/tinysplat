/**
 * pipeline.cpp
 *
 * JIT-compiled forward + backward pipelines for 2D Gaussian splatting.
 * Applies target-specific schedules (CPU/CUDA/Metal) before realize.
 * Exposes a C API callable from Python via ctypes.
 *
 * Build:
 *   HL_PATH=/path/to/halide make
 *
 *   Produces: libtinysplat_halide_pipeline.so
 */

#include <Halide.h>
#include <HalideRuntime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "algorithm.h"

using namespace Halide;

namespace {

// Detect target from environment
enum class TargetBackend { Auto, CPU, CUDA, Metal };

TargetBackend get_target_backend() {
    const char* env = getenv("HL_TARGET");
    if (!env) return TargetBackend::Auto;

    std::string target(env);
    if (target.find("cuda") != std::string::npos) return TargetBackend::CUDA;
    if (target.find("metal") != std::string::npos) return TargetBackend::Metal;
    return TargetBackend::CPU;
}

Target get_halide_target() {
    TargetBackend backend = get_target_backend();

    switch (backend) {
        case TargetBackend::CUDA: {
            // Try CUDA target, fall back to host if unavailable
            Target t = get_host_target();
            t.set_feature(Target::CUDA);
            return t;
        }
        case TargetBackend::Metal: {
            Target t;
            t.set_feature(Target::Metal);
            return t;
        }
        case TargetBackend::CPU:
        case TargetBackend::Auto:
        default:
            return get_host_target();
    }
}

}  // anonymous namespace


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
        Buffer<float> means_buf(means, {N, 2});
        Buffer<float> cov_buf(covariances, {N, 2, 2});
        Buffer<float> colors_buf(colors, {N, C});
        Buffer<float> opacities_buf(opacities, {N});
        Buffer<float> output_buf(output, {width, height, C});

        means_buf.set_host_dirty(true);
        cov_buf.set_host_dirty(true);
        colors_buf.set_host_dirty(true);
        opacities_buf.set_host_dirty(true);

        // Build pipeline
        auto p = tinysplat_halide::build_forward_pipeline(
            means_buf, cov_buf, colors_buf, opacities_buf,
            height, width, C);

        // Apply target-specific schedule
        Target target = get_halide_target();
        TargetBackend backend = get_target_backend();

        if (backend == TargetBackend::CUDA) {
            tinysplat_halide::apply_cuda_schedule_forward(p, height, width, C);
        } else if (backend == TargetBackend::Metal) {
            tinysplat_halide::apply_metal_schedule_forward(p, height, width, C);
        } else {
            // CPU schedule or auto
            tinysplat_halide::apply_cpu_schedule_forward(p, height, width, C);
        }

        // Realize
        p.output.realize(output_buf, target);
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
    try {
        Buffer<float> grad_out_buf(grad_output, {height, width, C});
        Buffer<float> means_buf(means, {N, 2});
        Buffer<float> cov_buf(covariances, {N, 2, 2});
        Buffer<float> colors_buf(colors, {N, C});
        Buffer<float> opacities_buf(opacities, {N});

        Buffer<float> grad_means_buf(grad_means, {N, 2});
        Buffer<float> grad_cov_buf(grad_cov, {N, 2, 2});
        Buffer<float> grad_colors_buf(grad_colors, {N, C});
        Buffer<float> grad_opacities_buf(grad_opacities, {N});

        grad_out_buf.set_host_dirty(true);

        // Build backward pipeline
        auto g = tinysplat_halide::build_backward_pipeline(
            grad_out_buf, means_buf, cov_buf, colors_buf, opacities_buf,
            height, width, C);

        Target target = get_halide_target();
        TargetBackend backend = get_target_backend();

        if (backend == TargetBackend::CUDA) {
            tinysplat_halide::apply_cuda_schedule_backward(
                g, height, width, C, N);
        } else if (backend == TargetBackend::Metal) {
            tinysplat_halide::apply_metal_schedule_backward(
                g, height, width, C, N);
        } else {
            tinysplat_halide::apply_cpu_schedule_backward(
                g, height, width, C, N);
        }

        // Realize gradients
        g.grad_means.realize(grad_means_buf, target);
        g.grad_cov.realize(grad_cov_buf, target);
        g.grad_colors.realize(grad_colors_buf, target);
        g.grad_opacities.realize(grad_opacities_buf, target);

        grad_means_buf.copy_to_host();
        grad_cov_buf.copy_to_host();
        grad_colors_buf.copy_to_host();
        grad_opacities_buf.copy_to_host();

        return 0;

    } catch (const Halide::Error& e) {
        fprintf(stderr, "Halide backward error: %s\n", e.what());
        return -1;
    } catch (const std::exception& e) {
        fprintf(stderr, "std backward error: %s\n", e.what());
        return -1;
    } catch (...) {
        fprintf(stderr, "Unknown backward error\n");
        return -1;
    }
}

}  // extern "C"
