/**
 * pipeline.cpp
 *
 * JIT-compiled forward + backward pipelines for 2D Gaussian splatting.
 *
 * Key performance feature: pipelines are built and JIT-compiled ONCE per
 * (H, W, C) configuration. Subsequent calls with the same dimensions
 * rebind input buffers and re-execute without recompilation.
 * N (number of Gaussians) is a runtime parameter via ImageParam extents.
 *
 * Build:
 *   HL_PATH=/path/to/halide make
 *   Produces: libtinysplat_halide_pipeline.so
 */

#include <Halide.h>
#include <HalideRuntime.h>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include "algorithm.h"

using namespace Halide;

namespace {

// -------------------------------------------------------------------------
// Target detection
// -------------------------------------------------------------------------

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
            Target t = get_host_target();
            t.set_feature(Target::CUDA);
            return t;
        }
        case TargetBackend::Metal: {
            Target t;
            t.set_feature(Target::Metal);
            return t;
        }
        default:
            return get_host_target();
    }
}

// -------------------------------------------------------------------------
// Pipeline cache: avoid JIT recompilation on every call
// -------------------------------------------------------------------------

struct ForwardCache {
    std::unique_ptr<tinysplat_halide::ForwardPipeline> pipeline;
    int H = 0, W = 0, C = 0;
};

struct BackwardCache {
    std::unique_ptr<tinysplat_halide::GradientPipeline> pipeline;
    int H = 0, W = 0, C = 0;
};

static ForwardCache  g_fwd_cache;
static BackwardCache g_bwd_cache;

tinysplat_halide::ForwardPipeline&
ensure_forward_pipeline(int H, int W, int C, const Target& target) {
    if (g_fwd_cache.pipeline &&
        g_fwd_cache.H == H && g_fwd_cache.W == W && g_fwd_cache.C == C) {
        return *g_fwd_cache.pipeline;
    }

    g_fwd_cache.pipeline = std::make_unique<tinysplat_halide::ForwardPipeline>();
    auto& p = *g_fwd_cache.pipeline;

    tinysplat_halide::build_forward(p, H, W, C);

    TargetBackend backend = get_target_backend();
    if (backend == TargetBackend::CUDA) {
        tinysplat_halide::apply_cuda_schedule_forward(p, H, W, C);
    } else if (backend == TargetBackend::Metal) {
        tinysplat_halide::apply_metal_schedule_forward(p, H, W, C);
    } else {
        tinysplat_halide::apply_cpu_schedule_forward(p, H, W, C);
    }

    p.output.compile_jit(target);

    g_fwd_cache.H = H;
    g_fwd_cache.W = W;
    g_fwd_cache.C = C;
    return p;
}

tinysplat_halide::GradientPipeline&
ensure_backward_pipeline(int H, int W, int C, const Target& target) {
    if (g_bwd_cache.pipeline &&
        g_bwd_cache.H == H && g_bwd_cache.W == W && g_bwd_cache.C == C) {
        return *g_bwd_cache.pipeline;
    }

    g_bwd_cache.pipeline = std::make_unique<tinysplat_halide::GradientPipeline>();
    auto& g = *g_bwd_cache.pipeline;

    tinysplat_halide::build_backward(g, H, W, C);

    TargetBackend backend = get_target_backend();
    if (backend == TargetBackend::CUDA) {
        tinysplat_halide::apply_cuda_schedule_backward(g, H, W, C);
    } else if (backend == TargetBackend::Metal) {
        tinysplat_halide::apply_metal_schedule_backward(g, H, W, C);
    } else {
        tinysplat_halide::apply_cpu_schedule_backward(g, H, W, C);
    }

    g.grad_means.compile_jit(target);
    g.grad_cov.compile_jit(target);
    g.grad_colors.compile_jit(target);
    g.grad_opacities.compile_jit(target);

    g_bwd_cache.H = H;
    g_bwd_cache.W = W;
    g_bwd_cache.C = C;
    return g;
}

}  // anonymous namespace


extern "C" {

// -------------------------------------------------------------------------
// Forward pass: C API
// -------------------------------------------------------------------------

int gaussian_splat_forward(float* means, float* covariances, float* colors,
                           float* opacities, int N, int height, int width,
                           int C, float* output) {
    try {
        Target target = get_halide_target();
        auto& p = ensure_forward_pipeline(height, width, C, target);

        Buffer<float> means_buf(means, {N, 2});
        Buffer<float> cov_buf(covariances, {N, 2, 2});
        Buffer<float> colors_buf(colors, {N, C});
        Buffer<float> opacities_buf(opacities, {N});
        Buffer<float> output_buf(output, {height, width, C});

        means_buf.set_host_dirty(true);
        cov_buf.set_host_dirty(true);
        colors_buf.set_host_dirty(true);
        opacities_buf.set_host_dirty(true);

        p.means_ip.set(means_buf);
        p.cov_ip.set(cov_buf);
        p.colors_ip.set(colors_buf);
        p.opacities_ip.set(opacities_buf);

        p.output.realize(output_buf, target);
        output_buf.copy_to_host();
        return 0;

    } catch (const Halide::Error& e) {
        fprintf(stderr, "Halide forward error: %s\n", e.what());
        return -1;
    } catch (const std::exception& e) {
        fprintf(stderr, "std forward error: %s\n", e.what());
        return -1;
    } catch (...) {
        fprintf(stderr, "Unknown forward error\n");
        return -1;
    }
}


// -------------------------------------------------------------------------
// Backward pass: C API
// -------------------------------------------------------------------------

int gaussian_splat_backward(float* grad_output, float* means,
                            float* covariances, float* colors,
                            float* opacities, int N, int height,
                            int width, int C,
                            float* grad_means, float* grad_cov,
                            float* grad_colors, float* grad_opacities) {
    try {
        Target target = get_halide_target();
        auto& g = ensure_backward_pipeline(height, width, C, target);

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
        means_buf.set_host_dirty(true);
        cov_buf.set_host_dirty(true);
        colors_buf.set_host_dirty(true);
        opacities_buf.set_host_dirty(true);

        g.grad_output_ip.set(grad_out_buf);
        g.means_ip.set(means_buf);
        g.cov_ip.set(cov_buf);
        g.colors_ip.set(colors_buf);
        g.opacities_ip.set(opacities_buf);

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
