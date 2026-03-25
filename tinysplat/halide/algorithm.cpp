/**
 * algorithm.cpp
 *
 * Halide forward + backward pipelines for 2D Gaussian splatting.
 *
 * Key design:
 *   Forward:   Σᵢ αᵢ·cᵢ / Σᵢ αᵢ   (alpha-composite N Gaussians)
 *   Backward: analytical gradients (Phase 2; Phase 1 falls back to PyTorch)
 */

#include "algorithm.h"
#include <Halide.h>
#include <vector>

using Halide::Buffer;
using Halide::cast;
using Halide::Expr;
using Halide::Func;
using Halide::RDom;
using Halide::TailStrategy;
using Halide::Var;
using Halide::_;
using Halide::sum;

namespace tinysplat_halide {
namespace {

constexpr float kPi  = 3.14159265358979323846f;
constexpr float kEps = 1e-8f;

constexpr int x_bufsize = 512;   // internal scratch for forward


// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

std::pair<Func, std::vector<Buffer<float>>>
build_forward_pipeline_impl(const Buffer<float>& means_in,
                            const Buffer<float>& covariances_in,
                            const Buffer<float>& colors_in,
                            const Buffer<float>& opacities_in,
                            int height,
                            int width,
                            int num_channels) {
    // ------------------------------------------------------------------
    // Halide variables
    // ------------------------------------------------------------------
    Var x("x"), y("y"), c("c"), n("n"), r("r");

    // ------------------------------------------------------------------
    // 1. Wrap input Buffers as Funcs
    // ------------------------------------------------------------------
    Func means_f("means_f"), cov_f("cov_f"), colors_f("colors_f"),
         opacities_f("opacities_f");

    means_f(n, _)   = means_in(n, _);
    cov_f(n, _, _)  = covariances_in(n, _, _);
    colors_f(n, c)  = colors_in(n, c);
    opacities_f(n)  = opacities_in(n);

    // ------------------------------------------------------------------
    // 2. Per-Gaussian constants
    // ------------------------------------------------------------------
    //   determinant = a*d - b*b  for 2x2 symmetric matrix [[a,b],[b,d]]
    Func det("det");
    det(n)  = cov_f(n, 0, 0) * cov_f(n, 1, 1)
            - cov_f(n, 0, 1) * cov_f(n, 0, 1);

    //   normalization = 1 / (2π × √det)
    Func norm_factor("norm_factor");
    norm_factor(n) = 1.0f / (2.0f * kPi * sqrt(max(det(n), kEps)));

    //   inverse 2x2 matrix: inv([[a,b],[b,d]]) = [[d,-b],[-b,a]] / det
    Func inv_cov("inv_cov");
    inv_cov(n, c, r) = cast<float>(0.0f);
    inv_cov(n, 0, 0) =  cov_f(n, 1, 1) / max(det(n), kEps);
    inv_cov(n, 1, 1) =  cov_f(n, 0, 0) / max(det(n), kEps);
    inv_cov(n, 0, 1) = -cov_f(n, 0, 1) / max(det(n), kEps);
    inv_cov(n, 1, 0) = -cov_f(n, 0, 1) / max(det(n), kEps);

    // ------------------------------------------------------------------
    // 3. Pixel coordinate grids (1D → 2D via broadcasting)
    // ------------------------------------------------------------------
    Func px("px"), py("py");
    px(x, y, n) = cast<float>(x);
    py(x, y, n) = cast<float>(y);

    // ------------------------------------------------------------------
    // 4. Mahalanobis + weight
    // ------------------------------------------------------------------
    //   d = [x, y] - means[n]
    //   mahal = dᵀ × inv_cov × d
    //   weight = opacities[n] × norm[n] × exp(-0.5 × mahal)
    Func dx("dx"), dy("dy"), mahal("mahal"), weight("weight");
    dx(x, y, n)    = px(x, y, n) - means_f(n, 0);
    dy(x, y, n)    = py(x, y, n) - means_f(n, 1);

    mahal(x, y, n) = dx(x, y, n) * (inv_cov(n, 0, 0) * dx(x, y, n)
                                  + inv_cov(n, 0, 1) * dy(x, y, n))
                   + dy(x, y, n) * (inv_cov(n, 1, 0) * dx(x, y, n)
                                  + inv_cov(n, 1, 1) * dy(x, y, n));

    weight(x, y, n) = opacities_f(n)
                    * norm_factor(n)
                    * exp(-0.5f * mahal(x, y, n));

    // ------------------------------------------------------------------
    // 5. Accumulate over Gaussians
    // ------------------------------------------------------------------
    RDom r_n(0, means_in.dim(0).extent(), "r_n");  // iterate N Gaussians

    Func accum_color("accum_color"), accum_weight("accum_weight");
    accum_color(x, y, c) = cast<float>(0.0f);
    accum_weight(x, y)   = cast<float>(0.0f);

    accum_color(x, y, c) += weight(x, y, r_n) * colors_f(r_n, c);
    accum_weight(x, y)   += weight(x, y, r_n);

    // ------------------------------------------------------------------
    // 6. Normalise + alpha-composite
    // ------------------------------------------------------------------
    Func output("forward_output");
    output(x, y, c) = accum_color(x, y, c)
                    / max(accum_weight(x, y), kEps);

    // ------------------------------------------------------------------
    // 7. Bounds
    // ------------------------------------------------------------------
    output.bound(x, 0, width);
    output.bound(y, 0, height);
    output.bound(c, 0, num_channels);

    std::vector<Buffer<float>> intermediates;
    return {output, intermediates};
}


// ---------------------------------------------------------------------------
// Backward pass — Phase 1 stub (returns empty, PyTorch fallback used)
// ---------------------------------------------------------------------------

std::vector<Buffer<float>>
build_backward_pipeline_impl(const Buffer<float>& grad_output_in,
                              const Buffer<float>& means_in,
                              const Buffer<float>& covariances_in,
                              const Buffer<float>& colors_in,
                              const Buffer<float>& opacities_in,
                              int height,
                              int width,
                              int num_channels) {
    // Phase 1: backward is handled by the Python PyTorch fallback.
    // Phase 2 will implement proper analytical Halide gradients.
    return {};
}

}  // anonymous namespace


// ---------------------------------------------------------------------------
// Public wrappers
// ---------------------------------------------------------------------------

std::pair<Func, std::vector<Buffer<float>>>
build_forward_pipeline(const Buffer<float>& means,
                       const Buffer<float>& covariances,
                       const Buffer<float>& colors,
                       const Buffer<float>& opacities,
                       int height,
                       int width,
                       int num_channels) {
    return build_forward_pipeline_impl(
        means, covariances, colors, opacities, height, width, num_channels);
}


std::vector<Buffer<float>>
build_backward_pipeline(const Buffer<float>& grad_output,
                         const Buffer<float>& means,
                         const Buffer<float>& covariances,
                         const Buffer<float>& colors,
                         const Buffer<float>& opacities,
                         int height,
                         int width,
                         int num_channels) {
    return build_backward_pipeline_impl(
        grad_output, means, covariances, colors, opacities,
        height, width, num_channels);
}

}  // namespace tinysplat_halide
