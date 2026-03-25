/**
 * algorithm.cpp
 *
 * Halide forward + backward pipelines for 2D Gaussian splatting.
 * Dimension convention: (y, x, c) to match PyTorch (H, W, C) row-major layout.
 *   - y: row index (height), maps to buffer dim(0)
 *   - x: column index (width), maps to buffer dim(1)
 *   - c: channel, maps to buffer dim(2)
 */

#include "algorithm.h"
#include "schedule_cpu.h"
#include "schedule_cuda.h"
#include "schedule_metal.h"

#include <Halide.h>
#include <vector>

using namespace Halide;

namespace tinysplat_halide {
namespace {

constexpr float kPi  = 3.14159265358979323846f;
constexpr float kEps = 1e-8f;


// ---------------------------------------------------------------------------
// Forward pipeline — (y, x, c) convention
// ---------------------------------------------------------------------------

ForwardPipeline
build_forward_pipeline_impl(const Buffer<float>& means_in,
                           const Buffer<float>& covariances_in,
                           const Buffer<float>& colors_in,
                           const Buffer<float>& opacities_in,
                           int height,
                           int width,
                           int num_channels) {
    Var x("x"), y("y"), c("c"), n("n"), r("r");
    const int N = means_in.dim(0).extent();

    // ---- Input wrappers ----
    Func means_f("means_f"), cov_f("cov_f"),
         colors_f("colors_f"), opacities_f("opacities_f");
    means_f(n, _)   = means_in(n, _);
    cov_f(n, _, _)  = covariances_in(n, _, _);
    colors_f(n, c)  = colors_in(n, c);
    opacities_f(n)  = opacities_in(n);

    // ---- Per-Gaussian constants ----
    Func det("det");
    det(n) = cov_f(n, 0, 0) * cov_f(n, 1, 1)
           - cov_f(n, 0, 1) * cov_f(n, 0, 1);

    Func norm_factor("norm_factor");
    norm_factor(n) = 1.0f / (2.0f * kPi * sqrt(max(det(n), kEps)));

    Func inv_cov("inv_cov");
    inv_cov(n, c, r) = cast<float>(0.0f);
    inv_cov(n, 0, 0) =  cov_f(n, 1, 1) / max(det(n), kEps);
    inv_cov(n, 1, 1) =  cov_f(n, 0, 0) / max(det(n), kEps);
    inv_cov(n, 0, 1) = -cov_f(n, 0, 1) / max(det(n), kEps);
    inv_cov(n, 1, 0) = -cov_f(n, 0, 1) / max(det(n), kEps);

    // ---- Pixel coordinates (y, x, n) ----
    // px = column (x-coord), py = row (y-coord)
    Func px("px"), py("py");
    px(y, x, n) = cast<float>(x);
    py(y, x, n) = cast<float>(y);

    // ---- Mahalanobis distance + weight ----
    Func dx("dx"), dy("dy"), mahal("mahal"), weight("weight");
    dx(y, x, n)    = px(y, x, n) - means_f(n, 0);
    dy(y, x, n)    = py(y, x, n) - means_f(n, 1);

    mahal(y, x, n) = dx(y, x, n) * (inv_cov(n, 0, 0) * dx(y, x, n)
                                    + inv_cov(n, 0, 1) * dy(y, x, n))
                   + dy(y, x, n) * (inv_cov(n, 1, 0) * dx(y, x, n)
                                    + inv_cov(n, 1, 1) * dy(y, x, n));

    weight(y, x, n) = opacities_f(n)
                    * norm_factor(n)
                    * exp(-0.5f * mahal(y, x, n));

    // ---- Accumulate over Gaussians ----
    RDom r_n(0, N, "r_n");

    Func accum_color("accum_color"), accum_weight("accum_weight");
    accum_color(y, x, c) = cast<float>(0.0f);
    accum_weight(y, x)   = cast<float>(0.0f);
    accum_color(y, x, c) += weight(y, x, r_n) * colors_f(r_n, c);
    accum_weight(y, x)   += weight(y, x, r_n);
    // Suppress warning: update(0) is the RDom accumulation
    accum_weight.update(0).unscheduled();
    accum_color.update(0).unscheduled();

    // ---- Normalize ----
    Func output_norm("output_norm");
    output_norm(y, x, c) = accum_color(y, x, c)
                         / max(accum_weight(y, x), kEps);

    Func output("forward_output");
    output(y, x, c) = output_norm(y, x, c);
    output.bound(y, 0, height);
    output.bound(x, 0, width);
    output.bound(c, 0, num_channels);

    return {output, accum_color, accum_weight, weight,
            accum_weight, accum_color, output_norm};
}


// ---------------------------------------------------------------------------
// Backward pipeline — (y, x, c) convention
// ---------------------------------------------------------------------------

GradientFuncs
build_backward_pipeline_impl(const Buffer<float>& grad_output_in,
                            const Buffer<float>& means_in,
                            const Buffer<float>& covariances_in,
                            const Buffer<float>& colors_in,
                            const Buffer<float>& opacities_in,
                            int height,
                            int width,
                            int num_channels) {
    Var x("x"), y("y"), c("c"), n("n"), r("r");
    const int N = means_in.dim(0).extent();

    // ---- Input wrappers ----
    // grad_output_in has shape (H, W, C) matching PyTorch layout
    Func grad_output("grad_output"), means_b("means_b"),
         cov_b("cov_b"), colors_b("colors_b"), opacities_b("opacities_b");
    grad_output(y, x, c) = grad_output_in(y, x, c);
    means_b(n, _)   = means_in(n, _);
    cov_b(n, _, _)  = covariances_in(n, _, _);
    colors_b(n, c)  = colors_in(n, c);
    opacities_b(n)  = opacities_in(n);

    // ---- Forward intermediates (recomputed) ----
    Func det_fn("det_b"), norm_factor_b("norm_b");
    det_fn(n)        = cov_b(n, 0, 0) * cov_b(n, 1, 1)
                     - cov_b(n, 0, 1) * cov_b(n, 0, 1);
    norm_factor_b(n) = 1.0f / (2.0f * kPi * sqrt(max(det_fn(n), kEps)));

    Func inv_cov_b("inv_cov_b");
    inv_cov_b(n, c, r) = cast<float>(0.0f);
    inv_cov_b(n, 0, 0) =  cov_b(n, 1, 1) / max(det_fn(n), kEps);
    inv_cov_b(n, 1, 1) =  cov_b(n, 0, 0) / max(det_fn(n), kEps);
    inv_cov_b(n, 0, 1) = -cov_b(n, 0, 1) / max(det_fn(n), kEps);
    inv_cov_b(n, 1, 0) = -cov_b(n, 0, 1) / max(det_fn(n), kEps);

    Func px_b("px_b"), py_b("py_b");
    px_b(y, x, n) = cast<float>(x);
    py_b(y, x, n) = cast<float>(y);

    Func dx_b("dx_b"), dy_b("dy_b"), mahal_b("mahal_b"), weight_b("weight_b");
    dx_b(y, x, n)    = px_b(y, x, n) - means_b(n, 0);
    dy_b(y, x, n)    = py_b(y, x, n) - means_b(n, 1);
    mahal_b(y, x, n) = dx_b(y, x, n) * (inv_cov_b(n, 0, 0) * dx_b(y, x, n)
                                         + inv_cov_b(n, 0, 1) * dy_b(y, x, n))
                      + dy_b(y, x, n) * (inv_cov_b(n, 1, 0) * dx_b(y, x, n)
                                         + inv_cov_b(n, 1, 1) * dy_b(y, x, n));
    weight_b(y, x, n) = opacities_b(n) * norm_factor_b(n)
                       * exp(-0.5f * mahal_b(y, x, n));

    // Forward recompute
    RDom r_n(0, N, "r_n");
    Func total_weight_pix_b("total_weight_pix_b"),
         total_color_pix_b("total_color_pix_b");
    total_weight_pix_b(y, x)   = cast<float>(0.0f);
    total_color_pix_b(y, x, c) = cast<float>(0.0f);
    total_weight_pix_b(y, x)   += weight_b(y, x, r_n);
    total_color_pix_b(y, x, c) += weight_b(y, x, r_n) * colors_b(r_n, c);

    Func output_norm_b("output_norm_b");
    output_norm_b(y, x, c) = total_color_pix_b(y, x, c)
                            / max(total_weight_pix_b(y, x), kEps);

    // ---- Reduction domains ----
    // r_hw: iterate over all pixels, r_hw.x = height dim, r_hw.y = width dim
    RDom r_hw(0, height, 0, width, "r_hw");
    // r_all: iterate over all pixels + channels
    RDom r_all(0, height, 0, width, 0, num_channels, "r_all");

    // ---- grad_colors[n,c] ----
    Func grad_colors_out("grad_colors_out");
    grad_colors_out(n, c) = cast<float>(0.0f);
    grad_colors_out(n, c) +=
        grad_output(r_hw.x, r_hw.y, c)
        * weight_b(r_hw.x, r_hw.y, n)
        / max(total_weight_pix_b(r_hw.x, r_hw.y), kEps);

    // ---- grad_opacities[n] ----
    Func grad_opacities_out("grad_opacities_out");
    grad_opacities_out(n) = cast<float>(0.0f);
    grad_opacities_out(n) +=
        grad_output(r_all.x, r_all.y, r_all.z)
        * norm_factor_b(n)
        * exp(-0.5f * mahal_b(r_all.x, r_all.y, n))
        * (colors_b(n, r_all.z) - output_norm_b(r_all.x, r_all.y, r_all.z))
        / max(total_weight_pix_b(r_all.x, r_all.y), kEps);

    // ---- grad_means[2, n] ----
    Func chain_term("chain_term");
    chain_term(y, x, c, n) =
        grad_output(y, x, c)
        * (colors_b(n, c) - output_norm_b(y, x, c))
        / max(total_weight_pix_b(y, x), kEps);

    Func inv_dot_dx("inv_dot_dx"), inv_dot_dy("inv_dot_dy");
    inv_dot_dx(y, x, n) = inv_cov_b(n, 0, 0) * dx_b(y, x, n)
                           + inv_cov_b(n, 0, 1) * dy_b(y, x, n);
    inv_dot_dy(y, x, n) = inv_cov_b(n, 1, 0) * dx_b(y, x, n)
                           + inv_cov_b(n, 1, 1) * dy_b(y, x, n);

    Func contrib_x("contrib_x"), contrib_y("contrib_y");
    contrib_x(y, x, c, n) = -weight_b(y, x, n) * inv_dot_dx(y, x, n)
                             * chain_term(y, x, c, n);
    contrib_y(y, x, c, n) = -weight_b(y, x, n) * inv_dot_dy(y, x, n)
                             * chain_term(y, x, c, n);

    Func grad_means_x("grad_means_x"), grad_means_y("grad_means_y");
    grad_means_x(n) = cast<float>(0.0f);
    grad_means_y(n) = cast<float>(0.0f);
    grad_means_x(n) += contrib_x(r_all.x, r_all.y, r_all.z, n);
    grad_means_y(n) += contrib_y(r_all.x, r_all.y, r_all.z, n);

    Func grad_means_out("grad_means_out");
    grad_means_out(n, c) = select(c == 0, grad_means_x(n), grad_means_y(n));

    // ---- grad_cov[2x2] — analytical ----
    Func det_pow15("det_pow15");
    det_pow15(n) = pow(max(det_fn(n), kEps), cast<float>(1.5f));

    Func grad_norm_a("grad_norm_a"), grad_norm_bv("grad_norm_bv"), grad_norm_d("grad_norm_d");
    grad_norm_a(n)  = -cov_b(n, 1, 1) / (cast<float>(4.0f) * kPi * det_pow15(n));
    grad_norm_bv(n) =  cov_b(n, 0, 1) / (cast<float>(2.0f) * kPi * det_pow15(n));
    grad_norm_d(n)  = -cov_b(n, 0, 0) / (cast<float>(4.0f) * kPi * det_pow15(n));

    Func mahal_grad_a("mahal_grad_a"), mahal_grad_b("mahal_grad_b"), mahal_grad_d("mahal_grad_d");
    mahal_grad_a(y, x, n) = -(inv_cov_b(n, 0, 0) * dx_b(y, x, n) * inv_cov_b(n, 0, 0) * dx_b(y, x, n)
                             + inv_cov_b(n, 0, 0) * dx_b(y, x, n) * inv_cov_b(n, 0, 1) * dy_b(y, x, n)
                             + inv_cov_b(n, 0, 1) * dy_b(y, x, n) * inv_cov_b(n, 0, 0) * dx_b(y, x, n)
                             + inv_cov_b(n, 0, 1) * dy_b(y, x, n) * inv_cov_b(n, 0, 1) * dy_b(y, x, n));
    mahal_grad_b(y, x, n) = -(inv_cov_b(n, 0, 0) * dx_b(y, x, n) * inv_cov_b(n, 1, 0) * dx_b(y, x, n)
                             + inv_cov_b(n, 0, 0) * dx_b(y, x, n) * inv_cov_b(n, 1, 1) * dy_b(y, x, n)
                             + inv_cov_b(n, 0, 1) * dy_b(y, x, n) * inv_cov_b(n, 1, 0) * dx_b(y, x, n)
                             + inv_cov_b(n, 0, 1) * dy_b(y, x, n) * inv_cov_b(n, 1, 1) * dy_b(y, x, n));
    mahal_grad_d(y, x, n) = -(inv_cov_b(n, 1, 0) * dx_b(y, x, n) * inv_cov_b(n, 1, 0) * dx_b(y, x, n)
                             + inv_cov_b(n, 1, 0) * dx_b(y, x, n) * inv_cov_b(n, 1, 1) * dy_b(y, x, n)
                             + inv_cov_b(n, 1, 1) * dy_b(y, x, n) * inv_cov_b(n, 1, 0) * dx_b(y, x, n)
                             + inv_cov_b(n, 1, 1) * dy_b(y, x, n) * inv_cov_b(n, 1, 1) * dy_b(y, x, n));

    Func contrib_cov_a("contrib_cov_a"), contrib_cov_bv("contrib_cov_bv"), contrib_cov_d("contrib_cov_d");
    contrib_cov_a(y, x, c, n) =
        weight_b(y, x, n)
        * (cast<float>(-0.5f) * mahal_grad_a(y, x, n)
           + grad_norm_a(n) / max(norm_factor_b(n), kEps))
        * chain_term(y, x, c, n);
    contrib_cov_bv(y, x, c, n) =
        weight_b(y, x, n)
        * (cast<float>(-0.5f) * mahal_grad_b(y, x, n)
           + grad_norm_bv(n) / max(norm_factor_b(n), kEps))
        * chain_term(y, x, c, n);
    contrib_cov_d(y, x, c, n) =
        weight_b(y, x, n)
        * (cast<float>(-0.5f) * mahal_grad_d(y, x, n)
           + grad_norm_d(n) / max(norm_factor_b(n), kEps))
        * chain_term(y, x, c, n);

    Func grad_cov_a("grad_cov_a"), grad_cov_b("grad_cov_b"), grad_cov_d("grad_cov_d");
    grad_cov_a(n) = cast<float>(0.0f);
    grad_cov_b(n) = cast<float>(0.0f);
    grad_cov_d(n) = cast<float>(0.0f);
    grad_cov_a(n) += contrib_cov_a(r_all.x, r_all.y, r_all.z, n);
    grad_cov_b(n) += contrib_cov_bv(r_all.x, r_all.y, r_all.z, n);
    grad_cov_d(n) += contrib_cov_d(r_all.x, r_all.y, r_all.z, n);

    Func grad_cov_out("grad_cov_out");
    grad_cov_out(n, c, r) = cast<float>(0.0f);
    grad_cov_out(n, 0, 0) = grad_cov_a(n);
    grad_cov_out(n, 0, 1) = grad_cov_b(n);
    grad_cov_out(n, 1, 0) = grad_cov_b(n);
    grad_cov_out(n, 1, 1) = grad_cov_d(n);

    return {grad_means_out, grad_cov_out, grad_colors_out, grad_opacities_out};
}

}  // anonymous namespace


// ---------------------------------------------------------------------------
// Public wrappers
// ---------------------------------------------------------------------------

ForwardPipeline
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


GradientFuncs
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


// ---------------------------------------------------------------------------
// Schedule implementations
// ---------------------------------------------------------------------------

void apply_cpu_schedule_forward(ForwardPipeline& p,
                               int height, int width, int num_channels) {
    schedule::apply_cpu_schedule(
        p.output, p.accum_color, p.accum_weight,
        height, width, num_channels);
}

void apply_cuda_schedule_forward(ForwardPipeline& p,
                                int height, int width, int num_channels) {
    schedule::apply_cuda_schedule(
        p.output, p.accum_color, p.accum_weight,
        height, width, num_channels);
}

void apply_metal_schedule_forward(ForwardPipeline& p,
                                 int height, int width, int num_channels) {
    schedule::apply_metal_schedule(
        p.output, p.accum_color, p.accum_weight,
        height, width, num_channels);
}

void apply_cpu_schedule_backward(GradientFuncs& g,
                                int height, int width, int num_channels, int N) {
    (void)g; (void)height; (void)width; (void)num_channels; (void)N;
}

void apply_cuda_schedule_backward(GradientFuncs& g,
                                  int height, int width, int num_channels, int N) {
    schedule::apply_cuda_schedule_backward(
        g.grad_means, g.grad_cov, g.grad_colors, g.grad_opacities,
        height, width, num_channels, N);
}

void apply_metal_schedule_backward(GradientFuncs& g,
                                  int height, int width, int num_channels, int N) {
    schedule::apply_metal_schedule_backward(
        g.grad_means, g.grad_cov, g.grad_colors, g.grad_opacities,
        height, width, num_channels, N);
}

}  // namespace tinysplat_halide
