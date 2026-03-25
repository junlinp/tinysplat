/**
 * algorithm.cpp
 *
 * Forward + Backward pipelines for 2D Gaussian splatting in Halide.
 */

#include "algorithm.h"
#include <Halide.h>
#include <vector>

using Halide::Buffer;
using Halide::Expr;
using Halide::Func;
using Halide::RDom;
using Halide::TailStrategy;
using Halide::Type;
using Halide::Var;
using Halide::_;
using Halide::exp;
using Halide::max;
using Halide::sqrt;
using Halide::sum;

namespace tinysplat_halide {
namespace {

constexpr float kPi  = 3.14159265358979323846f;
constexpr float kEps = 1e-8f;


// ---------------------------------------------------------------------------
// Forward pass only — Phase 2 backward uses PyTorch fallback
// ---------------------------------------------------------------------------

std::pair<Func, std::vector<Buffer<float>>>
build_forward_pipeline_impl(const Buffer<float>& means_in,
                            const Buffer<float>& covariances_in,
                            const Buffer<float>& colors_in,
                            const Buffer<float>& opacities_in,
                            int height,
                            int width,
                            int num_channels) {
    Var x("x"), y("y"), c("c"), n("n"), r("r");

    // Wrap inputs
    Func means_f("means_f"), cov_f("cov_f"), colors_f("colors_f"), opacities_f("opacities_f");
    means_f(n, _)   = means_in(n, _);
    cov_f(n, _, _)  = covariances_in(n, _, _);
    colors_f(n, c)  = colors_in(n, c);
    opacities_f(n)  = opacities_in(n);

    // --- Per-Gaussian constants ---
    Func det("det");
    det(n) = cov_f(n, 0, 0) * cov_f(n, 1, 1) - cov_f(n, 0, 1) * cov_f(n, 0, 1);

    Func norm_factor("norm_factor");
    norm_factor(n) = 1.0f / (2.0f * kPi * sqrt(max(det(n), kEps)));

    Func inv_cov("inv_cov");
    inv_cov(n, c, r) = Halide::cast(Halide::Float(32), 0.0f);
    inv_cov(n, 0, 0) =  cov_f(n, 1, 1) / max(det(n), kEps);
    inv_cov(n, 1, 1) =  cov_f(n, 0, 0) / max(det(n), kEps);
    inv_cov(n, 0, 1) = -cov_f(n, 0, 1) / max(det(n), kEps);
    inv_cov(n, 1, 0) = -cov_f(n, 0, 1) / max(det(n), kEps);

    // --- Pixel coordinate grids ---
    Func px("px"), py("py");
    px(x, y, n) = Halide::cast(Halide::Float(32), x);
    py(x, y, n) = Halide::cast(Halide::Float(32), y);

    // --- Mahalanobis + weight per Gaussian per pixel ---
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

    // --- Accumulate over Gaussians ---
    RDom r_n(0, means_in.dim(0).extent(), "r_n");

    Func accum_color("accum_color"), accum_weight("accum_weight");
    accum_color(x, y, c) = Halide::cast(Halide::Float(32), 0.0f);
    accum_weight(x, y)   = Halide::cast(Halide::Float(32), 0.0f);
    accum_color(x, y, c) += weight(x, y, r_n) * colors_f(r_n, c);
    accum_weight(x, y)   += weight(x, y, r_n);

    // --- Normalise ---
    Func output("forward_output");
    output(x, y, c) = accum_color(x, y, c) / max(accum_weight(x, y), kEps);
    output.bound(x, 0, width);
    output.bound(y, 0, height);
    output.bound(c, 0, num_channels);

    return {output, {}};
}


// ---------------------------------------------------------------------------
// Backward pass — Phase 2: analytical gradients
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

    // Wrap inputs
    Func grad_output("grad_output"), means_b("means_b"),
         cov_b("cov_b"), colors_b("colors_b"), opacities_b("opacities_b");
    grad_output(x, y, c) = grad_output_in(y, x, c);   // HWCB storage order
    means_b(n, _)   = means_in(n, _);
    cov_b(n, _, _)  = covariances_in(n, _, _);
    colors_b(n, c)  = colors_in(n, c);
    opacities_b(n)  = opacities_in(n);

    // ---- Forward intermediates (recomputed) ----
    Func det_fn("det_b"), norm_factor_b("norm_b");
    det_fn(n)        = cov_b(n, 0, 0) * cov_b(n, 1, 1) - cov_b(n, 0, 1) * cov_b(n, 0, 1);
    norm_factor_b(n) = 1.0f / (2.0f * kPi * sqrt(max(det_fn(n), kEps)));

    Func inv_cov_b("inv_cov_b");
    inv_cov_b(n, c, r) = Halide::cast(Halide::Float(32), 0.0f);
    inv_cov_b(n, 0, 0) =  cov_b(n, 1, 1) / max(det_fn(n), kEps);
    inv_cov_b(n, 1, 1) =  cov_b(n, 0, 0) / max(det_fn(n), kEps);
    inv_cov_b(n, 0, 1) = -cov_b(n, 0, 1) / max(det_fn(n), kEps);
    inv_cov_b(n, 1, 0) = -cov_b(n, 0, 1) / max(det_fn(n), kEps);

    Func px_b("px_b"), py_b("py_b");
    px_b(x, y, n) = Halide::cast(Halide::Float(32), x);
    py_b(x, y, n) = Halide::cast(Halide::Float(32), y);

    Func dx_b("dx_b"), dy_b("dy_b"), mahal_b("mahal_b"), weight_b("weight_b");
    dx_b(x, y, n)    = px_b(x, y, n) - means_b(n, 0);
    dy_b(x, y, n)    = py_b(x, y, n) - means_b(n, 1);
    mahal_b(x, y, n) = dx_b(x, y, n) * (inv_cov_b(n, 0, 0) * dx_b(x, y, n)
                                         + inv_cov_b(n, 0, 1) * dy_b(x, y, n))
                      + dy_b(x, y, n) * (inv_cov_b(n, 1, 0) * dx_b(x, y, n)
                                         + inv_cov_b(n, 1, 1) * dy_b(x, y, n));
    weight_b(x, y, n) = opacities_b(n) * norm_factor_b(n) * exp(-0.5f * mahal_b(x, y, n));

    // Forward recompute (needed for backward)
    RDom r_n(0, means_in.dim(0).extent(), "r_n");
    Func total_weight_pix("total_weight_pix"), total_color_pix("total_color_pix");
    total_weight_pix(x, y)   = Halide::cast(Halide::Float(32), 0.0f);
    total_color_pix(x, y, c) = Halide::cast(Halide::Float(32), 0.0f);
    total_weight_pix(x, y)   += weight_b(x, y, r_n);
    total_color_pix(x, y, c) += weight_b(x, y, r_n) * colors_b(r_n, c);

    Func output_norm("output_norm_b");
    output_norm(x, y, c) = total_color_pix(x, y, c)
                         / max(total_weight_pix(x, y), kEps);

    // ---- Combined RDom for all backward reductions ----
    // Single RDom over (height, width, num_channels) to avoid multiple reduction domains
    RDom r_all(0, height, 0, width, 0, num_channels, "r_all");

    // ---- grad_colors[n,c] ----
    // ∂L/∂colors[n,c] = Σ_{x,y} grad_out[x,y,c] × weight_b[n,x,y] / total_weight_pix[x,y]
    Func grad_colors_out("grad_colors_out");
    grad_colors_out(n, c) = Halide::cast(Halide::Float(32), 0.0f);
    grad_colors_out(n, c) +=
        grad_output(r_all.x, r_all.y, c)
        * weight_b(r_all.x, r_all.y, n)
        / max(total_weight_pix(r_all.x, r_all.y), kEps);

    // ---- grad_opacities[n] ----
    // ∂L/∂opacities[n] = Σ_{x,y,c} grad_out × norm × exp(-0.5×mahal)
    //                     × (colors - output_norm) / total_weight
    Func grad_opacities_out("grad_opacities_out");
    grad_opacities_out(n) = Halide::cast(Halide::Float(32), 0.0f);
    grad_opacities_out(n) +=
        grad_output(r_all.x, r_all.y, r_all.z)
        * norm_factor_b(n)
        * exp(-0.5f * mahal_b(r_all.x, r_all.y, n))
        * (colors_b(n, r_all.z) - output_norm(r_all.x, r_all.y, r_all.z))
        / max(total_weight_pix(r_all.x, r_all.y), kEps);

    // ---- grad_means[2, n] ----
    // x-component: Σ -weight × inv00*dx × chain_term
    // y-component: Σ -weight × inv10*dx × chain_term (actually inv11*dy + inv10*dx for y)
    // chain_term = grad_out × (colors - output_norm) / total_weight
    Func chain_term("chain_term");
    chain_term(x, y, c, n) =
        grad_output(x, y, c)
        * (colors_b(n, c) - output_norm(x, y, c))
        / max(total_weight_pix(x, y), kEps);
    // Note: chain_term uses pure dimensions, called with r_all.z for c in reduction

    Func inv_dot_dx("inv_dot_dx"), inv_dot_dy("inv_dot_dy");
    inv_dot_dx(x, y, n) = inv_cov_b(n, 0, 0) * dx_b(x, y, n)
                         + inv_cov_b(n, 0, 1) * dy_b(x, y, n);
    inv_dot_dy(x, y, n) = inv_cov_b(n, 1, 0) * dx_b(x, y, n)
                         + inv_cov_b(n, 1, 1) * dy_b(x, y, n);

    Func contrib_x("contrib_x"), contrib_y("contrib_y");
    contrib_x(x, y, c, n) = -weight_b(x, y, n) * inv_dot_dx(x, y, n)
                            * chain_term(x, y, c, n);
    contrib_y(x, y, c, n) = -weight_b(x, y, n) * inv_dot_dy(x, y, n)
                            * chain_term(x, y, c, n);

    Func grad_means_x("grad_means_x"), grad_means_y("grad_means_y");
    grad_means_x(n) = Halide::cast(Halide::Float(32), 0.0f);
    grad_means_y(n) = Halide::cast(Halide::Float(32), 0.0f);
    grad_means_x(n) += contrib_x(r_all.x, r_all.y, r_all.z, n);
    grad_means_y(n) += contrib_y(r_all.x, r_all.y, r_all.z, n);

    Func grad_means_out("grad_means_out");
    grad_means_out(n, c) = Halide::select(c == 0, grad_means_x(n), grad_means_y(n));

    // ---- grad_cov — stub ----
    Func grad_cov_out("grad_cov_out");
    grad_cov_out(n, c, r) = Halide::cast(Halide::Float(32), 0.0f);

    GradientFuncs result;
    result.grad_means = grad_means_out;
    result.grad_cov = grad_cov_out;
    result.grad_colors = grad_colors_out;
    result.grad_opacities = grad_opacities_out;
    return result;
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

}  // namespace tinysplat_halide
