/**
 * algorithm.cpp
 *
 * Halide forward + backward pipelines for 2D Gaussian splatting.
 * Returns Funcs via structs so schedules can be applied before realize.
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
// Forward pipeline
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
    // means_in shape: (N, 2) → access as means_in(n, dim=0) or means_in(n, 0/1)
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

    // ---- Pixel coordinates ----
    Func px("px"), py("py");
    px(x, y, n) = cast<float>(x);
    py(x, y, n) = cast<float>(y);

    // ---- Mahalanobis distance + weight ----
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

    // ---- Accumulate over Gaussians ----
    RDom r_n(0, N, "r_n");

    Func accum_color("accum_color"), accum_weight("accum_weight");
    accum_color(x, y, c) = cast<float>(0.0f);
    accum_weight(x, y)   = cast<float>(0.0f);
    accum_color(x, y, c) += weight(x, y, r_n) * colors_f(r_n, c);
    accum_weight(x, y)   += weight(x, y, r_n);

    // ---- Forward intermediates ----
    Func total_weight_pix("total_weight_pix"), total_color_pix("total_color_pix");
    total_weight_pix(x, y)   = accum_weight(x, y);
    total_color_pix(x, y, c) = accum_color(x, y, c);

    // ---- Normalize ----
    Func output_norm("output_norm");
    output_norm(x, y, c) = total_color_pix(x, y, c)
                         / max(total_weight_pix(x, y), kEps);

    Func output("forward_output");
    output(x, y, c) = output_norm(x, y, c);
    output.bound(x, 0, width);
    output.bound(y, 0, height);
    output.bound(c, 0, num_channels);

    return {output, accum_color, accum_weight, weight,
            total_weight_pix, total_color_pix, output_norm};
}


// ---------------------------------------------------------------------------
// Backward pipeline
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
    Func grad_output("grad_output"), means_b("means_b"),
         cov_b("cov_b"), colors_b("colors_b"), opacities_b("opacities_b");
    grad_output(x, y, c) = grad_output_in(y, x, c);  // HWCB storage
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
    px_b(x, y, n) = cast<float>(x);
    py_b(x, y, n) = cast<float>(y);

    Func dx_b("dx_b"), dy_b("dy_b"), mahal_b("mahal_b"), weight_b("weight_b");
    dx_b(x, y, n)    = px_b(x, y, n) - means_b(n, 0);
    dy_b(x, y, n)    = py_b(x, y, n) - means_b(n, 1);
    mahal_b(x, y, n) = dx_b(x, y, n) * (inv_cov_b(n, 0, 0) * dx_b(x, y, n)
                                         + inv_cov_b(n, 0, 1) * dy_b(x, y, n))
                      + dy_b(x, y, n) * (inv_cov_b(n, 1, 0) * dx_b(x, y, n)
                                         + inv_cov_b(n, 1, 1) * dy_b(x, y, n));
    weight_b(x, y, n) = opacities_b(n) * norm_factor_b(n)
                       * exp(-0.5f * mahal_b(x, y, n));

    // Forward recompute
    RDom r_n(0, N, "r_n");
    Func total_weight_pix_b("total_weight_pix_b"),
         total_color_pix_b("total_color_pix_b");
    total_weight_pix_b(x, y)   = cast<float>(0.0f);
    total_color_pix_b(x, y, c) = cast<float>(0.0f);
    total_weight_pix_b(x, y)   += weight_b(x, y, r_n);
    total_color_pix_b(x, y, c) += weight_b(x, y, r_n) * colors_b(r_n, c);

    Func output_norm_b("output_norm_b");
    output_norm_b(x, y, c) = total_color_pix_b(x, y, c)
                            / max(total_weight_pix_b(x, y), kEps);

    // ---- grad_colors[n,c] ----
    RDom r_hw(0, height, 0, width, "r_hw");

    Func grad_colors_out("grad_colors_out");
    grad_colors_out(n, c) = cast<float>(0.0f);
    grad_colors_out(n, c) +=
        grad_output(r_hw.x, r_hw.y, c)
        * weight_b(r_hw.x, r_hw.y, n)
        / max(total_weight_pix_b(r_hw.x, r_hw.y), kEps);

    // ---- grad_opacities[n] ----
    RDom r_all(0, height, 0, width, 0, num_channels, "r_all");

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
    chain_term(x, y, c, n) =
        grad_output(x, y, c)
        * (colors_b(n, c) - output_norm_b(x, y, c))
        / max(total_weight_pix_b(x, y), kEps);

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
    grad_means_x(n) = cast<float>(0.0f);
    grad_means_y(n) = cast<float>(0.0f);
    grad_means_x(n) += contrib_x(r_all.x, r_all.y, r_all.z, n);
    grad_means_y(n) += contrib_y(r_all.x, r_all.y, r_all.z, n);

    Func grad_means_out("grad_means_out");
    grad_means_out(n, c) = select(c == 0, grad_means_x(n), grad_means_y(n));

    // ---- grad_cov[2x2] — analytical ----
    // ∂weight/∂Σ[i,j] = -0.5 * op * norm * exp(-0.5*mahal) * ∂mahal/∂Σ[i,j]
    //                 + weight * ∂(1/det)/∂Σ[i,j] / (2π*sqrt(det))
    // For symmetric Σ with params (a,b,d) = (Σ_00, Σ_01, Σ_11):
    //   ∂det/∂a = d, ∂det/∂b = -2b, ∂det/∂d = a
    //   ∂norm/∂a = -d / (4π*det^1.5), ∂norm/∂b = b / (2π*det^1.5), ∂norm/∂d = -a / (4π*det^1.5)
    //
    // mahal = [dx,dy] · Σ⁻¹ · [dx,dy]ᵀ
    // For diagonal Σ: mahal = dx²/a + dy²/d
    // For full Σ: mahal = (a*dx² + 2b*dx*dy + d*dy²) / det
    //
    // Chain rule: ∂weight/∂Σ = -0.5 * op * norm * exp(-0.5*mahal) * ∂mahal/∂Σ
    //                        + weight * ∂norm/∂Σ / norm
    //
    // We implement the full 2x2 case using the matrix derivative identity:
    //   ∂mahal/∂Σ = -Σ⁻¹ · d · dᵀ · Σ⁻¹   (d = pixel - mean)
    //
    // Combined with chain rule through the exponential and normalization,
    // we get the gradient by summing over all pixel contributions.

    // Precompute mahal and weight for each (pixel, gaussian)
    Func mahal_term("mahal_term");
    mahal_term(x, y, n) = cast<float>(-0.5f) * mahal_b(x, y, n);

    Func norm_term("norm_term");
    norm_term(n) = norm_factor_b(n);

    Func weight_term("weight_term");
    weight_term(x, y, n) = weight_b(x, y, n);

    // grad_norm/∂Σ[i,j] for 2x2 symmetric
    // ∂norm/∂a = -cov[1,1] / (4π * det^1.5)
    // ∂norm/∂d = -cov[0,0] / (4π * det^1.5)  
    // ∂norm/∂b =  cov[0,1] / (2π * det^1.5)
    Func det_pow15("det_pow15");
    det_pow15(n) = pow(max(det_fn(n), kEps), cast<float>(1.5f));

    Func grad_norm_a("grad_norm_a"), grad_norm_b("grad_norm_b"), grad_norm_d("grad_norm_d");
    grad_norm_a(n) = -cov_b(n, 1, 1) / (cast<float>(4.0f) * kPi * det_pow15(n));
    grad_norm_b(n) =  cov_b(n, 0, 1) / (cast<float>(2.0f) * kPi * det_pow15(n));
    grad_norm_d(n) = -cov_b(n, 0, 0) / (cast<float>(4.0f) * kPi * det_pow15(n));

    // ∂mahal/∂Σ for each (pixel, gaussian) using matrix identity
    // mahal = dᵀ Σ⁻¹ d = (dᵀ * inv * d)
    // ∂mahal/∂Σ = -inv * d * dᵀ * inv
    //   = -(1/det²) * [[d², dx*dy],[dx*dy, dy²]] * [[d,-b],[-b,a]]
    //   = -(1/det²) * [[d*dx²-b*dx*dy, d*dx*dy - ab*dy], ...]
    //
    // For each of the 3 unique elements (a,b,d) of symmetric Σ:
    Func mahal_grad_a("mahal_grad_a"), mahal_grad_b("mahal_grad_b"), mahal_grad_d("mahal_grad_d");
    mahal_grad_a(x, y, n) = -inv_cov_b(n, 0, 0) * dx_b(x, y, n) * inv_cov_b(n, 0, 0) * dx_b(x, y, n)
                           - inv_cov_b(n, 0, 0) * dx_b(x, y, n) * inv_cov_b(n, 0, 1) * dy_b(x, y, n)
                           - inv_cov_b(n, 0, 1) * dy_b(x, y, n) * inv_cov_b(n, 0, 0) * dx_b(x, y, n)
                           - inv_cov_b(n, 0, 1) * dy_b(x, y, n) * inv_cov_b(n, 0, 1) * dy_b(x, y, n);
    mahal_grad_b(x, y, n) = -inv_cov_b(n, 0, 0) * dx_b(x, y, n) * inv_cov_b(n, 1, 0) * dx_b(x, y, n)
                           - inv_cov_b(n, 0, 0) * dx_b(x, y, n) * inv_cov_b(n, 1, 1) * dy_b(x, y, n)
                           - inv_cov_b(n, 0, 1) * dy_b(x, y, n) * inv_cov_b(n, 1, 0) * dx_b(x, y, n)
                           - inv_cov_b(n, 0, 1) * dy_b(x, y, n) * inv_cov_b(n, 1, 1) * dy_b(x, y, n);
    mahal_grad_d(x, y, n) = -inv_cov_b(n, 1, 0) * dx_b(x, y, n) * inv_cov_b(n, 1, 0) * dx_b(x, y, n)
                           - inv_cov_b(n, 1, 0) * dx_b(x, y, n) * inv_cov_b(n, 1, 1) * dy_b(x, y, n)
                           - inv_cov_b(n, 1, 1) * dy_b(x, y, n) * inv_cov_b(n, 1, 0) * dx_b(x, y, n)
                           - inv_cov_b(n, 1, 1) * dy_b(x, y, n) * inv_cov_b(n, 1, 1) * dy_b(x, y, n);

    // Combined gradient: ∂weight/∂Σ = op * ( -0.5*exp(-0.5*mahal)*norm*∂mahal/∂Σ + weight*grad_norm/det_term )
    // Chain rule through Σ: we sum over all pixels and channels
    // ∂L/∂Σ = Σ_{pixels,channels} grad_out * ∂output/∂weight * ∂weight/∂Σ
    // output_norm = total_color / total_weight
    // ∂output/∂weight is complex; we use chain_term from earlier
    //
    // Simplified: use weight_b * chain_term already computed for gradient

    // For grad_cov, we use the full chain:
    // ∂L/∂Σ = Σ grad_out * (∂composite/∂weight) * ∂weight/∂Σ
    // where composite includes both color weighting and normalization

    // grad_cov[0,0] = Σ_x,y,c grad_out * chain_term * (-0.5 * op * norm * mahal_grad_a)
    //                 + Σ_x,y,c grad_out * weight * grad_norm_a / norm * (2π*sqrt(det))
    // But this is complex. Use weight_b * chain_term already accumulated.
    //
    // Final: grad_cov[i,j] = Σ (weight_b * chain_term * mahal_grad + weight_b * grad_norm_term)

    Func contrib_cov_a("contrib_cov_a"), contrib_cov_b("contrib_cov_b"), contrib_cov_d("contrib_cov_d");
    contrib_cov_a(x, y, c, n) =
        weight_b(x, y, n)
        * (cast<float>(-0.5f) * mahal_grad_a(x, y, n)
           + grad_norm_a(n) / max(norm_factor_b(n), kEps))
        * chain_term(x, y, c, n);

    contrib_cov_b(x, y, c, n) =
        weight_b(x, y, n)
        * (cast<float>(-0.5f) * mahal_grad_b(x, y, n)
           + grad_norm_b(n) / max(norm_factor_b(n), kEps))
        * chain_term(x, y, c, n);

    contrib_cov_d(x, y, c, n) =
        weight_b(x, y, n)
        * (cast<float>(-0.5f) * mahal_grad_d(x, y, n)
           + grad_norm_d(n) / max(norm_factor_b(n), kEps))
        * chain_term(x, y, c, n);

    Func grad_cov_a("grad_cov_a"), grad_cov_b("grad_cov_b"), grad_cov_d("grad_cov_d");
    grad_cov_a(n) = cast<float>(0.0f);
    grad_cov_b(n) = cast<float>(0.0f);
    grad_cov_d(n) = cast<float>(0.0f);
    grad_cov_a(n) += contrib_cov_a(r_all.x, r_all.y, r_all.z, n);
    grad_cov_b(n) += contrib_cov_b(r_all.x, r_all.y, r_all.z, n);
    grad_cov_d(n) += contrib_cov_d(r_all.x, r_all.y, r_all.z, n);

    Func grad_cov_out("grad_cov_out");
    grad_cov_out(n, c, r) = cast<float>(0.0f);
    grad_cov_out(n, 0, 0) = grad_cov_a(n);  // Σ_00
    grad_cov_out(n, 0, 1) = grad_cov_b(n);  // Σ_01
    grad_cov_out(n, 1, 0) = grad_cov_b(n);  // Σ_10 = Σ_01
    grad_cov_out(n, 1, 1) = grad_cov_d(n);  // Σ_11

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
    (void)height; (void)width; (void)num_channels;
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
    // CPU backward: auto-scheduler handles it
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
