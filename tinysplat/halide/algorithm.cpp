/**
 * algorithm.cpp
 *
 * Halide forward + backward pipelines for 2D Gaussian splatting.
 *
 * Key optimisation decisions:
 *   - Inputs are ImageParam so the JIT-compiled code can be reused across
 *     calls (N is a runtime extent, H/W/C are compile-time bounds).
 *   - Per-Gaussian invariants (det, inv_cov, norm_factor) are exposed as
 *     separate Funcs so the schedule can compute_root them once.
 *   - A Mahalanobis cutoff avoids expensive exp() for distant pixels.
 */

#include "algorithm.h"

#include <Halide.h>

using namespace Halide;

namespace tinysplat_halide {
namespace {

constexpr float kPi  = 3.14159265358979323846f;
constexpr float kEps = 1e-8f;
constexpr float kMahalCutoff = 32.0f;  // ~4sigma: exp(-16) ~ 1.1e-7

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

void build_forward(ForwardPipeline& p, int height, int width, int num_channels) {
    Var x("x"), y("y"), c("c"), n("n"), r("r");
    Expr N = p.means_ip.dim(0).extent();

    // Per-Gaussian invariants (depend only on n)
    p.det(n) = p.cov_ip(n, 0, 0) * p.cov_ip(n, 1, 1)
             - p.cov_ip(n, 0, 1) * p.cov_ip(n, 0, 1);

    p.norm_factor(n) = 1.0f / (2.0f * kPi * sqrt(max(p.det(n), kEps)));

    p.inv_cov(n, c, r) = cast<float>(0.0f);
    p.inv_cov(n, 0, 0) =  p.cov_ip(n, 1, 1) / max(p.det(n), kEps);
    p.inv_cov(n, 1, 1) =  p.cov_ip(n, 0, 0) / max(p.det(n), kEps);
    p.inv_cov(n, 0, 1) = -p.cov_ip(n, 0, 1) / max(p.det(n), kEps);
    p.inv_cov(n, 1, 0) = -p.cov_ip(n, 0, 1) / max(p.det(n), kEps);

    // Mahalanobis distance + weight per (y, x, n)
    Func dx_f("dx_f"), dy_f("dy_f");
    dx_f(y, x, n) = cast<float>(x) - p.means_ip(n, 0);
    dy_f(y, x, n) = cast<float>(y) - p.means_ip(n, 1);

    Func mahal("mahal");
    mahal(y, x, n) = dx_f(y, x, n) * (p.inv_cov(n, 0, 0) * dx_f(y, x, n) + p.inv_cov(n, 0, 1) * dy_f(y, x, n))
                   + dy_f(y, x, n) * (p.inv_cov(n, 1, 0) * dx_f(y, x, n) + p.inv_cov(n, 1, 1) * dy_f(y, x, n));

    p.weight(y, x, n) = select(
        mahal(y, x, n) < kMahalCutoff,
        p.opacities_ip(n) * p.norm_factor(n) * exp(-0.5f * mahal(y, x, n)),
        0.0f);

    // Accumulate over Gaussians
    RDom r_n(0, N, "r_n");

    p.accum_color(y, x, c) = cast<float>(0.0f);
    p.accum_weight(y, x)   = cast<float>(0.0f);
    p.accum_color(y, x, c) += p.weight(y, x, r_n) * p.colors_ip(r_n, c);
    p.accum_weight(y, x)   += p.weight(y, x, r_n);

    // Normalize
    p.output_norm(y, x, c) = p.accum_color(y, x, c)
                            / max(p.accum_weight(y, x), kEps);

    p.output(y, x, c) = p.output_norm(y, x, c);
    p.output.bound(y, 0, height);
    p.output.bound(x, 0, width);
    p.output.bound(c, 0, num_channels);

    p.built = true;
}

// ---------------------------------------------------------------------------
// Backward
// ---------------------------------------------------------------------------

void build_backward(GradientPipeline& g, int height, int width, int num_channels) {
    Var x("x"), y("y"), c("c"), n("n"), r("r");
    Expr N = g.means_ip.dim(0).extent();

    // Recompute forward intermediates
    g.det(n) = g.cov_ip(n, 0, 0) * g.cov_ip(n, 1, 1)
             - g.cov_ip(n, 0, 1) * g.cov_ip(n, 0, 1);

    g.norm_factor(n) = 1.0f / (2.0f * kPi * sqrt(max(g.det(n), kEps)));

    g.inv_cov(n, c, r) = cast<float>(0.0f);
    g.inv_cov(n, 0, 0) =  g.cov_ip(n, 1, 1) / max(g.det(n), kEps);
    g.inv_cov(n, 1, 1) =  g.cov_ip(n, 0, 0) / max(g.det(n), kEps);
    g.inv_cov(n, 0, 1) = -g.cov_ip(n, 0, 1) / max(g.det(n), kEps);
    g.inv_cov(n, 1, 0) = -g.cov_ip(n, 0, 1) / max(g.det(n), kEps);

    Func dx_b("dx_b"), dy_b("dy_b"), mahal_b("mahal_b");
    dx_b(y, x, n) = cast<float>(x) - g.means_ip(n, 0);
    dy_b(y, x, n) = cast<float>(y) - g.means_ip(n, 1);
    mahal_b(y, x, n) = dx_b(y, x, n) * (g.inv_cov(n, 0, 0) * dx_b(y, x, n)
                                        + g.inv_cov(n, 0, 1) * dy_b(y, x, n))
                      + dy_b(y, x, n) * (g.inv_cov(n, 1, 0) * dx_b(y, x, n)
                                        + g.inv_cov(n, 1, 1) * dy_b(y, x, n));

    g.weight(y, x, n) = select(
        mahal_b(y, x, n) < kMahalCutoff,
        g.opacities_ip(n) * g.norm_factor(n) * exp(-0.5f * mahal_b(y, x, n)),
        0.0f);

    // Forward recompute: total weight + color per pixel
    RDom r_n(0, N, "r_n");
    g.total_weight_pix(y, x)   = cast<float>(0.0f);
    g.total_color_pix(y, x, c) = cast<float>(0.0f);
    g.total_weight_pix(y, x)   += g.weight(y, x, r_n);
    g.total_color_pix(y, x, c) += g.weight(y, x, r_n) * g.colors_ip(r_n, c);

    g.output_norm(y, x, c) = g.total_color_pix(y, x, c)
                            / max(g.total_weight_pix(y, x), kEps);

    // Reduction domains for backward accumulation
    RDom r_hw(0, height, 0, width, "r_hw");
    RDom r_all(0, height, 0, width, 0, num_channels, "r_all");

    // grad_colors[n,c]
    g.grad_colors(n, c) = cast<float>(0.0f);
    g.grad_colors(n, c) +=
        g.grad_output_ip(r_hw.x, r_hw.y, c)
        * g.weight(r_hw.x, r_hw.y, n)
        / max(g.total_weight_pix(r_hw.x, r_hw.y), kEps);

    // grad_opacities[n]
    g.grad_opacities(n) = cast<float>(0.0f);
    g.grad_opacities(n) +=
        g.grad_output_ip(r_all.x, r_all.y, r_all.z)
        * g.norm_factor(n)
        * exp(-0.5f * mahal_b(r_all.x, r_all.y, n))
        * (g.colors_ip(n, r_all.z) - g.output_norm(r_all.x, r_all.y, r_all.z))
        / max(g.total_weight_pix(r_all.x, r_all.y), kEps);

    // grad_means[n, coord]
    Func chain_term("chain_term");
    chain_term(y, x, c, n) =
        g.grad_output_ip(y, x, c)
        * (g.colors_ip(n, c) - g.output_norm(y, x, c))
        / max(g.total_weight_pix(y, x), kEps);

    Func inv_dot_dx("inv_dot_dx"), inv_dot_dy("inv_dot_dy");
    inv_dot_dx(y, x, n) = g.inv_cov(n, 0, 0) * dx_b(y, x, n)
                         + g.inv_cov(n, 0, 1) * dy_b(y, x, n);
    inv_dot_dy(y, x, n) = g.inv_cov(n, 1, 0) * dx_b(y, x, n)
                         + g.inv_cov(n, 1, 1) * dy_b(y, x, n);

    Func contrib_x("contrib_x"), contrib_y("contrib_y");
    contrib_x(y, x, c, n) = -g.weight(y, x, n) * inv_dot_dx(y, x, n) * chain_term(y, x, c, n);
    contrib_y(y, x, c, n) = -g.weight(y, x, n) * inv_dot_dy(y, x, n) * chain_term(y, x, c, n);

    Func grad_means_x("grad_means_x"), grad_means_y("grad_means_y");
    grad_means_x(n) = cast<float>(0.0f);
    grad_means_y(n) = cast<float>(0.0f);
    grad_means_x(n) += contrib_x(r_all.x, r_all.y, r_all.z, n);
    grad_means_y(n) += contrib_y(r_all.x, r_all.y, r_all.z, n);

    g.grad_means(n, c) = select(c == 0, grad_means_x(n), grad_means_y(n));

    // grad_cov[n, 2, 2]
    Func det_pow15("det_pow15");
    det_pow15(n) = pow(max(g.det(n), kEps), cast<float>(1.5f));

    Func gn_a("gn_a"), gn_bv("gn_bv"), gn_d("gn_d");
    gn_a(n)  = -g.cov_ip(n, 1, 1) / (4.0f * kPi * det_pow15(n));
    gn_bv(n) =  g.cov_ip(n, 0, 1) / (2.0f * kPi * det_pow15(n));
    gn_d(n)  = -g.cov_ip(n, 0, 0) / (4.0f * kPi * det_pow15(n));

    Func mg_a("mg_a"), mg_b("mg_b"), mg_d("mg_d");
    mg_a(y, x, n) = -(g.inv_cov(n, 0, 0) * dx_b(y, x, n) * g.inv_cov(n, 0, 0) * dx_b(y, x, n)
                     + g.inv_cov(n, 0, 0) * dx_b(y, x, n) * g.inv_cov(n, 0, 1) * dy_b(y, x, n)
                     + g.inv_cov(n, 0, 1) * dy_b(y, x, n) * g.inv_cov(n, 0, 0) * dx_b(y, x, n)
                     + g.inv_cov(n, 0, 1) * dy_b(y, x, n) * g.inv_cov(n, 0, 1) * dy_b(y, x, n));
    mg_b(y, x, n) = -(g.inv_cov(n, 0, 0) * dx_b(y, x, n) * g.inv_cov(n, 1, 0) * dx_b(y, x, n)
                     + g.inv_cov(n, 0, 0) * dx_b(y, x, n) * g.inv_cov(n, 1, 1) * dy_b(y, x, n)
                     + g.inv_cov(n, 0, 1) * dy_b(y, x, n) * g.inv_cov(n, 1, 0) * dx_b(y, x, n)
                     + g.inv_cov(n, 0, 1) * dy_b(y, x, n) * g.inv_cov(n, 1, 1) * dy_b(y, x, n));
    mg_d(y, x, n) = -(g.inv_cov(n, 1, 0) * dx_b(y, x, n) * g.inv_cov(n, 1, 0) * dx_b(y, x, n)
                     + g.inv_cov(n, 1, 0) * dx_b(y, x, n) * g.inv_cov(n, 1, 1) * dy_b(y, x, n)
                     + g.inv_cov(n, 1, 1) * dy_b(y, x, n) * g.inv_cov(n, 1, 0) * dx_b(y, x, n)
                     + g.inv_cov(n, 1, 1) * dy_b(y, x, n) * g.inv_cov(n, 1, 1) * dy_b(y, x, n));

    Func cc_a("cc_a"), cc_b("cc_b"), cc_d("cc_d");
    cc_a(y, x, c, n) = g.weight(y, x, n)
        * (-0.5f * mg_a(y, x, n) + gn_a(n) / max(g.norm_factor(n), kEps))
        * chain_term(y, x, c, n);
    cc_b(y, x, c, n) = g.weight(y, x, n)
        * (-0.5f * mg_b(y, x, n) + gn_bv(n) / max(g.norm_factor(n), kEps))
        * chain_term(y, x, c, n);
    cc_d(y, x, c, n) = g.weight(y, x, n)
        * (-0.5f * mg_d(y, x, n) + gn_d(n) / max(g.norm_factor(n), kEps))
        * chain_term(y, x, c, n);

    Func gc_a("gc_a"), gc_bv("gc_bv"), gc_d("gc_d");
    gc_a(n) = cast<float>(0.0f);
    gc_bv(n) = cast<float>(0.0f);
    gc_d(n) = cast<float>(0.0f);
    gc_a(n) += cc_a(r_all.x, r_all.y, r_all.z, n);
    gc_bv(n) += cc_b(r_all.x, r_all.y, r_all.z, n);
    gc_d(n) += cc_d(r_all.x, r_all.y, r_all.z, n);

    g.grad_cov(n, c, r) = cast<float>(0.0f);
    g.grad_cov(n, 0, 0) = gc_a(n);
    g.grad_cov(n, 0, 1) = gc_bv(n);
    g.grad_cov(n, 1, 0) = gc_bv(n);
    g.grad_cov(n, 1, 1) = gc_d(n);

    g.built = true;
}

}  // namespace tinysplat_halide
