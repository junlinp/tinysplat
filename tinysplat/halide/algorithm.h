#ifndef TINYSPLAT_HALIDE_ALGORITHM_H
#define TINYSPLAT_HALIDE_ALGORITHM_H

/**
 * TinySplat Halide Algorithm Definitions
 * ======================================
 * Forward + Backward pipelines for 2D Gaussian splatting.
 *
 * Uses ImageParam inputs so pipelines can be JIT-compiled once and reused
 * across calls with different input data (N may vary at runtime).
 *
 * Build:  HL_PATH=$HOME/halide make
 */

#include <Halide.h>

namespace tinysplat_halide {

// ---------------------------------------------------------------------------
// Forward pipeline
// ---------------------------------------------------------------------------

struct ForwardPipeline {
    Halide::ImageParam means_ip{Halide::Float(32), 2, "means"};
    Halide::ImageParam cov_ip{Halide::Float(32), 3, "cov"};
    Halide::ImageParam colors_ip{Halide::Float(32), 2, "colors"};
    Halide::ImageParam opacities_ip{Halide::Float(32), 1, "opacities"};

    Halide::Func output{"forward_output"};

    Halide::Func det{"det"};
    Halide::Func norm_factor{"norm_factor"};
    Halide::Func inv_cov{"inv_cov"};
    Halide::Func accum_color{"accum_color"};
    Halide::Func accum_weight{"accum_weight"};
    Halide::Func weight{"weight"};
    Halide::Func output_norm{"output_norm"};

    bool built = false;
};

void build_forward(ForwardPipeline& p, int height, int width, int num_channels);
void apply_cpu_schedule_forward(ForwardPipeline& p, int height, int width, int num_channels);
void apply_cuda_schedule_forward(ForwardPipeline& p, int height, int width, int num_channels);
void apply_metal_schedule_forward(ForwardPipeline& p, int height, int width, int num_channels);

// ---------------------------------------------------------------------------
// Backward pipeline
// ---------------------------------------------------------------------------

struct GradientPipeline {
    Halide::ImageParam grad_output_ip{Halide::Float(32), 3, "grad_out"};
    Halide::ImageParam means_ip{Halide::Float(32), 2, "means_b"};
    Halide::ImageParam cov_ip{Halide::Float(32), 3, "cov_b"};
    Halide::ImageParam colors_ip{Halide::Float(32), 2, "colors_b"};
    Halide::ImageParam opacities_ip{Halide::Float(32), 1, "opacities_b"};

    Halide::Func grad_means{"grad_means_out"};
    Halide::Func grad_cov{"grad_cov_out"};
    Halide::Func grad_colors{"grad_colors_out"};
    Halide::Func grad_opacities{"grad_opacities_out"};

    Halide::Func det{"det_b"};
    Halide::Func norm_factor{"norm_b"};
    Halide::Func inv_cov{"inv_cov_b"};
    Halide::Func total_weight_pix{"total_weight_pix_b"};
    Halide::Func total_color_pix{"total_color_pix_b"};
    Halide::Func weight{"weight_b"};
    Halide::Func output_norm{"output_norm_b"};

    bool built = false;
};

void build_backward(GradientPipeline& g, int height, int width, int num_channels);
void apply_cpu_schedule_backward(GradientPipeline& g, int height, int width, int num_channels);
void apply_cuda_schedule_backward(GradientPipeline& g, int height, int width, int num_channels);
void apply_metal_schedule_backward(GradientPipeline& g, int height, int width, int num_channels);

}  // namespace tinysplat_halide

#endif  // TINYSPLAT_HALIDE_ALGORITHM_H
