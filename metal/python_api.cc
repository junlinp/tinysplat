#include "tinysplat/gaussian_3d_metal.h"

#include <cstdint>

extern "C" {

int tinysplat_metal_available() { return tinysplat::metal::metal_available() ? 1 : 0; }

int tinysplat_metal_forward(const float* means, const float* covs, const float* colors,
                            const float* opacities, int n, int c, const float* intrinsics,
                            const float* c2w, int h, int w, float* out, float near_plane,
                            float min_cov, float sigma, float beta, int use_cb) {
  tinysplat::metal::Splat3DMetalOptions opts;
  opts.near_plane = near_plane;
  opts.min_covariance = min_cov;
  opts.sigma_radius = sigma;
  opts.compact_box_beta = beta;
  opts.use_compact_box = use_cb != 0;
  return tinysplat::metal::gaussian_splat_3d_forward(means, covs, colors, opacities, n, c,
                                                     intrinsics, c2w, h, w, out, opts)
             ? 1
             : 0;
}

int tinysplat_metal_forward_qs(const float* means, const float* log_scales, const float* quats,
                               const float* colors, const float* opacities, int n, int c,
                               const float* intrinsics, const float* c2w, int h, int w, float* out,
                               float near_plane, float min_cov, float sigma, float beta,
                               int use_cb) {
  tinysplat::metal::Splat3DMetalOptions opts;
  opts.near_plane = near_plane;
  opts.min_covariance = min_cov;
  opts.sigma_radius = sigma;
  opts.compact_box_beta = beta;
  opts.use_compact_box = use_cb != 0;
  return tinysplat::metal::gaussian_splat_3d_forward_qs(means, log_scales, quats, colors, opacities,
                                                        n, c, intrinsics, c2w, h, w, out, opts)
             ? 1
             : 0;
}

int tinysplat_metal_forward_qs_sh(const float* means, const float* log_scales, const float* quats,
                                  const float* sh, const float* opacities, int n, int c,
                                  const float* intrinsics, const float* c2w, int h, int w,
                                  float* out, float near_plane, float min_cov, float sigma,
                                  float beta, int use_cb, int sh_degree) {
  tinysplat::metal::Splat3DMetalOptions opts;
  opts.near_plane = near_plane;
  opts.min_covariance = min_cov;
  opts.sigma_radius = sigma;
  opts.compact_box_beta = beta;
  opts.use_compact_box = use_cb != 0;
  opts.sh_degree = sh_degree;
  return tinysplat::metal::gaussian_splat_3d_forward_qs_sh(means, log_scales, quats, sh, opacities,
                                                           n, c, intrinsics, c2w, h, w, out, opts)
             ? 1
             : 0;
}

int tinysplat_metal_count_hits(const float* proj_means, const float* proj_covs,
                               const float* opacities, int n, int h, int w,
                               const uint8_t* error_mask, int* counts, float min_cov, float sigma,
                               float beta, int use_cb) {
  tinysplat::metal::Splat3DMetalOptions opts;
  opts.min_covariance = min_cov;
  opts.sigma_radius = sigma;
  opts.compact_box_beta = beta;
  opts.use_compact_box = use_cb != 0;
  return tinysplat::metal::count_footprint_hits(proj_means, proj_covs, opacities, n, h, w,
                                                error_mask, counts, opts)
             ? 1
             : 0;
}

int tinysplat_metal_session_count_hits(const uint8_t* error_mask, int* counts, int n, int h,
                                       int w) {
  return tinysplat::metal::count_session_metric_hits(error_mask, counts, n, h, w) ? 1 : 0;
}

int tinysplat_metal_projected_backward(const float* grad_output, const float* proj_means,
                                       const float* proj_covs, const float* colors,
                                       const float* opacities, int n, int c, int h, int w,
                                       float* grad_proj_means, float* grad_proj_covs,
                                       float* grad_colors, float* grad_opacities, float min_cov,
                                       float sigma, float beta, int use_cb, const float* depths,
                                       int force_cpu) {
  tinysplat::metal::Splat3DMetalOptions opts;
  opts.min_covariance = min_cov;
  opts.sigma_radius = sigma;
  opts.compact_box_beta = beta;
  opts.use_compact_box = use_cb != 0;
  opts.force_cpu = force_cpu != 0;
  return tinysplat::metal::gaussian_splat_3d_projected_backward(
             grad_output, proj_means, proj_covs, colors, opacities, n, c, h, w, grad_proj_means,
             grad_proj_covs, grad_colors, grad_opacities, opts, depths)
             ? 1
             : 0;
}

int tinysplat_metal_session_backward(const float* grad_output, int n, int c, int h, int w,
                                     float* grad_means3d, float* grad_covs3d, float* grad_colors,
                                     float* grad_opacities, float min_cov, float sigma, float beta,
                                     int use_cb, int force_cpu) {
  tinysplat::metal::Splat3DMetalOptions opts;
  opts.min_covariance = min_cov;
  opts.sigma_radius = sigma;
  opts.compact_box_beta = beta;
  opts.use_compact_box = use_cb != 0;
  opts.force_cpu = force_cpu != 0;
  return tinysplat::metal::gaussian_splat_3d_session_backward(grad_output, n, c, h, w, grad_means3d,
                                                             grad_covs3d, grad_colors,
                                                             grad_opacities, opts)
             ? 1
             : 0;
}

int tinysplat_metal_session_backward_qs(const float* grad_output, int n, int c, int h, int w,
                                        float* grad_means3d, float* grad_log_scales,
                                        float* grad_quats, float* grad_colors,
                                        float* grad_opacities, float min_cov, float sigma,
                                        float beta, int use_cb, int force_cpu) {
  tinysplat::metal::Splat3DMetalOptions opts;
  opts.min_covariance = min_cov;
  opts.sigma_radius = sigma;
  opts.compact_box_beta = beta;
  opts.use_compact_box = use_cb != 0;
  opts.force_cpu = force_cpu != 0;
  return tinysplat::metal::gaussian_splat_3d_session_backward_qs(
             grad_output, n, c, h, w, grad_means3d, grad_log_scales, grad_quats, grad_colors,
             grad_opacities, opts)
             ? 1
             : 0;
}

int tinysplat_metal_session_backward_qs_sh(const float* grad_output, int n, int c, int h, int w,
                                           float* grad_means3d, float* grad_log_scales,
                                           float* grad_quats, float* grad_sh,
                                           float* grad_opacities, float min_cov, float sigma,
                                           float beta, int use_cb, int force_cpu) {
  tinysplat::metal::Splat3DMetalOptions opts;
  opts.min_covariance = min_cov;
  opts.sigma_radius = sigma;
  opts.compact_box_beta = beta;
  opts.use_compact_box = use_cb != 0;
  opts.force_cpu = force_cpu != 0;
  return tinysplat::metal::gaussian_splat_3d_session_backward_qs_sh(
             grad_output, n, c, h, w, grad_means3d, grad_log_scales, grad_quats, grad_sh,
             grad_opacities, opts)
             ? 1
             : 0;
}

int tinysplat_metal_last_grad_means2d_n() { return tinysplat::metal::last_grad_means2d_count(); }

int tinysplat_metal_last_grad_means2d(float* out, int max_n) {
  return tinysplat::metal::copy_last_grad_means2d(out, max_n);
}

int tinysplat_metal_last_grad_means2d_abs_n() {
  return tinysplat::metal::last_grad_means2d_abs_count();
}

int tinysplat_metal_last_grad_means2d_abs(float* out, int max_n) {
  return tinysplat::metal::copy_last_grad_means2d_abs(out, max_n);
}

int tinysplat_metal_last_radii2d_n() { return tinysplat::metal::last_radii2d_count(); }

int tinysplat_metal_last_radii2d(float* out, int max_n) {
  return tinysplat::metal::copy_last_radii2d(out, max_n);
}

}  // extern "C"
