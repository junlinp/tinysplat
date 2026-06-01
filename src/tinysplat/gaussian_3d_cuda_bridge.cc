#ifdef TINYSPLAT_CUDA

#include "gaussian_3d_cuda_bridge.h"

#include "tinysplat/gaussian_2d.h"
#include "tinysplat/gaussian_3d.h"

#include <tinysplat/gaussian_2d_cuda.h>
#include <tinysplat/gaussian_3d_cuda.h>

#include <vector>

namespace tinysplat {
namespace {

void pack_gaussians3d(const Gaussians3D& g, std::vector<float>& means, std::vector<float>& covs,
                      std::vector<float>& colors, std::vector<float>& opacities) {
  const int n = static_cast<int>(g.means.size());
  const int c = g.colors.empty() ? 3 : static_cast<int>(g.colors[0].size());
  means.resize(static_cast<size_t>(n) * 3);
  covs.resize(static_cast<size_t>(n) * 9);
  colors.resize(static_cast<size_t>(n) * c);
  opacities.resize(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    means[static_cast<size_t>(i) * 3 + 0] = g.means[static_cast<size_t>(i)].x;
    means[static_cast<size_t>(i) * 3 + 1] = g.means[static_cast<size_t>(i)].y;
    means[static_cast<size_t>(i) * 3 + 2] = g.means[static_cast<size_t>(i)].z;
    for (int r = 0; r < 3; ++r) {
      for (int col = 0; col < 3; ++col) {
        covs[static_cast<size_t>(i) * 9 + r * 3 + col] = g.covariances[static_cast<size_t>(i)].m[r][col];
      }
    }
    for (int ch = 0; ch < c; ++ch) {
      colors[static_cast<size_t>(i) * c + ch] = g.colors[static_cast<size_t>(i)][ch];
    }
    opacities[static_cast<size_t>(i)] = g.opacities[static_cast<size_t>(i)];
  }
}

void pack_intrinsics(const CameraIntrinsics& k, float out[9]) {
  out[0] = k.fx;
  out[1] = 0.0f;
  out[2] = k.cx;
  out[3] = 0.0f;
  out[4] = k.fy;
  out[5] = k.cy;
  out[6] = 0.0f;
  out[7] = 0.0f;
  out[8] = 1.0f;
}

void pack_c2w(const Mat4& m, float out[16]) {
  for (int r = 0; r < 4; ++r) {
    for (int col = 0; col < 4; ++col) {
      out[r * 4 + col] = m.m[r][col];
    }
  }
}

void pack_projected(const ProjectedGaussians2D& g, std::vector<float>& means,
                    std::vector<float>& covs, std::vector<float>& colors,
                    std::vector<float>& opacities) {
  const int n = static_cast<int>(g.means.size());
  const int c = g.colors.empty() ? 3 : static_cast<int>(g.colors[0].size());
  means.resize(static_cast<size_t>(n) * 2);
  covs.resize(static_cast<size_t>(n) * 4);
  colors.resize(static_cast<size_t>(n) * c);
  opacities.resize(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    means[static_cast<size_t>(i) * 2 + 0] = g.means[static_cast<size_t>(i)].x;
    means[static_cast<size_t>(i) * 2 + 1] = g.means[static_cast<size_t>(i)].y;
    covs[static_cast<size_t>(i) * 4 + 0] = g.covariances[static_cast<size_t>(i)].m00;
    covs[static_cast<size_t>(i) * 4 + 1] = g.covariances[static_cast<size_t>(i)].m01;
    covs[static_cast<size_t>(i) * 4 + 2] = g.covariances[static_cast<size_t>(i)].m10;
    covs[static_cast<size_t>(i) * 4 + 3] = g.covariances[static_cast<size_t>(i)].m11;
    for (int ch = 0; ch < c; ++ch) {
      colors[static_cast<size_t>(i) * c + ch] = g.colors[static_cast<size_t>(i)][ch];
    }
    opacities[static_cast<size_t>(i)] = g.opacities[static_cast<size_t>(i)];
  }
}

}  // namespace

bool cuda_device_available() {
  return cuda::cuda_available();
}

Image gaussian_splat_3d_forward_cuda_impl(const Gaussians3D& gaussians,
                                        const CameraIntrinsics& intrinsics,
                                        const Mat4& camera_to_world, int height, int width,
                                        const Splat3DOptions& opts) {
  std::vector<float> means;
  std::vector<float> covs;
  std::vector<float> colors;
  std::vector<float> opacities;
  pack_gaussians3d(gaussians, means, covs, colors, opacities);

  const int c = gaussians.colors.empty() ? 3 : static_cast<int>(gaussians.colors[0].size());
  Image image(height, width, c);
  float intr[9];
  float c2w[16];
  pack_intrinsics(intrinsics, intr);
  pack_c2w(camera_to_world, c2w);

  cuda::Splat3DCudaOptions cuda_opts;
  cuda_opts.near_plane = opts.near_plane;
  cuda_opts.min_covariance = opts.min_covariance;
  cuda_opts.sigma_radius = opts.sigma_radius;

  if (!cuda::gaussian_splat_3d_forward(means.data(), covs.data(), colors.data(), opacities.data(),
                                       static_cast<int>(gaussians.means.size()), c, intr, c2w,
                                       height, width, image.data(), cuda_opts)) {
    return Image();
  }
  return image;
}

GradientsProjected2D gaussian_splat_3d_projected_backward_cuda_impl(
    const Image& grad_output, const ProjectedGaussians2D& gaussians, int height, int width) {
  GradientsProjected2D grads;
  const int n = static_cast<int>(gaussians.means.size());
  const int c = gaussians.colors.empty() ? 3 : static_cast<int>(gaussians.colors[0].size());
  if (n == 0) {
    return grads;
  }

  std::vector<float> means;
  std::vector<float> covs;
  std::vector<float> colors;
  std::vector<float> opacities;
  pack_projected(gaussians, means, covs, colors, opacities);

  std::vector<float> grad_means(static_cast<size_t>(n) * 2);
  std::vector<float> grad_covs(static_cast<size_t>(n) * 4);
  std::vector<float> grad_colors(static_cast<size_t>(n) * c);
  std::vector<float> grad_opacities(static_cast<size_t>(n));

  if (!cuda::gaussian_splat_3d_projected_backward(
          grad_output.data(), means.data(), covs.data(), colors.data(), opacities.data(), n, c,
          height, width, grad_means.data(), grad_covs.data(), grad_colors.data(),
          grad_opacities.data())) {
    return gaussian_splat_3d_projected_backward(grad_output, gaussians, height, width);
  }

  grads.grad_means.resize(static_cast<size_t>(n));
  grads.grad_covariances.resize(static_cast<size_t>(n));
  grads.grad_colors.resize(static_cast<size_t>(n));
  grads.grad_opacities.resize(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    grads.grad_means[static_cast<size_t>(i)] = {grad_means[static_cast<size_t>(i) * 2 + 0],
                                                grad_means[static_cast<size_t>(i) * 2 + 1]};
    grads.grad_covariances[static_cast<size_t>(i)] = {
        grad_covs[static_cast<size_t>(i) * 4 + 0], grad_covs[static_cast<size_t>(i) * 4 + 1],
        grad_covs[static_cast<size_t>(i) * 4 + 2], grad_covs[static_cast<size_t>(i) * 4 + 3]};
    grads.grad_colors[static_cast<size_t>(i)].resize(static_cast<size_t>(c));
    for (int ch = 0; ch < c; ++ch) {
      grads.grad_colors[static_cast<size_t>(i)][ch] = grad_colors[static_cast<size_t>(i) * c + ch];
    }
    grads.grad_opacities[static_cast<size_t>(i)] = grad_opacities[static_cast<size_t>(i)];
  }
  return grads;
}

}  // namespace tinysplat

#endif
