#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tinysplat/gaussian_2d.h"
#include "tinysplat/types.h"

#ifdef TINYSPLAT_CUDA
#include <tinysplat/gaussian_2d_cuda.h>
#endif

namespace {

int failures = 0;

void check(bool cond, const char* msg) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    ++failures;
  }
}

float max_abs_diff(const tinysplat::Image& a, const tinysplat::Image& b) {
  float m = 0.0f;
  for (int y = 0; y < a.height(); ++y) {
    for (int x = 0; x < a.width(); ++x) {
      for (int c = 0; c < a.channels(); ++c) {
        m = std::max(m, std::abs(a.at(y, x, c) - b.at(y, x, c)));
      }
    }
  }
  return m;
}

tinysplat::Gaussians2D demo_gaussians() {
  tinysplat::Gaussians2D g;
  g.means = {{64.0f, 64.0f}, {96.0f, 96.0f}};
  g.covariances = {{{20.0f, 0.0f, 0.0f, 20.0f}, {15.0f, 0.0f, 0.0f, 15.0f}}};
  g.colors = {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
  g.opacities = {0.8f, 0.7f};
  return g;
}

void pack_2d(const tinysplat::Gaussians2D& g, std::vector<float>& means, std::vector<float>& covs,
             std::vector<float>& colors, std::vector<float>& opacities) {
  const int n = static_cast<int>(g.means.size());
  means.resize(static_cast<size_t>(n) * 2);
  covs.resize(static_cast<size_t>(n) * 4);
  colors.resize(static_cast<size_t>(n) * 3);
  opacities = g.opacities;
  for (int i = 0; i < n; ++i) {
    means[static_cast<size_t>(i) * 2 + 0] = g.means[static_cast<size_t>(i)].x;
    means[static_cast<size_t>(i) * 2 + 1] = g.means[static_cast<size_t>(i)].y;
    covs[static_cast<size_t>(i) * 4 + 0] = g.covariances[static_cast<size_t>(i)].m00;
    covs[static_cast<size_t>(i) * 4 + 1] = g.covariances[static_cast<size_t>(i)].m01;
    covs[static_cast<size_t>(i) * 4 + 2] = g.covariances[static_cast<size_t>(i)].m10;
    covs[static_cast<size_t>(i) * 4 + 3] = g.covariances[static_cast<size_t>(i)].m11;
    for (int c = 0; c < 3; ++c) {
      colors[static_cast<size_t>(i) * 3 + c] = g.colors[static_cast<size_t>(i)][c];
    }
  }
}

}  // namespace

int main() {
#ifndef TINYSPLAT_CUDA
  std::printf("SKIP: build with --define=cuda=1\n");
  return EXIT_SUCCESS;
#else
  if (!tinysplat::cuda::cuda_available()) {
    std::printf("SKIP: no CUDA device\n");
    return EXIT_SUCCESS;
  }

  const auto g = demo_gaussians();
  const int h = 128;
  const int w = 128;
  const auto cpu_img = tinysplat::gaussian_splat_2d_forward(g, h, w);

  std::vector<float> means;
  std::vector<float> covs;
  std::vector<float> colors;
  std::vector<float> opacities;
  pack_2d(g, means, covs, colors, opacities);

  tinysplat::Image cuda_img(h, w, 3);
  const bool ok = tinysplat::cuda::gaussian_splat_2d_forward(
      means.data(), covs.data(), colors.data(), opacities.data(),
      static_cast<int>(g.means.size()), 3, h, w, cuda_img.data(),
      tinysplat::cuda::CompositingMode::Weighted);
  check(ok, "cuda forward");
  const float diff = max_abs_diff(cpu_img, cuda_img);
  check(diff < 0.05f, "cpu vs cuda forward max diff < 0.05");

  tinysplat::Image grad_out(h, w, 3);
  grad_out.fill(0.0f);
  grad_out.at(70, 70, 0) = 1.0f;
  const auto cpu_grads = tinysplat::gaussian_splat_2d_backward(grad_out, g, h, w);

  std::vector<float> grad_means(g.means.size() * 2);
  std::vector<float> grad_covs(g.means.size() * 4);
  std::vector<float> grad_colors(g.means.size() * 3);
  std::vector<float> grad_opacities(g.means.size());
  const bool bwd_ok = tinysplat::cuda::gaussian_splat_2d_backward(
      grad_out.data(), means.data(), covs.data(), colors.data(), opacities.data(),
      static_cast<int>(g.means.size()), 3, h, w, grad_means.data(), grad_covs.data(),
      grad_colors.data(), grad_opacities.data());
  check(bwd_ok, "cuda backward");
  check(std::abs(cpu_grads.grad_means[0].x - grad_means[0]) +
            std::abs(cpu_grads.grad_means[0].y - grad_means[1]) >
        1e-8f,
        "mean grad non-zero spot check");
#endif

  if (failures == 0) {
    std::printf("All CUDA tests passed.\n");
    return EXIT_SUCCESS;
  }
  std::fprintf(stderr, "%d test(s) failed.\n", failures);
  return EXIT_FAILURE;
}
