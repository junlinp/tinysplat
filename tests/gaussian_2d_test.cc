#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "tinysplat/gaussian_2d.h"
#include "tinysplat/gaussian_3d.h"
#include "tinysplat/types.h"

namespace {

int failures = 0;

void check(bool cond, const char* msg) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    ++failures;
  }
}

tinysplat::Gaussians2D single_gaussian() {
  tinysplat::Gaussians2D g;
  g.means = {{128.0f, 128.0f}};
  g.covariances = {{{25.0f, 0.0f, 0.0f, 25.0f}}};
  g.colors = {{1.0f, 0.0f, 0.0f}};
  g.opacities = {1.0f};
  return g;
}

void test_forward_peak() {
  const auto g = single_gaussian();
  const auto image = tinysplat::gaussian_splat_2d_forward(g, 256, 256);
  const float center = image.at(128, 128, 0);
  const float corner = image.at(0, 0, 0);
  check(center > corner, "center brighter than corner");
  check(std::abs(center - 1.0f) < 0.15f, "center near 1.0");
}

void test_backward_mean() {
  const auto g = single_gaussian();
  tinysplat::Image grad_out(256, 256, 3);
  grad_out.fill(0.0f);
  grad_out.at(140, 128, 0) = 1.0f;
  const auto grads = tinysplat::gaussian_splat_2d_backward(grad_out, g, 256, 256);
  check(std::abs(grads.grad_colors[0][0]) > 1e-6f, "color grad");
}

void test_3d_forward() {
  tinysplat::Gaussians3D g;
  g.means = {{0.0f, 0.0f, 5.0f}};
  g.covariances = {{{0.1f, 0, 0, 0, 0.1f, 0, 0, 0, 0.1f}}};
  g.colors = {{0.5f, 0.5f, 1.0f}};
  g.opacities = {0.8f};

  tinysplat::CameraIntrinsics k{128.0f, 128.0f, 64.0f, 64.0f};
  tinysplat::Mat4 c2w = {};
  c2w.m[0][0] = c2w.m[1][1] = c2w.m[2][2] = c2w.m[3][3] = 1.0f;

  const auto image = tinysplat::gaussian_splat_3d_forward(g, k, c2w, 128, 128);
  float sum = 0.0f;
  for (int y = 0; y < image.height(); ++y) {
    for (int x = 0; x < image.width(); ++x) {
      for (int c = 0; c < image.channels(); ++c) {
        sum += image.at(y, x, c);
      }
    }
  }
  check(sum > 0.0f, "3d image non-empty");
}

}  // namespace

int main() {
  test_forward_peak();
  test_backward_mean();
  test_3d_forward();
  if (failures == 0) {
    std::printf("All tests passed.\n");
    return EXIT_SUCCESS;
  }
  std::fprintf(stderr, "%d test(s) failed.\n", failures);
  return EXIT_FAILURE;
}
