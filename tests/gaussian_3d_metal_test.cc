#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "tinysplat/gaussian_3d.h"
#include "tinysplat/types.h"

namespace {

tinysplat::Gaussians3D make_scene(int n, std::mt19937& rng) {
  std::uniform_real_distribution<float> pos(-1.5f, 1.5f);
  std::uniform_real_distribution<float> col(0.1f, 0.9f);
  std::uniform_real_distribution<float> op(0.4f, 0.9f);

  tinysplat::Gaussians3D g;
  g.means.resize(static_cast<size_t>(n));
  g.covariances.resize(static_cast<size_t>(n));
  g.colors.resize(static_cast<size_t>(n), std::vector<float>(3));
  g.opacities.resize(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    g.means[static_cast<size_t>(i)] = {pos(rng), pos(rng), 4.0f + pos(rng) * 0.3f};
    tinysplat::Mat3 cov = {};
    cov.m[0][0] = cov.m[1][1] = cov.m[2][2] = 0.04f;
    g.covariances[static_cast<size_t>(i)] = cov;
    for (int c = 0; c < 3; ++c) {
      g.colors[static_cast<size_t>(i)][c] = col(rng);
    }
    g.opacities[static_cast<size_t>(i)] = op(rng);
  }
  return g;
}

float max_abs_diff(const tinysplat::Image& a, const tinysplat::Image& b) {
  float m = 0.0f;
  const int n = a.height() * a.width() * a.channels();
  for (int i = 0; i < n; ++i) {
    m = std::max(m, std::fabs(a.data()[i] - b.data()[i]));
  }
  return m;
}

}  // namespace

int main(int argc, char** argv) {
  int n = 64;
  int h = 64;
  int w = 64;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--n" && i + 1 < argc) {
      n = std::atoi(argv[++i]);
    } else if (arg == "--h" && i + 1 < argc) {
      h = std::atoi(argv[++i]);
    } else if (arg == "--w" && i + 1 < argc) {
      w = std::atoi(argv[++i]);
    }
  }

  std::mt19937 rng(7);
  const auto g = make_scene(n, rng);
  tinysplat::CameraIntrinsics k{static_cast<float>(w) * 0.9f, static_cast<float>(h) * 0.9f,
                                static_cast<float>(w) * 0.5f, static_cast<float>(h) * 0.5f};
  tinysplat::Mat4 c2w = {};
  c2w.m[0][0] = c2w.m[1][1] = c2w.m[2][2] = c2w.m[3][3] = 1.0f;

  tinysplat::Splat3DOptions cpu_opts;
  cpu_opts.use_compact_box = true;
  cpu_opts.compact_box_beta = 3.0f;
  const auto cpu = tinysplat::gaussian_splat_3d_forward(g, k, c2w, h, w, cpu_opts);

  if (!tinysplat::metal_raster_available()) {
    std::printf("Metal unavailable; CPU render OK (%dx%d, N=%d)\n", w, h, n);
    return 0;
  }

  tinysplat::Splat3DOptions metal_opts = cpu_opts;
  metal_opts.use_metal = true;
  const auto metal = tinysplat::gaussian_splat_3d_forward(g, k, c2w, h, w, metal_opts);
  const float diff = max_abs_diff(cpu, metal);
  std::printf("CPU vs Metal max abs diff: %.6f (N=%d %dx%d)\n", diff, n, w, h);
  if (diff > 5e-3f) {
    std::fprintf(stderr, "FAIL: Metal diverges from CPU beyond tolerance\n");
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
