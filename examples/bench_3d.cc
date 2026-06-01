#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "tinysplat/gaussian_3d.h"
#include "tinysplat/types.h"

namespace {

tinysplat::Gaussians3D random_gaussians(int n, std::mt19937& rng) {
  std::uniform_real_distribution<float> pos(-2.0f, 2.0f);
  std::uniform_real_distribution<float> col(0.0f, 1.0f);
  std::uniform_real_distribution<float> op(0.3f, 0.9f);

  tinysplat::Gaussians3D g;
  g.means.resize(static_cast<size_t>(n));
  g.covariances.resize(static_cast<size_t>(n));
  g.colors.resize(static_cast<size_t>(n), std::vector<float>(3));
  g.opacities.resize(static_cast<size_t>(n));

  for (int i = 0; i < n; ++i) {
    g.means[static_cast<size_t>(i)] = {pos(rng), pos(rng), 3.0f + pos(rng) * 0.5f};
    tinysplat::Mat3 cov = {};
    cov.m[0][0] = cov.m[1][1] = cov.m[2][2] = 0.05f;
    g.covariances[static_cast<size_t>(i)] = cov;
    for (int c = 0; c < 3; ++c) {
      g.colors[static_cast<size_t>(i)][c] = col(rng);
    }
    g.opacities[static_cast<size_t>(i)] = op(rng);
  }
  return g;
}

double bench_ms(const tinysplat::Gaussians3D& g, const tinysplat::CameraIntrinsics& k,
                const tinysplat::Mat4& c2w, int h, int w, bool use_cuda, int repeats) {
  tinysplat::Splat3DOptions opts;
  opts.use_cuda = use_cuda;

  // warmup
  (void)tinysplat::gaussian_splat_3d_forward(g, k, c2w, h, w, opts);

  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < repeats; ++i) {
    (void)tinysplat::gaussian_splat_3d_forward(g, k, c2w, h, w, opts);
  }
  const auto t1 = std::chrono::steady_clock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count() / static_cast<double>(repeats);
  return ms;
}

}  // namespace

int main(int argc, char** argv) {
  int n = 10000;
  int h = 720;
  int w = 1280;
  int repeats = 5;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--n" && i + 1 < argc) {
      n = std::atoi(argv[++i]);
    } else if (arg == "--h" && i + 1 < argc) {
      h = std::atoi(argv[++i]);
    } else if (arg == "--w" && i + 1 < argc) {
      w = std::atoi(argv[++i]);
    } else if (arg == "--repeats" && i + 1 < argc) {
      repeats = std::atoi(argv[++i]);
    }
  }

  std::mt19937 rng(42);
  const auto g = random_gaussians(n, rng);

  tinysplat::CameraIntrinsics k{static_cast<float>(w) * 0.5f, static_cast<float>(h) * 0.5f,
                                static_cast<float>(w) * 0.5f, static_cast<float>(h) * 0.5f};
  tinysplat::Mat4 c2w = {};
  c2w.m[0][0] = c2w.m[1][1] = c2w.m[2][2] = c2w.m[3][3] = 1.0f;

  const double cpu_ms = bench_ms(g, k, c2w, h, w, false, repeats);
  std::printf("CPU 3D forward: %.2f ms/frame (N=%d, %dx%d, repeats=%d)\n", cpu_ms, n, w, h,
              repeats);

  if (tinysplat::cuda_raster_available()) {
    const double cuda_ms = bench_ms(g, k, c2w, h, w, true, repeats);
    std::printf("CUDA 3D forward: %.2f ms/frame (speedup %.2fx)\n", cuda_ms, cpu_ms / cuda_ms);
  } else {
    std::printf("CUDA unavailable (build with --define=cuda=1 and a GPU)\n");
  }

  return 0;
}
