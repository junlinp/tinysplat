#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "tinysplat/gaussian_3d.h"
#include "tinysplat/image.h"
#include "tinysplat/types.h"

namespace {

void write_ppm(const tinysplat::Image& img, const char* path) {
  FILE* f = std::fopen(path, "wb");
  if (!f) {
    return;
  }
  std::fprintf(f, "P6\n%d %d\n255\n", img.width(), img.height());
  for (int y = 0; y < img.height(); ++y) {
    for (int x = 0; x < img.width(); ++x) {
      for (int c = 0; c < 3; ++c) {
        const float v = img.channels() > c ? img.at(y, x, c) : 0.0f;
        const int b = static_cast<int>(std::max(0.0f, std::min(1.0f, v)) * 255.0f + 0.5f);
        std::fputc(b, f);
      }
    }
  }
  std::fclose(f);
}

}  // namespace

int main(int argc, char** argv) {
  bool use_metal = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--metal") {
      use_metal = true;
    }
  }

  const int n = 200;
  const int h = 256;
  const int w = 256;
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> pos(-1.2f, 1.2f);
  std::uniform_real_distribution<float> col(0.2f, 1.0f);

  tinysplat::Gaussians3D g;
  g.means.resize(n);
  g.covariances.resize(n);
  g.colors.resize(n, std::vector<float>(3));
  g.opacities.resize(n);
  for (int i = 0; i < n; ++i) {
    g.means[static_cast<size_t>(i)] = {pos(rng), pos(rng), 3.5f};
    tinysplat::Mat3 cov = {};
    cov.m[0][0] = cov.m[1][1] = cov.m[2][2] = 0.03f;
    g.covariances[static_cast<size_t>(i)] = cov;
    g.colors[static_cast<size_t>(i)] = {col(rng), col(rng), col(rng)};
    g.opacities[static_cast<size_t>(i)] = 0.7f;
  }

  tinysplat::CameraIntrinsics k{200.0f, 200.0f, 128.0f, 128.0f};
  tinysplat::Mat4 c2w = {};
  c2w.m[0][0] = c2w.m[1][1] = c2w.m[2][2] = c2w.m[3][3] = 1.0f;

  tinysplat::Splat3DOptions opts;
  opts.use_metal = use_metal;
  opts.use_compact_box = true;
  const auto img = tinysplat::gaussian_splat_3d_forward(g, k, c2w, h, w, opts);
  const char* out = use_metal ? "render_3d_metal.ppm" : "render_3d_cpu.ppm";
  write_ppm(img, out);
  std::printf("Wrote %s (metal=%d available=%d)\n", out, use_metal ? 1 : 0,
              tinysplat::metal_raster_available() ? 1 : 0);
  return 0;
}
