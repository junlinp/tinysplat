#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "tinysplat/gaussian_2d.h"
#include "tinysplat/types.h"

#ifdef TINYSPLAT_CUDA
#include "tinysplat/gaussian_2d_cuda.h"
#endif

namespace {

bool write_ppm(const char* path, const tinysplat::Image& image) {
  FILE* f = std::fopen(path, "wb");
  if (!f) {
    return false;
  }
  std::fprintf(f, "P6\n%d %d\n255\n", image.width(), image.height());
  for (int y = 0; y < image.height(); ++y) {
    for (int x = 0; x < image.width(); ++x) {
      for (int c = 0; c < 3; ++c) {
        const float v = image.at(y, x, std::min(c, image.channels() - 1));
        const int byte = static_cast<int>(std::round(std::clamp(v, 0.0f, 1.0f) * 255.0f));
        std::fputc(byte, f);
      }
    }
  }
  std::fclose(f);
  return true;
}

tinysplat::Gaussians2D make_demo_gaussians() {
  tinysplat::Gaussians2D g;
  g.means = {{96.0f, 96.0f}, {160.0f, 160.0f}};
  g.covariances = {{{100.0f, 0.0f, 0.0f, 100.0f}, {80.0f, 0.0f, 0.0f, 80.0f}}};
  g.colors = {{1.0f, 0.2f, 0.2f}, {0.2f, 1.0f, 0.3f}};
  g.opacities = {0.85f, 0.9f};
  return g;
}

}  // namespace

int main(int argc, char** argv) {
  const int height = 256;
  const int width = 256;
  const bool use_cuda = argc > 1 && std::string(argv[1]) == "--cuda";

  auto gaussians = make_demo_gaussians();
  tinysplat::Image image;

#ifdef TINYSPLAT_CUDA
  if (use_cuda) {
    std::vector<float> means(gaussians.means.size() * 2);
    std::vector<float> covs(gaussians.covariances.size() * 4);
    std::vector<float> colors(gaussians.colors.size() * 3);
    std::vector<float> opacities = gaussians.opacities;
    for (std::size_t i = 0; i < gaussians.means.size(); ++i) {
      means[i * 2 + 0] = gaussians.means[i].x;
      means[i * 2 + 1] = gaussians.means[i].y;
      covs[i * 4 + 0] = gaussians.covariances[i].m00;
      covs[i * 4 + 1] = gaussians.covariances[i].m01;
      covs[i * 4 + 2] = gaussians.covariances[i].m10;
      covs[i * 4 + 3] = gaussians.covariances[i].m11;
      for (int c = 0; c < 3; ++c) {
        colors[i * 3 + c] = gaussians.colors[i][c];
      }
    }
    image = tinysplat::Image(height, width, 3);
    if (!tinysplat::cuda::gaussian_splat_2d_forward(
            means.data(), covs.data(), colors.data(), opacities.data(),
            static_cast<int>(gaussians.means.size()), 3, height, width, image.data())) {
      std::fprintf(stderr, "CUDA render failed; falling back to CPU.\n");
      image = tinysplat::gaussian_splat_2d_forward(gaussians, height, width);
    }
  } else
#endif
  {
    image = tinysplat::gaussian_splat_2d_forward(gaussians, height, width);
  }

  const char* out_path = "render_2d.ppm";
  if (!write_ppm(out_path, image)) {
    std::fprintf(stderr, "Failed to write %s\n", out_path);
    return 1;
  }
  std::printf("Wrote %s (%dx%d, backend=%s)\n", out_path, width, height,
              use_cuda ? "cuda" : "cpu");
  return 0;
}
