#include <Halide.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

extern "C" int gaussian_splat_forward(float* means, float* covariances, float* colors,
                                      float* opacities, int N, int height, int width,
                                      int C, float* output);

int main() {
    const int N = 64, H = 50, W = 50, C = 5;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> mean_x_dist(0.0f, float(W - 1));
    std::uniform_real_distribution<float> mean_y_dist(0.0f, float(H - 1));
    std::uniform_real_distribution<float> color_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> opacity_dist(0.1f, 1.0f);
    std::normal_distribution<float> a_dist(0.0f, 1.0f);

    std::vector<float> means(N * 2);
    std::vector<float> cov(N * 2 * 2);
    std::vector<float> colors(N * C);
    std::vector<float> opacities(N);
    std::vector<float> output(H * W * C, 0.0f);

    for (int n = 0; n < N; ++n) {
        means[n * 2 + 0] = mean_x_dist(rng);
        means[n * 2 + 1] = mean_y_dist(rng);

        // A*A^T + eps*I -> PSD covariance.
        float a00 = a_dist(rng), a01 = a_dist(rng), a10 = a_dist(rng), a11 = a_dist(rng);
        float c00 = a00 * a00 + a01 * a01 + 1e-2f;
        float c01 = a00 * a10 + a01 * a11;
        float c11 = a10 * a10 + a11 * a11 + 1e-2f;
        cov[n * 4 + 0] = c00;
        cov[n * 4 + 1] = c01;
        cov[n * 4 + 2] = c01;
        cov[n * 4 + 3] = c11;

        for (int c = 0; c < C; ++c) colors[n * C + c] = color_dist(rng);
        opacities[n] = opacity_dist(rng);
    }

    const int rc = gaussian_splat_forward(
        means.data(), cov.data(), colors.data(), opacities.data(),
        N, H, W, C, output.data());
    if (rc != 0) {
        std::fprintf(stderr, "gaussian_splat_forward failed with rc=%d\n", rc);
        return 2;
    }

    int nan_count = 0;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {
                float v = output[(y * W + x) * C + c];
                if (std::isnan(v)) {
                    if (nan_count < 80) {
                        std::printf("NaN at (y=%d, x=%d, c=%d)\n", y, x, c);
                    }
                    ++nan_count;
                }
            }
        }
    }

    std::printf("nan_count=%d total=%d\n", nan_count, H * W * C);
    std::printf("sample center=%f edge00=%f edge0w=%f edgeh0=%f edgehw=%f\n",
                output[((H / 2) * W + (W / 2)) * C + 0],
                output[(0 * W + 0) * C + 0],
                output[(0 * W + (W - 1)) * C + 0],
                output[((H - 1) * W + 0) * C + 0],
                output[((H - 1) * W + (W - 1)) * C + 0]);

    return nan_count == 0 ? 0 : 1;
}
