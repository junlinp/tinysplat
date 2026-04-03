#include <Halide.h>
#include <cstdio>
#include <cmath>
#include "algorithm.h"

int main() {
    int sizes[][2] = {{16,16}, {16,32}, {32,16}, {32,32}, {64,64}};
    for (auto& s : sizes) {
        int H = s[0], W = s[1], N = 2, C = 3;
        float means[N*2] = {float(H/2), float(W/2), float(H/2+1), float(W/2-1)};
        float cov[N*2*2] = {8.0f,0,0,8.0f, 8.0f,0,0,8.0f};
        float colors[N*C] = {1,0,0, 0,1,0};
        float opacities[N] = {0.5f, 0.5f};
        float output[H*W*C];
        try {
            auto p = tinysplat_halide::build_forward_pipeline(
                Halide::Buffer<float>(means, {N, 2}),
                Halide::Buffer<float>(cov, {N, 2, 2}),
                Halide::Buffer<float>(colors, {N, C}),
                Halide::Buffer<float>(opacities, {N}),
                H, W, C);
            Halide::Buffer<float> out_buf(output, {H, W, C});
            p.output.realize(out_buf);
            float mn = 1e9, mx = -1e9;
            for (int i = 0; i < H*W*C; i++) {
                mn = fminf(mn, output[i]);
                mx = fmaxf(mx, output[i]);
            }
            printf("H=%d W=%d: OK [%.4f, %.4f]\n", H, W, mn, mx);
        } catch (const Halide::Error& e) {
            printf("H=%d W=%d: ERROR - %s\n", H, W, e.what());
        }
    }
    return 0;
}
