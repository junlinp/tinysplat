#include <Halide.h>
#include <cstdio>
#include <cmath>
#include "algorithm.h"

int main() {
    // Try larger dimensions to see if it's a size issue
    for (int H = 16; H <= 64; H += 16) {
        for (int W = 16; W <= 64; W += 16) {
            int N = 2, C = 3;
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
                tinysplat_halide::apply_cpu_schedule_forward(p, H, W, C);
                Halide::Buffer<float> out_buf(output, {H, W, C});
                p.output.realize(out_buf);
                printf("H=%d W=%d: OK\n", H, W);
            } catch (const Halide::Error& e) {
                printf("H=%d W=%d: ERROR - %s\n", H, W, e.what());
            }
        }
    }
    return 0;
}
