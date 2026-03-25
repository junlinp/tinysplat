#include <Halide.h>
#include <cstdio>
#include <cmath>
#include "algorithm.h"

int main() {
    const int N = 4, H = 32, W = 32, C = 3;
    
    float means[N*2] = {16.0f, 16.0f, 20.0f, 20.0f, 10.0f, 10.0f, 25.0f, 25.0f};
    float cov[N*2*2] = {64.0f,0,0,64.0f, 64.0f,0,0,64.0f, 64.0f,0,0,64.0f, 64.0f,0,0,64.0f};
    float colors[N*C] = {1,0,0, 0,1,0, 0,0,1, 1,1,0};
    float opacities[N] = {0.5f, 0.5f, 0.5f, 0.5f};
    
    Halide::Buffer<float> means_buf(means, {N, 2});
    Halide::Buffer<float> cov_buf(cov, {N, 2, 2});
    Halide::Buffer<float> colors_buf(colors, {N, C});
    Halide::Buffer<float> opacities_buf(opacities, {N});
    
    try {
        auto p = tinysplat_halide::build_forward_pipeline(
            means_buf, cov_buf, colors_buf, opacities_buf, H, W, C);
        
        // Apply CPU schedule
        tinysplat_halide::apply_cpu_schedule_forward(p, H, W, C);
        
        float output[H*W*C] = {0};
        Halide::Buffer<float> out_buf(output, {H, W, C});
        p.output.realize(out_buf);
        
        printf("Forward output[16,16]: %.4f %.4f %.4f\n",
               output[16*W*C + 16*C + 0],
               output[16*W*C + 16*C + 1],
               output[16*W*C + 16*C + 2]);
        
        bool has_nan = false;
        for (int i = 0; i < H*W*C; i++) {
            if (std::isnan(output[i])) { has_nan = true; break; }
        }
        float mx = -1e10, mn = 1e10;
        for (int i = 0; i < H*W*C; i++) {
            if (output[i] > mx) mx = output[i];
            if (output[i] < mn) mn = output[i];
        }
        printf("Forward range: [%.4f, %.4f], NaN: %s\n", mn, mx, has_nan ? "yes" : "no");
        printf("Forward: %s\n", has_nan ? "FAIL" : "OK");
        return has_nan ? 1 : 0;
    } catch (const Halide::Error& e) {
        printf("Halide error: %s\n", e.what());
        return 1;
    }
}
