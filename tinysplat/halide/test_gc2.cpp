#include <Halide.h>
#include <cstdio>
#include <cmath>
#include "algorithm.h"

int main() {
    const int N = 2, H = 16, W = 16, C = 3;
    
    float means[N*2] = {8.0f, 8.0f, 7.5f, 8.5f};
    float cov[N*2*2] = {8.0f,0,0,8.0f, 8.0f,0,0,8.0f};
    float colors[N*C] = {1,0,0, 0,1,0};
    float opacities[N] = {0.5f, 0.5f};
    
    float output[H*W*C] = {0};
    
    try {
        // Forward
        auto p = tinysplat_halide::build_forward_pipeline(
            Halide::Buffer<float>(means, {N, 2}),
            Halide::Buffer<float>(cov, {N, 2, 2}),
            Halide::Buffer<float>(colors, {N, C}),
            Halide::Buffer<float>(opacities, {N}),
            H, W, C);
        
        // Apply CPU schedule
        tinysplat_halide::apply_cpu_schedule_forward(p, H, W, C);
        
        Halide::Buffer<float> out_buf(output, {H, W, C});
        p.output.realize(out_buf);
        
        printf("Forward realized OK\n");
        printf("  Output[8,8]: %.4f %.4f %.4f\n",
               output[8*W*C + 8*C + 0],
               output[8*W*C + 8*C + 1],
               output[8*W*C + 8*C + 2]);
        
        return 0;
    } catch (const Halide::Error& e) {
        printf("Halide error: %s\n", e.what());
        return 1;
    }
}
