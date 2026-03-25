#include <Halide.h>
#include <cstdio>
#include <cmath>
#include "algorithm.h"

int main() {
    const int N = 2, H = 16, W = 16, C = 3;
    
    float means[N*2] = {8.0f, 8.0f, 8.0f, 8.0f};
    float cov[N*2*2] = {8.0f,0,0,8.0f, 8.0f,0,0,8.0f};
    float colors[N*C] = {1,0,0, 0,1,0};
    float opacities[N] = {0.8f, 0.8f};
    
    // Uniform gradient
    float grad_out[H*W*C];
    for (int i = 0; i < H*W*C; i++) grad_out[i] = 1.0f;
    
    Halide::Buffer<float> grad_out_buf(grad_out, {H, W, C});
    Halide::Buffer<float> means_buf(means, {N, 2});
    Halide::Buffer<float> cov_buf(cov, {N, 2, 2});
    Halide::Buffer<float> colors_buf(colors, {N, C});
    Halide::Buffer<float> opacities_buf(opacities, {N});
    
    // First compute forward to get total_weight_pix
    try {
        auto [out_func, _] = tinysplat_halide::build_forward_pipeline(
            means_buf, cov_buf, colors_buf, opacities_buf, H, W, C);
        
        float output[H*W*C] = {0};
        Halide::Buffer<float> out_buf(output, {H, W, C});
        out_func.realize(out_buf);
        
        printf("Forward output[8,8]: %f %f %f\n", output[8*W*C + 8*C + 0], output[8*W*C + 8*C + 1], output[8*W*C + 8*C + 2]);
        
        // Check if any NaN in forward
        bool has_nan = false;
        for (int i = 0; i < H*W*C; i++) {
            if (std::isnan(output[i])) { has_nan = true; break; }
        }
        printf("Forward has NaN: %s\n", has_nan ? "yes" : "no");
        
    } catch (const Halide::Error& e) {
        printf("Forward error: %s\n", e.what());
    }
    
    return 0;
}
