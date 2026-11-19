#include <Halide.h>
#include <cstdio>
#include <cmath>
#include "algorithm.h"

int main() {
    const int N = 2, H = 32, W = 32, C = 3;
    
    // Larger covariance for better coverage
    float means[N*2] = {16.0f, 16.0f, 16.0f, 16.0f};
    float cov[N*2*2] = {64.0f,0,0,64.0f, 64.0f,0,0,64.0f};
    float colors[N*C] = {1,0,0, 0,1,0};
    float opacities[N] = {0.8f, 0.8f};
    
    Halide::Buffer<float> means_buf(means, {N, 2});
    Halide::Buffer<float> cov_buf(cov, {N, 2, 2});
    Halide::Buffer<float> colors_buf(colors, {N, C});
    Halide::Buffer<float> opacities_buf(opacities, {N});
    
    try {
        auto [out_func, _] = tinysplat_halide::build_forward_pipeline(
            means_buf, cov_buf, colors_buf, opacities_buf, H, W, C);
        
        float output[H*W*C] = {0};
        Halide::Buffer<float> out_buf(output, {H, W, C});
        out_func.realize(out_buf);
        
        printf("Forward output[16,16]: %f %f %f\n", 
               output[16*W*C + 16*C + 0], 
               output[16*W*C + 16*C + 1], 
               output[16*W*C + 16*C + 2]);
        
        // Check min/max
        float min_val = 1e10, max_val = -1e10;
        bool has_nan = false;
        for (int i = 0; i < H*W*C; i++) {
            if (std::isnan(output[i])) has_nan = true;
            if (output[i] < min_val) min_val = output[i];
            if (output[i] > max_val) max_val = output[i];
        }
        printf("Forward range: [%f, %f], has NaN: %s\n", 
               min_val, max_val, has_nan ? "yes" : "no");
        
    } catch (const Halide::Error& e) {
        printf("Forward error: %s\n", e.what());
    }
    
    return 0;
}
