#include <Halide.h>
#include <cstdio>
#include "algorithm.h"

int main() {
    const int N = 4, H = 32, W = 32, C = 3;
    
    float means[N*2] = {16, 16, 20, 20, 10, 10, 25, 25};
    float cov[N*2*2] = {10,0,0,10, 10,0,0,10, 10,0,0,10, 10,0,0,10};
    float colors[N*C] = {1,0,0, 0,1,0, 0,0,1, 1,1,0};
    float opacities[N] = {0.5, 0.5, 0.5, 0.5};
    float output[H*W*C] = {0};
    
    Halide::Buffer<float> means_buf(means, {N, 2});
    Halide::Buffer<float> cov_buf(cov, {N, 2, 2});
    Halide::Buffer<float> colors_buf(colors, {N, C});
    Halide::Buffer<float> opacities_buf(opacities, {N});
    Halide::Buffer<float> output_buf(output, {H, W, C});
    
    try {
        auto [out_func, intermediates] = tinysplat_halide::build_forward_pipeline(
            means_buf, cov_buf, colors_buf, opacities_buf, H, W, C);
        
        Halide::Var x("x"), y("y"), c("c");
        out_func.bound(x, 0, W);
        out_func.bound(y, 0, H);
        out_func.bound(c, 0, C);
        
        out_func.realize(output_buf);
        
        printf("Success! Output[16,16,0] = %f\n", output[16*W*C + 16*C + 0]);
        return 0;
    } catch (const Halide::Error& e) {
        printf("Halide error: %s\n", e.what());
        return 1;
    } catch (const std::exception& e) {
        printf("std error: %s\n", e.what());
        return 1;
    }
}
