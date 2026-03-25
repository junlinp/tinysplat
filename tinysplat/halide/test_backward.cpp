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
    
    float grad_out[H*W*C];
    for (int i = 0; i < H*W*C; i++) grad_out[i] = 1.0f;
    
    float grad_means[N*2] = {0};
    float grad_cov[N*2*2] = {0};
    float grad_colors[N*C] = {0};
    float grad_opacities[N] = {0};
    
    Halide::Buffer<float> grad_out_buf(grad_out, {H, W, C});
    Halide::Buffer<float> means_buf(means, {N, 2});
    Halide::Buffer<float> cov_buf(cov, {N, 2, 2});
    Halide::Buffer<float> colors_buf(colors, {N, C});
    Halide::Buffer<float> opacities_buf(opacities, {N});
    
    Halide::Buffer<float> grad_means_buf(grad_means, {N, 2});
    Halide::Buffer<float> grad_cov_buf(grad_cov, {N, 2, 2});
    Halide::Buffer<float> grad_colors_buf(grad_colors, {N, C});
    Halide::Buffer<float> grad_opacities_buf(grad_opacities, {N});
    
    try {
        auto grad_funcs = tinysplat_halide::build_backward_pipeline(
            grad_out_buf, means_buf, cov_buf, colors_buf, opacities_buf, H, W, C);
        
        // Realize gradients
        grad_funcs.grad_means.realize(grad_means_buf);
        grad_funcs.grad_cov.realize(grad_cov_buf);
        grad_funcs.grad_colors.realize(grad_colors_buf);
        grad_funcs.grad_opacities.realize(grad_opacities_buf);
        
        printf("Backward gradients computed:\n");
        printf("  grad_colors[0]: %f %f %f\n", grad_colors[0], grad_colors[1], grad_colors[2]);
        printf("  grad_colors[1]: %f %f %f\n", grad_colors[3], grad_colors[4], grad_colors[5]);
        printf("  grad_opacities: %f %f\n", grad_opacities[0], grad_opacities[1]);
        printf("  grad_means[0]: %f %f\n", grad_means[0], grad_means[1]);
        printf("  grad_means[1]: %f %f\n", grad_means[2], grad_means[3]);
        return 0;
    } catch (const Halide::Error& e) {
        printf("Halide error: %s\n", e.what());
        return 1;
    }
}
