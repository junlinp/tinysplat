#include <Halide.h>
#include <cstdio>
#include "algorithm.h"

int main() {
    const int N = 2, H = 16, W = 16, C = 3;
    float means[N*2] = {8.0f, 8.0f, 7.5f, 8.5f};
    float cov[N*2*2] = {8.0f,0,0,8.0f, 8.0f,0,0,8.0f};
    float colors[N*C] = {1,0,0, 0,1,0};
    float opacities[N] = {0.5f, 0.5f};
    float grad_out[H*W*C]; for (int i = 0; i < H*W*C; i++) grad_out[i] = 1.0f;

    float grad_means[N*2] = {0}, grad_cov[N*2*2] = {0};
    float grad_colors[N*C] = {0}, grad_opacities[N] = {0};

    try {
        auto g = tinysplat_halide::build_backward_pipeline(
            Halide::Buffer<float>(grad_out, {H, W, C}),
            Halide::Buffer<float>(means, {N, 2}),
            Halide::Buffer<float>(cov, {N, 2, 2}),
            Halide::Buffer<float>(colors, {N, C}),
            Halide::Buffer<float>(opacities, {N}),
            H, W, C);
        printf("Pipeline built OK\n");
        
        Halide::Buffer<float> gm(grad_means, {N, 2});
        Halide::Buffer<float> gc(grad_cov, {N, 2, 2});
        Halide::Buffer<float> gcol(grad_colors, {N, C});
        Halide::Buffer<float> gop(grad_opacities, {N});
        
        g.grad_means.realize(gm);
        printf("grad_means realized OK\n");
        
        g.grad_cov.realize(gc);
        printf("grad_cov realized OK\n");
        
        g.grad_opacities.realize(gop);
        printf("grad_opacities realized OK\n");
        
        printf("All gradient Funcs realized successfully\n");
        return 0;
    } catch (const Halide::Error& e) {
        printf("Halide error: %s\n", e.what());
        return 1;
    } catch (const std::exception& e) {
        printf("std error: %s\n", e.what());
        return 1;
    }
}
