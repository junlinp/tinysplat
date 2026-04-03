#include <Halide.h>
#include <cstdio>
#include <cmath>
#include "algorithm.h"
int main() {
    int N = 2, H = 16, W = 32, C = 3;
    float means[N*2] = {float(H/2), float(W/2), float(H/2+1), float(W/2-1)};
    float cov[N*2*2] = {64.0f,0,0,64.0f, 64.0f,0,0,64.0f};
    float colors[N*C] = {1,0,0, 0,1,0};
    float opacities[N] = {0.5f, 0.5f};
    float grad_out[H*W*C]; for(int i=0;i<H*W*C;i++) grad_out[i]=1.0f;
    float grad_means[N*2]={0}, grad_cov[N*2*2]={0}, grad_colors[N*C]={0}, grad_opacities[N]={0};
    auto g = tinysplat_halide::build_backward_pipeline(
        Halide::Buffer<float>(grad_out, {W, H, C}),
        Halide::Buffer<float>(means, {N, 2}),
        Halide::Buffer<float>(cov, {N, 2, 2}),
        Halide::Buffer<float>(colors, {N, C}),
        Halide::Buffer<float>(opacities, {N}),
        H, W, C);
    Halide::Buffer<float> gm(grad_means, {N, 2});
    g.grad_means.realize(gm);
    Halide::Buffer<float> gc(grad_cov, {N, 2, 2});
    g.grad_cov.realize(gc);
    Halide::Buffer<float> gcol(grad_colors, {N, C});
    g.grad_colors.realize(gcol);
    Halide::Buffer<float> gop(grad_opacities, {N});
    g.grad_opacities.realize(gop);
    fprintf(stderr, "Backward H=16 W=32: all OK\n");
    fprintf(stderr, "grad_colors[0]=%f grad_colors[3]=%f\n", grad_colors[0], grad_colors[3]);
    return 0;
}
