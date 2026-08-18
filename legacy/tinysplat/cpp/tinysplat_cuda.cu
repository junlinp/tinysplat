#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>

namespace {

constexpr int kTileSize = 16;
constexpr float kEps = 1e-8f;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kSigmaRadius = 4.0f;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

struct __align__(8) Gaussian2D {
    float mean_x, mean_y;
    float inv_xx, inv_xy, inv_yx, inv_yy;
    float normalization;
    float opacity;
    int min_x, max_x, min_y, max_y;
};

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void precompute_gaussians_kernel(
    const float* __restrict__ means,  // (N, 2)
    const float* __restrict__ covs,   // (N, 4)
    Gaussian2D* __restrict__ out,
    int N, int H, int W, bool density_normalize
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= N) return;

    float a = covs[g * 4 + 0];
    float b = covs[g * 4 + 1];
    float c = covs[g * 4 + 2];
    float d = covs[g * 4 + 3];

    float det = a * d - b * c;
    if (det < kEps) det = kEps;

    float inv_det = 1.0f / det;
    float trace = a + d;
    float disc = sqrtf(fmaxf(0.0f, (a - d) * (a - d) + 4.0f * b * c));
    float lambda_max = fmaxf((trace + disc) * 0.5f, kEps);
    float radius = ceilf(kSigmaRadius * sqrtf(lambda_max));

    float mx = means[g * 2 + 0];
    float my = means[g * 2 + 1];

    int min_x = max(0, (int)floorf(mx - radius));
    int max_x = min(W - 1, (int)ceilf(mx + radius));
    int min_y = max(0, (int)floorf(my - radius));
    int max_y = min(H - 1, (int)ceilf(my + radius));

    out[g] = Gaussian2D{
        mx, my,
        d * inv_det, -b * inv_det, -c * inv_det, a * inv_det,
        density_normalize ? 1.0f / (2.0f * kPi * sqrtf(det + kEps)) : 1.0f,
        0.0f,
        min_x, max_x, min_y, max_y
    };
}

// Count tile memberships
__global__ void count_tile_membership_kernel(
    const Gaussian2D* __restrict__ gaussians,
    int* __restrict__ tile_counts,
    int tiles_x, int tiles_y, int N
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= N) return;

    const Gaussian2D& gk = gaussians[g];
    if (gk.max_x < gk.min_x || gk.max_y < gk.min_y) return;

    int tile_min_x = gk.min_x / kTileSize;
    int tile_max_x = gk.max_x / kTileSize;
    int tile_min_y = gk.min_y / kTileSize;
    int tile_max_y = gk.max_y / kTileSize;

    for (int ty = tile_min_y; ty <= tile_max_y; ++ty) {
        for (int tx = tile_min_x; tx <= tile_max_x; ++tx) {
            if (tx < 0 || tx >= tiles_x || ty < 0 || ty >= tiles_y) continue;
            atomicAdd(&tile_counts[ty * tiles_x + tx], 1);
        }
    }
}

// Assign gaussian IDs into tile bins
__global__ void assign_tile_bins_kernel(
    const Gaussian2D* __restrict__ gaussians,
    const int* __restrict__ tile_starts,   // exclusive prefix sum, num_tiles + 1
    int* __restrict__ tile_fill,           // per-tile fill counter, zeroed
    int* __restrict__ tile_bins,
    int* __restrict__ bin_tile_ids,        // tile index per slot, for sorting
    int tiles_x, int tiles_y, int N
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= N) return;

    const Gaussian2D& gk = gaussians[g];
    if (gk.max_x < gk.min_x || gk.max_y < gk.min_y) return;

    int tile_min_x = gk.min_x / kTileSize;
    int tile_max_x = gk.max_x / kTileSize;
    int tile_min_y = gk.min_y / kTileSize;
    int tile_max_y = gk.max_y / kTileSize;

    for (int ty = tile_min_y; ty <= tile_max_y; ++ty) {
        for (int tx = tile_min_x; tx <= tile_max_x; ++tx) {
            if (tx < 0 || tx >= tiles_x || ty < 0 || ty >= tiles_y) continue;
            const int tile_idx = ty * tiles_x + tx;
            // atomicAdd returns a per-tile local slot; it must be offset by the
            // tile's base in tile_bins. Without the base every tile wrote from
            // index 0 and the bins aliased each other.
            const int local = atomicAdd(&tile_fill[tile_idx], 1);
            const int slot = tile_starts[tile_idx] + local;
            tile_bins[slot] = g;
            bin_tile_ids[slot] = tile_idx;
        }
    }
}

// ---------------------------------------------------------------------------
// Forward rasterization — alpha compositing
// ---------------------------------------------------------------------------

__global__ void rasterize_forward_kernel(
    const Gaussian2D* __restrict__ gaussians,
    const float* __restrict__ colors,
    const float* __restrict__ opacities,
    const int* __restrict__ tile_starts,
    const int* __restrict__ tile_bins,
    float* __restrict__ output,
    float* __restrict__ total_weight,
    int N, int H, int W, int C, int tiles_x, int tiles_y
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int tile_x = x / kTileSize;
    int tile_y = y / kTileSize;
    int tile_idx = tile_y * tiles_x + tile_x;
    int bin_start = tile_starts[tile_idx];
    int bin_end = tile_starts[tile_idx + 1];

    float3 accum = {0.f, 0.f, 0.f};
    float T = 1.0f;

    for (int idx = bin_start; idx < bin_end; ++idx) {
        int g = tile_bins[idx];
        const Gaussian2D& gk = gaussians[g];

        if (x < gk.min_x || x > gk.max_x || y < gk.min_y || y > gk.max_y) continue;

        float dx = (float)x - gk.mean_x;
        float dy = (float)y - gk.mean_y;
        float qx = gk.inv_xx * dx + gk.inv_xy * dy;
        float qy = gk.inv_yx * dx + gk.inv_yy * dy;
        float quad = dx * qx + dy * qy;
        float gaussian = expf(-0.5f * quad) * gk.normalization;
        float alpha = fminf(0.999f, opacities[g] * gaussian);
        float w = alpha * T;

        accum.x += w * colors[g * C + 0];
        accum.y += w * colors[g * C + 1];
        accum.z += w * colors[g * C + 2];
        T *= (1.0f - alpha);
        if (T < 1e-4f) break;
    }

    output[(y * W + x) * C + 0] = accum.x;
    output[(y * W + x) * C + 1] = accum.y;
    output[(y * W + x) * C + 2] = accum.z;
    if (C > 3) output[(y * W + x) * C + 3] = 1.0f - T;
    total_weight[y * W + x] = 1.0f - T;
}



} // namespace

// ---------------------------------------------------------------------------
// Host-side wrappers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Per-pixel backward rasterization
//
// The previous backward ran one thread per Gaussian over that Gaussian's whole
// bounding box. Profiling a real iteration put it at 91.7% of CUDA time
// (92.5 ms/step): total work is sum of bbox areas, and one large Gaussian
// stalls its entire warp. It also dropped the transmittance factor, so the
// gradient it produced was only an approximation.
//
// This version mirrors the forward: one block per tile, one thread per pixel,
// Gaussians staged through shared memory. Two front-to-back passes avoid the
// numerically awkward T /= (1 - alpha) recovery used by reference 3DGS:
//   pass 1 computes final transmittance and the pixel's total colour,
//   pass 2 re-walks accumulating a prefix, so the suffix needed by the alpha
//          gradient is simply (total - prefix).
// ---------------------------------------------------------------------------

// Sum a value across the warp; lane 0 ends up with the total. Every thread in
// the warp is compositing the same Gaussian at the same loop index, so one
// atomic per warp replaces 32 -- the same trick Metal uses via simd_sum().
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

__global__ void rasterize_backward_perpixel_kernel(
    const float* __restrict__ grad_output,   // (H,W,C)
    const Gaussian2D* __restrict__ gaussians,
    const float* __restrict__ colors,
    const float* __restrict__ opacities,
    const int* __restrict__ tile_starts,
    const int* __restrict__ tile_bins,
    float* __restrict__ grad_means,          // (N,2)
    float* __restrict__ grad_means_abs,      // (N,2) AbsGS: sum of |contribution|
    float* __restrict__ grad_covs,           // (N,4)
    float* __restrict__ grad_colors,         // (N,C)
    float* __restrict__ grad_opacities,      // (N)
    int N, int H, int W, int C, int tiles_x, int tiles_y
) {
    const int tile_idx = blockIdx.y * tiles_x + blockIdx.x;
    const int x = blockIdx.x * kTileSize + threadIdx.x;
    const int y = blockIdx.y * kTileSize + threadIdx.y;
    const bool inside = (x < W) && (y < H);

    const int bin_start = tile_starts[tile_idx];
    const int bin_end   = tile_starts[tile_idx + 1];
    const int n_bin     = bin_end - bin_start;

    const int lane = threadIdx.y * kTileSize + threadIdx.x;   // 0..255
    constexpr int kChunk = kTileSize * kTileSize;             // 256

    __shared__ Gaussian2D sh_g[kChunk];
    __shared__ float sh_col[kChunk * 3];
    __shared__ float sh_opa[kChunk];

    float go[3] = {0.f, 0.f, 0.f};
    if (inside) {
        go[0] = grad_output[(y * W + x) * C + 0];
        if (C > 1) go[1] = grad_output[(y * W + x) * C + 1];
        if (C > 2) go[2] = grad_output[(y * W + x) * C + 2];
    }

    // ---- pass 1: total composited colour and final transmittance ----
    float T = 1.0f;
    float total[3] = {0.f, 0.f, 0.f};
    bool done = !inside;
    for (int base = 0; base < n_bin; base += kChunk) {
        const int m = min(kChunk, n_bin - base);
        if (lane < m) {
            const int g = tile_bins[bin_start + base + lane];
            sh_g[lane] = gaussians[g];
            sh_opa[lane] = opacities[g];
            sh_col[lane*3+0] = colors[g*C+0];
            sh_col[lane*3+1] = (C > 1) ? colors[g*C+1] : 0.f;
            sh_col[lane*3+2] = (C > 2) ? colors[g*C+2] : 0.f;
        }
        __syncthreads();
        if (!done) {
            for (int k = 0; k < m; ++k) {
                const Gaussian2D& gk = sh_g[k];
                if (x < gk.min_x || x > gk.max_x || y < gk.min_y || y > gk.max_y) continue;
                const float dx = (float)x - gk.mean_x, dy = (float)y - gk.mean_y;
                const float qx = gk.inv_xx*dx + gk.inv_xy*dy;
                const float qy = gk.inv_yx*dx + gk.inv_yy*dy;
                const float G  = expf(-0.5f * (dx*qx + dy*qy)) * gk.normalization;
                const float alpha = fminf(0.999f, sh_opa[k] * G);
                const float w = alpha * T;
                total[0] += w * sh_col[k*3+0];
                total[1] += w * sh_col[k*3+1];
                total[2] += w * sh_col[k*3+2];
                T *= (1.0f - alpha);
                if (T < 1e-4f) { done = true; break; }
            }
        }
        __syncthreads();
    }

    // ---- pass 2: gradients, using suffix = total - prefix ----
    T = 1.0f;
    float prefix[3] = {0.f, 0.f, 0.f};
    done = !inside;
    for (int base = 0; base < n_bin; base += kChunk) {
        const int m = min(kChunk, n_bin - base);
        if (lane < m) {
            const int g = tile_bins[bin_start + base + lane];
            sh_g[lane] = gaussians[g];
            sh_opa[lane] = opacities[g];
            sh_col[lane*3+0] = colors[g*C+0];
            sh_col[lane*3+1] = (C > 1) ? colors[g*C+1] : 0.f;
            sh_col[lane*3+2] = (C > 2) ? colors[g*C+2] : 0.f;
        }
        __syncthreads();

        // No `continue`/`break` in this loop: every lane must reach the warp
        // shuffles below, so misses and finished pixels contribute zero rather
        // than diverging. All lanes share the same Gaussian at a given k.
        for (int k = 0; k < m; ++k) {
            const Gaussian2D& gk = sh_g[k];
            const int g = tile_bins[bin_start + base + k];
            const bool hit = !done && inside &&
                             x >= gk.min_x && x <= gk.max_x &&
                             y >= gk.min_y && y <= gk.max_y;

            float d_c0 = 0.f, d_c1 = 0.f, d_c2 = 0.f;
            float d_opa = 0.f, gm_x = 0.f, gm_y = 0.f;
            float d_cv0 = 0.f, d_cv1 = 0.f, d_cv3 = 0.f;
            float alpha = 0.f;

            if (hit) {
                const float dx = (float)x - gk.mean_x, dy = (float)y - gk.mean_y;
                const float qx = gk.inv_xx*dx + gk.inv_xy*dy;
                const float qy = gk.inv_yx*dx + gk.inv_yy*dy;
                const float G  = expf(-0.5f * (dx*qx + dy*qy)) * gk.normalization;
                const float raw = sh_opa[k] * G;
                alpha = fminf(0.999f, raw);
                const float w = alpha * T;

                prefix[0] += w * sh_col[k*3+0];
                prefix[1] += w * sh_col[k*3+1];
                prefix[2] += w * sh_col[k*3+2];

                d_c0 = w * go[0]; d_c1 = w * go[1]; d_c2 = w * go[2];

                const float inv_1ma = 1.0f / fmaxf(1.0f - alpha, 1e-3f);
                float dalpha = 0.f;
                #pragma unroll
                for (int ch = 0; ch < 3; ++ch) {
                    const float suffix = total[ch] - prefix[ch];
                    dalpha += go[ch] * (T * sh_col[k*3+ch] - suffix * inv_1ma);
                }
                if (raw < 0.999f) {          // clamp has zero derivative above
                    d_opa = dalpha * G;
                    const float dquad = -0.5f * G * (dalpha * sh_opa[k]);
                    // d = pixel - mean, so d(mean) picks up the sign flip.
                    gm_x = -2.0f * dquad * qx;
                    gm_y = -2.0f * dquad * qy;
                    // quad = d^T S^-1 d  =>  dquad/dS = -(S^-1 d)(S^-1 d)^T
                    d_cv0 = -dquad * qx * qx;
                    d_cv1 = -dquad * qx * qy;
                    d_cv3 = -dquad * qy * qy;
                }
            }

            // One atomic per warp instead of one per pixel; Metal does the same
            // with simd_sum(). At 400k Gaussians the raw per-pixel atomics left
            // the GPU latency-bound at ~195W of a 450W budget.
            const float r_c0 = warp_sum(d_c0), r_c1 = warp_sum(d_c1), r_c2 = warp_sum(d_c2);
            const float r_op = warp_sum(d_opa);
            const float r_mx = warp_sum(gm_x), r_my = warp_sum(gm_y);
            const float r_ax = warp_sum(fabsf(gm_x)), r_ay = warp_sum(fabsf(gm_y));
            const float r_v0 = warp_sum(d_cv0), r_v1 = warp_sum(d_cv1), r_v3 = warp_sum(d_cv3);

            if ((lane & 31) == 0) {
                if (r_c0 != 0.f) atomicAdd(&grad_colors[g*C+0], r_c0);
                if (C > 1 && r_c1 != 0.f) atomicAdd(&grad_colors[g*C+1], r_c1);
                if (C > 2 && r_c2 != 0.f) atomicAdd(&grad_colors[g*C+2], r_c2);
                if (r_op != 0.f) atomicAdd(&grad_opacities[g], r_op);
                if (r_mx != 0.f) atomicAdd(&grad_means[g*2+0], r_mx);
                if (r_my != 0.f) atomicAdd(&grad_means[g*2+1], r_my);
                if (grad_means_abs != nullptr) {
                    if (r_ax != 0.f) atomicAdd(&grad_means_abs[g*2+0], r_ax);
                    if (r_ay != 0.f) atomicAdd(&grad_means_abs[g*2+1], r_ay);
                }
                if (r_v0 != 0.f) atomicAdd(&grad_covs[g*4+0], r_v0);
                if (r_v1 != 0.f) { atomicAdd(&grad_covs[g*4+1], r_v1);
                                   atomicAdd(&grad_covs[g*4+2], r_v1); }
                if (r_v3 != 0.f) atomicAdd(&grad_covs[g*4+3], r_v3);
            }

            if (hit) {
                T *= (1.0f - alpha);
                if (T < 1e-4f) done = true;
            }
        }
        __syncthreads();
    }
}


// ---------------------------------------------------------------------------
// FastGS VCD support: per-Gaussian high-error footprint hit counts.
// Mirrors the Metal footprint_hit_count kernel. Counts, for each projected
// Gaussian, how many high-error pixels inside its bounding box it actually
// contributes to (alpha >= 1e-4).
// ---------------------------------------------------------------------------

__global__ void footprint_hit_count_kernel(
    const Gaussian2D* __restrict__ gaussians,
    const float* __restrict__ opacities,
    const unsigned char* __restrict__ error_mask,
    int* __restrict__ counts,
    int N, int H, int W
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= N) return;
    const Gaussian2D& gk = gaussians[g];
    if (gk.max_x < gk.min_x || gk.max_y < gk.min_y) return;

    const int x0 = max(gk.min_x, 0), x1 = min(gk.max_x, W - 1);
    const int y0 = max(gk.min_y, 0), y1 = min(gk.max_y, H - 1);
    const float opa = opacities[g];
    int hits = 0;
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            if (error_mask[y * W + x] == 0) continue;
            const float dx = (float)x - gk.mean_x, dy = (float)y - gk.mean_y;
            const float quad = dx * (gk.inv_xx*dx + gk.inv_xy*dy)
                             + dy * (gk.inv_yx*dx + gk.inv_yy*dy);
            if (opa * expf(-0.5f * quad) < 1e-4f) continue;
            ++hits;
        }
    }
    counts[g] = hits;
}


// Build (tile_starts, tile_bins) for an already-precomputed Gaussian2D array.
// Shared so the backward can reuse the forward's bins instead of rebuilding
// them -- count + scan + fill + sort was ~13% of GPU time, run twice per step.
static std::vector<torch::Tensor> build_tile_bins(
    torch::Tensor gaussians, int N, int tiles_x, int tiles_y, int blocks
) {
    const int num_tiles = tiles_x * tiles_y;
    auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    auto tile_counts = torch::zeros({num_tiles}, i32);
    count_tile_membership_kernel<<<blocks, 256>>>(
        (Gaussian2D*)gaussians.data_ptr<float>(), tile_counts.data_ptr<int>(),
        tiles_x, tiles_y, N);

    auto tile_starts = torch::zeros({num_tiles + 1}, i32);
    tile_starts.slice(0, 1, num_tiles + 1).copy_(torch::cumsum(tile_counts, 0, torch::kInt32));
    const int total_bins = tile_starts[num_tiles].item<int>();

    auto tile_bins = torch::zeros({std::max(total_bins, 1)}, i32);
    auto tile_fill = torch::zeros({num_tiles}, i32);
    auto bin_tile_ids = torch::zeros_like(tile_bins);
    assign_tile_bins_kernel<<<blocks, 256>>>(
        (Gaussian2D*)gaussians.data_ptr<float>(), tile_starts.data_ptr<int>(),
        tile_fill.data_ptr<int>(), tile_bins.data_ptr<int>(),
        bin_tile_ids.data_ptr<int>(), tiles_x, tiles_y, N);

    // atomicAdd hands out slots nondeterministically; alpha compositing is
    // order dependent and the caller supplies Gaussians depth sorted, so index
    // order is depth order. Sorting the composite key (tile, index) restores it
    // in one pass and preserves the tile_starts layout.
    if (total_bins > 0) {
        auto keys = bin_tile_ids.to(torch::kInt64) * (int64_t)(N + 1)
                  + tile_bins.to(torch::kInt64);
        tile_bins = tile_bins.index({torch::argsort(keys)}).contiguous();
    }
    return {tile_starts, tile_bins};
}

torch::Tensor gaussian_splat_2d_forward_cuda(
    torch::Tensor means, torch::Tensor covariances, torch::Tensor colors,
    torch::Tensor opacities, int64_t height, int64_t width, bool density_normalize);

static std::vector<torch::Tensor> forward_core(
    torch::Tensor means,
    torch::Tensor covariances,
    torch::Tensor colors,
    torch::Tensor opacities,
    int64_t height,
    int64_t width,
    // 3DGS alpha compositing uses alpha = opacity * exp(-0.5 * quad) with no
    // density normalization; the 1/(2*pi*sqrt(det)) factor only belongs in the
    // weighted mode where it cancels. Applying it here scaled alpha down by the
    // screen-space Gaussian's peak density (~22x on truck) and made every CUDA
    // render far too dim. Default true to preserve the standalone 2D API.
    bool density_normalize
) {
    TORCH_CHECK(means.is_cuda(), "means must be CUDA");
    TORCH_CHECK(covariances.is_cuda(), "covariances must be CUDA");
    TORCH_CHECK(colors.is_cuda(), "colors must be CUDA");
    TORCH_CHECK(opacities.is_cuda(), "opacities must be CUDA");

    auto N = means.size(0);
    auto C = colors.size(1);
    int H = (int)height, W = (int)width;
    int tiles_x = (W + kTileSize - 1) / kTileSize;
    int tiles_y = (H + kTileSize - 1) / kTileSize;
    int num_tiles = tiles_x * tiles_y;

    // 1. Precompute Gaussian params
    // Gaussian2D is 48 bytes (8 floats + 4 ints); the buffer is reinterpreted as
    // Gaussian2D* below, so it must hold N structs, not N floats.
    constexpr int64_t kFloatsPerGaussian2D = sizeof(Gaussian2D) / sizeof(float);
    auto gaussians = torch::zeros({(int64_t)N * kFloatsPerGaussian2D},
                                  torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    int blocks = (N + 255) / 256;
    precompute_gaussians_kernel<<<blocks, 256>>>(
        means.data_ptr<float>(),
        covariances.data_ptr<float>(),
        (Gaussian2D*)gaussians.data_ptr<float>(),
        N, H, W, density_normalize
    );

    // 2-4. Tile bins (shared helper; also returned so the backward can reuse them)
    auto bins = build_tile_bins(gaussians, N, tiles_x, tiles_y, blocks);
    auto tile_starts_t = bins[0];
    auto tile_bins = bins[1];

    // 5. Rasterize
    auto output = torch::zeros({H, W, C}, torch::TensorOptions().dtype(colors.dtype()).device(torch::kCUDA));
    auto total_weight = torch::zeros({H, W}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    rasterize_forward_kernel<<<grid, block>>>(
        (Gaussian2D*)gaussians.data_ptr<float>(),
        colors.data_ptr<float>(),
        opacities.data_ptr<float>(),
        tile_starts_t.data_ptr<int>(),
        tile_bins.data_ptr<int>(),
        output.data_ptr<float>(),
        total_weight.data_ptr<float>(),
        N, H, W, C, tiles_x, tiles_y
    );

    return {output, tile_starts_t, tile_bins};
}

torch::Tensor gaussian_splat_2d_forward_cuda(
    torch::Tensor means, torch::Tensor covariances, torch::Tensor colors,
    torch::Tensor opacities, int64_t height, int64_t width, bool density_normalize
) {
    return forward_core(means, covariances, colors, opacities, height, width,
                        density_normalize)[0];
}

std::vector<torch::Tensor> gaussian_splat_2d_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor means,
    torch::Tensor covariances,
    torch::Tensor colors,
    torch::Tensor opacities,
    int64_t height,
    int64_t width,
    // Must match the forward's convention or the gradients are inconsistent
    // with the rendered image.
    bool density_normalize,
    // Bins from the forward. Empty means rebuild them (the 2D API path).
    torch::Tensor tile_starts_in,
    torch::Tensor tile_bins_in
) {
    auto N = means.size(0);
    auto C = colors.size(1);
    int H = (int)height, W = (int)width;

    // Gaussian2D is 48 bytes (8 floats + 4 ints); the buffer is reinterpreted as
    // Gaussian2D* below, so it must hold N structs, not N floats.
    constexpr int64_t kFloatsPerGaussian2D = sizeof(Gaussian2D) / sizeof(float);
    auto gaussians = torch::zeros({(int64_t)N * kFloatsPerGaussian2D},
                                  torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    int blocks = (N + 255) / 256;
    precompute_gaussians_kernel<<<blocks, 256>>>(
        means.contiguous().data_ptr<float>(),
        covariances.contiguous().data_ptr<float>(),
        (Gaussian2D*)gaussians.data_ptr<float>(),
        N, H, W, density_normalize
    );

    // The per-pixel backward walks the same tile bins as the forward. Reusing
    // the forward's bins avoids repeating count + scan + fill + sort, which was
    // ~13% of GPU time; rebuilding is kept as the fallback for callers that
    // cannot supply them.
    const int tiles_x = (W + kTileSize - 1) / kTileSize;
    const int tiles_y = (H + kTileSize - 1) / kTileSize;
    torch::Tensor tile_starts_t, tile_bins;
    if (tile_starts_in.defined() && tile_starts_in.numel() > 0) {
        tile_starts_t = tile_starts_in.contiguous();
        tile_bins = tile_bins_in.contiguous();
    } else {
        auto bins = build_tile_bins(gaussians, N, tiles_x, tiles_y, blocks);
        tile_starts_t = bins[0];
        tile_bins = bins[1];
    }

    auto grad_means = torch::zeros_like(means);
    auto grad_means_abs = torch::zeros_like(means);
    auto grad_covariances = torch::zeros_like(covariances);
    auto grad_colors = torch::zeros_like(colors);
    auto grad_opacities = torch::zeros_like(opacities);

    dim3 bblock(kTileSize, kTileSize);
    dim3 bgrid(tiles_x, tiles_y);
    rasterize_backward_perpixel_kernel<<<bgrid, bblock>>>(
        grad_output.contiguous().data_ptr<float>(),
        (Gaussian2D*)gaussians.data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        tile_starts_t.data_ptr<int>(),
        tile_bins.data_ptr<int>(),
        grad_means.data_ptr<float>(),
        grad_means_abs.data_ptr<float>(),
        grad_covariances.data_ptr<float>(),
        grad_colors.data_ptr<float>(),
        grad_opacities.data_ptr<float>(),
        N, H, W, C, tiles_x, tiles_y
    );

    return {grad_means, grad_covariances, grad_colors, grad_opacities, grad_means_abs};
}

torch::Tensor gaussian_splat_3d_projected_forward_cuda(
    torch::Tensor projected_means,
    torch::Tensor projected_covariances,
    torch::Tensor projected_colors,
    torch::Tensor projected_opacities,
    int64_t height,
    int64_t width,
    float min_covariance,
    float sigma_radius
) {
    // 3DGS alpha convention: no density normalization.
    return gaussian_splat_2d_forward_cuda(
        projected_means, projected_covariances, projected_colors, projected_opacities,
        height, width, /*density_normalize=*/false);
}

// Same forward, but also hands back the tile bins so the backward can skip
// rebuilding them. Returns {image, tile_starts, tile_bins}.
std::vector<torch::Tensor> gaussian_splat_3d_projected_forward_binned_cuda(
    torch::Tensor projected_means,
    torch::Tensor projected_covariances,
    torch::Tensor projected_colors,
    torch::Tensor projected_opacities,
    int64_t height,
    int64_t width,
    float min_covariance,
    float sigma_radius
) {
    return forward_core(projected_means, projected_covariances, projected_colors,
                        projected_opacities, height, width, /*density_normalize=*/false);
}

std::vector<torch::Tensor> gaussian_splat_2d_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor means,
    torch::Tensor covariances,
    torch::Tensor colors,
    torch::Tensor opacities,
    int64_t height,
    int64_t width,
    // Must match the forward's convention or the gradients are inconsistent
    // with the rendered image.
    bool density_normalize
) {
    auto N = means.size(0);
    auto C = colors.size(1);
    int H = (int)height, W = (int)width;

    // Gaussian2D is 48 bytes (8 floats + 4 ints); the buffer is reinterpreted as
    // Gaussian2D* below, so it must hold N structs, not N floats.
    constexpr int64_t kFloatsPerGaussian2D = sizeof(Gaussian2D) / sizeof(float);
    auto gaussians = torch::zeros({(int64_t)N * kFloatsPerGaussian2D},
                                  torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    int blocks = (N + 255) / 256;
    precompute_gaussians_kernel<<<blocks, 256>>>(
        means.contiguous().data_ptr<float>(),
        covariances.contiguous().data_ptr<float>(),
        (Gaussian2D*)gaussians.data_ptr<float>(),
        N, H, W, density_normalize
    );

    // The per-pixel backward walks the same tile bins as the forward, so the
    // binning has to be rebuilt here (identically, including the depth sort).
    const int tiles_x = (W + kTileSize - 1) / kTileSize;
    const int tiles_y = (H + kTileSize - 1) / kTileSize;
    const int num_tiles = tiles_x * tiles_y;
    auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    auto tile_counts = torch::zeros({num_tiles}, i32);
    count_tile_membership_kernel<<<blocks, 256>>>(
        (Gaussian2D*)gaussians.data_ptr<float>(), tile_counts.data_ptr<int>(),
        tiles_x, tiles_y, N);

    auto tile_starts_t = torch::zeros({num_tiles + 1}, i32);
    tile_starts_t.slice(0, 1, num_tiles + 1).copy_(torch::cumsum(tile_counts, 0, torch::kInt32));
    const int total_bins = tile_starts_t[num_tiles].item<int>();

    auto tile_bins = torch::zeros({std::max(total_bins, 1)}, i32);
    auto tile_fill = torch::zeros({num_tiles}, i32);
    auto bin_tile_ids = torch::zeros_like(tile_bins);
    assign_tile_bins_kernel<<<blocks, 256>>>(
        (Gaussian2D*)gaussians.data_ptr<float>(), tile_starts_t.data_ptr<int>(),
        tile_fill.data_ptr<int>(), tile_bins.data_ptr<int>(),
        bin_tile_ids.data_ptr<int>(), tiles_x, tiles_y, N);
    if (total_bins > 0) {
        auto keys = bin_tile_ids.to(torch::kInt64) * (int64_t)(N + 1)
                  + tile_bins.to(torch::kInt64);
        tile_bins = tile_bins.index({torch::argsort(keys)}).contiguous();
    }

    auto grad_means = torch::zeros_like(means);
    auto grad_means_abs = torch::zeros_like(means);
    auto grad_covariances = torch::zeros_like(covariances);
    auto grad_colors = torch::zeros_like(colors);
    auto grad_opacities = torch::zeros_like(opacities);

    dim3 bblock(kTileSize, kTileSize);
    dim3 bgrid(tiles_x, tiles_y);
    rasterize_backward_perpixel_kernel<<<bgrid, bblock>>>(
        grad_output.contiguous().data_ptr<float>(),
        (Gaussian2D*)gaussians.data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        tile_starts_t.data_ptr<int>(),
        tile_bins.data_ptr<int>(),
        grad_means.data_ptr<float>(),
        grad_means_abs.data_ptr<float>(),
        grad_covariances.data_ptr<float>(),
        grad_colors.data_ptr<float>(),
        grad_opacities.data_ptr<float>(),
        N, H, W, C, tiles_x, tiles_y
    );

    return {grad_means, grad_covariances, grad_colors, grad_opacities, grad_means_abs};
}


std::vector<torch::Tensor> gaussian_splat_3d_projected_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor projected_means,
    torch::Tensor projected_covariances,
    torch::Tensor projected_colors,
    torch::Tensor projected_opacities,
    int64_t height,
    int64_t width,
    float min_covariance,
    float sigma_radius,
    torch::Tensor tile_starts,
    torch::Tensor tile_bins
) {
    // 3DGS alpha convention, matching gaussian_splat_3d_projected_forward_cuda.
    return gaussian_splat_2d_backward_cuda(
        grad_output, projected_means, projected_covariances, projected_colors,
        projected_opacities, height, width, /*density_normalize=*/false,
        tile_starts, tile_bins);
}




// ---------------------------------------------------------------------------
// Fused 3D -> 2D projection (forward + VJP)
//
// Replaces the pure-PyTorch projection chain, which dominated training: an
// iteration cost ~112 ms of which the raster kernels were only 5-8 ms, the
// rest being autograd unwinding hundreds of small ops over N Gaussians.
//
// Semantics match _project_gaussians_3d_to_2d_pytorch exactly (no Inria
// low-pass), so this is a pure speedup and the CPU/CUDA parity test still holds.
// ---------------------------------------------------------------------------

namespace {

struct CamConst {
    float w[9];   // world->camera rotation, row major
    float t[3];   // world->camera translation
    float fx, fy, cx, cy;
    float near_plane, min_cov;
    float lim_x, lim_y;   // 1.3 * tan(fov/2), Inria EWA Jacobian clamp
};

__global__ void project_fwd_kernel(
    const float* __restrict__ means,   // (N,3)
    const float* __restrict__ cov3,    // (N,9)
    float* __restrict__ proj_means,    // (N,2)
    float* __restrict__ cov2d,         // (N,4)  xx,xy,yx,yy
    float* __restrict__ depth,         // (N)
    bool*  __restrict__ visible,       // (N)
    CamConst cam, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const float mx = means[i*3+0], my = means[i*3+1], mz = means[i*3+2];
    const float cx_ = cam.w[0]*mx + cam.w[1]*my + cam.w[2]*mz + cam.t[0];
    const float cy_ = cam.w[3]*mx + cam.w[4]*my + cam.w[5]*mz + cam.t[1];
    const float cz  = cam.w[6]*mx + cam.w[7]*my + cam.w[8]*mz + cam.t[2];

    depth[i]   = cz;
    visible[i] = cz > cam.near_plane;
    const float z = visible[i] ? cz : 1.0f;   // matches torch.where(safe_z)

    proj_means[i*2+0] = cam.fx * cx_ / z + cam.cx;
    proj_means[i*2+1] = cam.fy * cy_ / z + cam.cy;

    // Cc = W * cov3 * W^T
    float tmp[9], cc[9];
    #pragma unroll
    for (int r = 0; r < 3; ++r)
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            float s = 0.f;
            #pragma unroll
            for (int k = 0; k < 3; ++k) s += cam.w[r*3+k] * cov3[i*9 + k*3 + c];
            tmp[r*3+c] = s;
        }
    #pragma unroll
    for (int r = 0; r < 3; ++r)
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            float s = 0.f;
            #pragma unroll
            for (int k = 0; k < 3; ++k) s += tmp[r*3+k] * cam.w[c*3+k];
            cc[r*3+c] = s;
        }

    const float iz = 1.0f / z, iz2 = iz * iz;
    // Inria/FastGS clamp: an off-axis or near-plane Gaussian otherwise gets an
    // unbounded d(screen)/d(camera) term and its 2D covariance explodes.
    const float jx = fminf(fmaxf(cx_ * iz, -cam.lim_x), cam.lim_x) * z;
    const float jy = fminf(fmaxf(cy_ * iz, -cam.lim_y), cam.lim_y) * z;
    const float j00 = cam.fx * iz, j02 = -cam.fx * jx * iz2;
    const float j11 = cam.fy * iz, j12 = -cam.fy * jy * iz2;

    // C2 = J Cc J^T with J = [[j00,0,j02],[0,j11,j12]]
    const float a0 = j00*cc[0] + j02*cc[6];
    const float a1 = j00*cc[1] + j02*cc[7];
    const float a2 = j00*cc[2] + j02*cc[8];
    const float b0 = j11*cc[3] + j12*cc[6];
    const float b1 = j11*cc[4] + j12*cc[7];
    const float b2 = j11*cc[5] + j12*cc[8];

    cov2d[i*4+0] = a0*j00 + a2*j02 + cam.min_cov;
    cov2d[i*4+1] = a1*j11 + a2*j12;
    cov2d[i*4+2] = b0*j00 + b2*j02;
    cov2d[i*4+3] = b1*j11 + b2*j12 + cam.min_cov;
}

__global__ void project_bwd_kernel(
    const float* __restrict__ g_proj_means,  // (N,2)
    const float* __restrict__ g_cov2d,       // (N,4)
    const float* __restrict__ g_depth,       // (N)
    const float* __restrict__ means,         // (N,3)
    const float* __restrict__ cov3,          // (N,9)
    float* __restrict__ grad_means,          // (N,3)
    float* __restrict__ grad_cov3,           // (N,9)
    CamConst cam, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const float mx = means[i*3+0], my = means[i*3+1], mz = means[i*3+2];
    const float cx_ = cam.w[0]*mx + cam.w[1]*my + cam.w[2]*mz + cam.t[0];
    const float cy_ = cam.w[3]*mx + cam.w[4]*my + cam.w[5]*mz + cam.t[1];
    const float cz  = cam.w[6]*mx + cam.w[7]*my + cam.w[8]*mz + cam.t[2];
    const bool  vis = cz > cam.near_plane;
    const float z   = vis ? cz : 1.0f;
    const float iz = 1.0f / z, iz2 = iz*iz, iz3 = iz2*iz;

    float tmp[9], cc[9];
    #pragma unroll
    for (int r = 0; r < 3; ++r)
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            float s = 0.f;
            #pragma unroll
            for (int k = 0; k < 3; ++k) s += cam.w[r*3+k] * cov3[i*9 + k*3 + c];
            tmp[r*3+c] = s;
        }
    #pragma unroll
    for (int r = 0; r < 3; ++r)
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            float s = 0.f;
            #pragma unroll
            for (int k = 0; k < 3; ++k) s += tmp[r*3+k] * cam.w[c*3+k];
            cc[r*3+c] = s;
        }

    const float tx = cx_ * iz, ty = cy_ * iz;
    const bool clamp_x = fabsf(tx) > cam.lim_x;
    const bool clamp_y = fabsf(ty) > cam.lim_y;
    const float jx = (clamp_x ? copysignf(cam.lim_x, tx) : tx) * z;
    const float jy = (clamp_y ? copysignf(cam.lim_y, ty) : ty) * z;
    const float j00 = cam.fx * iz, j02 = -cam.fx * jx * iz2;
    const float j11 = cam.fy * iz, j12 = -cam.fy * jy * iz2;
    // The projected mean is NOT clamped, so d(mean)/d(camera) uses the exact
    // terms rather than the clamped j02/j12.
    const float p02 = -cam.fx * cx_ * iz2;
    const float p12 = -cam.fy * cy_ * iz2;

    const float G00 = g_cov2d[i*4+0], G01 = g_cov2d[i*4+1];
    const float G10 = g_cov2d[i*4+2], G11 = g_cov2d[i*4+3];

    // dCc = J^T G J   (3x3, J is 2x3)
    float Jm[6] = {j00, 0.f, j02, 0.f, j11, j12};   // row major 2x3
    float GJ[6];                                    // G * J  -> 2x3
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        GJ[0*3+c] = G00*Jm[0*3+c] + G01*Jm[1*3+c];
        GJ[1*3+c] = G10*Jm[0*3+c] + G11*Jm[1*3+c];
    }
    float dCc[9];
    #pragma unroll
    for (int r = 0; r < 3; ++r)
        #pragma unroll
        for (int c = 0; c < 3; ++c)
            dCc[r*3+c] = Jm[0*3+r]*GJ[0*3+c] + Jm[1*3+r]*GJ[1*3+c];

    // grad_cov3 = W^T dCc W
    #pragma unroll
    for (int r = 0; r < 3; ++r)
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            float s = 0.f;
            #pragma unroll
            for (int a = 0; a < 3; ++a)
                #pragma unroll
                for (int b = 0; b < 3; ++b)
                    s += cam.w[a*3+r] * dCc[a*3+b] * cam.w[b*3+c];
            grad_cov3[i*9 + r*3 + c] = s;
        }

    // dJ = G J Cc^T + G^T J Cc ; Cc symmetric so dJ = (G + G^T) J Cc
    float Gs00 = G00 + G00, Gs01 = G01 + G10, Gs10 = G10 + G01, Gs11 = G11 + G11;
    float JC[6];   // J * Cc -> 2x3
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        JC[0*3+c] = j00*cc[0*3+c] + j02*cc[2*3+c];
        JC[1*3+c] = j11*cc[1*3+c] + j12*cc[2*3+c];
    }
    // dL/dJ = (G + G^T) J Cc  -- no 1/2 factor.
    float dJ[6];
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        dJ[0*3+c] = Gs00*JC[0*3+c] + Gs01*JC[1*3+c];
        dJ[1*3+c] = Gs10*JC[0*3+c] + Gs11*JC[1*3+c];
    }

    // camera-space gradient
    const float dpx = g_proj_means[i*2+0], dpy = g_proj_means[i*2+1];
    float dcx = j00 * dpx;                 // d(proj mean)/d(camera), unclamped
    float dcy = j11 * dpy;
    float dcz = p02 * dpx + p12 * dpy;

    // through J's dependence on (cx_, cy_, z). Inside the clamp jx = cx_, so the
    // usual terms apply; outside it jx = +/-lim_x * z, which no longer depends on
    // cx_ at all and depends on z only linearly.
    dcz += dJ[0*3+0] * (-cam.fx * iz2) + dJ[1*3+1] * (-cam.fy * iz2);
    if (clamp_x) {
        dcz += dJ[0*3+2] * ( cam.fx * copysignf(cam.lim_x, tx) * iz2);
    } else {
        dcx += dJ[0*3+2] * (-cam.fx * iz2);
        dcz += dJ[0*3+2] * ( 2.0f * cam.fx * cx_ * iz3);
    }
    if (clamp_y) {
        dcz += dJ[1*3+2] * ( cam.fy * copysignf(cam.lim_y, ty) * iz2);
    } else {
        dcy += dJ[1*3+2] * (-cam.fy * iz2);
        dcz += dJ[1*3+2] * ( 2.0f * cam.fy * cy_ * iz3);
    }

    if (!vis) { dcx = 0.f; dcy = 0.f; dcz = 0.f; }   // safe_z branch is constant
    dcz += g_depth[i];                               // depth is emitted directly

    // grad_means = W^T dc
    grad_means[i*3+0] = cam.w[0]*dcx + cam.w[3]*dcy + cam.w[6]*dcz;
    grad_means[i*3+1] = cam.w[1]*dcx + cam.w[4]*dcy + cam.w[7]*dcz;
    grad_means[i*3+2] = cam.w[2]*dcx + cam.w[5]*dcy + cam.w[8]*dcz;
}

CamConst make_cam(torch::Tensor intrinsics, torch::Tensor camera_to_world,
                  float near_plane, float min_cov, float height, float width) {
    auto K = intrinsics.to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto M = camera_to_world.to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto k = K.accessor<float,2>();
    auto m = M.accessor<float,2>();
    CamConst c{};
    // world->camera is the inverse of camera_to_world (rotation transposed)
    for (int r = 0; r < 3; ++r)
        for (int col = 0; col < 3; ++col) c.w[r*3+col] = m[col][r];
    for (int r = 0; r < 3; ++r)
        c.t[r] = -(c.w[r*3+0]*m[0][3] + c.w[r*3+1]*m[1][3] + c.w[r*3+2]*m[2][3]);
    c.fx = k[0][0]; c.fy = k[1][1]; c.cx = k[0][2]; c.cy = k[1][2];
    c.near_plane = near_plane; c.min_cov = min_cov;
    // Metal: tan_fovx = (0.5 * width) / fx. Fall back to the principal point
    // when the image size is not supplied.
    const float tan_fovx = (width > 0.f ? 0.5f * width : c.cx) / fmaxf(c.fx, 1e-6f);
    const float tan_fovy = (height > 0.f ? 0.5f * height : c.cy) / fmaxf(c.fy, 1e-6f);
    c.lim_x = 1.3f * tan_fovx;
    c.lim_y = 1.3f * tan_fovy;
    return c;
}

}  // namespace

std::vector<torch::Tensor> project_3d_forward_cuda(
    torch::Tensor means, torch::Tensor cov3,
    torch::Tensor intrinsics, torch::Tensor camera_to_world,
    double near_plane, double min_covariance,
    double height, double width
) {
    TORCH_CHECK(means.is_cuda(), "means must be CUDA");
    auto m = means.contiguous(), c = cov3.contiguous().view({-1, 9});
    const int N = (int)m.size(0);
    auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto proj_means = torch::empty({N, 2}, f32);
    auto cov2d      = torch::empty({N, 4}, f32);
    auto depth      = torch::empty({N}, f32);
    auto visible    = torch::empty({N}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    if (N > 0) {
        project_fwd_kernel<<<(N + 255) / 256, 256>>>(
            m.data_ptr<float>(), c.data_ptr<float>(),
            proj_means.data_ptr<float>(), cov2d.data_ptr<float>(),
            depth.data_ptr<float>(), visible.data_ptr<bool>(),
            make_cam(intrinsics, camera_to_world, (float)near_plane, (float)min_covariance,
                     (float)height, (float)width), N);
    }
    return {proj_means, cov2d, depth, visible};
}

std::vector<torch::Tensor> project_3d_backward_cuda(
    torch::Tensor grad_proj_means, torch::Tensor grad_cov2d, torch::Tensor grad_depth,
    torch::Tensor means, torch::Tensor cov3,
    torch::Tensor intrinsics, torch::Tensor camera_to_world,
    double near_plane, double min_covariance,
    double height, double width
) {
    auto m  = means.contiguous(), c = cov3.contiguous().view({-1, 9});
    auto gp = grad_proj_means.contiguous(), gc = grad_cov2d.contiguous().view({-1, 4});
    auto gd = grad_depth.contiguous();
    const int N = (int)m.size(0);
    auto grad_means = torch::zeros_like(m);
    auto grad_cov3  = torch::zeros({N, 9}, m.options());
    if (N > 0) {
        project_bwd_kernel<<<(N + 255) / 256, 256>>>(
            gp.data_ptr<float>(), gc.data_ptr<float>(), gd.data_ptr<float>(),
            m.data_ptr<float>(), c.data_ptr<float>(),
            grad_means.data_ptr<float>(), grad_cov3.data_ptr<float>(),
            make_cam(intrinsics, camera_to_world, (float)near_plane, (float)min_covariance,
                     (float)height, (float)width), N);
    }
    return {grad_means, grad_cov3.view({N, 3, 3})};
}


torch::Tensor footprint_hit_count_cuda(
    torch::Tensor projected_means,        // (N,2)
    torch::Tensor projected_covariances,  // (N,2,2)
    torch::Tensor projected_opacities,    // (N,)
    torch::Tensor error_mask,             // (H,W) uint8
    int64_t height, int64_t width
) {
    TORCH_CHECK(projected_means.is_cuda(), "projected_means must be CUDA");
    auto pm = projected_means.contiguous();
    auto pc = projected_covariances.contiguous().view({-1, 4});
    auto po = projected_opacities.contiguous();
    auto mask = error_mask.to(torch::kUInt8).contiguous();
    const int N = (int)pm.size(0);
    const int H = (int)height, W = (int)width;
    auto counts = torch::zeros({N}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    if (N == 0) return counts;

    constexpr int64_t kFloatsPerGaussian2D = sizeof(Gaussian2D) / sizeof(float);
    auto gaussians = torch::zeros({(int64_t)N * kFloatsPerGaussian2D},
                                  torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    const int blocks = (N + 255) / 256;
    // 3DGS alpha convention, matching the forward used for this session.
    precompute_gaussians_kernel<<<blocks, 256>>>(
        pm.data_ptr<float>(), pc.data_ptr<float>(),
        (Gaussian2D*)gaussians.data_ptr<float>(), N, H, W, /*density_normalize=*/false);
    footprint_hit_count_kernel<<<blocks, 256>>>(
        (Gaussian2D*)gaussians.data_ptr<float>(), po.data_ptr<float>(),
        mask.data_ptr<unsigned char>(), counts.data_ptr<int>(), N, H, W);
    return counts;
}

// ---------------------------------------------------------------------------
// Fused quaternion + log-scale projection ("qs"), matching the Metal path.
//
// Without this the caller must build the 3x3 covariance in PyTorch, and
// covariance_matrices() plus its backward showed up as aten::bmm at 38% of GPU
// time -- work Metal never pays because it fuses quat -> covariance ->
// projection in one kernel. This provides the direct 3D Jacobian chain:
// gradients land on means, log_scales and quats without a cov3 intermediate.
// ---------------------------------------------------------------------------

namespace {

// Rotation matrix (row major) from a normalized quaternion (w, x, y, z).
__device__ __forceinline__ void quat_to_R(float w, float x, float y, float z, float* R) {
    const float xx=x*x, yy=y*y, zz=z*z, xy=x*y, xz=x*z, yz=y*z, wx=w*x, wy=w*y, wz=w*z;
    R[0]=1.f-2.f*(yy+zz); R[1]=2.f*(xy-wz);     R[2]=2.f*(xz+wy);
    R[3]=2.f*(xy+wz);     R[4]=1.f-2.f*(xx+zz); R[5]=2.f*(yz-wx);
    R[6]=2.f*(xz-wy);     R[7]=2.f*(yz+wx);     R[8]=1.f-2.f*(xx+yy);
}

// cov3 = R diag(s^2) R^T
__device__ __forceinline__ void qs_to_cov3(const float* q, const float* ls, float* cov3) {
    const float n = rsqrtf(fmaxf(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3], 1e-20f));
    float R[9]; quat_to_R(q[0]*n, q[1]*n, q[2]*n, q[3]*n, R);
    const float s0=expf(ls[0]), s1=expf(ls[1]), s2=expf(ls[2]);
    const float d0=s0*s0, d1=s1*s1, d2=s2*s2;
    #pragma unroll
    for (int r = 0; r < 3; ++r)
        #pragma unroll
        for (int c = 0; c < 3; ++c)
            cov3[r*3+c] = R[r*3+0]*d0*R[c*3+0] + R[r*3+1]*d1*R[c*3+1] + R[r*3+2]*d2*R[c*3+2];
}

// Given dL/dcov3, produce dL/dlog_scales and dL/dquat (unnormalized).
__device__ __forceinline__ void qs_vjp(const float* q, const float* ls, const float* dC,
                                       float* d_ls, float* d_q) {
    const float qn2 = fmaxf(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3], 1e-20f);
    const float inv = rsqrtf(qn2);
    const float w=q[0]*inv, x=q[1]*inv, y=q[2]*inv, z=q[3]*inv;
    float R[9]; quat_to_R(w, x, y, z, R);
    const float s0=expf(ls[0]), s1=expf(ls[1]), s2=expf(ls[2]);
    const float d[3] = {s0*s0, s1*s1, s2*s2};

    // Symmetrize: cov3 is symmetric so only the symmetric part of dC acts.
    float G[9];
    #pragma unroll
    for (int r = 0; r < 3; ++r)
        #pragma unroll
        for (int c = 0; c < 3; ++c) G[r*3+c] = dC[r*3+c] + dC[c*3+r];

    // dL/dD_kk = (R^T dC R)_kk  ->  dL/dls_k = 2 * s_k^2 * that
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        float acc = 0.f;
        #pragma unroll
        for (int a = 0; a < 3; ++a)
            #pragma unroll
            for (int b = 0; b < 3; ++b) acc += R[a*3+k] * dC[a*3+b] * R[b*3+k];
        d_ls[k] = 2.f * d[k] * acc;
    }

    // dL/dR = (dC + dC^T) R D
    float dR[9];
    #pragma unroll
    for (int r = 0; r < 3; ++r)
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            float acc = 0.f;
            #pragma unroll
            for (int a = 0; a < 3; ++a) acc += G[r*3+a] * R[a*3+c];
            dR[r*3+c] = acc * d[c];
        }

    // dR/dn for each quaternion component, contracted with dR.
    const float dn_w = 2.f*(-dR[1]*z + dR[2]*y + dR[3]*z - dR[5]*x - dR[6]*y + dR[7]*x);
    const float dn_x = 2.f*( dR[1]*y + dR[2]*z + dR[3]*y - 2.f*dR[4]*x - dR[5]*w
                           + dR[6]*z + dR[7]*w - 2.f*dR[8]*x);
    const float dn_y = 2.f*(-2.f*dR[0]*y + dR[1]*x + dR[2]*w + dR[3]*x + dR[5]*z
                           - dR[6]*w + dR[7]*z - 2.f*dR[8]*y);
    const float dn_z = 2.f*(-2.f*dR[0]*z - dR[1]*w + dR[2]*x + dR[3]*w - 2.f*dR[4]*z
                           + dR[5]*y + dR[6]*x + dR[7]*y);

    // Back through the normalization n = q / ||q||.
    const float dn[4] = {dn_w, dn_x, dn_y, dn_z};
    const float nq[4] = {w, x, y, z};
    float dot = 0.f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) dot += dn[k] * nq[k];
    #pragma unroll
    for (int k = 0; k < 4; ++k) d_q[k] = (dn[k] - dot * nq[k]) * inv;
}

__global__ void qs_to_cov3_kernel(
    const float* __restrict__ quats, const float* __restrict__ log_scales,
    float* __restrict__ cov3, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    qs_to_cov3(&quats[i*4], &log_scales[i*3], &cov3[i*9]);
}

__global__ void qs_vjp_kernel(
    const float* __restrict__ quats, const float* __restrict__ log_scales,
    const float* __restrict__ grad_cov3,
    float* __restrict__ grad_log_scales, float* __restrict__ grad_quats, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    qs_vjp(&quats[i*4], &log_scales[i*3], &grad_cov3[i*9],
           &grad_log_scales[i*3], &grad_quats[i*4]);
}

}  // namespace

torch::Tensor quat_scale_to_cov3_cuda(torch::Tensor quats, torch::Tensor log_scales) {
    TORCH_CHECK(quats.is_cuda(), "quats must be CUDA");
    auto q = quats.contiguous(), ls = log_scales.contiguous();
    const int N = (int)q.size(0);
    auto cov3 = torch::empty({N, 3, 3}, q.options());
    if (N > 0) {
        qs_to_cov3_kernel<<<(N + 255) / 256, 256>>>(
            q.data_ptr<float>(), ls.data_ptr<float>(), cov3.data_ptr<float>(), N);
    }
    return cov3;
}

std::vector<torch::Tensor> quat_scale_to_cov3_vjp_cuda(
    torch::Tensor quats, torch::Tensor log_scales, torch::Tensor grad_cov3
) {
    auto q = quats.contiguous(), ls = log_scales.contiguous();
    auto gc = grad_cov3.contiguous().view({-1, 9});
    const int N = (int)q.size(0);
    auto g_ls = torch::zeros_like(ls);
    auto g_q  = torch::zeros_like(q);
    if (N > 0) {
        qs_vjp_kernel<<<(N + 255) / 256, 256>>>(
            q.data_ptr<float>(), ls.data_ptr<float>(), gc.data_ptr<float>(),
            g_ls.data_ptr<float>(), g_q.data_ptr<float>(), N);
    }
    return {g_q, g_ls};
}
