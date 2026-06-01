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
    int N, int H, int W
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
        1.0f / (2.0f * kPi * sqrtf(det + kEps)),
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
    int* __restrict__ tile_counts,
    int* __restrict__ tile_bins,
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
            int slot = atomicAdd(&tile_counts[ty * tiles_x + tx], 1);
            tile_bins[slot] = g;
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
        float alpha = fminf(1.0f, opacities[g] * gaussian);
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

// ---------------------------------------------------------------------------
// Backward rasterization
// ---------------------------------------------------------------------------

__global__ void rasterize_backward_kernel(
    const float* __restrict__ grad_output,
    const Gaussian2D* __restrict__ gaussians,
    const float* __restrict__ colors,
    const float* __restrict__ opacities,
    const int* __restrict__ tile_starts,
    const int* __restrict__ tile_bins,
    float* __restrict__ grad_means,
    float* __restrict__ grad_covs,
    float* __restrict__ grad_colors,
    float* __restrict__ grad_opacities,
    int N, int H, int W, int C, int tiles_x, int tiles_y
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= N) return;

    const Gaussian2D& gk = gaussians[g];
    float gm_x = 0.0f, gm_y = 0.0f;
    float gop = 0.0f;
    float gcv[4] = {0.f, 0.f, 0.f, 0.f};
    float gcl[4] = {0.f, 0.f, 0.f, 0.f};

    for (int y = gk.min_y; y <= gk.max_y; ++y) {
        for (int x = gk.min_x; x <= gk.max_x; ++x) {
            if (x < 0 || x >= W || y < 0 || y >= H) continue;

            int tile_x = x / kTileSize;
            int tile_y = y / kTileSize;
            int tile_idx = tile_y * tiles_x + tile_x;
            int bin_start = tile_starts[tile_idx];
            int bin_end = tile_starts[tile_idx + 1];

            bool in_tile = false;
            for (int bi = bin_start; bi < bin_end; ++bi) {
                if (tile_bins[bi] == g) { in_tile = true; break; }
            }
            if (!in_tile) continue;

            float dx = (float)x - gk.mean_x;
            float dy = (float)y - gk.mean_y;
            float qx = gk.inv_xx * dx + gk.inv_xy * dy;
            float qy = gk.inv_yx * dx + gk.inv_yy * dy;
            float quad = dx * qx + dy * qy;
            float gaussian = expf(-0.5f * quad) * gk.normalization;
            float alpha = fminf(1.0f, opacities[g] * gaussian);

            float grad_out_r = grad_output[(y * W + x) * C + 0];
            float grad_out_g = C > 1 ? grad_output[(y * W + x) * C + 1] : 0.0f;
            float grad_out_b = C > 2 ? grad_output[(y * W + x) * C + 2] : 0.0f;

            // grad_color
            gcl[0] += grad_out_r * alpha;
            if (C > 1) gcl[1] += grad_out_g * alpha;
            if (C > 2) gcl[2] += grad_out_b * alpha;

            // grad_opacity
            gop += (grad_out_r * colors[g * C] +
                    grad_out_g * colors[g * C + 1] +
                    grad_out_b * colors[g * C + 2]) * gaussian;

            // grad_mean
            float contrib = (grad_out_r + grad_out_g + grad_out_b) * alpha;
            gm_x += contrib * qx;
            gm_y += contrib * qy;

            // grad_cov (simplified)
            float base = (grad_out_r + grad_out_g + grad_out_b) * alpha * opacities[g];
            float outer00 = qx * qx, outer01 = qx * qy;
            float outer10 = qy * qx, outer11 = qy * qy;
            gcv[0] += base * (outer00 - 0.5f * gk.inv_xx);
            gcv[1] += base * (outer01 - 0.5f * gk.inv_xy);
            gcv[2] += base * (outer10 - 0.5f * gk.inv_yx);
            gcv[3] += base * (outer11 - 0.5f * gk.inv_yy);
        }
    }

    grad_means[g * 2 + 0] = gm_x;
    grad_means[g * 2 + 1] = gm_y;
    grad_opacities[g] = gop;
    for (int c = 0; c < C && c < 4; ++c) {
        grad_colors[g * C + c] = gcl[c];
    }
    for (int i = 0; i < 4; ++i) {
        grad_covs[g * 4 + i] = gcv[i];
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Host-side wrappers
// ---------------------------------------------------------------------------

torch::Tensor gaussian_splat_2d_forward_cuda(
    torch::Tensor means,
    torch::Tensor covariances,
    torch::Tensor colors,
    torch::Tensor opacities,
    int64_t height,
    int64_t width
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
    auto gaussians = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    int blocks = (N + 255) / 256;
    precompute_gaussians_kernel<<<blocks, 256>>>(
        means.data_ptr<float>(),
        covariances.data_ptr<float>(),
        (Gaussian2D*)gaussians.data_ptr<float>(),
        N, H, W
    );

    // 2. Count tile memberships
    auto tile_counts = torch::zeros({num_tiles}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    count_tile_membership_kernel<<<blocks, 256>>>(
        (Gaussian2D*)gaussians.data_ptr<float>(),
        tile_counts.data_ptr<int>(),
        tiles_x, tiles_y, N
    );

    // 3. Prefix sum to get offsets
    auto counts_host = tile_counts.cpu();
    std::vector<int> counts(num_tiles);
    for (int i = 0; i < num_tiles; ++i) counts[i] = counts_host.data_ptr<int>()[i];

    std::vector<int> tile_starts(num_tiles + 1, 0);
    for (int i = 0; i < num_tiles; ++i) tile_starts[i + 1] = tile_starts[i] + counts[i];
    int total_bins = tile_starts[num_tiles];

    // 4. Build tile bins
    auto tile_bins = torch::zeros({total_bins}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto tile_counts_tmp = torch::zeros({num_tiles}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    assign_tile_bins_kernel<<<blocks, 256>>>(
        (Gaussian2D*)gaussians.data_ptr<float>(),
        tile_counts_tmp.data_ptr<int>(),
        tile_bins.data_ptr<int>(),
        tiles_x, tiles_y, N
    );

    auto tile_starts_t = torch::from_blob(tile_starts.data(), {num_tiles + 1}, torch::kInt32).clone().to(torch::kCUDA);

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

    return output;
}

std::vector<torch::Tensor> gaussian_splat_2d_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor means,
    torch::Tensor covariances,
    torch::Tensor colors,
    torch::Tensor opacities,
    int64_t height,
    int64_t width
) {
    auto N = means.size(0);
    auto C = colors.size(1);
    int H = (int)height, W = (int)width;
    int tiles_x = (W + kTileSize - 1) / kTileSize;
    int tiles_y = (H + kTileSize - 1) / kTileSize;
    int num_tiles = tiles_x * tiles_y;

    auto gaussians = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    int blocks = (N + 255) / 256;
    precompute_gaussians_kernel<<<blocks, 256>>>(
        means.contiguous().data_ptr<float>(),
        covariances.contiguous().data_ptr<float>(),
        (Gaussian2D*)gaussians.data_ptr<float>(),
        N, H, W
    );

    auto tile_counts = torch::zeros({num_tiles}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    count_tile_membership_kernel<<<blocks, 256>>>(
        (Gaussian2D*)gaussians.data_ptr<float>(),
        tile_counts.data_ptr<int>(),
        tiles_x, tiles_y, N
    );

    auto counts_host = tile_counts.cpu();
    std::vector<int> counts(num_tiles);
    for (int i = 0; i < num_tiles; ++i) counts[i] = counts_host.data_ptr<int>()[i];

    std::vector<int> tile_starts(num_tiles + 1, 0);
    for (int i = 0; i < num_tiles; ++i) tile_starts[i + 1] = tile_starts[i] + counts[i];
    int total_bins = tile_starts[num_tiles];

    auto tile_bins = torch::zeros({total_bins}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto tile_counts_tmp = torch::zeros({num_tiles}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    assign_tile_bins_kernel<<<blocks, 256>>>(
        (Gaussian2D*)gaussians.data_ptr<float>(),
        tile_counts_tmp.data_ptr<int>(),
        tile_bins.data_ptr<int>(),
        tiles_x, tiles_y, N
    );

    auto tile_starts_t = torch::from_blob(tile_starts.data(), {num_tiles + 1}, torch::kInt32).clone().to(torch::kCUDA);

    auto grad_means = torch::zeros_like(means);
    auto grad_covariances = torch::zeros_like(covariances);
    auto grad_colors = torch::zeros_like(colors);
    auto grad_opacities = torch::zeros_like(opacities);

    blocks = (N + 255) / 256;
    rasterize_backward_kernel<<<blocks, 256>>>(
        grad_output.contiguous().data_ptr<float>(),
        (Gaussian2D*)gaussians.data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        tile_starts_t.data_ptr<int>(),
        tile_bins.data_ptr<int>(),
        grad_means.data_ptr<float>(),
        grad_covariances.data_ptr<float>(),
        grad_colors.data_ptr<float>(),
        grad_opacities.data_ptr<float>(),
        N, H, W, C, tiles_x, tiles_y
    );

    return {grad_means, grad_covariances, grad_colors, grad_opacities};
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
    return gaussian_splat_2d_forward_cuda(
        projected_means, projected_covariances, projected_colors, projected_opacities, height, width);
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
    float sigma_radius
) {
    return gaussian_splat_2d_backward_cuda(
        grad_output, projected_means, projected_covariances, projected_colors, projected_opacities, height, width);
}
