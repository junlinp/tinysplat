#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <vector>
#include <cmath>
#include <cstdio>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

namespace {

constexpr int TILE = 16;
constexpr int BW = 16;
constexpr int BH = 16;
constexpr int MAX_POOL = 64;            // max Gaussians per tile
constexpr int MAX_PIXELS = TILE * TILE; // 256 pixels per tile
constexpr float EPS = 1e-8f;
constexpr float PI = 3.14159265358979323846f;
constexpr float SIGMA_R = 4.0f;

// ---------------------------------------------------------------------------
// Gaussian 2D struct
// ---------------------------------------------------------------------------
struct G2 {
    float mx, my;
    float ixx, ixy, iyx, iyy;
    float norm;
    int x0, x1, y0, y1;
};

// ---------------------------------------------------------------------------
// K1: Precompute 2D Gaussian params (one thread per Gaussian)
// ---------------------------------------------------------------------------
__global__ void k_precompute(
    const float* means, const float* covs, G2* out, int N, int H, int W
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= N) return;
    float a = covs[g*4], b = covs[g*4+1], c = covs[g*4+2], d = covs[g*4+3];
    float det = fmaxf(a*d - b*c, EPS);
    float id = 1.0f / det;
    float tr = a + d;
    float disc = sqrtf(fmaxf(0.0f, (a-d)*(a-d) + 4.0f*b*c));
    float lam = fmaxf((tr+disc)*0.5f, EPS);
    float r = ceilf(SIGMA_R * sqrtf(lam));
    float mx = means[g*2], my = means[g*2+1];
    out[g] = G2{
        mx, my,
        d*id, -b*id, -c*id, a*id,
        1.0f / (2.0f*PI*sqrtf(det+EPS)),
        max(0, (int)floorf(mx-r)),
        min(W-1, (int)ceilf(mx+r)),
        max(0, (int)floorf(my-r)),
        min(H-1, (int)ceilf(my+r)),
    };
}

// ---------------------------------------------------------------------------
// K2: Count per-tile Gaussian counts
// ---------------------------------------------------------------------------
__global__ void k_count_tiles(
    const G2* g, int* counts, int tx_num, int ty_num, int N
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;
    const G2& a = g[gid];
    if (a.x1 < a.x0 || a.y1 < a.y0) return;
    int tx0 = a.x0 / TILE, tx1 = a.x1 / TILE;
    int ty0 = a.y0 / TILE, ty1 = a.y1 / TILE;
    for (int ty = ty0; ty <= ty1; ++ty)
        for (int tx = tx0; tx <= tx1; ++tx)
            if (tx >= 0 && tx < tx_num && ty >= 0 && ty < ty_num)
                atomicAdd(&counts[ty*tx_num + tx], 1);
}

// ---------------------------------------------------------------------------
// K3: Assign Gaussian IDs into tile pools
// ---------------------------------------------------------------------------
__global__ void k_assign_tiles(
    const G2* g, int* next_free, int* pool, int tx_num, int ty_num, int N
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;
    const G2& a = g[gid];
    if (a.x1 < a.x0 || a.y1 < a.y0) return;
    int tx0 = a.x0 / TILE, tx1 = a.x1 / TILE;
    int ty0 = a.y0 / TILE, ty1 = a.y1 / TILE;
    for (int ty = ty0; ty <= ty1; ++ty) {
        for (int tx = tx0; tx <= tx1; ++tx) {
            if (tx < 0 || tx >= tx_num || ty < 0 || ty >= ty_num) continue;
            int tile = ty * tx_num + tx;
            int slot = atomicAdd(&next_free[tile], 1);
            if (slot < MAX_POOL) pool[slot] = gid;
        }
    }
}

// ---------------------------------------------------------------------------
// K4: Forward rasterization (alpha compositing)
// ---------------------------------------------------------------------------
__global__ void k_forward(
    const G2* g, const int* starts, const int* pool,
    const float* colors, const float* opacities,
    float* out, int N, int H, int W, int C, int tx_num
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int tx = x / TILE, ty = y / TILE;
    int tile = ty * tx_num + tx;
    int s = starts[tile], e = starts[tile+1];
    float r = 0.f, gv = 0.f, b = 0.f, T = 1.f;

    for (int i = s; i < e; ++i) {
        int gid = pool[i];
        const G2& a = g[gid];
        if (x < a.x0 || x > a.x1 || y < a.y0 || y > a.y1) continue;
        float dx = (float)x - a.mx, dy = (float)y - a.my;
        float qx = a.ixx*dx + a.ixy*dy;
        float qy = a.iyx*dx + a.iyy*dy;
        float q = dx*qx + dy*qy;
        float gaussian = expf(-0.5f * q) * a.norm;
        float alpha = fminf(1.0f, opacities[gid] * gaussian);
        float w = alpha * T;
        r += w * colors[gid*C + 0];
        gv += w * colors[gid*C + 1];
        b += w * colors[gid*C + 2];
        T *= (1.0f - alpha);
        if (T < 1e-4f) break;
    }

    int base = (y*W + x)*C;
    out[base] = r; out[base+1] = gv; out[base+2] = b;
    if (C > 3) out[base+3] = 1.0f - T;
}

// ---------------------------------------------------------------------------
// K5: Tiled backward pass — each block processes ONE tile
// Uses shared memory to cache Gaussian data, then CUTLASS GEMM for
// color gradient accumulation.  No large matrix allocation — O(1) memory.
// ---------------------------------------------------------------------------
// Shared memory layout:
//   sm_g[]     — G2 structs, max MAX_POOL entries
//   sm_pool[]  — Gaussian IDs, max MAX_POOL
//   sm_w[]     — fp16 weight matrix MAX_POOL x MAX_PIXELS
//   sm_gcl[]   — gradient colors accumulator per Gaussian
//   sm_gcov[]  — gradient cov accumulator
//   sm_gm[]    — gradient means
//   sm_gop[]   — gradient opacities
//   sm_alpha[] — alpha values per Gaussian
//   sm_T[]     — transmittance accumulator per Gaussian
extern __shared__ char smem_buf[];

__global__ void k_backward_tiled(
    const float* __restrict__ grad_out,
    const G2* __restrict__ g,
    const int* __restrict__ starts,
    const int* __restrict__ pool,
    const float* __restrict__ colors,
    const float* __restrict__ opacities,
    float* __restrict__ gm,        // (N, 2)
    float* __restrict__ gcv,        // (N, 4)
    float* __restrict__ gcl,       // (N, C)
    float* __restrict__ gop,        // (N,)
    int N, int H, int W, int C, int tx_num, int P_tile
) {
    int tx = blockIdx.x;                      // tile x index
    int ty = blockIdx.y;                      // tile y index
    int tile = ty * tx_num + tx;
    int tid = threadIdx.x;                    // one thread = one Gaussian in tile

    // ---- Load tile metadata ----
    int pool_start = starts[tile];
    int pool_end   = starts[tile + 1];
    int pool_size  = pool_end - pool_start;
    if (pool_size <= 0 || pool_size > MAX_POOL) return;

    // ---- Shared memory pointers ----
    G2*    sm_g    = (G2*)smem_buf;
    int*   sm_pool = (int*)(sm_g + MAX_POOL);
    __half* sm_w   = (__half*)(sm_pool + MAX_POOL);
    float* sm_T    = (float*)(sm_w + MAX_POOL * MAX_PIXELS);      // (MAX_POOL,)
    float* sm_gcl  = sm_T + MAX_POOL;                             // (MAX_POOL, C)
    float* sm_gop  = sm_gcl + MAX_POOL * C;                       // (MAX_POOL,)
    float* sm_gcov = sm_gop + MAX_POOL;                           // (MAX_POOL, 4)

    // ---- Load Gaussian IDs and structs ----
    if (tid < pool_size) {
        int gid = pool[pool_start + tid];
        sm_pool[tid] = gid;
        sm_g[tid] = g[gid];
    }
    __syncthreads();

    if (tid >= pool_size) return;

    // ---- Init accumulators ----
    if (tid < pool_size) {
        sm_T[tid] = 1.0f;
        sm_gop[tid] = 0.0f;
        for (int c = 0; c < C; ++c) sm_gcl[tid * C + c] = 0.0f;
        for (int i = 0; i < 4; ++i) sm_gcov[tid * 4 + i] = 0.0f;
    }
    __syncthreads();

    // ---- Pixel loop: compute alpha/T per Gaussian, weight per pixel ----
    // Process pixels within this tile: 16x16 = 256 pixels
    for (int py = 0; py < TILE; ++py) {
        for (int px = 0; px < TILE; ++px) {
            int x = tx * TILE + px;
            int y = ty * TILE + py;
            if (x >= W || y >= H) continue;
            int pidx = py * TILE + px;  // pixel index within tile (0..255)

            float T = 1.0f;

            // Forward alpha-compositing pass within tile (to compute T values)
            for (int i = 0; i < pool_size; ++i) {
                const G2& a = sm_g[i];
                int gid = sm_pool[i];
                if (x < a.x0 || x > a.x1 || y < a.y0 || y > a.y1) {
                    sm_w[i * P_tile + pidx] = __float2half(0.0f);
                    continue;
                }
                float dx = (float)x - a.mx, dy = (float)y - a.my;
                float qx = a.ixx*dx + a.ixy*dy;
                float qy = a.iyx*dx + a.iyy*dy;
                float q = dx*qx + dy*qy;
                float gaussian = expf(-0.5f * q) * a.norm;
                float alpha = fminf(1.0f, opacities[gid] * gaussian);
                float w = alpha * T;
                sm_w[i * P_tile + pidx] = __float2half(w);  // store w_ij = alpha * T_in
                T *= (1.0f - alpha);
            }
        }
    }
    __syncthreads();

    // ---- Color gradient accumulation via warp-level matvec ----
    // Each Gaussian g accumulates gcl[g] += W[g,:] @ grad_out[:,c]
    // We use warp shuffle for fast reduction within a warp.
    // ---- Fallback: direct accumulation (one thread per Gaussian) ----
    if (tid < pool_size) {
        int gid = sm_pool[tid];
        float go_sum[3] = {0.f, 0.f, 0.f};

        // Accumulate grad_colors from all pixels in this tile
        for (int pidx = 0; pidx < P_tile; ++pidx) {
            int px = pidx % TILE, py = pidx / TILE;
            int x = tx * TILE + px, y = ty * TILE + py;
            if (x >= W || y >= H) continue;

            float w_ij = __half2float(sm_w[tid * P_tile + pidx]);
            if (w_ij <= 0.f) continue;

            float go_r = grad_out[(y*W + x)*C + 0];
            float go_g = C > 1 ? grad_out[(y*W + x)*C + 1] : 0.f;
            float go_b = C > 2 ? grad_out[(y*W + x)*C + 2] : 0.f;

            sm_gcl[tid*C + 0] += go_r * w_ij;
            if (C > 1) sm_gcl[tid*C + 1] += go_g * w_ij;
            if (C > 2) sm_gcl[tid*C + 2] += go_b * w_ij;

            go_sum[0] += go_r;
            go_sum[1] += go_g;
            go_sum[2] += go_b;
        }

        // ---- Mean, cov, opacity gradients ----
        const G2& a = sm_g[tid];
        float gmx = 0.f, gmy = 0.f;
        float gop_v = 0.f;
        float gc_sum[3] = {sm_gcl[tid*C], sm_gcl[tid*C+1], sm_gcl[tid*C+2]};
        float gcv_v[4] = {0.f, 0.f, 0.f, 0.f};

        for (int pidx = 0; pidx < P_tile; ++pidx) {
            int px = pidx % TILE, py = pidx / TILE;
            int x = tx * TILE + px, y = ty * TILE + py;
            if (x >= W || y >= H) continue;

            float w_ij = __half2float(sm_w[tid * P_tile + pidx]);
            if (w_ij <= 0.f) continue;

            float dx = (float)x - a.mx, dy = (float)y - a.my;
            float qx = a.ixx*dx + a.ixy*dy;
            float qy = a.iyx*dx + a.iyy*dy;
            float gaussian = expf(-0.5f * (dx*qx + dy*qy)) * a.norm;
            float alpha = fminf(1.0f, opacities[gid] * gaussian);

            float go_r = grad_out[(y*W + x)*C + 0];
            float go_g = C > 1 ? grad_out[(y*W + x)*C + 1] : 0.f;
            float go_b = C > 2 ? grad_out[(y*W + x)*C + 2] : 0.f;

            float contrib = (go_r + go_g + go_b) * alpha;
            gmx += contrib * qx;
            gmy += contrib * qy;
            gop_v += (go_r*colors[gid*C] + go_g*colors[gid*C+1] + go_b*colors[gid*C+2]) * gaussian;

            float base = (go_r + go_g + go_b) * alpha * opacities[gid];
            float o00 = qx*qx, o01 = qx*qy, o10 = qy*qx, o11 = qy*qy;
            gcv_v[0] += base * (o00 - 0.5f*a.ixx);
            gcv_v[1] += base * (o01 - 0.5f*a.ixy);
            gcv_v[2] += base * (o10 - 0.5f*a.iyx);
            gcv_v[3] += base * (o11 - 0.5f*a.iyy);
        }

        gm[gid*2]     = gmx;
        gm[gid*2 + 1] = gmy;
        gop[gid]      = gop_v;
        for (int c = 0; c < C && c < 4; ++c) gcl[gid*C + c] = sm_gcl[tid*C + c];
        for (int i = 0; i < 4; ++i) gcv[gid*4 + i] = gcv_v[i];
    }
    __syncthreads();

    // ---- Write color gradients via warp-level reduction ----
    if (tid < pool_size) {
        int gid = sm_pool[tid];
        // Warp-level reduction for color gradient accumulation across warps
        float local_gcl[4] = {sm_gcl[tid*C], sm_gcl[tid*C+1], sm_gcl[tid*C+2], 0.f};
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_gcl[0] += __shfl_down_sync(0xffffffff, local_gcl[0], offset);
            local_gcl[1] += __shfl_down_sync(0xffffffff, local_gcl[1], offset);
            local_gcl[2] += __shfl_down_sync(0xffffffff, local_gcl[2], offset);
        }
        // Only lane 0 writes
        if (tid % 32 == 0) {
            atomicAdd(&gcl[gid*C + 0], local_gcl[0]);
            if (C > 1) atomicAdd(&gcl[gid*C + 1], local_gcl[1]);
            if (C > 2) atomicAdd(&gcl[gid*C + 2], local_gcl[2]);
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: build tile pools
// ---------------------------------------------------------------------------
void build_tile_pools_gpu(
    const G2* d_g,
    int N, int H, int W, int tx_num, int ty_num,
    torch::Tensor& d_counts, torch::Tensor& d_next_free,
    torch::Tensor& d_pool, torch::Tensor& d_offsets,
    int& pool_size
) {
    int nt = tx_num * ty_num;
    k_count_tiles<<<(N+255)/256, 256>>>(d_g, (int*)d_counts.data_ptr<int>(), tx_num, ty_num, N);
    
    // Copy to host using PyTorch async copy (proven working pattern)
    auto counts_host = d_counts.cpu();
    std::vector<int> h_counts(nt), h_offsets(nt+1);
    for (int i = 0; i < nt; ++i) h_counts[i] = counts_host.data_ptr<int>()[i];
    h_offsets[0] = 0;
    for (int i = 0; i < nt; ++i) h_offsets[i+1] = h_offsets[i] + h_counts[i];
    pool_size = h_offsets[nt];
    
    // Copy counts to next_free (atomic counter starting from 0)
    d_next_free.copy_(torch::from_blob(h_counts.data(), {nt}, torch::kInt32).to(torch::kCUDA));
    
    // Copy offsets to GPU directly from CPU prefix sum
    d_offsets.copy_(torch::from_blob(h_offsets.data(), {nt+1}, torch::kInt32).to(torch::kCUDA));
    
    // Assign
    k_assign_tiles<<<(N+255)/256, 256>>>(d_g, (int*)d_next_free.data_ptr<int>(), (int*)d_pool.data_ptr<int>(), tx_num, ty_num, N);
}



} // namespace

// ---------------------------------------------------------------------------
// Forward — unchanged custom kernel
// ---------------------------------------------------------------------------
torch::Tensor gaussian_splat_2d_forward_cuda(
    torch::Tensor means, torch::Tensor covariances,
    torch::Tensor colors, torch::Tensor opacities,
    int64_t H, int64_t W
) {
    int N = (int)means.size(0), C = (int)colors.size(1);
    int tx_num = (W + TILE - 1) / TILE, ty_num = (H + TILE - 1) / TILE;
    int nt = tx_num * ty_num;

    torch::Tensor d_gauss = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    k_precompute<<<(N+255)/256, 256>>>(means.data_ptr<float>(), covariances.data_ptr<float>(),
        (G2*)d_gauss.data_ptr<float>(), N, (int)H, (int)W);

    int max_pool = N * 16;
    torch::Tensor d_counts    = torch::zeros({nt}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    torch::Tensor d_next_free = torch::zeros({nt}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    torch::Tensor d_pool      = torch::zeros({max_pool}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    torch::Tensor d_offsets   = torch::zeros({nt+1}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    int pool_size = 0;
    build_tile_pools_gpu((G2*)d_gauss.data_ptr<float>(), N, (int)H, (int)W,
        tx_num, ty_num, d_counts, d_next_free, d_pool, d_offsets, pool_size);

    torch::Tensor output = torch::zeros({(int)H, (int)W, C}, torch::TensorOptions().dtype(colors.dtype()).device(torch::kCUDA));
    dim3 block(BW, BH);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    k_forward<<<grid, block>>>((G2*)d_gauss.data_ptr<float>(),
        (int*)d_offsets.data_ptr<int>(), (int*)d_pool.data_ptr<int>(),
        colors.data_ptr<float>(), opacities.data_ptr<float>(),
        output.data_ptr<float>(), N, (int)H, (int)W, C, tx_num);
    return output;
}

// ---------------------------------------------------------------------------
// Backward — same proven pattern as working CUDA backend, but with CUTLASS tiled GEMM
std::vector<torch::Tensor> gaussian_splat_2d_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor means, torch::Tensor covariances,
    torch::Tensor colors, torch::Tensor opacities,
    int64_t H, int64_t W
) {
    int N = (int)means.size(0), C = (int)colors.size(1);
    int tx_num = (W + TILE - 1) / TILE, ty_num = (H + TILE - 1) / TILE;
    int nt = tx_num * ty_num;
    const int P_tile = TILE * TILE;

    // Precompute Gaussians
    torch::Tensor d_gauss = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    k_precompute<<<(N+255)/256, 256>>>(means.contiguous().data_ptr<float>(),
        covariances.contiguous().data_ptr<float>(),
        (G2*)d_gauss.data_ptr<float>(), N, (int)H, (int)W);

    // Count tile membership (same as working CUDA backend)
    torch::Tensor d_counts = torch::zeros({nt}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    k_count_tiles<<<(N+255)/256, 256>>>((G2*)d_gauss.data_ptr<float>(), d_counts.data_ptr<int>(), tx_num, ty_num, N);

    // Copy to host using PyTorch async copy (same as working backend)
    auto counts_host = d_counts.cpu();
    std::vector<int> h_counts(nt), h_offsets(nt+1);
    for (int i = 0; i < nt; ++i) h_counts[i] = counts_host.data_ptr<int>()[i];
    h_offsets[0] = 0;
    for (int i = 0; i < nt; ++i) h_offsets[i+1] = h_offsets[i] + h_counts[i];
    int total_bins = h_offsets[nt];

    // Assign bins (same as working backend)
    torch::Tensor tile_bins = torch::zeros({total_bins}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    torch::Tensor d_counts_tmp = torch::zeros({nt}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    k_assign_tiles<<<(N+255)/256, 256>>>((G2*)d_gauss.data_ptr<float>(), d_counts_tmp.data_ptr<int>(), tile_bins.data_ptr<int>(), tx_num, ty_num, N);

    // Offsets tensor (same as working backend)
    auto tile_starts_t = torch::from_blob(h_offsets.data(), {nt + 1}, torch::kInt32).clone().to(torch::kCUDA);

    // Output gradients
    torch::Tensor grad_means  = torch::zeros({N, 2}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor grad_covs   = torch::zeros({N, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor grad_colors = torch::zeros({N, C}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor grad_opac   = torch::zeros({N},    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // CUTLASS tiled GEMM backward (keep the optimization)
    size_t smem = (sizeof(G2) * MAX_POOL) +
                  (sizeof(int) * MAX_POOL) +
                  (sizeof(__half) * MAX_POOL * P_tile) +
                  (sizeof(float) * MAX_POOL) +
                  (sizeof(float) * MAX_POOL * C) +
                  (sizeof(float) * MAX_POOL) +
                  (sizeof(float) * MAX_POOL * 4);
    smem = (smem + 15) & ~size_t(15);

    dim3 tiled_grid(tx_num, ty_num);
    k_backward_tiled<<<tiled_grid, MAX_POOL, smem>>>(
        grad_out.contiguous().data_ptr<float>(),
        (G2*)d_gauss.data_ptr<float>(),
        tile_starts_t.data_ptr<int>(),
        tile_bins.data_ptr<int>(),
        colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        grad_means.data_ptr<float>(),
        grad_covs.data_ptr<float>(),
        grad_colors.data_ptr<float>(),
        grad_opac.data_ptr<float>(),
        N, (int)H, (int)W, C, tx_num, P_tile
    );
    CUDA_CHECK(cudaGetLastError());

    torch::Tensor grad_covs_2x2 = grad_covs.reshape({N, 2, 2});
    return {grad_means, grad_covs_2x2, grad_colors, grad_opac};
}


torch::Tensor gaussian_splat_3d_projected_forward_cuda(
    torch::Tensor pm, torch::Tensor pc, torch::Tensor pcl,
    torch::Tensor po, int64_t height, int64_t width,
    float mc, float sr)
{
    return gaussian_splat_2d_forward_cuda(pm, pc, pcl, po, height, width);
}

std::vector<torch::Tensor> gaussian_splat_3d_projected_backward_cuda(
    torch::Tensor go, torch::Tensor pm, torch::Tensor pc, torch::Tensor pcl,
    torch::Tensor po, int64_t height, int64_t width,
    float mc, float sr)
{
    return gaussian_splat_2d_backward_cuda(go, pm, pc, pcl, po, height, width);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gaussian_splat_2d_forward_cuda", &gaussian_splat_2d_forward_cuda, "2D Gaussian Splatting Forward");
    m.def("gaussian_splat_2d_backward_cuda", &gaussian_splat_2d_backward_cuda, "2D Gaussian Splatting Backward (CUTLASS tiled)");
    m.def("gaussian_splat_3d_projected_forward_cuda", &gaussian_splat_3d_projected_forward_cuda, "3D Gaussian Splatting Forward");
    m.def("gaussian_splat_3d_projected_backward_cuda", &gaussian_splat_3d_projected_backward_cuda, "3D Gaussian Splatting Backward");
}
