#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <ATen/Parallel.h>
#include <torch/extension.h>

namespace {

constexpr int64_t kTileSize = 16;
constexpr double kSigmaRadius = 4.0;

template <typename scalar_t>
struct GaussianPrecomputed {
    scalar_t inv_xx;
    scalar_t inv_xy;
    scalar_t inv_yx;
    scalar_t inv_yy;
    scalar_t normalization;
    scalar_t det_ratio;
    int64_t min_x;
    int64_t max_x;
    int64_t min_y;
    int64_t max_y;
};


template <typename scalar_t>
std::vector<GaussianPrecomputed<scalar_t>> precompute_gaussians(
    const torch::TensorAccessor<scalar_t, 2>& means_a,
    const torch::TensorAccessor<scalar_t, 3>& covs_a,
    int64_t num_gaussians,
    int64_t height,
    int64_t width
) {
    std::vector<GaussianPrecomputed<scalar_t>> params(static_cast<size_t>(num_gaussians));

    constexpr scalar_t kEps = static_cast<scalar_t>(1e-8);
    constexpr scalar_t kPi = static_cast<scalar_t>(3.14159265358979323846);

    at::parallel_for(0, num_gaussians, 0, [&](int64_t begin, int64_t end) {
        for (int64_t g = begin; g < end; ++g) {
            const scalar_t a = covs_a[g][0][0];
            const scalar_t b = covs_a[g][0][1];
            const scalar_t c = covs_a[g][1][0];
            const scalar_t d = covs_a[g][1][1];

            scalar_t det = a * d - b * c;
            if (det < kEps) {
                det = kEps;
            }

            const scalar_t inv_det = static_cast<scalar_t>(1.0) / det;
            const scalar_t trace = a + d;
            const scalar_t disc =
                std::sqrt(std::max(static_cast<scalar_t>(0), (a - d) * (a - d) + 4 * b * c));
            const scalar_t lambda_max = std::max(
                (trace + disc) * static_cast<scalar_t>(0.5),
                kEps
            );
            const scalar_t radius = static_cast<scalar_t>(
                std::ceil(kSigmaRadius * std::sqrt(lambda_max))
            );
            const int64_t min_x = std::max(
                static_cast<int64_t>(0),
                static_cast<int64_t>(std::floor(means_a[g][0] - radius))
            );
            const int64_t max_x = std::min(
                width - 1,
                static_cast<int64_t>(std::ceil(means_a[g][0] + radius))
            );
            const int64_t min_y = std::max(
                static_cast<int64_t>(0),
                static_cast<int64_t>(std::floor(means_a[g][1] - radius))
            );
            const int64_t max_y = std::min(
                height - 1,
                static_cast<int64_t>(std::ceil(means_a[g][1] + radius))
            );

            params[static_cast<size_t>(g)] = GaussianPrecomputed<scalar_t>{
                d * inv_det,
                -b * inv_det,
                -c * inv_det,
                a * inv_det,
                static_cast<scalar_t>(1.0) /
                    (static_cast<scalar_t>(2.0) * kPi * std::sqrt(det + kEps)),
                det / (det + kEps),
                min_x,
                max_x,
                min_y,
                max_y,
            };
        }
    });

    return params;
}


template <typename scalar_t>
std::vector<std::vector<int64_t>> build_tile_bins(
    const std::vector<GaussianPrecomputed<scalar_t>>& gaussian_params,
    int64_t tiles_x,
    int64_t tiles_y
) {
    std::vector<std::vector<int64_t>> tile_bins(static_cast<size_t>(tiles_x * tiles_y));
    for (int64_t g = 0; g < static_cast<int64_t>(gaussian_params.size()); ++g) {
        const auto& gp = gaussian_params[static_cast<size_t>(g)];
        if (gp.max_x < gp.min_x || gp.max_y < gp.min_y) {
            continue;
        }
        const int64_t tile_min_x = gp.min_x / kTileSize;
        const int64_t tile_max_x = gp.max_x / kTileSize;
        const int64_t tile_min_y = gp.min_y / kTileSize;
        const int64_t tile_max_y = gp.max_y / kTileSize;
        for (int64_t ty = tile_min_y; ty <= tile_max_y; ++ty) {
            for (int64_t tx = tile_min_x; tx <= tile_max_x; ++tx) {
                tile_bins[static_cast<size_t>(ty * tiles_x + tx)].push_back(g);
            }
        }
    }
    return tile_bins;
}

template <typename scalar_t>
struct ProjectedGaussian3D {
    scalar_t mean_x;
    scalar_t mean_y;
    scalar_t depth;
    scalar_t inv_xx;
    scalar_t inv_xy;
    scalar_t inv_yx;
    scalar_t inv_yy;
    int64_t min_x;
    int64_t max_x;
    int64_t min_y;
    int64_t max_y;
    int64_t source_index;
};

template <typename scalar_t>
struct RasterGaussian2D {
    scalar_t mean_x;
    scalar_t mean_y;
    scalar_t inv_xx;
    scalar_t inv_xy;
    scalar_t inv_yx;
    scalar_t inv_yy;
    int64_t min_x;
    int64_t max_x;
    int64_t min_y;
    int64_t max_y;
};

template <typename scalar_t>
std::vector<RasterGaussian2D<scalar_t>> precompute_raster_gaussians_2d(
    const torch::TensorAccessor<scalar_t, 2>& means_a,
    const torch::TensorAccessor<scalar_t, 3>& covs_a,
    int64_t num_gaussians,
    int64_t height,
    int64_t width,
    scalar_t min_covariance,
    scalar_t sigma_radius
) {
    std::vector<RasterGaussian2D<scalar_t>> gaussians(static_cast<size_t>(num_gaussians));
    at::parallel_for(0, num_gaussians, 0, [&](int64_t begin, int64_t end) {
        for (int64_t g = begin; g < end; ++g) {
            scalar_t s00 = covs_a[g][0][0] + min_covariance;
            scalar_t s01 = covs_a[g][0][1];
            scalar_t s10 = covs_a[g][1][0];
            scalar_t s11 = covs_a[g][1][1] + min_covariance;
            scalar_t det = s00 * s11 - s01 * s10;
            if (det <= min_covariance) {
                det = min_covariance;
            }
            scalar_t inv_det = static_cast<scalar_t>(1.0) / det;
            scalar_t inv_xx = s11 * inv_det;
            scalar_t inv_xy = -s01 * inv_det;
            scalar_t inv_yx = -s10 * inv_det;
            scalar_t inv_yy = s00 * inv_det;

            scalar_t trace = s00 + s11;
            scalar_t disc = std::sqrt(
                std::max(
                    static_cast<scalar_t>(0),
                    (s00 - s11) * (s00 - s11) + 4 * s01 * s10
                )
            );
            scalar_t lambda_max = std::max(
                (trace + disc) * static_cast<scalar_t>(0.5),
                min_covariance
            );
            scalar_t radius = sigma_radius * std::sqrt(lambda_max);

            int64_t min_x = static_cast<int64_t>(std::floor(means_a[g][0] - radius));
            int64_t max_x = static_cast<int64_t>(std::ceil(means_a[g][0] + radius));
            int64_t min_y = static_cast<int64_t>(std::floor(means_a[g][1] - radius));
            int64_t max_y = static_cast<int64_t>(std::ceil(means_a[g][1] + radius));
            if (max_x < 0 || min_x >= width || max_y < 0 || min_y >= height) {
                min_x = 1;
                max_x = 0;
                min_y = 1;
                max_y = 0;
            }

            gaussians[static_cast<size_t>(g)] = RasterGaussian2D<scalar_t>{
                means_a[g][0],
                means_a[g][1],
                inv_xx,
                inv_xy,
                inv_yx,
                inv_yy,
                min_x,
                max_x,
                min_y,
                max_y,
            };
        }
    });
    return gaussians;
}

template <typename scalar_t>
std::vector<ProjectedGaussian3D<scalar_t>> project_gaussians_3d(
    const torch::TensorAccessor<scalar_t, 2>& means_a,
    const torch::TensorAccessor<scalar_t, 3>& covs_a,
    const torch::TensorAccessor<scalar_t, 2>& intrinsics_a,
    const torch::TensorAccessor<scalar_t, 2>& camera_to_world_a,
    int64_t num_gaussians,
    int64_t height,
    int64_t width,
    scalar_t near_plane,
    scalar_t min_covariance,
    scalar_t sigma_radius
) {
    std::vector<ProjectedGaussian3D<scalar_t>> projected;
    projected.reserve(static_cast<size_t>(num_gaussians));

    const scalar_t fx = intrinsics_a[0][0];
    const scalar_t fy = intrinsics_a[1][1];
    const scalar_t cx = intrinsics_a[0][2];
    const scalar_t cy = intrinsics_a[1][2];

    scalar_t r00 = camera_to_world_a[0][0];
    scalar_t r01 = camera_to_world_a[0][1];
    scalar_t r02 = camera_to_world_a[0][2];
    scalar_t r10 = camera_to_world_a[1][0];
    scalar_t r11 = camera_to_world_a[1][1];
    scalar_t r12 = camera_to_world_a[1][2];
    scalar_t r20 = camera_to_world_a[2][0];
    scalar_t r21 = camera_to_world_a[2][1];
    scalar_t r22 = camera_to_world_a[2][2];

    scalar_t tx = camera_to_world_a[0][3];
    scalar_t ty = camera_to_world_a[1][3];
    scalar_t tz = camera_to_world_a[2][3];

    scalar_t rwc00 = r00;
    scalar_t rwc01 = r10;
    scalar_t rwc02 = r20;
    scalar_t rwc10 = r01;
    scalar_t rwc11 = r11;
    scalar_t rwc12 = r21;
    scalar_t rwc20 = r02;
    scalar_t rwc21 = r12;
    scalar_t rwc22 = r22;

    scalar_t twc0 = -(rwc00 * tx + rwc01 * ty + rwc02 * tz);
    scalar_t twc1 = -(rwc10 * tx + rwc11 * ty + rwc12 * tz);
    scalar_t twc2 = -(rwc20 * tx + rwc21 * ty + rwc22 * tz);

    for (int64_t g = 0; g < num_gaussians; ++g) {
        const scalar_t mx = means_a[g][0];
        const scalar_t my = means_a[g][1];
        const scalar_t mz = means_a[g][2];

        const scalar_t cam_x = rwc00 * mx + rwc01 * my + rwc02 * mz + twc0;
        const scalar_t cam_y = rwc10 * mx + rwc11 * my + rwc12 * mz + twc1;
        const scalar_t cam_z = rwc20 * mx + rwc21 * my + rwc22 * mz + twc2;
        if (cam_z <= near_plane) {
            continue;
        }

        scalar_t wcov00 = covs_a[g][0][0];
        scalar_t wcov01 = covs_a[g][0][1];
        scalar_t wcov02 = covs_a[g][0][2];
        scalar_t wcov10 = covs_a[g][1][0];
        scalar_t wcov11 = covs_a[g][1][1];
        scalar_t wcov12 = covs_a[g][1][2];
        scalar_t wcov20 = covs_a[g][2][0];
        scalar_t wcov21 = covs_a[g][2][1];
        scalar_t wcov22 = covs_a[g][2][2];

        scalar_t t00 = rwc00 * wcov00 + rwc01 * wcov10 + rwc02 * wcov20;
        scalar_t t01 = rwc00 * wcov01 + rwc01 * wcov11 + rwc02 * wcov21;
        scalar_t t02 = rwc00 * wcov02 + rwc01 * wcov12 + rwc02 * wcov22;
        scalar_t t10 = rwc10 * wcov00 + rwc11 * wcov10 + rwc12 * wcov20;
        scalar_t t11 = rwc10 * wcov01 + rwc11 * wcov11 + rwc12 * wcov21;
        scalar_t t12 = rwc10 * wcov02 + rwc11 * wcov12 + rwc12 * wcov22;
        scalar_t t20 = rwc20 * wcov00 + rwc21 * wcov10 + rwc22 * wcov20;
        scalar_t t21 = rwc20 * wcov01 + rwc21 * wcov11 + rwc22 * wcov21;
        scalar_t t22 = rwc20 * wcov02 + rwc21 * wcov12 + rwc22 * wcov22;

        scalar_t ccov00 = t00 * rwc00 + t01 * rwc01 + t02 * rwc02;
        scalar_t ccov01 = t00 * rwc10 + t01 * rwc11 + t02 * rwc12;
        scalar_t ccov02 = t00 * rwc20 + t01 * rwc21 + t02 * rwc22;
        scalar_t ccov10 = t10 * rwc00 + t11 * rwc01 + t12 * rwc02;
        scalar_t ccov11 = t10 * rwc10 + t11 * rwc11 + t12 * rwc12;
        scalar_t ccov12 = t10 * rwc20 + t11 * rwc21 + t12 * rwc22;
        scalar_t ccov20 = t20 * rwc00 + t21 * rwc01 + t22 * rwc02;
        scalar_t ccov21 = t20 * rwc10 + t21 * rwc11 + t22 * rwc12;
        scalar_t ccov22 = t20 * rwc20 + t21 * rwc21 + t22 * rwc22;

        const scalar_t proj_x = fx * cam_x / cam_z + cx;
        const scalar_t proj_y = fy * cam_y / cam_z + cy;

        scalar_t j00 = fx / cam_z;
        scalar_t j01 = static_cast<scalar_t>(0);
        scalar_t j02 = -fx * cam_x / (cam_z * cam_z);
        scalar_t j10 = static_cast<scalar_t>(0);
        scalar_t j11 = fy / cam_z;
        scalar_t j12 = -fy * cam_y / (cam_z * cam_z);

        scalar_t s00 =
            j00 * (ccov00 * j00 + ccov01 * j01 + ccov02 * j02) +
            j01 * (ccov10 * j00 + ccov11 * j01 + ccov12 * j02) +
            j02 * (ccov20 * j00 + ccov21 * j01 + ccov22 * j02);
        scalar_t s01 =
            j00 * (ccov00 * j10 + ccov01 * j11 + ccov02 * j12) +
            j01 * (ccov10 * j10 + ccov11 * j11 + ccov12 * j12) +
            j02 * (ccov20 * j10 + ccov21 * j11 + ccov22 * j12);
        scalar_t s10 =
            j10 * (ccov00 * j00 + ccov01 * j01 + ccov02 * j02) +
            j11 * (ccov10 * j00 + ccov11 * j01 + ccov12 * j02) +
            j12 * (ccov20 * j00 + ccov21 * j01 + ccov22 * j02);
        scalar_t s11 =
            j10 * (ccov00 * j10 + ccov01 * j11 + ccov02 * j12) +
            j11 * (ccov10 * j10 + ccov11 * j11 + ccov12 * j12) +
            j12 * (ccov20 * j10 + ccov21 * j11 + ccov22 * j12);

        s00 += min_covariance;
        s11 += min_covariance;
        scalar_t det = s00 * s11 - s01 * s10;
        if (det <= min_covariance) {
            det = min_covariance;
        }
        scalar_t inv_det = static_cast<scalar_t>(1.0) / det;
        scalar_t inv_xx = s11 * inv_det;
        scalar_t inv_xy = -s01 * inv_det;
        scalar_t inv_yx = -s10 * inv_det;
        scalar_t inv_yy = s00 * inv_det;

        scalar_t trace = s00 + s11;
        scalar_t disc = std::sqrt(std::max(static_cast<scalar_t>(0), (s00 - s11) * (s00 - s11) + 4 * s01 * s10));
        scalar_t lambda_max = std::max((trace + disc) * static_cast<scalar_t>(0.5), min_covariance);
        scalar_t radius = sigma_radius * std::sqrt(lambda_max);

        int64_t min_x = static_cast<int64_t>(std::floor(proj_x - radius));
        int64_t max_x = static_cast<int64_t>(std::ceil(proj_x + radius));
        int64_t min_y = static_cast<int64_t>(std::floor(proj_y - radius));
        int64_t max_y = static_cast<int64_t>(std::ceil(proj_y + radius));
        if (max_x < 0 || min_x >= width || max_y < 0 || min_y >= height) {
            continue;
        }

        projected.push_back(ProjectedGaussian3D<scalar_t>{
            proj_x,
            proj_y,
            cam_z,
            inv_xx,
            inv_xy,
            inv_yx,
            inv_yy,
            min_x,
            max_x,
            min_y,
            max_y,
            g,
        });
    }

    std::sort(
        projected.begin(),
        projected.end(),
        [](const ProjectedGaussian3D<scalar_t>& a, const ProjectedGaussian3D<scalar_t>& b) {
            return a.depth < b.depth;
        }
    );

    return projected;
}

template <typename scalar_t>
void compute_forward_buffers(
    const torch::TensorAccessor<scalar_t, 2>& means_a,
    const std::vector<GaussianPrecomputed<scalar_t>>& gaussian_params,
    const std::vector<std::vector<int64_t>>& tile_bins,
    const torch::TensorAccessor<scalar_t, 2>& colors_a,
    const torch::TensorAccessor<scalar_t, 1>& opacities_a,
    torch::TensorAccessor<scalar_t, 3> output_a,
    torch::TensorAccessor<scalar_t, 2> total_weight_a,
    int64_t tiles_x,
    int64_t tiles_y,
    int64_t num_channels,
    int64_t height,
    int64_t width
) {
    constexpr scalar_t kEps = static_cast<scalar_t>(1e-8);

    at::parallel_for(0, tiles_x * tiles_y, 0, [&](int64_t begin, int64_t end) {
        std::vector<scalar_t> accum_dynamic;
        std::array<scalar_t, 4> accum_small{};
        if (num_channels > 4) {
            accum_dynamic.resize(static_cast<size_t>(num_channels));
        }

        for (int64_t tile_idx = begin; tile_idx < end; ++tile_idx) {
            const int64_t tile_y = tile_idx / tiles_x;
            const int64_t tile_x = tile_idx % tiles_x;
            const int64_t start_x = tile_x * kTileSize;
            const int64_t end_x = std::min(start_x + kTileSize, width);
            const int64_t start_y = tile_y * kTileSize;
            const int64_t end_y = std::min(start_y + kTileSize, height);
            const auto& gaussian_ids = tile_bins[static_cast<size_t>(tile_idx)];

            for (int64_t y = start_y; y < end_y; ++y) {
                for (int64_t x = start_x; x < end_x; ++x) {
                    scalar_t* accum_ptr = nullptr;
                    if (num_channels <= 4) {
                        accum_small.fill(static_cast<scalar_t>(0));
                        accum_ptr = accum_small.data();
                    } else {
                        std::fill(
                            accum_dynamic.begin(),
                            accum_dynamic.end(),
                            static_cast<scalar_t>(0)
                        );
                        accum_ptr = accum_dynamic.data();
                    }
                    scalar_t total_weight = static_cast<scalar_t>(0);

                    for (const int64_t g : gaussian_ids) {
                        const auto& gp = gaussian_params[static_cast<size_t>(g)];
                        if (x < gp.min_x || x > gp.max_x || y < gp.min_y || y > gp.max_y) {
                            continue;
                        }
                        const scalar_t dx = static_cast<scalar_t>(x) - means_a[g][0];
                        const scalar_t dy = static_cast<scalar_t>(y) - means_a[g][1];
                        const scalar_t qx = gp.inv_xx * dx + gp.inv_xy * dy;
                        const scalar_t qy = gp.inv_yx * dx + gp.inv_yy * dy;
                        const scalar_t quad = dx * qx + dy * qy;
                        const scalar_t gaussian =
                            std::exp(static_cast<scalar_t>(-0.5) * quad) * gp.normalization;
                        const scalar_t weight = gaussian * opacities_a[g];

                        total_weight += weight;
                        for (int64_t c_idx = 0; c_idx < num_channels; ++c_idx) {
                            accum_ptr[c_idx] += weight * colors_a[g][c_idx];
                        }
                    }

                    const scalar_t denom = std::max(total_weight, kEps);
                    total_weight_a[y][x] = denom;
                    for (int64_t c_idx = 0; c_idx < num_channels; ++c_idx) {
                        output_a[y][x][c_idx] = accum_ptr[c_idx] / denom;
                    }
                }
            }
        }
    });

    if (num_channels == 4) {
        at::parallel_for(0, height * width, 0, [&](int64_t begin, int64_t end) {
            for (int64_t linear_idx = begin; linear_idx < end; ++linear_idx) {
                const int64_t y = linear_idx / width;
                const int64_t x = linear_idx % width;
                const scalar_t alpha = output_a[y][x][3];
                output_a[y][x][0] *= alpha;
                output_a[y][x][1] *= alpha;
                output_a[y][x][2] *= alpha;
            }
        });
    }
}

torch::Tensor gaussian_splat_2d_forward_cpu(
    const torch::Tensor& means,
    const torch::Tensor& covariances,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    int64_t height,
    int64_t width
) {
    TORCH_CHECK(!means.is_cuda(), "means must be a CPU tensor");
    TORCH_CHECK(!covariances.is_cuda(), "covariances must be a CPU tensor");
    TORCH_CHECK(!colors.is_cuda(), "colors must be a CPU tensor");
    TORCH_CHECK(!opacities.is_cuda(), "opacities must be a CPU tensor");
    TORCH_CHECK(means.dim() == 2 && means.size(1) == 2, "means must have shape (N, 2)");
    TORCH_CHECK(
        covariances.dim() == 3 && covariances.size(1) == 2 && covariances.size(2) == 2,
        "covariances must have shape (N, 2, 2)"
    );
    TORCH_CHECK(colors.dim() == 2, "colors must have shape (N, C)");
    TORCH_CHECK(opacities.dim() == 1, "opacities must have shape (N)");
    TORCH_CHECK(means.size(0) == covariances.size(0), "means/covariances mismatch");
    TORCH_CHECK(means.size(0) == colors.size(0), "means/colors mismatch");
    TORCH_CHECK(means.size(0) == opacities.size(0), "means/opacities mismatch");
    TORCH_CHECK(height > 0 && width > 0, "height and width must be positive");

    auto means_c = means.contiguous();
    auto covariances_c = covariances.contiguous();
    auto colors_c = colors.contiguous();
    auto opacities_c = opacities.contiguous();

    auto num_gaussians = means_c.size(0);
    auto num_channels = colors_c.size(1);

    auto output = torch::zeros(
        {height, width, num_channels},
        torch::TensorOptions().dtype(colors_c.dtype()).device(colors_c.device())
    );
    auto total_weight = torch::zeros(
        {height, width},
        torch::TensorOptions().dtype(means_c.dtype()).device(means_c.device())
    );

    AT_DISPATCH_FLOATING_TYPES(means_c.scalar_type(), "gaussian_splat_2d_forward_cpu", [&] {
        auto means_a = means_c.accessor<scalar_t, 2>();
        auto covs_a = covariances_c.accessor<scalar_t, 3>();
        auto colors_a = colors_c.accessor<scalar_t, 2>();
        auto opacities_a = opacities_c.accessor<scalar_t, 1>();
        auto output_a = output.accessor<scalar_t, 3>();
        auto total_weight_a = total_weight.accessor<scalar_t, 2>();
        auto gaussian_params = precompute_gaussians<scalar_t>(
            means_a,
            covs_a,
            num_gaussians,
            height,
            width
        );
        const int64_t tiles_x = (width + kTileSize - 1) / kTileSize;
        const int64_t tiles_y = (height + kTileSize - 1) / kTileSize;
        auto tile_bins = build_tile_bins(gaussian_params, tiles_x, tiles_y);

        compute_forward_buffers<scalar_t>(
            means_a,
            gaussian_params,
            tile_bins,
            colors_a,
            opacities_a,
            output_a,
            total_weight_a,
            tiles_x,
            tiles_y,
            num_channels,
            height,
            width
        );
    });

    return output;
}

std::vector<torch::Tensor> gaussian_splat_2d_backward_cpu(
    const torch::Tensor& grad_output,
    const torch::Tensor& means,
    const torch::Tensor& covariances,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    int64_t height,
    int64_t width
) {
    TORCH_CHECK(!grad_output.is_cuda(), "grad_output must be a CPU tensor");
    TORCH_CHECK(!means.is_cuda(), "means must be a CPU tensor");
    TORCH_CHECK(!covariances.is_cuda(), "covariances must be a CPU tensor");
    TORCH_CHECK(!colors.is_cuda(), "colors must be a CPU tensor");
    TORCH_CHECK(!opacities.is_cuda(), "opacities must be a CPU tensor");

    auto grad_output_c = grad_output.contiguous();
    auto means_c = means.contiguous();
    auto covariances_c = covariances.contiguous();
    auto colors_c = colors.contiguous();
    auto opacities_c = opacities.contiguous();

    auto num_gaussians = means_c.size(0);
    auto num_channels = colors_c.size(1);

    auto output = torch::zeros(
        {height, width, num_channels},
        torch::TensorOptions().dtype(colors_c.dtype()).device(colors_c.device())
    );
    auto total_weight = torch::zeros(
        {height, width},
        torch::TensorOptions().dtype(means_c.dtype()).device(means_c.device())
    );

    auto grad_means = torch::zeros_like(means_c);
    auto grad_covariances = torch::zeros_like(covariances_c);
    auto grad_colors = torch::zeros_like(colors_c);
    auto grad_opacities = torch::zeros_like(opacities_c);

    AT_DISPATCH_FLOATING_TYPES(means_c.scalar_type(), "gaussian_splat_2d_backward_cpu", [&] {
        auto means_a = means_c.accessor<scalar_t, 2>();
        auto covs_a = covariances_c.accessor<scalar_t, 3>();
        auto colors_a = colors_c.accessor<scalar_t, 2>();
        auto opacities_a = opacities_c.accessor<scalar_t, 1>();
        auto grad_output_a = grad_output_c.accessor<scalar_t, 3>();
        auto output_a = output.accessor<scalar_t, 3>();
        auto total_weight_a = total_weight.accessor<scalar_t, 2>();
        auto gaussian_params = precompute_gaussians<scalar_t>(
            means_a,
            covs_a,
            num_gaussians,
            height,
            width
        );
        const int64_t tiles_x = (width + kTileSize - 1) / kTileSize;
        const int64_t tiles_y = (height + kTileSize - 1) / kTileSize;
        auto tile_bins = build_tile_bins(gaussian_params, tiles_x, tiles_y);

        compute_forward_buffers<scalar_t>(
            means_a,
            gaussian_params,
            tile_bins,
            colors_a,
            opacities_a,
            output_a,
            total_weight_a,
            tiles_x,
            tiles_y,
            num_channels,
            height,
            width
        );

        auto grad_means_a = grad_means.accessor<scalar_t, 2>();
        auto grad_covs_a = grad_covariances.accessor<scalar_t, 3>();
        auto grad_colors_a = grad_colors.accessor<scalar_t, 2>();
        auto grad_opacities_a = grad_opacities.accessor<scalar_t, 1>();
        auto grad_image_dot_output = torch::zeros(
            {height, width},
            torch::TensorOptions().dtype(grad_output_c.dtype()).device(grad_output_c.device())
        );
        auto grad_image_dot_output_a = grad_image_dot_output.accessor<scalar_t, 2>();

        at::parallel_for(0, height * width, 0, [&](int64_t begin, int64_t end) {
            for (int64_t linear_idx = begin; linear_idx < end; ++linear_idx) {
                const int64_t y = linear_idx / width;
                const int64_t x = linear_idx % width;
                scalar_t value = static_cast<scalar_t>(0);
                for (int64_t c_idx = 0; c_idx < num_channels; ++c_idx) {
                    value += grad_output_a[y][x][c_idx] * output_a[y][x][c_idx];
                }
                grad_image_dot_output_a[y][x] = value;
            }
        });

        at::parallel_for(0, num_gaussians, 0, [&](int64_t begin, int64_t end) {
            std::vector<scalar_t> grad_color_dynamic;
            std::array<scalar_t, 4> grad_color_small{};
            if (num_channels > 4) {
                grad_color_dynamic.resize(static_cast<size_t>(num_channels));
            }

            for (int64_t g = begin; g < end; ++g) {
                const auto& gp = gaussian_params[static_cast<size_t>(g)];

                scalar_t grad_mean_x = static_cast<scalar_t>(0);
                scalar_t grad_mean_y = static_cast<scalar_t>(0);
                scalar_t grad_opacity = static_cast<scalar_t>(0);
                scalar_t* grad_color_ptr = nullptr;
                if (num_channels <= 4) {
                    grad_color_small.fill(static_cast<scalar_t>(0));
                    grad_color_ptr = grad_color_small.data();
                } else {
                    std::fill(
                        grad_color_dynamic.begin(),
                        grad_color_dynamic.end(),
                        static_cast<scalar_t>(0)
                    );
                    grad_color_ptr = grad_color_dynamic.data();
                }
                scalar_t grad_cov_00 = static_cast<scalar_t>(0);
                scalar_t grad_cov_01 = static_cast<scalar_t>(0);
                scalar_t grad_cov_10 = static_cast<scalar_t>(0);
                scalar_t grad_cov_11 = static_cast<scalar_t>(0);

                for (int64_t y = gp.min_y; y <= gp.max_y; ++y) {
                    for (int64_t x = gp.min_x; x <= gp.max_x; ++x) {
                        const scalar_t dx = static_cast<scalar_t>(x) - means_a[g][0];
                        const scalar_t dy = static_cast<scalar_t>(y) - means_a[g][1];
                        const scalar_t v0 = gp.inv_xx * dx + gp.inv_yx * dy;
                        const scalar_t v1 = gp.inv_xy * dx + gp.inv_yy * dy;
                        const scalar_t quad = dx * v0 + dy * v1;
                        const scalar_t gaussian =
                            std::exp(static_cast<scalar_t>(-0.5) * quad) * gp.normalization;
                        const scalar_t weight = gaussian * opacities_a[g];
                        const scalar_t denom = total_weight_a[y][x];

                        scalar_t dot_grad_color = static_cast<scalar_t>(0);
                        for (int64_t c_idx = 0; c_idx < num_channels; ++c_idx) {
                            const scalar_t grad_val = grad_output_a[y][x][c_idx];
                            dot_grad_color += grad_val * colors_a[g][c_idx];
                            grad_color_ptr[c_idx] += grad_val * weight / denom;
                        }

                        const scalar_t gamma =
                            (dot_grad_color - grad_image_dot_output_a[y][x]) / denom;
                        grad_opacity += gamma * gaussian;
                        grad_mean_x += gamma * weight * v0;
                        grad_mean_y += gamma * weight * v1;

                        const scalar_t outer00 = v0 * v0;
                        const scalar_t outer01 = v0 * v1;
                        const scalar_t outer10 = v1 * v0;
                        const scalar_t outer11 = v1 * v1;

                        grad_cov_00 += gamma * weight * static_cast<scalar_t>(0.5) *
                            (outer00 - gp.det_ratio * gp.inv_xx);
                        grad_cov_01 += gamma * weight * static_cast<scalar_t>(0.5) *
                            (outer01 - gp.det_ratio * gp.inv_yx);
                        grad_cov_10 += gamma * weight * static_cast<scalar_t>(0.5) *
                            (outer10 - gp.det_ratio * gp.inv_xy);
                        grad_cov_11 += gamma * weight * static_cast<scalar_t>(0.5) *
                            (outer11 - gp.det_ratio * gp.inv_yy);
                    }
                }

                grad_means_a[g][0] = grad_mean_x;
                grad_means_a[g][1] = grad_mean_y;
                grad_opacities_a[g] = grad_opacity;
                for (int64_t c_idx = 0; c_idx < num_channels; ++c_idx) {
                    grad_colors_a[g][c_idx] = grad_color_ptr[c_idx];
                }
                grad_covs_a[g][0][0] = grad_cov_00;
                grad_covs_a[g][0][1] = grad_cov_01;
                grad_covs_a[g][1][0] = grad_cov_10;
                grad_covs_a[g][1][1] = grad_cov_11;
            }
        });
    });

    return {grad_means, grad_covariances, grad_colors, grad_opacities};
}

torch::Tensor gaussian_splat_3d_forward_cpu(
    const torch::Tensor& means,
    const torch::Tensor& covariances,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& intrinsics,
    const torch::Tensor& camera_to_world,
    int64_t height,
    int64_t width,
    double near_plane,
    double min_covariance,
    double sigma_radius
) {
    TORCH_CHECK(!means.is_cuda(), "means must be a CPU tensor");
    TORCH_CHECK(!covariances.is_cuda(), "covariances must be a CPU tensor");
    TORCH_CHECK(!colors.is_cuda(), "colors must be a CPU tensor");
    TORCH_CHECK(!opacities.is_cuda(), "opacities must be a CPU tensor");
    TORCH_CHECK(!intrinsics.is_cuda(), "intrinsics must be a CPU tensor");
    TORCH_CHECK(!camera_to_world.is_cuda(), "camera_to_world must be a CPU tensor");
    TORCH_CHECK(means.dim() == 2 && means.size(1) == 3, "means must have shape (N, 3)");
    TORCH_CHECK(
        covariances.dim() == 3 && covariances.size(1) == 3 && covariances.size(2) == 3,
        "covariances must have shape (N, 3, 3)"
    );
    TORCH_CHECK(colors.dim() == 2, "colors must have shape (N, C)");
    TORCH_CHECK(opacities.dim() == 1, "opacities must have shape (N)");
    TORCH_CHECK(intrinsics.sizes() == torch::IntArrayRef({3, 3}), "intrinsics must have shape (3, 3)");
    TORCH_CHECK(camera_to_world.sizes() == torch::IntArrayRef({4, 4}), "camera_to_world must have shape (4, 4)");

    auto means_c = means.contiguous();
    auto covariances_c = covariances.contiguous();
    auto colors_c = colors.contiguous();
    auto opacities_c = opacities.contiguous();
    auto intrinsics_c = intrinsics.contiguous();
    auto camera_to_world_c = camera_to_world.contiguous();

    auto num_gaussians = means_c.size(0);
    auto num_channels = colors_c.size(1);
    auto image = torch::zeros(
        {height, width, num_channels},
        torch::TensorOptions().dtype(colors_c.dtype()).device(colors_c.device())
    );
    auto transmittance = torch::ones(
        {height, width},
        torch::TensorOptions().dtype(colors_c.dtype()).device(colors_c.device())
    );

    AT_DISPATCH_FLOATING_TYPES(means_c.scalar_type(), "gaussian_splat_3d_forward_cpu", [&] {
        auto means_a = means_c.accessor<scalar_t, 2>();
        auto covs_a = covariances_c.accessor<scalar_t, 3>();
        auto colors_a = colors_c.accessor<scalar_t, 2>();
        auto opacities_a = opacities_c.accessor<scalar_t, 1>();
        auto intrinsics_a = intrinsics_c.accessor<scalar_t, 2>();
        auto camera_to_world_a = camera_to_world_c.accessor<scalar_t, 2>();
        auto image_a = image.accessor<scalar_t, 3>();
        auto transmittance_a = transmittance.accessor<scalar_t, 2>();

        auto projected = project_gaussians_3d<scalar_t>(
            means_a,
            covs_a,
            intrinsics_a,
            camera_to_world_a,
            num_gaussians,
            height,
            width,
            static_cast<scalar_t>(near_plane),
            static_cast<scalar_t>(min_covariance),
            static_cast<scalar_t>(sigma_radius)
        );

        for (const auto& pg : projected) {
            const int64_t x0 = std::max<int64_t>(0, pg.min_x);
            const int64_t x1 = std::min<int64_t>(width - 1, pg.max_x);
            const int64_t y0 = std::max<int64_t>(0, pg.min_y);
            const int64_t y1 = std::min<int64_t>(height - 1, pg.max_y);
            const int64_t src = pg.source_index;

            for (int64_t y = y0; y <= y1; ++y) {
                for (int64_t x = x0; x <= x1; ++x) {
                    const scalar_t dx = static_cast<scalar_t>(x) - pg.mean_x;
                    const scalar_t dy = static_cast<scalar_t>(y) - pg.mean_y;
                    const scalar_t quad =
                        dx * (pg.inv_xx * dx + pg.inv_xy * dy) +
                        dy * (pg.inv_yx * dx + pg.inv_yy * dy);
                    const scalar_t gaussian = std::exp(static_cast<scalar_t>(-0.5) * quad);
                    scalar_t alpha = opacities_a[src] * gaussian;
                    if (alpha < static_cast<scalar_t>(0)) {
                        alpha = static_cast<scalar_t>(0);
                    }
                    if (alpha > static_cast<scalar_t>(0.999)) {
                        alpha = static_cast<scalar_t>(0.999);
                    }

                    const scalar_t t = transmittance_a[y][x];
                    for (int64_t c_idx = 0; c_idx < num_channels; ++c_idx) {
                        image_a[y][x][c_idx] += t * alpha * colors_a[src][c_idx];
                    }
                    transmittance_a[y][x] = t * (static_cast<scalar_t>(1.0) - alpha);
                }
            }
        }
    });

    return image;
}

torch::Tensor gaussian_splat_3d_projected_forward_cpu(
    const torch::Tensor& means,
    const torch::Tensor& covariances,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    int64_t height,
    int64_t width,
    double min_covariance,
    double sigma_radius
) {
    TORCH_CHECK(!means.is_cuda(), "means must be a CPU tensor");
    TORCH_CHECK(!covariances.is_cuda(), "covariances must be a CPU tensor");
    TORCH_CHECK(!colors.is_cuda(), "colors must be a CPU tensor");
    TORCH_CHECK(!opacities.is_cuda(), "opacities must be a CPU tensor");
    TORCH_CHECK(means.dim() == 2 && means.size(1) == 2, "means must have shape (N, 2)");
    TORCH_CHECK(covariances.dim() == 3 && covariances.size(1) == 2 && covariances.size(2) == 2, "covariances must have shape (N, 2, 2)");

    auto means_c = means.contiguous();
    auto covs_c = covariances.contiguous();
    auto colors_c = colors.contiguous();
    auto opacities_c = opacities.contiguous();
    auto num_gaussians = means_c.size(0);
    auto num_channels = colors_c.size(1);

    auto image = torch::zeros({height, width, num_channels}, torch::TensorOptions().dtype(colors_c.dtype()).device(colors_c.device()));
    auto transmittance = torch::ones({height, width}, torch::TensorOptions().dtype(colors_c.dtype()).device(colors_c.device()));

    AT_DISPATCH_FLOATING_TYPES(means_c.scalar_type(), "gaussian_splat_3d_projected_forward_cpu", [&] {
        auto means_a = means_c.accessor<scalar_t, 2>();
        auto covs_a = covs_c.accessor<scalar_t, 3>();
        auto colors_a = colors_c.accessor<scalar_t, 2>();
        auto opacities_a = opacities_c.accessor<scalar_t, 1>();
        auto image_a = image.accessor<scalar_t, 3>();
        auto trans_a = transmittance.accessor<scalar_t, 2>();

        auto gaussians = precompute_raster_gaussians_2d(
            means_a,
            covs_a,
            num_gaussians,
            height,
            width,
            static_cast<scalar_t>(min_covariance),
            static_cast<scalar_t>(sigma_radius)
        );

        for (int64_t g = 0; g < num_gaussians; ++g) {
            const auto& rg = gaussians[static_cast<size_t>(g)];
            if (rg.max_x < rg.min_x || rg.max_y < rg.min_y) {
                continue;
            }
            const int64_t x0 = std::max<int64_t>(0, rg.min_x);
            const int64_t x1 = std::min<int64_t>(width - 1, rg.max_x);
            const int64_t y0 = std::max<int64_t>(0, rg.min_y);
            const int64_t y1 = std::min<int64_t>(height - 1, rg.max_y);
            for (int64_t y = y0; y <= y1; ++y) {
                for (int64_t x = x0; x <= x1; ++x) {
                    const scalar_t dx = static_cast<scalar_t>(x) - rg.mean_x;
                    const scalar_t dy = static_cast<scalar_t>(y) - rg.mean_y;
                    const scalar_t quad =
                        dx * (rg.inv_xx * dx + rg.inv_xy * dy) +
                        dy * (rg.inv_yx * dx + rg.inv_yy * dy);
                    const scalar_t gaussian = std::exp(static_cast<scalar_t>(-0.5) * quad);
                    scalar_t alpha = opacities_a[g] * gaussian;
                    if (alpha <= static_cast<scalar_t>(0)) {
                        continue;
                    }
                    if (alpha > static_cast<scalar_t>(0.999)) {
                        alpha = static_cast<scalar_t>(0.999);
                    }
                    const scalar_t t = trans_a[y][x];
                    for (int64_t c = 0; c < num_channels; ++c) {
                        image_a[y][x][c] += t * alpha * colors_a[g][c];
                    }
                    trans_a[y][x] = t * (static_cast<scalar_t>(1.0) - alpha);
                }
            }
        }
    });

    return image;
}

std::vector<torch::Tensor> gaussian_splat_3d_projected_backward_cpu(
    const torch::Tensor& grad_output,
    const torch::Tensor& means,
    const torch::Tensor& covariances,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    int64_t height,
    int64_t width,
    double min_covariance,
    double sigma_radius
) {
    auto means_c = means.contiguous();
    auto covs_c = covariances.contiguous();
    auto colors_c = colors.contiguous();
    auto opacities_c = opacities.contiguous();
    auto grad_output_c = grad_output.contiguous();
    auto num_gaussians = means_c.size(0);
    auto num_channels = colors_c.size(1);

    auto grad_means = torch::zeros_like(means_c);
    auto grad_covs = torch::zeros_like(covs_c);
    auto grad_colors = torch::zeros_like(colors_c);
    auto grad_opacities = torch::zeros_like(opacities_c);

    AT_DISPATCH_FLOATING_TYPES(means_c.scalar_type(), "gaussian_splat_3d_projected_backward_cpu", [&] {
        auto means_a = means_c.accessor<scalar_t, 2>();
        auto covs_a = covs_c.accessor<scalar_t, 3>();
        auto colors_a = colors_c.accessor<scalar_t, 2>();
        auto opacities_a = opacities_c.accessor<scalar_t, 1>();
        auto grad_out_a = grad_output_c.accessor<scalar_t, 3>();
        auto grad_means_a = grad_means.accessor<scalar_t, 2>();
        auto grad_covs_a = grad_covs.accessor<scalar_t, 3>();
        auto grad_colors_a = grad_colors.accessor<scalar_t, 2>();
        auto grad_opacities_a = grad_opacities.accessor<scalar_t, 1>();

        auto gaussians = precompute_raster_gaussians_2d(
            means_a,
            covs_a,
            num_gaussians,
            height,
            width,
            static_cast<scalar_t>(min_covariance),
            static_cast<scalar_t>(sigma_radius)
        );

        std::vector<std::vector<int64_t>> pixel_lists(static_cast<size_t>(height * width));
        for (int64_t g = 0; g < num_gaussians; ++g) {
            const auto& rg = gaussians[static_cast<size_t>(g)];
            if (rg.max_x < rg.min_x || rg.max_y < rg.min_y) {
                continue;
            }
            const int64_t x0 = std::max<int64_t>(0, rg.min_x);
            const int64_t x1 = std::min<int64_t>(width - 1, rg.max_x);
            const int64_t y0 = std::max<int64_t>(0, rg.min_y);
            const int64_t y1 = std::min<int64_t>(height - 1, rg.max_y);
            for (int64_t y = y0; y <= y1; ++y) {
                for (int64_t x = x0; x <= x1; ++x) {
                    pixel_lists[static_cast<size_t>(y * width + x)].push_back(g);
                }
            }
        }

        for (int64_t y = 0; y < height; ++y) {
            for (int64_t x = 0; x < width; ++x) {
                const auto& ids = pixel_lists[static_cast<size_t>(y * width + x)];
                if (ids.empty()) {
                    continue;
                }

                const int64_t m = static_cast<int64_t>(ids.size());
                std::vector<scalar_t> alpha(static_cast<size_t>(m), static_cast<scalar_t>(0));
                std::vector<scalar_t> gaussian(static_cast<size_t>(m), static_cast<scalar_t>(0));
                std::vector<scalar_t> trans_before(static_cast<size_t>(m), static_cast<scalar_t>(1));
                std::vector<std::array<scalar_t, 4>> suffix_color(static_cast<size_t>(m));
                scalar_t trans = static_cast<scalar_t>(1);

                for (int64_t i = 0; i < m; ++i) {
                    const int64_t g = ids[static_cast<size_t>(i)];
                    const auto& rg = gaussians[static_cast<size_t>(g)];
                    const scalar_t dx = static_cast<scalar_t>(x) - rg.mean_x;
                    const scalar_t dy = static_cast<scalar_t>(y) - rg.mean_y;
                    const scalar_t quad =
                        dx * (rg.inv_xx * dx + rg.inv_xy * dy) +
                        dy * (rg.inv_yx * dx + rg.inv_yy * dy);
                    const scalar_t gauss = std::exp(static_cast<scalar_t>(-0.5) * quad);
                    scalar_t a = opacities_a[g] * gauss;
                    if (a < static_cast<scalar_t>(0)) {
                        a = static_cast<scalar_t>(0);
                    }
                    if (a > static_cast<scalar_t>(0.999)) {
                        a = static_cast<scalar_t>(0.999);
                    }
                    gaussian[static_cast<size_t>(i)] = gauss;
                    alpha[static_cast<size_t>(i)] = a;
                    trans_before[static_cast<size_t>(i)] = trans;
                    trans = trans * (static_cast<scalar_t>(1.0) - a);
                }

                std::array<scalar_t, 4> suffix{};
                suffix.fill(static_cast<scalar_t>(0));
                for (int64_t i = m - 1; i >= 0; --i) {
                    suffix_color[static_cast<size_t>(i)] = suffix;
                    const int64_t g = ids[static_cast<size_t>(i)];
                    for (int64_t c = 0; c < num_channels; ++c) {
                        suffix[static_cast<size_t>(c)] =
                            alpha[static_cast<size_t>(i)] * colors_a[g][c] +
                            (static_cast<scalar_t>(1.0) - alpha[static_cast<size_t>(i)]) * suffix[static_cast<size_t>(c)];
                    }
                }

                for (int64_t i = 0; i < m; ++i) {
                    const int64_t g = ids[static_cast<size_t>(i)];
                    const auto& rg = gaussians[static_cast<size_t>(g)];
                    scalar_t dot_grad = static_cast<scalar_t>(0);
                    for (int64_t c = 0; c < num_channels; ++c) {
                        grad_colors_a[g][c] += grad_out_a[y][x][c] * trans_before[static_cast<size_t>(i)] * alpha[static_cast<size_t>(i)];
                        dot_grad += grad_out_a[y][x][c] * (
                            colors_a[g][c] - suffix_color[static_cast<size_t>(i)][static_cast<size_t>(c)]
                        );
                    }

                    scalar_t raw_alpha = opacities_a[g] * gaussian[static_cast<size_t>(i)];
                    scalar_t grad_alpha = static_cast<scalar_t>(0);
                    if (raw_alpha > static_cast<scalar_t>(0) && raw_alpha < static_cast<scalar_t>(0.999)) {
                        grad_alpha = trans_before[static_cast<size_t>(i)] * dot_grad;
                    }

                    grad_opacities_a[g] += grad_alpha * gaussian[static_cast<size_t>(i)];
                    scalar_t grad_gaussian = grad_alpha * opacities_a[g];

                    const scalar_t dx = static_cast<scalar_t>(x) - rg.mean_x;
                    const scalar_t dy = static_cast<scalar_t>(y) - rg.mean_y;
                    const scalar_t v0 = rg.inv_xx * dx + rg.inv_xy * dy;
                    const scalar_t v1 = rg.inv_yx * dx + rg.inv_yy * dy;
                    const scalar_t common = grad_gaussian * gaussian[static_cast<size_t>(i)];

                    grad_means_a[g][0] += common * v0;
                    grad_means_a[g][1] += common * v1;

                    grad_covs_a[g][0][0] += common * static_cast<scalar_t>(0.5) * v0 * v0;
                    grad_covs_a[g][0][1] += common * static_cast<scalar_t>(0.5) * v0 * v1;
                    grad_covs_a[g][1][0] += common * static_cast<scalar_t>(0.5) * v1 * v0;
                    grad_covs_a[g][1][1] += common * static_cast<scalar_t>(0.5) * v1 * v1;
                }
            }
        }
    });

    return {grad_means, grad_covs, grad_colors, grad_opacities};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "gaussian_splat_2d_forward_cpu",
        &gaussian_splat_2d_forward_cpu,
        "TinySplat CPU forward kernel scaffold"
    );
    m.def(
        "gaussian_splat_2d_backward_cpu",
        &gaussian_splat_2d_backward_cpu,
        "TinySplat CPU backward kernel scaffold"
    );
    m.def(
        "gaussian_splat_3d_forward_cpu",
        &gaussian_splat_3d_forward_cpu,
        "TinySplat CPU native 3D forward rasterizer"
    );
    m.def(
        "gaussian_splat_3d_projected_forward_cpu",
        &gaussian_splat_3d_projected_forward_cpu,
        "TinySplat CPU native projected 3D forward rasterizer"
    );
    m.def(
        "gaussian_splat_3d_projected_backward_cpu",
        &gaussian_splat_3d_projected_backward_cpu,
        "TinySplat CPU native projected 3D backward rasterizer"
    );
}
