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

    for (int64_t g = 0; g < num_gaussians; ++g) {
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
        const scalar_t lambda_max = std::max((trace + disc) * static_cast<scalar_t>(0.5), kEps);
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
}
