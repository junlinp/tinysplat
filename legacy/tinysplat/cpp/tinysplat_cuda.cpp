#include <torch/extension.h>

torch::Tensor gaussian_splat_2d_forward_cuda(
    torch::Tensor means,
    torch::Tensor covariances,
    torch::Tensor colors,
    torch::Tensor opacities,
    int64_t height,
    int64_t width,
    // Density normalization (1/(2*pi*sqrt(det))) belongs to the weighted
    // compositing mode. The 3D projected path passes false for the 3DGS alpha
    // convention; defaulted true so the standalone 2D API is unchanged.
    bool density_normalize = true
);

std::vector<torch::Tensor> gaussian_splat_2d_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor means,
    torch::Tensor covariances,
    torch::Tensor colors,
    torch::Tensor opacities,
    int64_t height,
    int64_t width,
    bool density_normalize = true,
    torch::Tensor tile_starts_in = torch::Tensor(),
    torch::Tensor tile_bins_in = torch::Tensor(),
    torch::Tensor valid = torch::Tensor()
);

torch::Tensor gaussian_splat_3d_projected_forward_cuda(
    torch::Tensor projected_means,
    torch::Tensor projected_covariances,
    torch::Tensor projected_colors,
    torch::Tensor projected_opacities,
    int64_t height,
    int64_t width,
    float min_covariance,
    float sigma_radius
);

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
    torch::Tensor tile_starts = torch::Tensor(),
    torch::Tensor tile_bins = torch::Tensor(),
    torch::Tensor valid = torch::Tensor()
);


std::vector<torch::Tensor> gaussian_splat_3d_projected_forward_binned_cuda(
    torch::Tensor projected_means,
    torch::Tensor projected_covariances,
    torch::Tensor projected_colors,
    torch::Tensor projected_opacities,
    int64_t height, int64_t width, float min_covariance, float sigma_radius,
    torch::Tensor depths, torch::Tensor valid);

std::vector<torch::Tensor> project_3d_forward_cuda(
    torch::Tensor means, torch::Tensor cov3,
    torch::Tensor intrinsics, torch::Tensor camera_to_world,
    double near_plane, double min_covariance,
    double height, double width
);

std::vector<torch::Tensor> project_3d_backward_cuda(
    torch::Tensor grad_proj_means, torch::Tensor grad_cov2d, torch::Tensor grad_depth,
    torch::Tensor means, torch::Tensor cov3,
    torch::Tensor intrinsics, torch::Tensor camera_to_world,
    double near_plane, double min_covariance,
    double height, double width
);

torch::Tensor footprint_hit_count_cuda(
    torch::Tensor projected_means,
    torch::Tensor projected_covariances,
    torch::Tensor projected_opacities,
    torch::Tensor error_mask,
    int64_t height, int64_t width
);

torch::Tensor quat_scale_to_cov3_cuda(torch::Tensor quats, torch::Tensor log_scales);
std::vector<torch::Tensor> quat_scale_to_cov3_vjp_cuda(
    torch::Tensor quats, torch::Tensor log_scales, torch::Tensor grad_cov3);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gaussian_splat_2d_forward_cuda", &gaussian_splat_2d_forward_cuda, "2D Gaussian splatting forward (CUDA)");
    m.def("gaussian_splat_2d_backward_cuda", &gaussian_splat_2d_backward_cuda,
          "2D Gaussian splatting backward (CUDA)",
          py::arg("grad_output"), py::arg("means"), py::arg("covariances"),
          py::arg("colors"), py::arg("opacities"), py::arg("height"), py::arg("width"),
          py::arg("density_normalize") = true,
          py::arg("tile_starts_in") = torch::empty({0}, torch::kInt32),
          py::arg("tile_bins_in") = torch::empty({0}, torch::kInt32),
          py::arg("valid") = torch::empty({0}, torch::kBool));
    m.def("gaussian_splat_3d_projected_forward_cuda", &gaussian_splat_3d_projected_forward_cuda, "3D projected Gaussian splatting forward (CUDA)");
    // pybind11 does not inherit C++ default arguments, so the optional bins
    // have to be declared here or every caller must pass them.
    m.def("gaussian_splat_3d_projected_backward_cuda", &gaussian_splat_3d_projected_backward_cuda,
          "3D projected Gaussian splatting backward (CUDA)",
          py::arg("grad_output"), py::arg("projected_means"), py::arg("projected_covariances"),
          py::arg("projected_colors"), py::arg("projected_opacities"),
          py::arg("height"), py::arg("width"),
          py::arg("min_covariance"), py::arg("sigma_radius"),
          py::arg("tile_starts") = torch::empty({0}, torch::kInt32),
          py::arg("tile_bins") = torch::empty({0}, torch::kInt32),
          py::arg("valid") = torch::empty({0}, torch::kBool));
    m.def("project_3d_forward_cuda", &project_3d_forward_cuda, "Fused 3D->2D projection (CUDA)");
    m.def("project_3d_backward_cuda", &project_3d_backward_cuda, "Fused 3D->2D projection VJP (CUDA)");
    m.def("footprint_hit_count_cuda", &footprint_hit_count_cuda, "FastGS VCD footprint hit counts (CUDA)");
    m.def("quat_scale_to_cov3_cuda", &quat_scale_to_cov3_cuda, "Fused quat+log-scale -> 3x3 covariance (CUDA)");
    m.def("quat_scale_to_cov3_vjp_cuda", &quat_scale_to_cov3_vjp_cuda, "VJP of the above (CUDA)");
    m.def("gaussian_splat_3d_projected_forward_binned_cuda",
          &gaussian_splat_3d_projected_forward_binned_cuda,
          "3D projected forward returning {image, tile_starts, tile_bins} (CUDA)",
          py::arg("projected_means"), py::arg("projected_covariances"),
          py::arg("projected_colors"), py::arg("projected_opacities"),
          py::arg("height"), py::arg("width"),
          py::arg("min_covariance"), py::arg("sigma_radius"),
          py::arg("depths") = torch::empty({0}, torch::kFloat32),
          py::arg("valid") = torch::empty({0}, torch::kBool));
}
