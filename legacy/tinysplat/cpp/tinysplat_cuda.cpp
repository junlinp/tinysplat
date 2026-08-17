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
    bool density_normalize = true
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
    float sigma_radius
);


std::vector<torch::Tensor> project_3d_forward_cuda(
    torch::Tensor means, torch::Tensor cov3,
    torch::Tensor intrinsics, torch::Tensor camera_to_world,
    double near_plane, double min_covariance
);

std::vector<torch::Tensor> project_3d_backward_cuda(
    torch::Tensor grad_proj_means, torch::Tensor grad_cov2d, torch::Tensor grad_depth,
    torch::Tensor means, torch::Tensor cov3,
    torch::Tensor intrinsics, torch::Tensor camera_to_world,
    double near_plane, double min_covariance
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gaussian_splat_2d_forward_cuda", &gaussian_splat_2d_forward_cuda, "2D Gaussian splatting forward (CUDA)");
    m.def("gaussian_splat_2d_backward_cuda", &gaussian_splat_2d_backward_cuda, "2D Gaussian splatting backward (CUDA)");
    m.def("gaussian_splat_3d_projected_forward_cuda", &gaussian_splat_3d_projected_forward_cuda, "3D projected Gaussian splatting forward (CUDA)");
    m.def("gaussian_splat_3d_projected_backward_cuda", &gaussian_splat_3d_projected_backward_cuda, "3D projected Gaussian splatting backward (CUDA)");
    m.def("project_3d_forward_cuda", &project_3d_forward_cuda, "Fused 3D->2D projection (CUDA)");
    m.def("project_3d_backward_cuda", &project_3d_backward_cuda, "Fused 3D->2D projection VJP (CUDA)");
}
