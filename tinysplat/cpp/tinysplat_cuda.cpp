#include <torch/extension.h>

torch::Tensor gaussian_splat_2d_forward_cuda(
    torch::Tensor means,
    torch::Tensor covariances,
    torch::Tensor colors,
    torch::Tensor opacities,
    int64_t height,
    int64_t width
);

std::vector<torch::Tensor> gaussian_splat_2d_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor means,
    torch::Tensor covariances,
    torch::Tensor colors,
    torch::Tensor opacities,
    int64_t height,
    int64_t width
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gaussian_splat_2d_forward_cuda", &gaussian_splat_2d_forward_cuda, "2D Gaussian splatting forward (CUDA)");
    m.def("gaussian_splat_2d_backward_cuda", &gaussian_splat_2d_backward_cuda, "2D Gaussian splatting backward (CUDA)");
    m.def("gaussian_splat_3d_projected_forward_cuda", &gaussian_splat_3d_projected_forward_cuda, "3D projected Gaussian splatting forward (CUDA)");
    m.def("gaussian_splat_3d_projected_backward_cuda", &gaussian_splat_3d_projected_backward_cuda, "3D projected Gaussian splatting backward (CUDA)");
}
