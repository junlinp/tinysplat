"""CUDA backend registration."""

try:
    from tinysplat_cuda import gaussian_splat_2d_backward_cuda, gaussian_splat_2d_forward_cuda

    from .common import BackendOps

    CUDA_BACKEND = BackendOps(
        name="cuda",
        forward=gaussian_splat_2d_forward_cuda,
        backward=gaussian_splat_2d_backward_cuda,
        is_compiled=True,
    )
except ImportError:
    from .common import BackendOps
    from .python import backward_pytorch, forward_pytorch

    CUDA_BACKEND = BackendOps(
        name="cuda",
        forward=forward_pytorch,
        backward=backward_pytorch,
        is_compiled=False,
    )
