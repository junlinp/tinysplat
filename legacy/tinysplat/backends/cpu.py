"""CPU backend registration."""

from .common import BackendOps
from .python import backward_pytorch


def _make_compiled_backend(extension_module) -> BackendOps:
    """Adapt a low-level compiled extension module to the backend contract."""

    def forward(means, covariances, colors, opacities, height, width):
        image = extension_module.gaussian_splat_2d_forward_cpu(
            means,
            covariances,
            colors,
            opacities,
            height,
            width,
        )
        return image, []

    def backward(
        grad_output,
        means,
        covariances,
        colors,
        opacities,
        height,
        width,
        intermediates,
        needs_input_grad,
    ):
        if hasattr(extension_module, "gaussian_splat_2d_backward_cpu"):
            try:
                grads = extension_module.gaussian_splat_2d_backward_cpu(
                    grad_output,
                    means,
                    covariances,
                    colors,
                    opacities,
                    height,
                    width,
                )
                return tuple(grads)
            except RuntimeError:
                pass

        return backward_pytorch(
            grad_output,
            means,
            covariances,
            colors,
            opacities,
            height,
            width,
            intermediates,
            needs_input_grad,
        )

    return BackendOps(
        name="cpu",
        forward=forward,
        backward=backward,
        is_compiled=True,
    )


try:
    from tinysplat_cpp import gaussian_splat_2d_backward_cpu, gaussian_splat_2d_forward_cpu

    class _ImportedExtension:
        gaussian_splat_2d_forward_cpu = staticmethod(gaussian_splat_2d_forward_cpu)
        gaussian_splat_2d_backward_cpu = staticmethod(gaussian_splat_2d_backward_cpu)

    CPU_BACKEND = _make_compiled_backend(_ImportedExtension)
except ImportError:
    try:
        from ..cpp import load_cpu_extension

        _cpu_extension = load_cpu_extension()
    except ImportError:
        _cpu_extension = None

    if _cpu_extension is not None:
        CPU_BACKEND = _make_compiled_backend(_cpu_extension)
    else:
        from .python import forward_pytorch

        CPU_BACKEND = BackendOps(
            name="cpu",
            forward=forward_pytorch,
            backward=backward_pytorch,
            is_compiled=False,
        )
