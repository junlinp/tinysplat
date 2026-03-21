"""MPS backend registration."""

try:
    from tinysplat_mps import (
        HAS_COMPILED_MPS_EXTENSION,
        gaussian_splat_2d_backward_mps,
        gaussian_splat_2d_forward_mps,
    )

    from .common import BackendOps

    MPS_BACKEND = BackendOps(
        name="mps",
        forward=gaussian_splat_2d_forward_mps,
        backward=gaussian_splat_2d_backward_mps,
        is_compiled=HAS_COMPILED_MPS_EXTENSION,
    )
except ImportError:
    from .common import BackendOps
    from .python import backward_pytorch, forward_pytorch

    MPS_BACKEND = BackendOps(
        name="mps",
        forward=forward_pytorch,
        backward=backward_pytorch,
        is_compiled=False,
    )
