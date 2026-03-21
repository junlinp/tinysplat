"""MPS backend for 3D Gaussian splatting."""

try:
    from tinysplat_mps import HAS_COMPILED_MPS_EXTENSION, gaussian_splat_3d_forward_mps

    from .common import Backend3DOps

    MPS_BACKEND_3D = Backend3DOps(
        name="mps",
        render=gaussian_splat_3d_forward_mps,
        is_compiled=HAS_COMPILED_MPS_EXTENSION,
    )
except ImportError:
    from .projected import make_projected_backend

    MPS_BACKEND_3D = make_projected_backend("mps")
