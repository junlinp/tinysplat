"""MPS device type: prefer FastGS Metal tiled rasterizer; fall back to legacy naive MPS."""

from __future__ import annotations

try:
    from .metal import METAL_BACKEND_3D, _HAS_METAL
except Exception:
    METAL_BACKEND_3D = None
    _HAS_METAL = False

if _HAS_METAL and METAL_BACKEND_3D is not None:
    MPS_BACKEND_3D = METAL_BACKEND_3D
else:
    try:
        from tinysplat.mps import HAS_COMPILED_MPS_EXTENSION, gaussian_splat_3d_forward_mps

        from .common import Backend3DOps

        MPS_BACKEND_3D = Backend3DOps(
            name="mps",
            render=gaussian_splat_3d_forward_mps,
            is_compiled=HAS_COMPILED_MPS_EXTENSION,
        )
    except ImportError:
        from .projected import make_projected_backend

        MPS_BACKEND_3D = make_projected_backend("mps")
