"""Metal tiled 3DGS backend (FastGS-compatible). Replaces naive MPS when available."""

from __future__ import annotations

import torch

from .common import Backend3DOps


try:
    from ..metal_backend import metal_available, render_metal_3d as _render_metal_3d

    _HAS_METAL = bool(metal_available())
except Exception:
    _HAS_METAL = False
    _render_metal_3d = None


def render_metal_backend_3d(
    means: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_to_world: torch.Tensor,
    height: int,
    width: int,
    near_plane: float = 1e-4,
    min_covariance: float = 1e-4,
    sigma_radius: float = 4.0,
    compact_box_beta: float = 3.0,
    use_compact_box: bool = True,
    covariances: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    if not _HAS_METAL or _render_metal_3d is None:
        raise RuntimeError("Metal 3D backend unavailable")
    return _render_metal_3d(
        means=means,
        covariances=covariances,
        colors=colors,
        opacities=opacities,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
        height=height,
        width=width,
        near_plane=near_plane,
        min_covariance=min_covariance,
        sigma_radius=sigma_radius,
        compact_box_beta=compact_box_beta,
        use_compact_box=use_compact_box,
        **kwargs,
    )


if _HAS_METAL:
    METAL_BACKEND_3D = Backend3DOps(
        name="metal",
        render=render_metal_backend_3d,
        is_compiled=True,
    )
else:
    METAL_BACKEND_3D = None
