"""Metal tiled 3DGS backend (FastGS-compatible). Replaces naive MPS when available."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from .common import Backend3DOps


def _import_metal_backend():
    # Prefer package module; also allow repo-root / tinysplat/ layouts.
    try:
        from tinysplat.metal_backend import metal_available, render_metal_3d

        return metal_available, render_metal_3d
    except ImportError:
        pass
    repo = Path(__file__).resolve().parents[3]
    for extra in (repo, repo / "tinysplat"):
        s = str(extra)
        if s not in sys.path:
            sys.path.insert(0, s)
    try:
        from metal_backend import metal_available, render_metal_3d  # type: ignore

        return metal_available, render_metal_3d
    except ImportError:
        from tinysplat.metal_backend import metal_available, render_metal_3d

        return metal_available, render_metal_3d


try:
    _metal_available, _render_metal_3d = _import_metal_backend()
    _HAS_METAL = bool(_metal_available())
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
