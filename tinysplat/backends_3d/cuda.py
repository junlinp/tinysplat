"""CUDA backend for 3D Gaussian splatting using gsplat."""

from __future__ import annotations

import inspect
from typing import Any, Dict

import torch

from .common import Backend3DOps

LAST_RENDER_INFO = None


def get_last_render_info():
    return LAST_RENDER_INFO


def _import_gsplat_rasterization():
    try:
        import gsplat  # type: ignore

        if hasattr(gsplat, "rasterization"):
            return gsplat.rasterization
    except Exception:
        pass

    try:
        from gsplat.rendering import rasterization  # type: ignore

        return rasterization
    except Exception as exc:
        raise RuntimeError(
            "gsplat rasterization is unavailable. Install with: "
            "python3 -m pip install --user git+https://github.com/nerfstudio-project/gsplat.git"
        ) from exc


def _rotation_matrix_to_quaternion(rot: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix batch (..., 3, 3) to quaternion (..., 4) in wxyz."""
    m = rot
    q = torch.zeros((*m.shape[:-2], 4), dtype=m.dtype, device=m.device)

    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    pos = trace > 0

    if pos.any():
        t = torch.sqrt(trace[pos] + 1.0) * 2.0
        q[pos, 0] = 0.25 * t
        q[pos, 1] = (m[pos, 2, 1] - m[pos, 1, 2]) / t
        q[pos, 2] = (m[pos, 0, 2] - m[pos, 2, 0]) / t
        q[pos, 3] = (m[pos, 1, 0] - m[pos, 0, 1]) / t

    npos = ~pos
    if npos.any():
        m2 = m[npos]
        q2 = q[npos]
        c0 = (m2[:, 0, 0] > m2[:, 1, 1]) & (m2[:, 0, 0] > m2[:, 2, 2])
        c1 = ~c0 & (m2[:, 1, 1] > m2[:, 2, 2])
        c2 = ~(c0 | c1)

        if c0.any():
            t = torch.sqrt(1.0 + m2[c0, 0, 0] - m2[c0, 1, 1] - m2[c0, 2, 2]) * 2.0
            q2[c0, 0] = (m2[c0, 2, 1] - m2[c0, 1, 2]) / t
            q2[c0, 1] = 0.25 * t
            q2[c0, 2] = (m2[c0, 0, 1] + m2[c0, 1, 0]) / t
            q2[c0, 3] = (m2[c0, 0, 2] + m2[c0, 2, 0]) / t

        if c1.any():
            t = torch.sqrt(1.0 + m2[c1, 1, 1] - m2[c1, 0, 0] - m2[c1, 2, 2]) * 2.0
            q2[c1, 0] = (m2[c1, 0, 2] - m2[c1, 2, 0]) / t
            q2[c1, 1] = (m2[c1, 0, 1] + m2[c1, 1, 0]) / t
            q2[c1, 2] = 0.25 * t
            q2[c1, 3] = (m2[c1, 1, 2] + m2[c1, 2, 1]) / t

        if c2.any():
            t = torch.sqrt(1.0 + m2[c2, 2, 2] - m2[c2, 0, 0] - m2[c2, 1, 1]) * 2.0
            q2[c2, 0] = (m2[c2, 1, 0] - m2[c2, 0, 1]) / t
            q2[c2, 1] = (m2[c2, 0, 2] + m2[c2, 2, 0]) / t
            q2[c2, 2] = (m2[c2, 1, 2] + m2[c2, 2, 1]) / t
            q2[c2, 3] = 0.25 * t

        q[npos] = q2

    q = q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-12)
    return q


def _covariance_to_scales_quats(covariances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    evals, evecs = torch.linalg.eigh(covariances)
    evals = torch.clamp(evals, min=1e-12)

    det = torch.linalg.det(evecs)
    neg = det < 0
    if neg.any():
        evecs[neg, :, 2] *= -1.0

    scales = torch.sqrt(evals)
    quats = _rotation_matrix_to_quaternion(evecs)
    return scales, quats


def _build_viewmat(camera_to_world: torch.Tensor) -> torch.Tensor:
    rot_c2w = camera_to_world[:3, :3]
    trans_c2w = camera_to_world[:3, 3]
    rot_w2c = rot_c2w.transpose(0, 1)
    trans_w2c = -rot_w2c @ trans_c2w

    viewmat = torch.eye(4, dtype=camera_to_world.dtype, device=camera_to_world.device)
    viewmat[:3, :3] = rot_w2c
    viewmat[:3, 3] = trans_w2c
    return viewmat


def _call_gsplat(
    rasterization,
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
    near_plane: float,
):
    sig = inspect.signature(rasterization)
    params = set(sig.parameters.keys())

    common: Dict[str, Any] = {
        "means": means,
        "means3d": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "colors": colors,
        "viewmats": viewmats,
        "viewmatrix": viewmats,
        "Ks": Ks,
        "K": Ks,
        "width": width,
        "W": width,
        "height": height,
        "H": height,
        "near_plane": near_plane,
        "near": near_plane,
        "packed": False,
    }

    kwargs = {k: v for k, v in common.items() if k in params}

    out = rasterization(**kwargs)

    if isinstance(out, tuple):
        # gsplat commonly returns (render, alpha, meta)
        if len(out) >= 3:
            return out[0], out[2]
        return out[0], {}
    return out, {}


def render_cuda_3d(
    means: torch.Tensor,
    covariances: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_to_world: torch.Tensor,
    height: int,
    width: int,
    near_plane: float,
    min_covariance: float,
    sigma_radius: float,
    scales: torch.Tensor | None = None,
    quats: torch.Tensor | None = None,
) -> torch.Tensor:
    del min_covariance, sigma_radius

    rasterization = _import_gsplat_rasterization()

    device = torch.device("cuda")
    means = means.to(device)
    covariances = covariances.to(device)
    colors = colors.to(device)
    opacities = opacities.to(device)
    intrinsics = intrinsics.to(device)
    camera_to_world = camera_to_world.to(device)

    if scales is None or quats is None:
        scales, quats = _covariance_to_scales_quats(covariances)
    else:
        scales = scales.to(device)
        quats = quats.to(device)
        quats = quats / torch.clamp(torch.linalg.norm(quats, dim=-1, keepdim=True), min=1e-12)
    viewmats = _build_viewmat(camera_to_world).unsqueeze(0)
    Ks = intrinsics.unsqueeze(0)

    global LAST_RENDER_INFO
    render, meta = _call_gsplat(
        rasterization=rasterization,
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        near_plane=near_plane,
    )

    # Save gsplat meta for training strategy (e.g., means2d grads/radii/ids).
    info = {
        "width": width,
        "height": height,
        "n_cameras": 1,
    }
    if isinstance(meta, dict):
        info.update(meta)
    LAST_RENDER_INFO = info

    # Normalize possible output layouts to (H, W, C)
    if render.ndim == 4:
        # likely (B, H, W, C)
        render = render[0]
    elif render.ndim == 3 and render.shape[0] in (1, 3, 4) and render.shape[-1] not in (1, 3, 4):
        # possible (C, H, W)
        render = render.permute(1, 2, 0)

    return render


CUDA_BACKEND_3D = Backend3DOps(
    name="cuda",
    render=render_cuda_3d,
    is_compiled=True,
)
