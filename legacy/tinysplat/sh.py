"""Spherical harmonics helpers for view-dependent Gaussian color (3DGS schedule)."""

from __future__ import annotations

import math
from typing import Optional

import torch

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)

NUM_SH_BASES = 16  # degree 3


def sh_degree_from_step(step: int, max_degree: int = 3) -> int:
    """3DGS / nerfstudio schedule: introduce one SH band every 1000 steps."""
    if max_degree <= 0:
        return 0
    return min(max_degree, step // 1000)


def num_sh_coeffs(degree: int) -> int:
    return (degree + 1) ** 2


def rgb_to_sh_dc(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB in [0,1] to SH DC coefficients (N,3)."""
    return (rgb - 0.5) / SH_C0


def init_sh_from_rgb(rgb: torch.Tensor, max_degree: int = 3) -> torch.Tensor:
    """Allocate (N, 16, 3) SH features with DC from RGB and rest zero."""
    n = rgb.shape[0]
    sh = torch.zeros(n, NUM_SH_BASES, 3, dtype=rgb.dtype, device=rgb.device)
    sh[:, 0, :] = rgb_to_sh_dc(rgb.clamp(0.0, 1.0))
    return sh


def eval_sh(dirs: torch.Tensor, sh: torch.Tensor, degree: int) -> torch.Tensor:
    """Evaluate SH to RGB. dirs: (N,3) unit vectors, sh: (N,16,3). Returns (N,3)."""
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    result = SH_C0 * sh[:, 0, :]
    if degree >= 1:
        result = (
            result
            - SH_C1 * y.unsqueeze(-1) * sh[:, 1, :]
            + SH_C1 * z.unsqueeze(-1) * sh[:, 2, :]
            - SH_C1 * x.unsqueeze(-1) * sh[:, 3, :]
        )
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        result = (
            result
            + SH_C2[0] * xy.unsqueeze(-1) * sh[:, 4, :]
            + SH_C2[1] * yz.unsqueeze(-1) * sh[:, 5, :]
            + SH_C2[2] * (2.0 * zz - xx - yy).unsqueeze(-1) * sh[:, 6, :]
            + SH_C2[3] * xz.unsqueeze(-1) * sh[:, 7, :]
            + SH_C2[4] * (xx - yy).unsqueeze(-1) * sh[:, 8, :]
        )
    if degree >= 3:
        result = (
            result
            + SH_C3[0] * (y * (3.0 * xx - yy)).unsqueeze(-1) * sh[:, 9, :]
            + SH_C3[1] * (xy * z).unsqueeze(-1) * sh[:, 10, :]
            + SH_C3[2] * (y * (4.0 * zz - xx - yy)).unsqueeze(-1) * sh[:, 11, :]
            + SH_C3[3] * (z * (2.0 * zz - 3.0 * xx - 3.0 * yy)).unsqueeze(-1) * sh[:, 12, :]
            + SH_C3[4] * (x * (4.0 * zz - xx - yy)).unsqueeze(-1) * sh[:, 13, :]
            + SH_C3[5] * (z * (xx - yy)).unsqueeze(-1) * sh[:, 14, :]
            + SH_C3[6] * (x * (xx - 3.0 * yy)).unsqueeze(-1) * sh[:, 15, :]
        )
    return (result + 0.5).clamp(0.0, 1.0)


def colors_from_sh(
    means: torch.Tensor,
    sh_coeffs: torch.Tensor,
    camera_to_world: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    """View-dependent RGB from means + SH + camera center."""
    cam = camera_to_world[:3, 3]
    dirs = means - cam.unsqueeze(0)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return eval_sh(dirs, sh_coeffs, degree)


def active_sh_mask(degree: int, device=None, dtype=None) -> torch.Tensor:
    """Boolean mask over 16 bases that are active at this degree."""
    n = num_sh_coeffs(degree)
    mask = torch.zeros(NUM_SH_BASES, dtype=torch.bool, device=device)
    mask[:n] = True
    return mask
