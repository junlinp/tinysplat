"""FastGS statistics for the CUDA path.

Mirrors the subset of ``metal_backend`` that FastGS depends on. Without these
the trainer falls back to stubs returning ``None``, the VCD counts become all
zero, no Gaussian ever clears the densify threshold, and training silently
never densifies -- which is why ``--fastgs`` was Metal-only.

Values are read from the per-render cache in ``backends_3d.cuda``; each getter
returns results indexed by *original* Gaussian, scattering from the projected
(visible, depth-sorted) ordering via the cached ``visible_indices``.
"""

from __future__ import annotations

from typing import Optional

import torch


def cuda_available() -> bool:
    try:
        from .cpp import load_cuda_extension

        return torch.cuda.is_available() and load_cuda_extension() is not None
    except Exception:
        return False


def _session() -> dict:
    from .backends_3d.cuda import last_session

    return last_session()


def _scatter_to_original(
    values: torch.Tensor, indices: torch.Tensor, n: int
) -> torch.Tensor:
    """Place per-projected-splat values back at their original Gaussian slots."""
    out = torch.zeros(n, dtype=values.dtype, device=values.device)
    idx = indices.to(values.device).reshape(-1)
    k = min(idx.numel(), values.numel())
    if k:
        out.index_copy_(0, idx[:k], values.reshape(-1)[:k])
    return out


def count_session_hits(
    error_mask: torch.Tensor,
    num_gaussians: int,
    height: int,
    width: int,
) -> Optional[torch.Tensor]:
    """FastGS VCD counts: high-error pixels each Gaussian actually contributes to."""
    from .cpp import load_cuda_extension

    sess = _session()
    ext = load_cuda_extension()
    if ext is None or not hasattr(ext, "footprint_hit_count_cuda") or not sess:
        return None
    if sess.get("height") != int(height) or sess.get("width") != int(width):
        return None  # mask does not belong to the cached render

    mask = error_mask.detach()
    if mask.dtype != torch.uint8:
        mask = mask.to(dtype=torch.uint8)
    mask = mask.reshape(int(height), int(width)).contiguous().cuda()

    counts = ext.footprint_hit_count_cuda(
        sess["proj_means"].cuda(),
        sess["proj_covs"].cuda(),
        sess["proj_opacities"].cuda(),
        mask,
        int(height),
        int(width),
    )
    return _scatter_to_original(
        counts.to(torch.int64), sess["visible_indices"], int(num_gaussians)
    )


def _grad2d():
    from .backends_3d.cuda import last_grad2d

    return last_grad2d()


def last_grad_means2d_abs() -> Optional[torch.Tensor]:
    """AbsGS per-splat |dL/dmean2d| sums (N, 2) from the last backward, or None."""
    return _grad2d().get("abs")


def last_grad_means2d() -> Optional[torch.Tensor]:
    """Signed per-splat dL/dmean2d (N, 2) from the last backward, or None."""
    return _grad2d().get("signed")


def last_radii2d() -> Optional[torch.Tensor]:
    """Screen-space radii in pixels (N,), or None.

    Largest 3-sigma axis of the projected covariance, matching the
    ``sigma_radius`` support used when building bounding boxes.
    """
    sess = _session()
    if not sess or "proj_covs" not in sess:
        return None
    cov = sess["proj_covs"].reshape(-1, 2, 2).float()
    xx, xy, yy = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
    trace = xx + yy
    disc = torch.sqrt(torch.clamp((xx - yy) ** 2 + 4.0 * xy * xy, min=0.0))
    lambda_max = torch.clamp(0.5 * (trace + disc), min=0.0)
    radii = 3.0 * torch.sqrt(lambda_max)
    return _scatter_to_original(
        radii, sess["visible_indices"], int(sess.get("num_gaussians", radii.numel()))
    )
