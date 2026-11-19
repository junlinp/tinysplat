"""Device dispatch for the per-render statistics FastGS needs.

FastGS requires three things the rasterizer must supply: VCD footprint hit
counts, AbsGS |dL/dmean2d| sums, and screen-space radii. These were previously
imported straight from ``metal_backend``, so on CUDA they resolved to stubs
returning ``None`` -- the trainer then substituted all-zero VCD counts and
densification silently never fired.

Each getter here tries Metal first (unchanged behaviour on Apple Silicon), then
CUDA, and returns ``None`` only when neither backend can serve the call.
"""

from __future__ import annotations

from typing import Optional

import torch


def _metal():
    try:
        from . import metal_backend

        if metal_backend.metal_available():
            return metal_backend
    except Exception:
        pass
    return None


def _cuda():
    try:
        from . import cuda_backend

        if cuda_backend.cuda_available():
            return cuda_backend
    except Exception:
        pass
    return None


def stats_available() -> bool:
    return _metal() is not None or _cuda() is not None


def stats_backend_name() -> str:
    if _metal() is not None:
        return "metal"
    if _cuda() is not None:
        return "cuda"
    return "none"


def count_session_hits(
    error_mask: torch.Tensor, num_gaussians: int, height: int, width: int
) -> Optional[torch.Tensor]:
    for backend in (_metal(), _cuda()):
        if backend is None:
            continue
        hits = backend.count_session_hits(error_mask, num_gaussians, height, width)
        if hits is not None:
            return hits
    return None


def last_grad_means2d_abs() -> Optional[torch.Tensor]:
    for backend in (_metal(), _cuda()):
        if backend is None:
            continue
        g = backend.last_grad_means2d_abs()
        if g is not None:
            return g
    return None


def last_grad_means2d() -> Optional[torch.Tensor]:
    for backend in (_metal(), _cuda()):
        if backend is None:
            continue
        g = backend.last_grad_means2d()
        if g is not None:
            return g
    return None


def last_radii2d() -> Optional[torch.Tensor]:
    for backend in (_metal(), _cuda()):
        if backend is None:
            continue
        r = backend.last_radii2d()
        if r is not None:
            return r
    return None
