"""
halide_backend.py

Halide backend for TinySplat — bridges Halide JIT-compiled Gaussian splatting
pipelines to the torch.autograd.Function interface used by GaussianSplat2DFunction.

Usage:
    export HL_PATH=$HOME/halide
    export TINYSPLAT_HALIDE_LIB=/path/to/libtinysplat_halide_pipeline.so
    export TINYSPLAT_BACKEND=halide   # optional, forces Halide over other backends

    # Then tinyplat will auto-detect and use the Halide backend.
"""

import ctypes
import os
import torch
import numpy as np
from typing import Tuple, List

# ---------------------------------------------------------------------------
# Load the Halide shared library (deferred to first use)
# ---------------------------------------------------------------------------

_HALIDE_LIB_PATH = os.environ.get("TINYSPLAT_HALIDE_LIB", "")
_halide_lib = None


def _try_load_halide_lib():
    global _halide_lib
    if _halide_lib is not None:
        return True

    if not _HALIDE_LIB_PATH or not os.path.exists(_HALIDE_LIB_PATH):
        return False

    try:
        _halide_lib = ctypes.CDLL(_HALIDE_LIB_PATH)

        # C function signatures
        _halide_lib.gaussian_splat_forward.argtypes = [
            ctypes.c_void_p,   # means
            ctypes.c_void_p,   # covariances
            ctypes.c_void_p,   # colors
            ctypes.c_void_p,   # opacities
            ctypes.c_int,      # N
            ctypes.c_int,      # height
            ctypes.c_int,      # width
            ctypes.c_int,      # C
            ctypes.c_void_p,   # output
        ]
        _halide_lib.gaussian_splat_forward.restype = ctypes.c_int

        _halide_lib.gaussian_splat_backward.argtypes = [
            ctypes.c_void_p,   # grad_output
            ctypes.c_void_p,   # means
            ctypes.c_void_p,   # covariances
            ctypes.c_void_p,   # colors
            ctypes.c_void_p,   # opacities
            ctypes.c_int,      # N
            ctypes.c_int,      # height
            ctypes.c_int,      # width
            ctypes.c_int,      # C
            ctypes.c_void_p,   # grad_means
            ctypes.c_void_p,   # grad_cov
            ctypes.c_void_p,   # grad_colors
            ctypes.c_void_p,   # grad_opacities
        ]
        _halide_lib.gaussian_splat_backward.restype = ctypes.c_int

        return True
    except (OSError, AttributeError) as e:
        print(f"[tinysplat halide] Failed to load Halide library: {e}")
        return False


def _ensure_halide():
    if _halide_lib is None and not _try_load_halide_lib():
        raise ImportError(
            "Halide backend not available. Set TINYSPLAT_HALIDE_LIB "
            "to the path of libtinysplat_halide_pipeline.so"
        )


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def forward_halide(
    means: torch.Tensor,          # (N, 2)
    covariances: torch.Tensor,    # (N, 2, 2)
    colors: torch.Tensor,          # (N, C)
    opacities: torch.Tensor,      # (N,)
    height: int,
    width: int,
) -> Tuple[torch.Tensor, list]:
    """
    Forward pass using the Halide JIT-compiled pipeline.
    Falls back to pure-PyTorch if Halide library unavailable.
    """
    if _halide_lib is None and not _try_load_halide_lib():
        from .backends.python import forward_pytorch
        return forward_pytorch(means, covariances, colors, opacities, height, width)

    N, C = colors.shape
    device = means.device

    # Move to CPU, contiguous numpy arrays (Halide reads host memory)
    means_np     = means.detach().cpu().numpy().astype(np.float32)
    cov_np       = covariances.detach().cpu().numpy().astype(np.float32)
    colors_np    = colors.detach().cpu().numpy().astype(np.float32)
    opacities_np = opacities.detach().cpu().numpy().astype(np.float32)
    output_np    = np.zeros((height, width, C), dtype=np.float32)

    rc = _halide_lib.gaussian_splat_forward(
        means_np.ctypes.data,
        cov_np.ctypes.data,
        colors_np.ctypes.data,
        opacities_np.ctypes.data,
        N, height, width, C,
        output_np.ctypes.data,
    )

    if rc != 0:
        # Fall back on error
        from .backends.python import forward_pytorch
        return forward_pytorch(means, covariances, colors, opacities, height, width)

    output = torch.from_numpy(output_np).to(device)
    return output, []


# ---------------------------------------------------------------------------
# Backward pass (Phase 2 — for now always fall back to PyTorch)
# ---------------------------------------------------------------------------

def backward_halide(
    grad_output: torch.Tensor,
    means: torch.Tensor,
    covariances: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    height: int,
    width: int,
    intermediates: list,
    needs_input_grad: tuple,
):
    """
    Backward pass. Phase 1: always falls back to PyTorch.
    Phase 2: will call _halide_lib.gaussian_splat_backward.
    """
    from .backends.python import backward_pytorch
    return backward_pytorch(
        grad_output, means, covariances, colors, opacities,
        height, width, intermediates, needs_input_grad,
    )


# ---------------------------------------------------------------------------
# BackendOps registration
# ---------------------------------------------------------------------------

from .backends.common import BackendOps

HALIDE_BACKEND = BackendOps(
    name="halide",
    forward=forward_halide,
    backward=backward_halide,
    is_compiled=True,
)


# ---------------------------------------------------------------------------
# Smoke test (run with: python -m tinysplat.halide_backend)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    print("Smoke test: Halide backend")
    print(f"  Library path: {_HALIDE_LIB_PATH}")
    print(f"  Library loaded: {_halide_lib is not None}")

    N, H, W, C = 8, 64, 64, 3
    means       = torch.randn(N, 2) * 10 + torch.tensor([W/2, H/2])
    covariances = torch.eye(2).unsqueeze(0).repeat(N, 1, 1) * 10.0
    colors      = torch.rand(N, C)
    opacities   = torch.rand(N) * 0.5 + 0.1

    out, _ = forward_halide(means, covariances, colors, opacities, H, W)
    print(f"  Output shape: {out.shape}")
    print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
    print("  OK")
