"""
halide_backend.py

Halide backend for TinySplat — bridges Halide JIT-compiled Gaussian splatting
pipelines to the torch.autograd.Function interface used by GaussianSplat2DFunction.

Usage:
    export HL_PATH=$HOME/halide
    export TINYSPLAT_HALIDE_LIB=/path/to/libtinysplat_halide_pipeline.so
    export TINYSPLAT_BACKEND=halide
"""

import ctypes
import os
import torch
from typing import Tuple

_HALIDE_LIB_PATH = os.environ.get("TINYSPLAT_HALIDE_LIB", "")
_halide_lib = None
_halide_load_attempted = False


def _try_load_halide_lib():
    global _halide_lib, _halide_load_attempted
    if _halide_lib is not None:
        return True
    if _halide_load_attempted:
        return False
    _halide_load_attempted = True

    if not _HALIDE_LIB_PATH or not os.path.exists(_HALIDE_LIB_PATH):
        return False

    try:
        _halide_lib = ctypes.CDLL(_HALIDE_LIB_PATH)

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
    if _halide_lib is not None:
        return
    if not _try_load_halide_lib():
        raise ImportError(
            "Halide backend not available. Set TINYSPLAT_HALIDE_LIB "
            "to the path of libtinysplat_halide_pipeline.so and ensure "
            "the Halide runtime loads (e.g. matching libstdc++)."
        )


def _require_cpu_halide(means: torch.Tensor):
    if means.device.type != "cpu":
        raise RuntimeError(
            "TINYSPLAT_BACKEND=halide requires CPU tensors; "
            f"got device {means.device!r}. Use CPU tensors or pick cuda/mps/cpu backend."
        )


def _to_contiguous_float32_cpu(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is contiguous float32 on CPU."""
    t = t.detach()
    if t.device.type != "cpu":
        t = t.cpu()
    if t.dtype != torch.float32:
        t = t.to(torch.float32)
    return t.contiguous()


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def forward_halide(
    means: torch.Tensor,
    covariances: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    height: int,
    width: int,
) -> Tuple[torch.Tensor, list]:
    _require_cpu_halide(means)
    _ensure_halide()

    N, C = colors.shape
    device = means.device

    # Halide runtime buffers in pipeline.cpp are created as:
    # means: [N,2], cov: [N,2,2], colors: [N,C], output: [H,W,C]
    # with compact dim0-major storage. Convert PyTorch row-major tensors into
    # matching compact memory layouts before passing raw pointers.
    means_c     = _to_contiguous_float32_cpu(means).transpose(0, 1).contiguous()      # [2, N]
    cov_c       = _to_contiguous_float32_cpu(covariances).permute(1, 2, 0).contiguous()  # [2, 2, N]
    colors_c    = _to_contiguous_float32_cpu(colors).transpose(0, 1).contiguous()     # [C, N]
    opacities_c = _to_contiguous_float32_cpu(opacities)
    output_planar = torch.zeros(C, width, height, dtype=torch.float32)                # [C, W, H]

    rc = _halide_lib.gaussian_splat_forward(
        means_c.data_ptr(),
        cov_c.data_ptr(),
        colors_c.data_ptr(),
        opacities_c.data_ptr(),
        N, height, width, C,
        output_planar.data_ptr(),
    )

    if rc != 0:
        raise RuntimeError(f"gaussian_splat_forward failed (rc={rc})")

    output_t = output_planar.permute(2, 1, 0).contiguous()  # [H, W, C]
    return output_t.to(device), []


# ---------------------------------------------------------------------------
# Backward pass — uses Halide analytical gradients
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
    _require_cpu_halide(means)
    _ensure_halide()

    N, C = colors.shape
    device = means.device

    grad_out_c  = _to_contiguous_float32_cpu(grad_output).permute(2, 1, 0).contiguous()  # [C, W, H]
    means_c     = _to_contiguous_float32_cpu(means).transpose(0, 1).contiguous()          # [2, N]
    cov_c       = _to_contiguous_float32_cpu(covariances).permute(1, 2, 0).contiguous()   # [2, 2, N]
    colors_c    = _to_contiguous_float32_cpu(colors).transpose(0, 1).contiguous()         # [C, N]
    opacities_c = _to_contiguous_float32_cpu(opacities)

    need_means = len(needs_input_grad) > 0 and needs_input_grad[0]
    need_cov = len(needs_input_grad) > 1 and needs_input_grad[1]
    need_colors = len(needs_input_grad) > 2 and needs_input_grad[2]
    need_opacities = len(needs_input_grad) > 3 and needs_input_grad[3]

    grad_means_planar = torch.zeros(2, N, dtype=torch.float32) if need_means else None
    grad_cov_planar = torch.zeros(2, 2, N, dtype=torch.float32) if need_cov else None
    grad_colors_planar = torch.zeros(C, N, dtype=torch.float32) if need_colors else None
    grad_opacities_t = torch.zeros(N, dtype=torch.float32) if need_opacities else None

    rc = _halide_lib.gaussian_splat_backward(
        grad_out_c.data_ptr(),
        means_c.data_ptr(),
        cov_c.data_ptr(),
        colors_c.data_ptr(),
        opacities_c.data_ptr(),
        N, height, width, C,
        grad_means_planar.data_ptr() if grad_means_planar is not None else None,
        grad_cov_planar.data_ptr() if grad_cov_planar is not None else None,
        grad_colors_planar.data_ptr() if grad_colors_planar is not None else None,
        grad_opacities_t.data_ptr() if grad_opacities_t is not None else None,
    )

    if rc != 0:
        raise RuntimeError(f"gaussian_splat_backward failed (rc={rc})")

    grad_means_t = grad_means_planar.transpose(0, 1).contiguous() if need_means else None
    grad_cov_t = grad_cov_planar.permute(2, 0, 1).contiguous() if need_cov else None
    grad_colors_t = grad_colors_planar.transpose(0, 1).contiguous() if need_colors else None

    grad_means = grad_means_t.to(device) if need_means else None
    grad_cov = grad_cov_t.to(device) if need_cov else None
    grad_colors = grad_colors_t.to(device) if need_colors else None
    grad_opacities = grad_opacities_t.to(device) if need_opacities else None

    return grad_means, grad_cov, grad_colors, grad_opacities


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


if __name__ == "__main__":
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
