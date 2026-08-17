"""ctypes bridge to the C++/Metal 3DGS rasterizer (FastGS-compatible).

Build: metal/build_python_dylib.sh
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional, Tuple

import torch


def _find_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / "metal" / "build").is_dir() or (
            candidate / "metal" / "BUILD.bazel"
        ).is_file():
            return candidate
    if here.name == "tinysplat" and here.parent.name == "legacy":
        return here.parents[1]
    return here.parents[1] if here.name == "tinysplat" else here


_REPO = _find_repo_root()
_DEFAULT_LIB = _REPO / "metal" / "build" / "libtinysplat_metal_py.dylib"

_lib = None


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    path = Path(os.environ.get("TINYSPLAT_METAL_LIB", str(_DEFAULT_LIB)))
    if not path.is_file():
        return None
    lib = ctypes.CDLL(str(path))
    lib.tinysplat_metal_available.restype = ctypes.c_int
    lib.tinysplat_metal_forward.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
    ]
    lib.tinysplat_metal_forward.restype = ctypes.c_int
    lib.tinysplat_metal_forward_qs.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
    ]
    lib.tinysplat_metal_forward_qs.restype = ctypes.c_int
    lib.tinysplat_metal_count_hits.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
    ]
    lib.tinysplat_metal_count_hits.restype = ctypes.c_int
    lib.tinysplat_metal_projected_backward.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    lib.tinysplat_metal_projected_backward.restype = ctypes.c_int
    lib.tinysplat_metal_session_backward.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.tinysplat_metal_session_backward.restype = ctypes.c_int
    lib.tinysplat_metal_session_backward_qs.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.tinysplat_metal_session_backward_qs.restype = ctypes.c_int
    lib.tinysplat_metal_forward_qs_sh.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.tinysplat_metal_forward_qs_sh.restype = ctypes.c_int
    lib.tinysplat_metal_session_backward_qs_sh.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.tinysplat_metal_session_backward_qs_sh.restype = ctypes.c_int
    if hasattr(lib, "tinysplat_metal_last_grad_means2d"):
        lib.tinysplat_metal_last_grad_means2d_n.restype = ctypes.c_int
        lib.tinysplat_metal_last_grad_means2d.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        lib.tinysplat_metal_last_grad_means2d.restype = ctypes.c_int
    if hasattr(lib, "tinysplat_metal_session_count_hits"):
        lib.tinysplat_metal_session_count_hits.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.tinysplat_metal_session_count_hits.restype = ctypes.c_int
    if hasattr(lib, "tinysplat_metal_last_grad_means2d_abs"):
        lib.tinysplat_metal_last_grad_means2d_abs_n.restype = ctypes.c_int
        lib.tinysplat_metal_last_grad_means2d_abs.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        lib.tinysplat_metal_last_grad_means2d_abs.restype = ctypes.c_int
    if hasattr(lib, "tinysplat_metal_last_radii2d"):
        lib.tinysplat_metal_last_radii2d_n.restype = ctypes.c_int
        lib.tinysplat_metal_last_radii2d.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        lib.tinysplat_metal_last_radii2d.restype = ctypes.c_int
    _lib = lib
    return _lib


def metal_available() -> bool:
    lib = _load_lib()
    if lib is None:
        return False
    return bool(lib.tinysplat_metal_available())


def _host_f32(t: torch.Tensor) -> torch.Tensor:
    """Contiguous float32 CPU tensor; keeps storage alive for ctypes."""
    x = t.detach()
    if x.dtype != torch.float32:
        x = x.to(dtype=torch.float32)
    x = x.contiguous()
    if x.device.type != "cpu":
        x = x.cpu().contiguous()
    return x


def _f32_ptr(t: torch.Tensor):
    return ctypes.cast(t.data_ptr(), ctypes.POINTER(ctypes.c_float))


def _i32_ptr(t: torch.Tensor):
    return ctypes.cast(t.data_ptr(), ctypes.POINTER(ctypes.c_int))


def _u8_ptr(t: torch.Tensor):
    return ctypes.cast(t.data_ptr(), ctypes.POINTER(ctypes.c_uint8))


def forward_3d(
    means: torch.Tensor,
    covs: torch.Tensor,
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
) -> Optional[torch.Tensor]:
    """Rasterize with Metal tiled forward. Returns HxWxC on CPU, or None."""
    lib = _load_lib()
    if lib is None or not lib.tinysplat_metal_available():
        return None

    means_h = _host_f32(means).view(-1)
    covs_h = _host_f32(covs).view(-1)
    colors_h = _host_f32(colors).view(-1)
    opa_h = _host_f32(opacities).view(-1)
    n = int(means.shape[0])
    c = int(colors.shape[1])
    intr = _host_f32(intrinsics).view(-1).contiguous()
    c2w = _host_f32(camera_to_world).view(-1).contiguous()
    out = torch.zeros(height * width * c, dtype=torch.float32)

    ok = lib.tinysplat_metal_forward(
        _f32_ptr(means_h),
        _f32_ptr(covs_h),
        _f32_ptr(colors_h),
        _f32_ptr(opa_h),
        n,
        c,
        _f32_ptr(intr),
        _f32_ptr(c2w),
        height,
        width,
        _f32_ptr(out),
        ctypes.c_float(near_plane),
        ctypes.c_float(min_covariance),
        ctypes.c_float(sigma_radius),
        ctypes.c_float(compact_box_beta),
        ctypes.c_int(1 if use_compact_box else 0),
    )
    if not ok:
        return None
    return out.view(height, width, c)


def forward_3d_qs(
    means: torch.Tensor,
    log_scales: torch.Tensor,
    rotations: torch.Tensor,
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
) -> Optional[torch.Tensor]:
    """Rasterize with Metal, building world covariance from quat+log-scale on GPU."""
    lib = _load_lib()
    if lib is None or not lib.tinysplat_metal_available():
        return None

    means_h = _host_f32(means).view(-1)
    scales_h = _host_f32(log_scales).view(-1)
    quats_h = _host_f32(rotations).view(-1)
    colors_h = _host_f32(colors).view(-1)
    opa_h = _host_f32(opacities).view(-1)
    n = int(means.shape[0])
    c = int(colors.shape[1])
    intr = _host_f32(intrinsics).view(-1).contiguous()
    c2w = _host_f32(camera_to_world).view(-1).contiguous()
    out = torch.zeros(height * width * c, dtype=torch.float32)

    ok = lib.tinysplat_metal_forward_qs(
        _f32_ptr(means_h),
        _f32_ptr(scales_h),
        _f32_ptr(quats_h),
        _f32_ptr(colors_h),
        _f32_ptr(opa_h),
        n,
        c,
        _f32_ptr(intr),
        _f32_ptr(c2w),
        height,
        width,
        _f32_ptr(out),
        ctypes.c_float(near_plane),
        ctypes.c_float(min_covariance),
        ctypes.c_float(sigma_radius),
        ctypes.c_float(compact_box_beta),
        ctypes.c_int(1 if use_compact_box else 0),
    )
    if not ok:
        return None
    return out.view(height, width, c)


def projected_backward(
    grad_output: torch.Tensor,
    proj_means: torch.Tensor,
    proj_covs: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    height: int,
    width: int,
    min_covariance: float = 1e-4,
    sigma_radius: float = 4.0,
    compact_box_beta: float = 3.0,
    use_compact_box: bool = True,
    depths: Optional[torch.Tensor] = None,
    force_cpu: bool = False,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Per-splat projected backward. Returns grads for means, covs, colors, opacities."""
    lib = _load_lib()
    if lib is None:
        return None
    n = int(proj_means.shape[0])
    c = int(colors.shape[1])
    go = _host_f32(grad_output).view(-1)
    means_h = _host_f32(proj_means).view(-1)
    covs_h = _host_f32(proj_covs).view(-1)
    colors_h = _host_f32(colors).view(-1)
    opa_h = _host_f32(opacities).view(-1)
    depths_h = _host_f32(depths).view(-1) if depths is not None else None
    gm = torch.zeros(n * 2, dtype=torch.float32)
    gc = torch.zeros(n * 4, dtype=torch.float32)
    gcol = torch.zeros(n * c, dtype=torch.float32)
    gopa = torch.zeros(n, dtype=torch.float32)
    ok = lib.tinysplat_metal_projected_backward(
        _f32_ptr(go),
        _f32_ptr(means_h),
        _f32_ptr(covs_h),
        _f32_ptr(colors_h),
        _f32_ptr(opa_h),
        n,
        c,
        height,
        width,
        _f32_ptr(gm),
        _f32_ptr(gc),
        _f32_ptr(gcol),
        _f32_ptr(gopa),
        ctypes.c_float(min_covariance),
        ctypes.c_float(sigma_radius),
        ctypes.c_float(compact_box_beta),
        ctypes.c_int(1 if use_compact_box else 0),
        _f32_ptr(depths_h) if depths_h is not None else None,
        ctypes.c_int(1 if force_cpu else 0),
    )
    if not ok:
        return None
    return (
        gm.view(n, 2),
        gc.view(n, 2, 2),
        gcol.view(n, c),
        gopa,
    )


def session_backward(
    grad_output: torch.Tensor,
    num_gaussians: int,
    num_channels: int,
    height: int,
    width: int,
    min_covariance: float = 1e-4,
    sigma_radius: float = 4.0,
    compact_box_beta: float = 3.0,
    use_compact_box: bool = True,
    force_cpu: bool = False,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Reuse GPU tiles from the last forward_3d. Returns 3D mean/cov + color/opacity grads."""
    lib = _load_lib()
    if lib is None:
        return None
    n = int(num_gaussians)
    c = int(num_channels)
    go = _host_f32(grad_output).view(-1)
    gm = torch.zeros(n * 3, dtype=torch.float32)
    gc = torch.zeros(n * 9, dtype=torch.float32)
    gcol = torch.zeros(n * c, dtype=torch.float32)
    gopa = torch.zeros(n, dtype=torch.float32)
    ok = lib.tinysplat_metal_session_backward(
        _f32_ptr(go),
        n,
        c,
        height,
        width,
        _f32_ptr(gm),
        _f32_ptr(gc),
        _f32_ptr(gcol),
        _f32_ptr(gopa),
        ctypes.c_float(min_covariance),
        ctypes.c_float(sigma_radius),
        ctypes.c_float(compact_box_beta),
        ctypes.c_int(1 if use_compact_box else 0),
        ctypes.c_int(1 if force_cpu else 0),
    )
    if not ok:
        return None
    return gm.view(n, 3), gc.view(n, 3, 3), gcol.view(n, c), gopa


def session_backward_qs(
    grad_output: torch.Tensor,
    num_gaussians: int,
    num_channels: int,
    height: int,
    width: int,
    min_covariance: float = 1e-4,
    sigma_radius: float = 4.0,
    compact_box_beta: float = 3.0,
    use_compact_box: bool = True,
    force_cpu: bool = False,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Reuse GPU tiles from the last forward_3d_qs. Returns mean/log_scale/quat + color/opacity grads."""
    lib = _load_lib()
    if lib is None:
        return None
    n = int(num_gaussians)
    c = int(num_channels)
    go = _host_f32(grad_output).view(-1)
    gm = torch.zeros(n * 3, dtype=torch.float32)
    gls = torch.zeros(n * 3, dtype=torch.float32)
    gq = torch.zeros(n * 4, dtype=torch.float32)
    gcol = torch.zeros(n * c, dtype=torch.float32)
    gopa = torch.zeros(n, dtype=torch.float32)
    ok = lib.tinysplat_metal_session_backward_qs(
        _f32_ptr(go),
        n,
        c,
        height,
        width,
        _f32_ptr(gm),
        _f32_ptr(gls),
        _f32_ptr(gq),
        _f32_ptr(gcol),
        _f32_ptr(gopa),
        ctypes.c_float(min_covariance),
        ctypes.c_float(sigma_radius),
        ctypes.c_float(compact_box_beta),
        ctypes.c_int(1 if use_compact_box else 0),
        ctypes.c_int(1 if force_cpu else 0),
    )
    if not ok:
        return None
    return gm.view(n, 3), gls.view(n, 3), gq.view(n, 4), gcol.view(n, c), gopa


def forward_3d_qs_sh(
    means: torch.Tensor,
    log_scales: torch.Tensor,
    rotations: torch.Tensor,
    sh_coeffs: torch.Tensor,
    opacities: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_to_world: torch.Tensor,
    height: int,
    width: int,
    sh_degree: int,
    near_plane: float = 1e-4,
    min_covariance: float = 1e-4,
    sigma_radius: float = 4.0,
    compact_box_beta: float = 3.0,
    use_compact_box: bool = True,
) -> Optional[torch.Tensor]:
    """Rasterize with Metal, evaluating SH RGB on GPU."""
    lib = _load_lib()
    if lib is None or not lib.tinysplat_metal_available():
        return None

    means_h = _host_f32(means).view(-1)
    scales_h = _host_f32(log_scales).view(-1)
    quats_h = _host_f32(rotations).view(-1)
    sh_h = _host_f32(sh_coeffs).reshape(-1)
    opa_h = _host_f32(opacities).view(-1)
    n = int(means.shape[0])
    c = 3
    intr = _host_f32(intrinsics).view(-1).contiguous()
    c2w = _host_f32(camera_to_world).view(-1).contiguous()
    out = torch.zeros(height * width * c, dtype=torch.float32)

    ok = lib.tinysplat_metal_forward_qs_sh(
        _f32_ptr(means_h),
        _f32_ptr(scales_h),
        _f32_ptr(quats_h),
        _f32_ptr(sh_h),
        _f32_ptr(opa_h),
        n,
        c,
        _f32_ptr(intr),
        _f32_ptr(c2w),
        height,
        width,
        _f32_ptr(out),
        ctypes.c_float(near_plane),
        ctypes.c_float(min_covariance),
        ctypes.c_float(sigma_radius),
        ctypes.c_float(compact_box_beta),
        ctypes.c_int(1 if use_compact_box else 0),
        ctypes.c_int(int(sh_degree)),
    )
    if not ok:
        return None
    return out.view(height, width, c)


def session_backward_qs_sh(
    grad_output: torch.Tensor,
    num_gaussians: int,
    height: int,
    width: int,
    min_covariance: float = 1e-4,
    sigma_radius: float = 4.0,
    compact_box_beta: float = 3.0,
    use_compact_box: bool = True,
    force_cpu: bool = False,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Reuse GPU tiles from the last forward_3d_qs_sh. Returns mean/log_scale/quat/sh/opacity grads."""
    lib = _load_lib()
    if lib is None:
        return None
    n = int(num_gaussians)
    c = 3
    go = _host_f32(grad_output).view(-1)
    gm = torch.zeros(n * 3, dtype=torch.float32)
    gls = torch.zeros(n * 3, dtype=torch.float32)
    gq = torch.zeros(n * 4, dtype=torch.float32)
    gsh = torch.zeros(n * 48, dtype=torch.float32)
    gopa = torch.zeros(n, dtype=torch.float32)
    ok = lib.tinysplat_metal_session_backward_qs_sh(
        _f32_ptr(go),
        n,
        c,
        height,
        width,
        _f32_ptr(gm),
        _f32_ptr(gls),
        _f32_ptr(gq),
        _f32_ptr(gsh),
        _f32_ptr(gopa),
        ctypes.c_float(min_covariance),
        ctypes.c_float(sigma_radius),
        ctypes.c_float(compact_box_beta),
        ctypes.c_int(1 if use_compact_box else 0),
        ctypes.c_int(1 if force_cpu else 0),
    )
    if not ok:
        return None
    return gm.view(n, 3), gls.view(n, 3), gq.view(n, 4), gsh.view(n, 16, 3), gopa


def count_footprint_hits(
    proj_means: torch.Tensor,
    proj_covs: torch.Tensor,
    opacities: torch.Tensor,
    height: int,
    width: int,
    error_mask: torch.Tensor,
    sigma_radius: float = 4.0,
    compact_box_beta: float = 3.0,
    use_compact_box: bool = True,
    min_covariance: float = 1e-4,
) -> Optional[torch.Tensor]:
    """Return int64 counts[N] of high-error pixels in each Gaussian footprint."""
    lib = _load_lib()
    if lib is None:
        return None
    n = int(proj_means.shape[0])
    means_h = _host_f32(proj_means).view(-1)
    covs_h = _host_f32(proj_covs).view(-1)
    opa_h = _host_f32(opacities).view(-1)
    mask = error_mask.detach()
    if mask.dtype != torch.uint8:
        mask = mask.to(dtype=torch.uint8)
    mask = mask.contiguous()
    if mask.device.type != "cpu":
        mask = mask.cpu().contiguous()
    mask = mask.view(-1)
    counts = torch.zeros(n, dtype=torch.int32)
    ok = lib.tinysplat_metal_count_hits(
        _f32_ptr(means_h),
        _f32_ptr(covs_h),
        _f32_ptr(opa_h),
        n,
        height,
        width,
        _u8_ptr(mask),
        _i32_ptr(counts),
        ctypes.c_float(min_covariance),
        ctypes.c_float(sigma_radius),
        ctypes.c_float(compact_box_beta),
        ctypes.c_int(1 if use_compact_box else 0),
    )
    if not ok:
        return None
    return counts.to(dtype=torch.int64)


def count_session_hits(
    error_mask: torch.Tensor,
    num_gaussians: int,
    height: int,
    width: int,
) -> Optional[torch.Tensor]:
    """FastGS compositor counts from the last Metal forward session, or None."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "tinysplat_metal_session_count_hits"):
        return None
    mask = error_mask.detach()
    if mask.dtype != torch.uint8:
        mask = mask.to(dtype=torch.uint8)
    mask = mask.contiguous()
    if mask.device.type != "cpu":
        mask = mask.cpu().contiguous()
    mask = mask.view(-1)
    counts = torch.zeros(int(num_gaussians), dtype=torch.int32)
    ok = lib.tinysplat_metal_session_count_hits(
        _u8_ptr(mask),
        _i32_ptr(counts),
        int(num_gaussians),
        int(height),
        int(width),
    )
    if not ok:
        return None
    return counts.to(dtype=torch.int64)


def last_grad_means2d() -> Optional[torch.Tensor]:
    """Per-splat 2D mean grads (N,2) from the last Metal session backward, or None."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "tinysplat_metal_last_grad_means2d"):
        return None
    n = int(lib.tinysplat_metal_last_grad_means2d_n())
    if n <= 0:
        return None
    out = torch.zeros(n * 2, dtype=torch.float32)
    got = int(lib.tinysplat_metal_last_grad_means2d(_f32_ptr(out), n))
    if got != n:
        return None
    return out.view(n, 2)


def last_grad_means2d_abs() -> Optional[torch.Tensor]:
    """AbsGS per-splat |dL/dmean2d| sums (N,2) from the last Metal backward, or None."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "tinysplat_metal_last_grad_means2d_abs"):
        return None
    n = int(lib.tinysplat_metal_last_grad_means2d_abs_n())
    if n <= 0:
        return None
    out = torch.zeros(n * 2, dtype=torch.float32)
    got = int(lib.tinysplat_metal_last_grad_means2d_abs(_f32_ptr(out), n))
    if got != n:
        return None
    return out.view(n, 2)


def last_radii2d() -> Optional[torch.Tensor]:
    """Compact-box radii in pixels (N,) from the last Metal forward, or None."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "tinysplat_metal_last_radii2d"):
        return None
    n = int(lib.tinysplat_metal_last_radii2d_n())
    if n <= 0:
        return None
    out = torch.zeros(n, dtype=torch.float32)
    got = int(lib.tinysplat_metal_last_radii2d(_f32_ptr(out), n))
    if got != n:
        return None
    return out


class _MetalSplat3DFn(torch.autograd.Function):
    """Metal tiled forward; backward reuses the GPU tile session and 3D VJP."""

    @staticmethod
    def forward(
        ctx,
        means,
        covariances,
        colors,
        opacities,
        intrinsics,
        camera_to_world,
        height,
        width,
        near_plane,
        min_covariance,
        sigma_radius,
        compact_box_beta,
        use_compact_box,
    ):
        device = means.device
        dtype = means.dtype
        img = forward_3d(
            means.detach(),
            covariances.detach(),
            colors.detach(),
            opacities.detach(),
            intrinsics.detach(),
            camera_to_world.detach(),
            int(height),
            int(width),
            near_plane=float(near_plane),
            min_covariance=float(min_covariance),
            sigma_radius=float(sigma_radius),
            compact_box_beta=float(compact_box_beta),
            use_compact_box=bool(use_compact_box),
        )
        if img is None:
            raise RuntimeError("Metal forward failed")

        ctx.num_gaussians = int(means.shape[0])
        ctx.num_channels = int(colors.shape[1])
        ctx.height = int(height)
        ctx.width = int(width)
        ctx.min_covariance = float(min_covariance)
        ctx.sigma_radius = float(sigma_radius)
        ctx.compact_box_beta = float(compact_box_beta)
        ctx.use_compact_box = bool(use_compact_box)
        ctx.device = device
        ctx.dtype = dtype
        ctx.mean_shape = tuple(means.shape)
        ctx.cov_shape = tuple(covariances.shape)
        return img.to(device=device, dtype=dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grads = session_backward(
            grad_output.detach(),
            ctx.num_gaussians,
            ctx.num_channels,
            ctx.height,
            ctx.width,
            min_covariance=ctx.min_covariance,
            sigma_radius=ctx.sigma_radius,
            compact_box_beta=ctx.compact_box_beta,
            use_compact_box=ctx.use_compact_box,
        )
        if grads is None:
            raise RuntimeError("Metal session backward failed")
        g_mean, g_cov, g_col, g_opa = grads
        device = ctx.device
        dtype = ctx.dtype
        return (
            g_mean.to(device=device, dtype=dtype).view(ctx.mean_shape),
            g_cov.to(device=device, dtype=dtype).view(ctx.cov_shape),
            g_col.to(device=device, dtype=dtype),
            g_opa.to(device=device, dtype=dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _MetalSplat3DQsFn(torch.autograd.Function):
    """Metal tiled forward with fused quat+log-scale covariance; 3D VJP to mean/scale/quat."""

    @staticmethod
    def forward(
        ctx,
        means,
        log_scales,
        rotations,
        colors,
        opacities,
        intrinsics,
        camera_to_world,
        height,
        width,
        near_plane,
        min_covariance,
        sigma_radius,
        compact_box_beta,
        use_compact_box,
    ):
        device = means.device
        dtype = means.dtype
        img = forward_3d_qs(
            means.detach(),
            log_scales.detach(),
            rotations.detach(),
            colors.detach(),
            opacities.detach(),
            intrinsics.detach(),
            camera_to_world.detach(),
            int(height),
            int(width),
            near_plane=float(near_plane),
            min_covariance=float(min_covariance),
            sigma_radius=float(sigma_radius),
            compact_box_beta=float(compact_box_beta),
            use_compact_box=bool(use_compact_box),
        )
        if img is None:
            raise RuntimeError("Metal qs forward failed")

        ctx.num_gaussians = int(means.shape[0])
        ctx.num_channels = int(colors.shape[1])
        ctx.height = int(height)
        ctx.width = int(width)
        ctx.min_covariance = float(min_covariance)
        ctx.sigma_radius = float(sigma_radius)
        ctx.compact_box_beta = float(compact_box_beta)
        ctx.use_compact_box = bool(use_compact_box)
        ctx.device = device
        ctx.dtype = dtype
        ctx.mean_shape = tuple(means.shape)
        ctx.scale_shape = tuple(log_scales.shape)
        ctx.quat_shape = tuple(rotations.shape)
        return img.to(device=device, dtype=dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grads = session_backward_qs(
            grad_output.detach(),
            ctx.num_gaussians,
            ctx.num_channels,
            ctx.height,
            ctx.width,
            min_covariance=ctx.min_covariance,
            sigma_radius=ctx.sigma_radius,
            compact_box_beta=ctx.compact_box_beta,
            use_compact_box=ctx.use_compact_box,
        )
        if grads is None:
            raise RuntimeError("Metal qs session backward failed")
        g_mean, g_ls, g_q, g_col, g_opa = grads
        device = ctx.device
        dtype = ctx.dtype
        return (
            g_mean.to(device=device, dtype=dtype).view(ctx.mean_shape),
            g_ls.to(device=device, dtype=dtype).view(ctx.scale_shape),
            g_q.to(device=device, dtype=dtype).view(ctx.quat_shape),
            g_col.to(device=device, dtype=dtype),
            g_opa.to(device=device, dtype=dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _MetalSplat3DQsShFn(torch.autograd.Function):
    """Metal tiled forward with fused quat+log-scale covariance and SH color."""

    @staticmethod
    def forward(
        ctx,
        means,
        log_scales,
        rotations,
        sh_coeffs,
        opacities,
        intrinsics,
        camera_to_world,
        height,
        width,
        sh_degree,
        near_plane,
        min_covariance,
        sigma_radius,
        compact_box_beta,
        use_compact_box,
    ):
        device = means.device
        dtype = means.dtype
        img = forward_3d_qs_sh(
            means.detach(),
            log_scales.detach(),
            rotations.detach(),
            sh_coeffs.detach(),
            opacities.detach(),
            intrinsics.detach(),
            camera_to_world.detach(),
            int(height),
            int(width),
            int(sh_degree),
            near_plane=float(near_plane),
            min_covariance=float(min_covariance),
            sigma_radius=float(sigma_radius),
            compact_box_beta=float(compact_box_beta),
            use_compact_box=bool(use_compact_box),
        )
        if img is None:
            raise RuntimeError("Metal qs+SH forward failed")

        ctx.num_gaussians = int(means.shape[0])
        ctx.height = int(height)
        ctx.width = int(width)
        ctx.min_covariance = float(min_covariance)
        ctx.sigma_radius = float(sigma_radius)
        ctx.compact_box_beta = float(compact_box_beta)
        ctx.use_compact_box = bool(use_compact_box)
        ctx.device = device
        ctx.dtype = dtype
        ctx.mean_shape = tuple(means.shape)
        ctx.scale_shape = tuple(log_scales.shape)
        ctx.quat_shape = tuple(rotations.shape)
        ctx.sh_shape = tuple(sh_coeffs.shape)
        return img.to(device=device, dtype=dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grads = session_backward_qs_sh(
            grad_output.detach(),
            ctx.num_gaussians,
            ctx.height,
            ctx.width,
            min_covariance=ctx.min_covariance,
            sigma_radius=ctx.sigma_radius,
            compact_box_beta=ctx.compact_box_beta,
            use_compact_box=ctx.use_compact_box,
        )
        if grads is None:
            raise RuntimeError("Metal qs+SH session backward failed")
        g_mean, g_ls, g_q, g_sh, g_opa = grads
        device = ctx.device
        dtype = ctx.dtype
        return (
            g_mean.to(device=device, dtype=dtype).view(ctx.mean_shape),
            g_ls.to(device=device, dtype=dtype).view(ctx.scale_shape),
            g_q.to(device=device, dtype=dtype).view(ctx.quat_shape),
            g_sh.to(device=device, dtype=dtype).view(ctx.sh_shape),
            g_opa.to(device=device, dtype=dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def render_metal_3d(
    means: torch.Tensor,
    covariances: Optional[torch.Tensor] = None,
    colors: torch.Tensor = None,
    opacities: torch.Tensor = None,
    intrinsics: torch.Tensor = None,
    camera_to_world: torch.Tensor = None,
    height: int = 0,
    width: int = 0,
    near_plane: float = 1e-4,
    min_covariance: float = 1e-4,
    sigma_radius: float = 4.0,
    compact_box_beta: float = 3.0,
    use_compact_box: bool = True,
    log_scales: Optional[torch.Tensor] = None,
    rotations: Optional[torch.Tensor] = None,
    sh_coeffs: Optional[torch.Tensor] = None,
    sh_degree: int = 0,
    **_kwargs,
) -> torch.Tensor:
    if not metal_available():
        raise RuntimeError("Metal rasterizer dylib is not available")
    opts = dict(
        near_plane=near_plane,
        min_covariance=min_covariance,
        sigma_radius=sigma_radius,
        compact_box_beta=compact_box_beta,
        use_compact_box=use_compact_box,
    )
    if log_scales is not None and rotations is not None and sh_coeffs is not None:
        return _MetalSplat3DQsShFn.apply(
            means,
            log_scales,
            rotations,
            sh_coeffs,
            opacities,
            intrinsics,
            camera_to_world,
            height,
            width,
            int(sh_degree),
            opts["near_plane"],
            opts["min_covariance"],
            opts["sigma_radius"],
            opts["compact_box_beta"],
            opts["use_compact_box"],
        )
    if log_scales is not None and rotations is not None:
        return _MetalSplat3DQsFn.apply(
            means,
            log_scales,
            rotations,
            colors,
            opacities,
            intrinsics,
            camera_to_world,
            height,
            width,
            opts["near_plane"],
            opts["min_covariance"],
            opts["sigma_radius"],
            opts["compact_box_beta"],
            opts["use_compact_box"],
        )
    if covariances is None:
        raise ValueError("render_metal_3d requires covariances or log_scales+rotations")
    return _MetalSplat3DFn.apply(
        means,
        covariances,
        colors,
        opacities,
        intrinsics,
        camera_to_world,
        height,
        width,
        near_plane,
        min_covariance,
        sigma_radius,
        compact_box_beta,
        use_compact_box,
    )
