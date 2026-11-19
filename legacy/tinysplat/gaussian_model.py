"""Trainable Gaussian parameters, sparse-point initialization, and covariance kernels."""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .gaussian_splat_3d import gaussian_splat_3d

try:
    from .metal_backend import metal_available
    from .sh import colors_from_sh, init_sh_from_rgb, sh_degree_from_step
except ImportError:
    _HAS_SH = False

    def metal_available():
        return False
else:
    _HAS_SH = True


def build_sparse_points_gaussians_3d(
    points3d: List[Dict[str, object]],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    if not points3d:
        raise ValueError("Dataset JSON does not contain points3d for sparse initialization.")

    xyz = torch.tensor([p["xyz"] for p in points3d], dtype=dtype, device=device)
    rgb = torch.tensor([p.get("rgb", [128, 128, 128]) for p in points3d], dtype=dtype, device=device) / 255.0
    rgb = torch.clamp(rgb, 0.0, 1.0)

    n = xyz.shape[0]
    if n < 2:
        scales = torch.full((n, 3), 0.01, dtype=dtype, device=device)
    else:
        # 3DGS: log(sqrt(mean squared distance to 3 nearest neighbors)).
        nn = None
        try:
            from scipy.spatial import cKDTree

            pts = xyz.detach().cpu().numpy()
            k = int(min(4, n))
            dist, _ = cKDTree(pts).query(pts, k=k)
            dist = np.asarray(dist, dtype=np.float64)
            if dist.ndim == 1:
                nn_np = dist
            else:
                nn_np = dist[:, 1:].mean(axis=1)
            nn = torch.tensor(nn_np, dtype=dtype, device=device)
        except Exception:
            ref_n = min(4096, n)
            ref_idx = torch.randperm(n, device=device)[:ref_n]
            refs = xyz[ref_idx]
            nn = torch.full((n,), 1e9, dtype=dtype, device=device)
            chunk = 8192
            for st in range(0, n, chunk):
                ed = min(st + chunk, n)
                nn[st:ed] = torch.cdist(xyz[st:ed], refs).min(dim=1).values
        med = xyz.median(dim=0).values
        radius = (xyz - med).norm(dim=-1)
        max_init = max(float(torch.quantile(radius, 0.5).item()) * 0.02, 1e-3)
        base = torch.clamp(nn, min=1e-7, max=max_init)
        scales = base.unsqueeze(1).repeat(1, 3)

    log_scales = torch.log(scales)
    rotations = torch.zeros(n, 4, dtype=dtype, device=device)
    rotations[:, 0] = 1.0
    opa = torch.full((n,), torch.logit(torch.tensor(0.1, dtype=dtype, device=device)).item(), dtype=dtype, device=device)

    out = {
        "means": xyz,
        "log_scales": log_scales,
        "rotations": rotations,
        "colors": rgb,
        "opacities": opa,
    }
    if _HAS_SH:
        out["sh_coeffs"] = init_sh_from_rgb(rgb, max_degree=3)
    return out


class GaussianData:
    """Central manager for 3D Gaussian Splatting parameters.

    Holds tensors with requires_grad=True so the optimizer can update in-place
    and all consumers (rasterizer, visualizer) see the same data.
    """

    def __init__(self, params: Dict[str, torch.Tensor], device: str, sh_degree: int = 0):
        self._device = torch.device(device)
        self.max_sh_degree = int(sh_degree)
        self.active_sh_degree = 0 if self.max_sh_degree > 0 else 0
        self.means = params["means"].to(self._device).requires_grad_(True)
        self.log_scales = params["log_scales"].to(self._device).requires_grad_(True)
        self.rotations = params["rotations"].to(self._device).requires_grad_(True)
        self.colors = params["colors"].to(self._device).requires_grad_(True)
        self.opacities = params["opacities"].to(self._device).requires_grad_(True)
        if "sh_coeffs" in params and self.max_sh_degree > 0:
            self.sh_coeffs = params["sh_coeffs"].to(self._device).requires_grad_(True)
        elif self.max_sh_degree > 0 and _HAS_SH:
            self.sh_coeffs = init_sh_from_rgb(self.colors.detach()).requires_grad_(True)
        else:
            self.sh_coeffs = None

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[0]

    @property
    def num_channels(self) -> int:
        return 3

    def parameters(self):
        params = [self.means, self.log_scales, self.rotations, self.opacities]
        if self.sh_coeffs is not None:
            params.append(self.sh_coeffs)
        else:
            params.append(self.colors)
        return params

    def set_sh_degree_for_step(self, step: int):
        if self.max_sh_degree <= 0 or not _HAS_SH:
            return
        self.active_sh_degree = sh_degree_from_step(step, self.max_sh_degree)

    def covariance_matrices(self) -> torch.Tensor:
        # On CUDA this is a single fused kernel. In PyTorch the quat -> 3x3
        # covariance construction plus its backward showed up as aten::bmm at
        # 38% of GPU time -- work Metal never pays because it fuses the same
        # thing into its projection kernels.
        if self.log_scales.is_cuda:
            fused = _fused_quat_scale_to_cov3(self.rotations, self.log_scales)
            if fused is not None:
                return fused
        scales = torch.exp(self.log_scales)
        norm = self.rotations.norm(dim=1, keepdim=True).clamp(min=1e-8)
        q = self.rotations / norm
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        rotation = torch.stack(
            [
                torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=1),
                torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=1),
                torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=1),
            ],
            dim=1,
        )
        scale_matrix = torch.diag_embed(scales * scales)
        covariance = rotation @ scale_matrix @ rotation.transpose(1, 2)
        epsilon = torch.eye(3, device=self._device).unsqueeze(0) * 1e-6
        return covariance + epsilon

    def visible_colors(self, camera_to_world: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.sh_coeffs is not None and camera_to_world is not None and _HAS_SH:
            return colors_from_sh(
                self.means, self.sh_coeffs, camera_to_world, self.active_sh_degree
            )
        return torch.clamp(self.colors, 0.0, 1.0)

    def visible_opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.opacities)

    def replace(self, params: Dict[str, torch.Tensor]):
        self.means = params["means"].to(self._device).requires_grad_(True)
        self.log_scales = params["log_scales"].to(self._device).requires_grad_(True)
        self.rotations = params["rotations"].to(self._device).requires_grad_(True)
        self.colors = params["colors"].to(self._device).requires_grad_(True)
        self.opacities = params["opacities"].to(self._device).requires_grad_(True)
        if "sh_coeffs" in params and params["sh_coeffs"] is not None:
            self.sh_coeffs = params["sh_coeffs"].to(self._device).requires_grad_(True)
        elif self.sh_coeffs is not None and _HAS_SH:
            self.sh_coeffs = init_sh_from_rgb(self.colors.detach()).requires_grad_(True)

    def export_params(self) -> Dict[str, torch.Tensor]:
        out = {
            "means": self.means.detach().cpu(),
            "log_scales": self.log_scales.detach().cpu(),
            "rotations": self.rotations.detach().cpu(),
            "colors": self.colors.detach().cpu(),
            "opacities": self.opacities.detach().cpu(),
        }
        if self.sh_coeffs is not None:
            out["sh_coeffs"] = self.sh_coeffs.detach().cpu()
            out["sh_degree"] = torch.tensor(self.active_sh_degree)
        return out

    def render(
        self,
        intrinsics: torch.Tensor,
        camera_to_world: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        use_metal_qs = self._device.type == "mps" and metal_available()
        use_metal_sh = use_metal_qs and self.sh_coeffs is not None
        kwargs = dict(
            means=self.means,
            opacities=self.visible_opacities(),
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            height=height,
            width=width,
            device=self._device.type,
        )
        if use_metal_sh:
            kwargs["colors"] = torch.zeros(
                self.means.shape[0], 3, device=self._device, dtype=self.means.dtype
            )
            kwargs["sh_coeffs"] = self.sh_coeffs
            kwargs["sh_degree"] = int(self.active_sh_degree)
            kwargs["log_scales"] = self.log_scales
            kwargs["rotations"] = self.rotations
        else:
            kwargs["colors"] = self.visible_colors(camera_to_world)
            if use_metal_qs:
                kwargs["log_scales"] = self.log_scales
                kwargs["rotations"] = self.rotations
            else:
                kwargs["covariances"] = self.covariance_matrices()
                try:
                    import inspect

                    params = inspect.signature(gaussian_splat_3d).parameters
                    if "scales" in params:
                        kwargs["scales"] = torch.exp(self.log_scales)
                    if "quats" in params:
                        kwargs["quats"] = F.normalize(self.rotations, dim=-1)
                except (TypeError, ValueError):
                    pass
        return gaussian_splat_3d(**kwargs)

    def snapshot_for_visualizer(self) -> Dict[str, torch.Tensor]:
        return {
            "means": self.means.detach().cpu(),
            "colors": self.visible_colors().detach().cpu(),
            "opacities": self.visible_opacities().detach().cpu(),
            "covariances": self.covariance_matrices().detach().cpu(),
        }


class _QuatScaleToCov3CUDA(torch.autograd.Function):
    """quat + log-scale -> 3x3 covariance, fused, with a hand-written VJP."""

    @staticmethod
    def forward(ctx, quats, log_scales):
        from tinysplat.cpp import load_cuda_extension

        ext = load_cuda_extension()
        cov3 = ext.quat_scale_to_cov3_cuda(quats.contiguous(), log_scales.contiguous())
        ctx.save_for_backward(quats, log_scales)
        return cov3

    @staticmethod
    def backward(ctx, grad_cov3):
        from tinysplat.cpp import load_cuda_extension

        quats, log_scales = ctx.saved_tensors
        ext = load_cuda_extension()
        g_q, g_ls = ext.quat_scale_to_cov3_vjp_cuda(
            quats.contiguous(), log_scales.contiguous(), grad_cov3.contiguous()
        )
        return g_q, g_ls


def _fused_quat_scale_to_cov3(quats, log_scales):
    """Fused covariance on CUDA, or None to fall back to the PyTorch path."""
    try:
        from tinysplat.cpp import load_cuda_extension

        ext = load_cuda_extension()
        if ext is None or not hasattr(ext, "quat_scale_to_cov3_cuda"):
            return None
        return _QuatScaleToCov3CUDA.apply(quats, log_scales)
    except Exception:
        return None
