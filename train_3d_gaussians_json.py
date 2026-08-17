#!/usr/bin/env python3
"""Train 3D Gaussian splats from a JSON dataset generated from COLMAP."""

import argparse
import json
import math
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import viser
from tqdm.auto import tqdm

from tinysplat.gaussian_splat_3d import gaussian_splat_3d
from tinysplat.gaussian_splat_3d_core import project_gaussians_3d_to_2d

try:
    from tinysplat.backends_3d.cuda import get_last_render_info
except ImportError:

    def get_last_render_info():
        return None

try:
    from tinysplat.fastgs import (
        FastGSConfig,
        accumulate_vcd_scores,
        accumulate_vcp_scores,
        densify_mask_vcd,
        high_error_mask,
        photometric_loss_value,
        prune_mask_fastgs_densify,
        prune_mask_vcp,
        should_step_optimizer,
        split_child_log_scales,
    )
    from tinysplat.metal_backend import (
        count_session_hits,
        last_grad_means2d,
        last_grad_means2d_abs,
        last_radii2d,
        metal_available,
    )
    from tinysplat.sh import colors_from_sh, init_sh_from_rgb, sh_degree_from_step

    _HAS_FASTGS = True
except ImportError:
    _HAS_FASTGS = False

    def metal_available():
        return False

    def last_grad_means2d():
        return None

    def last_grad_means2d_abs():
        return None

    def last_radii2d():
        return None

    def count_session_hits(*args, **kwargs):
        return None

    def screen_frac_from_proj_covs(proj_covs, height, width):
        return torch.zeros(proj_covs.shape[0], device=proj_covs.device)

    def split_child_log_scales(log_scales: torch.Tensor, shrink: float) -> torch.Tensor:
        scale = float(shrink)
        if scale <= 0:
            scale = 0.8
        delta = math.log(scale) if scale < 1.0 else -math.log(scale)
        return log_scales + delta

try:
    from gsplat.strategy.default import DefaultStrategy
    _HAS_GSPLAT_STRATEGY = True
except Exception:
    DefaultStrategy = None
    _HAS_GSPLAT_STRATEGY = False

try:
    from pytorch_msssim import ssim as _compute_ssim

    _HAS_SSIM = True
except ImportError:
    _HAS_SSIM = False

try:
    import lpips as _lpips_pkg

    _HAS_LPIPS = True
except ImportError:
    _lpips_pkg = None
    _HAS_LPIPS = False


@dataclass
class FrameSample:
    image_id: int
    file_path: Path
    width: int
    height: int
    intrinsics: torch.Tensor
    camera_to_world: torch.Tensor


@dataclass
class PreparedFrame:
    frame: FrameSample
    image: torch.Tensor
    intrinsics: torch.Tensor
    height: int
    width: int


class ViserVisualizer:
    def __init__(self, port: int):
        self.server = viser.ViserServer(port=port, host="0.0.0.0")
        self.server.scene.set_up_direction("+z")
        self.gaussian_handle = None
        self.camera_handles = {}
        self.selected_frame_idx = 0
        self.paused = False
        self.render_requested = False
        self.loss_handle = self.server.gui.add_markdown("**Status:** waiting for training")
        self.step_handle = self.server.gui.add_markdown("**Step:** 0")
        self.gaussian_stats_handle = self.server.gui.add_markdown("**Gaussians:** 0")
        self.frame_dropdown = None
        self.pause_button = self.server.gui.add_button("Pause", color="yellow")
        self.resume_button = self.server.gui.add_button("Resume", color="green")
        self.refresh_button = self.server.gui.add_button("Render Selected Frame", color="blue")
        self.frame_info_handle = self.server.gui.add_markdown("**Selected frame:** none")
        blank = np.zeros((32, 32, 3), dtype=np.uint8)
        self.target_image_handle = self.server.gui.add_image(blank, label="Target", format="jpeg")
        self.render_image_handle = self.server.gui.add_image(blank, label="Rendered", format="jpeg")

        @self.pause_button.on_click
        def _pause(_event):
            self.paused = True
            self.update_status("**Status:** paused")

        @self.resume_button.on_click
        def _resume(_event):
            self.paused = False
            self.update_status("**Status:** training")

        @self.refresh_button.on_click
        def _refresh(_event):
            self.render_requested = True

    def update_status(self, text: str):
        self.loss_handle.content = text

    def update_step(self, step: int, loss: float, psnr: float, frame_id: int):
        self.step_handle.content = (
            f"**Step:** {step}\n\n"
            f"**Frame:** {frame_id}\n\n"
            f"**Loss:** {loss:.6f}\n\n"
            f"**PSNR:** {psnr:.2f} dB"
        )

    def update_gaussian_stats(self, count: int):
        self.gaussian_stats_handle.content = f"**Gaussians:** {count}"

    def set_cameras(self, frames: List["FrameSample"]):
        for idx, frame in enumerate(frames):
            intr = frame.intrinsics.detach().cpu().numpy()
            c2w = frame.camera_to_world.detach().cpu().numpy()
            rotation = c2w[:3, :3]
            position = c2w[:3, 3]
            self.camera_handles[idx] = self.server.scene.add_camera_frustum(
                f"/cameras/{idx:04d}_{frame.image_id}",
                fov=2.0 * math.atan2(frame.height * 0.5, float(intr[1, 1])),
                aspect=float(frame.width) / float(frame.height),
                scale=0.08,
                line_width=1.0,
                color=(40, 120, 255),
                wxyz=rotation_matrix_to_wxyz(rotation),
                position=position,
            )
        options = tuple(f"{idx}: image_id={frame.image_id}" for idx, frame in enumerate(frames))
        self.frame_dropdown = self.server.gui.add_dropdown(
            "Frame", options=options, initial_value=options[0]
        )

        @self.frame_dropdown.on_update
        def _select_frame(_event):
            self.selected_frame_idx = int(self.frame_dropdown.value.split(":", 1)[0])

    def update_gaussians(
        self,
        means: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        covariances: torch.Tensor,
    ):
        centers = np.ascontiguousarray(means.detach().cpu().numpy().astype(np.float32))
        point_colors = np.ascontiguousarray(
            np.clip(colors.detach().cpu().numpy(), 0.0, 1.0).astype(np.float32)
        )
        point_opacities = np.ascontiguousarray(
            opacities.detach().cpu().numpy().reshape(-1, 1).astype(np.float32)
        )
        point_covariances = np.ascontiguousarray(
            covariances.detach().cpu().numpy().astype(np.float32)
        )
        if (
            self.gaussian_handle is not None
            and centers.shape[0] == self.gaussian_handle.centers.shape[0]
        ):
            self.gaussian_handle.centers = centers
            self.gaussian_handle.rgbs = point_colors
            self.gaussian_handle.opacities = point_opacities
            self.gaussian_handle.covariances = point_covariances
            return
        if self.gaussian_handle is not None:
            self.gaussian_handle.remove()
        self.gaussian_handle = self.server.scene.add_gaussian_splats(
            "/gaussians",
            centers=centers,
            rgbs=point_colors,
            opacities=point_opacities,
            covariances=point_covariances,
        )

    def update_frame_preview(
        self,
        frame_idx: int,
        frame: "FrameSample",
        target: torch.Tensor,
        rendered: torch.Tensor,
    ):
        self.selected_frame_idx = frame_idx
        if self.frame_dropdown is not None:
            self.frame_dropdown.value = f"{frame_idx}: image_id={frame.image_id}"
        self.frame_info_handle.content = (
            f"**Selected frame:** {frame_idx}\n\n"
            f"**Image id:** {frame.image_id}\n\n"
            f"**Path:** `{frame.file_path.name}`"
        )
        self.target_image_handle.image = tensor_image_to_uint8(target)
        self.render_image_handle.image = tensor_image_to_uint8(rendered)

    def should_render_selected_frame(self, step: int, update_every: int) -> bool:
        if self.render_requested:
            self.render_requested = False
            return True
        if update_every and step % update_every == 0:
            return True
        return False

    def wait_if_paused(self):
        while self.paused:
            time.sleep(0.1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_json", type=Path, help="Path to the JSON dataset file.")
    parser.add_argument("--iterations", type=int, default=500, help="Number of optimization steps.")
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        help="Global multiplier on 3DGS per-parameter Adam rates (default 1.0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use for training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=50,
        help="Render and save the first frame every N steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for outputs. Defaults to a temp dir under the repo root.",
    )
    parser.add_argument(
        "--viser-port",
        type=int,
        default=8080,
        help="Port for the viser server.",
    )
    parser.add_argument(
        "--viser-update-every",
        type=int,
        default=10,
        help="Update the viser scene every N steps. Use 0 to disable updates after startup.",
    )
    parser.add_argument(
        "--init-grid-long-side",
        type=int,
        default=256,
        help="Cap the longer side of the initial Gaussian grid while keeping training at full image resolution.",
    )
    parser.add_argument(
        "--ssim-lambda",
        type=float,
        default=0.2,
        help="Weight of SSIM loss (0 to disable, nerfstudio default is 0.2).",
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=0,
        help="Downsample images so the long side is at most this value. Use 0 to keep original resolution.",
    )
    parser.add_argument(
        "--num-downscales",
        type=int,
        default=2,
        help="Start training at 1/2^d resolution (nerfstudio default: 2).",
    )
    parser.add_argument(
        "--resolution-schedule",
        type=int,
        default=3000,
        help="Double resolution every N steps (nerfstudio default: 3000).",
    )
    parser.add_argument(
        "--limit-frames",
        type=int,
        default=0,
        help="Use only the first N frames for training. Use 0 to use all frames.",
    )
    parser.add_argument(
        "--densify-every",
        type=int,
        default=100,
        help="Run densify/prune every N steps. Use 0 to disable. (nerfstudio default: 100)",
    )
    parser.add_argument(
        "--densify-from",
        type=int,
        default=500,
        help="Start densification after this many steps. (nerfstudio default: 500)",
    )
    parser.add_argument(
        "--densify-until",
        type=int,
        default=15000,
        help="Stop densification after this many steps. (nerfstudio default: 15000)",
    )
    parser.add_argument(
        "--densify-grad-thresh",
        type=float,
        default=None,
        help="Split/duplicate gaussians whose mean gradient norm exceeds this value. "
        "Default 8e-4 (nerfstudio) or 2e-4 with --fastgs.",
    )
    parser.add_argument(
        "--densify-grad-abs-thresh",
        type=float,
        default=None,
        help="FastGS AbsGS split threshold (official truck 0.0009, code default 0.0012).",
    )
    parser.add_argument(
        "--prune-opacity-thresh",
        type=float,
        default=0.03,
        help="Prune gaussians whose sigmoid opacity falls below this threshold.",
    )
    parser.add_argument(
        "--reset-opacity-every",
        type=int,
        default=3000,
        help="Reset opacities to a low value every N steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-gaussians",
        type=int,
        default=5000000,
        help="Maximum number of gaussians after densification.",
    )
    parser.add_argument(
        "--split-scale-shrink",
        type=float,
        default=0.8,
        help="Scale shrink for split children: <1 multiplies (nerfstudio 0.8), >=1 divides. "
        "With --fastgs the 0.8 default is replaced by 3DGS /1.6.",
    )
    parser.add_argument(
        "--use-scale-regularization",
        action="store_true",
        help="Enable scale regularization to prevent huge spikey gaussians (PhysGaussian).",
    )
    parser.add_argument(
        "--max-gauss-ratio",
        type=float,
        default=10.0,
        help="Max ratio of gaussian max to min scale before applying regularization.",
    )
    parser.add_argument(
        "--cull-screen-size",
        type=float,
        default=0.15,
        help="Prune gaussians covering more than this fraction of screen. Set 0 to disable.",
    )
    parser.add_argument(
        "--split-screen-size",
        type=float,
        default=0.05,
        help="Split gaussians covering more than this fraction of screen. Set 0 to disable.",
    )
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=0,
        help="Set torch intra-op CPU threads. Use 0 to keep the runtime default.",
    )
    parser.add_argument(
        "--torch-num-inter-op-threads",
        type=int,
        default=0,
        help="Set torch inter-op CPU threads. Use 0 to keep the runtime default.",
    )
    parser.add_argument(
        "--cache-images",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Preload training images in memory. Defaults to on for MPS.",
    )
    parser.add_argument(
        "--eval-hold",
        type=int,
        default=0,
        help="Hold every N-th frame (by sorted index) for evaluation. "
        "3DGS / LLFF protocol uses 8. Use 0 to disable (train on all frames).",
    )
    parser.add_argument(
        "--no-viser",
        action="store_true",
        help="Disable the viser UI (recommended for headless benchmarks).",
    )
    parser.add_argument(
        "--fastgs",
        action="store_true",
        help="Enable FastGS VCD/VCP densify/prune (official compositor scoring + AbsGS split).",
    )
    parser.add_argument(
        "--sh-degree",
        type=int,
        default=3,
        help="Max spherical-harmonics degree (0=RGB only). Scheduled 0→max every 1000 steps.",
    )
    parser.add_argument(
        "--use-metal",
        action="store_true",
        help="Prefer Metal tiled rasterizer when the dylib is available.",
    )
    return parser.parse_args()


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return choose_device()
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if device_arg == "mps" and (
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
    ):
        raise ValueError("MPS was requested but is not available.")
    return device_arg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_torch_threads(num_threads: int, num_inter_op_threads: int):
    if num_threads and num_threads > 0:
        torch.set_num_threads(num_threads)
    if num_inter_op_threads and num_inter_op_threads > 0:
        try:
            torch.set_num_interop_threads(num_inter_op_threads)
        except RuntimeError:
            pass


def rotation_matrix_to_wxyz(rotation: np.ndarray) -> Tuple[float, float, float, float]:
    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return (
            0.25 * s,
            (rotation[2, 1] - rotation[1, 2]) / s,
            (rotation[0, 2] - rotation[2, 0]) / s,
            (rotation[1, 0] - rotation[0, 1]) / s,
        )
    if rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        return (
            (rotation[2, 1] - rotation[1, 2]) / s,
            0.25 * s,
            (rotation[0, 1] + rotation[1, 0]) / s,
            (rotation[0, 2] + rotation[2, 0]) / s,
        )
    if rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        return (
            (rotation[0, 2] - rotation[2, 0]) / s,
            (rotation[0, 1] + rotation[1, 0]) / s,
            0.25 * s,
            (rotation[1, 2] + rotation[2, 1]) / s,
        )
    s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
    return (
        (rotation[1, 0] - rotation[0, 1]) / s,
        (rotation[0, 2] + rotation[2, 0]) / s,
        (rotation[1, 2] + rotation[2, 1]) / s,
        0.25 * s,
    )


def save_image(image: torch.Tensor, output_path: Path):
    image_np = image.detach().cpu().clamp(0.0, 1.0).numpy()
    image_np = (image_np * 255.0).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), image_bgr)


def tensor_image_to_uint8(image: torch.Tensor) -> np.ndarray:
    return (image.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)


def compute_initial_grid_shape(
    height: int,
    width: int,
    long_side_limit: int,
) -> Tuple[int, int]:
    if long_side_limit <= 0:
        return max(1, height), max(1, width)
    long_side = max(height, width)
    scale = min(1.0, float(long_side_limit) / float(long_side))
    grid_height = max(1, int(round(height * scale)))
    grid_width = max(1, int(round(width * scale)))
    return grid_height, grid_width


def backproject_pixels_to_world(
    pixel_centers: torch.Tensor,
    depth,
    intrinsics: torch.Tensor,
    camera_to_world: torch.Tensor,
) -> torch.Tensor:
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    if isinstance(depth, torch.Tensor):
        d = depth
    else:
        d = torch.full(
            (pixel_centers.shape[0],), depth, dtype=pixel_centers.dtype, device=pixel_centers.device
        )

    x_cam = (pixel_centers[:, 0] - cx) * d / fx
    y_cam = (pixel_centers[:, 1] - cy) * d / fy
    z_cam = d
    points_camera = torch.stack([x_cam, y_cam, z_cam], dim=1)

    rotation = camera_to_world[:3, :3]
    translation = camera_to_world[:3, 3]
    return points_camera @ rotation.transpose(0, 1) + translation


def _pose_centers_bbox(frames: List[FrameSample], dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    centers = torch.stack([frame.camera_to_world[:3, 3] for frame in frames], dim=0).to(device=device, dtype=dtype)
    bbox_min = centers.min(dim=0).values
    bbox_max = centers.max(dim=0).values
    eps = torch.tensor([0.1, 0.1, 0.1], device=device, dtype=dtype)
    return bbox_min - eps, bbox_max + eps


def scene_scale_from_frames(frames: List[FrameSample]) -> float:
    """3DGS cameras_extent: 1.1 × max camera distance from the camera centroid."""
    if not frames:
        return 1.0
    centers = torch.stack([frame.camera_to_world[:3, 3] for frame in frames], dim=0)
    radius = (centers - centers.mean(dim=0)).norm(dim=-1).max()
    return max(float(radius.item()) * 1.1, 1e-3)


def build_pose_bbox_gaussians_3d(
    target: torch.Tensor,
    frames: List[FrameSample],
    voxel_size: float = 0.1,
) -> Dict[str, torch.Tensor]:
    channels = target.shape[2]
    dtype = target.dtype
    device = target.device

    bbox_min, bbox_max = _pose_centers_bbox(frames, dtype=dtype, device=device)

    step = float(voxel_size)
    xs = torch.arange(float(bbox_min[0]), float(bbox_max[0]) + 0.5 * step, step, device=device, dtype=dtype)
    ys = torch.arange(float(bbox_min[1]), float(bbox_max[1]) + 0.5 * step, step, device=device, dtype=dtype)
    zs = torch.arange(float(bbox_min[2]), float(bbox_max[2]) + 0.5 * step, step, device=device, dtype=dtype)

    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing='ij')
    means = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3).contiguous()

    num_gaussians = means.shape[0]
    if num_gaussians <= 0:
        raise ValueError('Pose bounding-box initialization produced zero gaussians.')
    if num_gaussians > 2_000_000:
        raise ValueError(
            f'Pose bounding-box initialization produced {num_gaussians} gaussians (>2,000,000). '
            'Increase voxel size or reduce pose extent.'
        )

    init_scale = max(step * 0.5, 1e-3)
    log_scales = torch.full((num_gaussians, 3), math.log(init_scale), device=device, dtype=dtype)

    flat_colors = target.reshape(-1, channels)
    color_idx = torch.randint(0, flat_colors.shape[0], (num_gaussians,), device=device)
    colors = flat_colors[color_idx].contiguous()
    # Keep initial splats visible in viser even when source images are very dark.
    colors = torch.clamp(colors * 0.9 + 0.1, 0.0, 1.0)

    initial_alpha = 0.99
    initial_opacity_logit = torch.logit(torch.tensor(initial_alpha, device=device, dtype=dtype)).item()
    opacity_logits = torch.full((num_gaussians,), initial_opacity_logit, device=device, dtype=dtype)

    rotations = torch.zeros(num_gaussians, 4, device=device, dtype=dtype)
    rotations[:, 0] = 1.0

    out = {
        'means': means,
        'log_scales': log_scales,
        'rotations': rotations,
        'colors': colors,
        'opacities': opacity_logits,
    }
    if _HAS_FASTGS:
        out['sh_coeffs'] = init_sh_from_rgb(colors, max_degree=3)
    return out


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
    if _HAS_FASTGS:
        out["sh_coeffs"] = init_sh_from_rgb(rgb, max_degree=3)
    return out


def load_dataset_frames(dataset_json: Path, device: str) -> Tuple[Path, List[FrameSample], List[Dict[str, object]]]:
    dataset = json.loads(dataset_json.read_text(encoding="utf-8"))
    scene_dir = Path(dataset["scene_dir"])
    points3d = dataset.get("points3d", [])
    frames: List[FrameSample] = []
    for frame in dataset["frames"]:
        intr = frame["intrinsics"]
        intrinsics = torch.tensor(
            [
                [intr["fx"], 0.0, intr["cx"]],
                [0.0, intr["fy"], intr["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        camera_to_world = torch.tensor(
            frame["transform_matrix"],
            dtype=torch.float32,
            device=device,
        )
        frames.append(
            FrameSample(
                image_id=frame["image_id"],
                file_path=scene_dir / frame["file_path"],
                width=frame["width"],
                height=frame["height"],
                intrinsics=intrinsics,
                camera_to_world=camera_to_world,
            )
        )
    return scene_dir, frames, points3d


def load_frame_image(
    frame: FrameSample,
    device: str,
    max_resolution: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    image_bgr = cv2.imread(str(frame.file_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image from {frame.file_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    intrinsics = frame.intrinsics.clone()
    if frame.width > 0 and frame.height > 0 and (width != frame.width or height != frame.height):
        sx = width / float(frame.width)
        sy = height / float(frame.height)
        intrinsics[0, 0] *= sx
        intrinsics[0, 2] *= sx
        intrinsics[1, 1] *= sy
        intrinsics[1, 2] *= sy

    if max_resolution > 0 and max(height, width) > max_resolution:
        scale = max_resolution / max(height, width)
        new_h = int(round(height * scale))
        new_w = int(round(width * scale))
        image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        intrinsics[:2, :3] *= scale
        height, width = new_h, new_w

    image = torch.from_numpy(image_rgb.astype(np.float32) / 255.0).to(device)
    return image, intrinsics, height, width


def prepare_dataset_frames(
    frames: List[FrameSample],
    device: str,
    max_resolution: int = 0,
) -> List[PreparedFrame]:
    prepared: List[PreparedFrame] = []
    for frame in frames:
        image, intrinsics, height, width = load_frame_image(
            frame,
            device=device,
            max_resolution=max_resolution,
        )
        prepared.append(
            PreparedFrame(
                frame=frame,
                image=image,
                intrinsics=intrinsics,
                height=height,
                width=width,
            )
        )
    return prepared


def split_train_eval(
    frames: List[FrameSample],
    eval_hold: int,
) -> Tuple[List[FrameSample], List[FrameSample]]:
    """LLFF / 3DGS split: every eval_hold-th frame (0-based) is held out."""
    if eval_hold <= 0:
        return list(frames), []
    train_frames: List[FrameSample] = []
    eval_frames: List[FrameSample] = []
    for idx, frame in enumerate(frames):
        if idx % eval_hold == 0:
            eval_frames.append(frame)
        else:
            train_frames.append(frame)
    if not train_frames:
        raise ValueError(
            f"eval_hold={eval_hold} left zero training frames "
            f"(dataset has {len(frames)} frames)."
        )
    return train_frames, eval_frames


def _psnr(mse: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(mse.clamp_min(1e-10))


@torch.no_grad()
def evaluate_heldout(
    gauss_data: "GaussianData",
    frames: List[FrameSample],
    device: str,
    max_resolution: int = 0,
) -> Dict[str, float]:
    """Mean PSNR / SSIM / LPIPS over held-out views."""
    if not frames:
        return {"psnr": float("nan"), "ssim": float("nan"), "lpips": float("nan"), "num_views": 0}

    if not _HAS_SSIM:
        raise RuntimeError("pytorch_msssim is required for eval: pip install pytorch-msssim")

    lpips_fn = None
    if _HAS_LPIPS:
        # Official 3DGS metrics use AlexNet LPIPS.
        lpips_fn = _lpips_pkg.LPIPS(net="alex").to(device)
        lpips_fn.eval()
    else:
        print("Warning: lpips not installed; LPIPS will be reported as NaN. pip install lpips")

    psnrs: List[float] = []
    ssims: List[float] = []
    lpipses: List[float] = []

    for frame in tqdm(frames, desc="Eval", unit="view", leave=False):
        target, intrinsics, height, width = load_frame_image(
            frame, device=device, max_resolution=max_resolution
        )
        rendered = gauss_data.render(
            intrinsics=intrinsics,
            camera_to_world=frame.camera_to_world,
            height=height,
            width=width,
        )
        mse = F.mse_loss(rendered, target)
        psnrs.append(float(_psnr(mse).item()))
        ssims.append(
            float(
                _compute_ssim(
                    rendered.permute(2, 0, 1).unsqueeze(0),
                    target.permute(2, 0, 1).unsqueeze(0),
                    data_range=1.0,
                    size_average=True,
                ).item()
            )
        )
        if lpips_fn is not None:
            # LPIPS expects NCHW in [-1, 1]
            pred_n = rendered.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
            tgt_n = target.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
            lpipses.append(float(lpips_fn(pred_n, tgt_n).mean().item()))

    return {
        "psnr": sum(psnrs) / len(psnrs),
        "ssim": sum(ssims) / len(ssims),
        "lpips": (sum(lpipses) / len(lpipses)) if lpipses else float("nan"),
        "num_views": float(len(frames)),
    }


class _NullVisualizer:
    """No-op stand-in when --no-viser is set."""

    selected_frame_idx = 0
    paused = False
    render_requested = False

    def update_status(self, text: str):
        pass

    def update_step(self, step: int, loss: float, psnr: float, frame_id: int):
        pass

    def update_gaussian_stats(self, n: int):
        pass

    def set_cameras(self, frames):
        pass

    def update_gaussians(self, *args, **kwargs):
        pass

    def update_frame_preview(self, *args, **kwargs):
        pass

    def wait_if_paused(self):
        pass

    def should_render_selected_frame(self, step: int, update_every: int) -> bool:
        return False


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
        elif self.max_sh_degree > 0 and _HAS_FASTGS:
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
        if self.max_sh_degree <= 0 or not _HAS_FASTGS:
            return
        self.active_sh_degree = sh_degree_from_step(step, self.max_sh_degree)

    def covariance_matrices(self) -> torch.Tensor:
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
        if self.sh_coeffs is not None and camera_to_world is not None and _HAS_FASTGS:
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
        elif self.sh_coeffs is not None and _HAS_FASTGS:
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

    def sync_from_strategy_params(self, params: Dict[str, torch.Tensor]):
        self.means = params["means"]
        self.log_scales = params["scales"]
        self.rotations = params["quats"]
        self.colors = params["colors"]
        self.opacities = params["opacities"]


def save_checkpoint(
    data: GaussianData,
    output_path: Path,
):
    torch.save(data.export_params(), output_path)


def save_ply(
    data: GaussianData,
    output_path: Path,
):
    """Export 3D Gaussians as PLY file (compatible with standard 3DGS viewers)."""
    import numpy as np
    from plyfile import PlyData, PlyElement

    n = data.num_gaussians
    means = data.means.detach().cpu().numpy()
    log_scales = data.log_scales.detach().cpu().numpy()
    rotations = data.rotations.detach().cpu().numpy()
    colors = data.visible_colors().detach().cpu().numpy()
    opacities = data.visible_opacities().detach().cpu().numpy()

    # Normalize rotations to unit quaternion
    norms = np.linalg.norm(rotations, axis=1, keepdims=True).clip(min=1e-8)
    rotations = rotations / norms

    # Opacity: inverse sigmoid for PLY (PLYSigmoid format)
    opacity_inv_sigmoid = np.log(opacities / (1.0 - opacities + 1e-8))

    # SH DC coefficients (constant term = color * SH_C0)
    SH_C0 = 0.28209479177387814
    sh_dc = colors / SH_C0

    # Build vertex dtype
    dtype_list = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
    ]
    # SH DC (one per channel)
    num_sh_coeffs = 1  # DC only
    for i in range(3):  # RGB
        for j in range(num_sh_coeffs):
            dtype_list.append((f"f_dc_{i}", "f4"))
    dtype_list.append(("opacity", "f4"))
    for i in range(3):
        dtype_list.append((f"scale_{i}", "f4"))
    for i in range(4):
        dtype_list.append((f"rot_{i}", "f4"))

    vertices = np.zeros(n, dtype=dtype_list)
    vertices["x"] = means[:, 0]
    vertices["y"] = means[:, 1]
    vertices["z"] = means[:, 2]
    vertices["nx"] = 0.0
    vertices["ny"] = 0.0
    vertices["nz"] = 0.0
    for i in range(3):
        vertices[f"f_dc_{i}"] = sh_dc[:, i]
    vertices["opacity"] = opacity_inv_sigmoid
    for i in range(3):
        vertices[f"scale_{i}"] = log_scales[:, i]
    for i in range(4):
        vertices[f"rot_{i}"] = rotations[:, i]

    el = PlyElement.describe(vertices, "vertex")
    PlyData([el], byte_order="<").write(str(output_path))


# 3DGS / FastGS per-parameter Adam rates (multiplied by --lr).
_MEANS_LR_INIT = 1.6e-4
_MEANS_LR_FINAL = 1.6e-6
_SCALES_LR = 5e-3
_QUATS_LR = 1e-3
_OPACITY_LR = 5e-2
_SH_LR = 2.5e-3
_SH_REST_GRAD_SCALE = 0.05  # rest / 20 relative to DC


def expon_lr(step: int, lr_init: float, lr_final: float, max_steps: int) -> float:
    """3DGS log-linear decay from lr_init to lr_final over max_steps."""
    if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
        return 0.0
    t = min(max(float(step) / float(max(max_steps, 1)), 0.0), 1.0)
    return math.exp(math.log(max(lr_init, 1e-16)) * (1.0 - t) + math.log(max(lr_final, 1e-16)) * t)


def means_lr_at(step: int, scene_scale: float, max_steps: int, lr_scale: float = 1.0) -> float:
    extent = max(float(scene_scale), 1e-3)
    return expon_lr(
        step,
        _MEANS_LR_INIT * extent * lr_scale,
        _MEANS_LR_FINAL * extent * lr_scale,
        max_steps,
    )


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def build_optimizers(
    data: GaussianData,
    scene_scale: float,
    step: int = 0,
    max_steps: int = 30000,
    lr_scale: float = 1.0,
):
    adam_kw = {"foreach": True}
    scale = float(lr_scale)
    opts = {
        "means": torch.optim.Adam(
            [data.means], lr=means_lr_at(step, scene_scale, max_steps, scale), **adam_kw
        ),
        "scales": torch.optim.Adam([data.log_scales], lr=_SCALES_LR * scale, **adam_kw),
        "quats": torch.optim.Adam([data.rotations], lr=_QUATS_LR * scale, **adam_kw),
        "opacities": torch.optim.Adam([data.opacities], lr=_OPACITY_LR * scale, **adam_kw),
    }
    if data.sh_coeffs is not None:
        opts["sh"] = torch.optim.Adam([data.sh_coeffs], lr=_SH_LR * scale, **adam_kw)
    else:
        opts["colors"] = torch.optim.Adam([data.colors], lr=_SH_LR * scale, **adam_kw)
    return opts


def zero_optimizers(optimizers):
    for opt in optimizers.values():
        opt.zero_grad(set_to_none=True)


def scale_sh_rest_grads(data: GaussianData) -> None:
    """3DGS: higher-order SH steps 20× slower than DC without splitting the tensor."""
    if data.sh_coeffs is not None and data.sh_coeffs.grad is not None:
        data.sh_coeffs.grad[:, 1:, :] *= _SH_REST_GRAD_SCALE


def step_optimizers(optimizers):
    for opt in optimizers.values():
        opt.step()


@torch.no_grad()
def sanitize_gaussians(data: GaussianData, scene_scale: float) -> None:
    """Drop non-finite params, clamp scales, and renormalize quaternions."""
    max_log = math.log(max(0.1 * float(scene_scale), 1e-4))
    min_log = math.log(1e-6)
    for tensor in data.parameters():
        bad = ~torch.isfinite(tensor)
        if bool(bad.any().item()):
            tensor.data[bad] = 0.0
    data.log_scales.clamp_(min=min_log, max=max_log)
    data.rotations.copy_(F.normalize(data.rotations, dim=-1))


def _tensor_pstats(x: torch.Tensor) -> Tuple[float, float, float]:
    """Return (p50, p90, max) for a 1D tensor; zeros if empty. CPU quantile (MPS is NaN-prone)."""
    x = x.detach().float().reshape(-1)
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return 0.0, 0.0, 0.0
    x = x.cpu()
    return (
        float(torch.quantile(x, 0.5).item()),
        float(torch.quantile(x, 0.9).item()),
        float(x.max().item()),
    )


def remap_optimizers_after_densify(
    old_optimizers,
    data: "GaussianData",
    keep_idx: torch.Tensor,
    n_new: int,
    step: int,
    scene_scale: float,
    max_steps: int,
    lr_scale: float,
):
    """Rebuild Adam but copy exp_avg/exp_avg_sq for surviving rows; zeros for new Gaussians."""
    new_opts = build_optimizers(
        data, scene_scale=scene_scale, step=step, max_steps=max_steps, lr_scale=lr_scale
    )
    keep_idx = keep_idx.long()
    n_keep = int(keep_idx.numel())
    for name, new_opt in new_opts.items():
        old_opt = old_optimizers.get(name)
        if old_opt is None:
            continue
        old_p = old_opt.param_groups[0]["params"][0]
        old_st = old_opt.state.get(old_p)
        if not old_st:
            continue
        new_p = new_opt.param_groups[0]["params"][0]
        if new_p.shape[0] != n_keep + n_new:
            continue
        new_st = new_opt.state[new_p]
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if key not in old_st:
                continue
            src = old_st[key]
            dst = torch.zeros_like(new_p)
            if n_keep > 0:
                dst[:n_keep] = src[keep_idx]
            new_st[key] = dst
        if "step" in old_st:
            new_st["step"] = old_st["step"]
    return new_opts


def reset_opacity_adam(optimizers) -> None:
    """Zero Adam moments for opacity only (3DGS opacity reset)."""
    opt = optimizers.get("opacities")
    if opt is None:
        return
    p = opt.param_groups[0]["params"][0]
    st = opt.state.get(p)
    if not st:
        return
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        if key in st and torch.is_tensor(st[key]):
            st[key].zero_()
    if "step" in st:
        step = st["step"]
        if torch.is_tensor(step):
            step.zero_()
        else:
            st["step"] = 0


@torch.no_grad()
def densify_and_prune_fastgs(
    data: GaussianData,
    frames: List[FrameSample],
    grad_accum: torch.Tensor,
    vis_count: torch.Tensor,
    cfg: "FastGSConfig",
    step: int,
    max_gaussians: int,
    max_resolution: int = 0,
    ssim_fn=None,
    do_densify: bool = True,
    do_prune: bool = True,
    scene_scale: float = 1.0,
    absgrad_accum: Optional[torch.Tensor] = None,
    grad3d_accum: Optional[torch.Tensor] = None,
    max_radii2d: Optional[torch.Tensor] = None,
    size_threshold: Optional[float] = None,
) -> Tuple[bool, torch.Tensor, int]:
    """Official FastGS densify/prune: compositor VCD, AbsGS split, 20px/opacity prune."""
    if not _HAS_FASTGS:
        raise RuntimeError("FastGS helpers are not available")
    n = data.num_gaussians
    device = data.device
    orig_idx = torch.arange(n, device=device)
    means = data.means.detach().clone()
    log_scales = data.log_scales.detach().clone()
    rotations = data.rotations.detach().clone()
    colors = data.colors.detach().clone()
    opacities = data.opacities.detach().clone()
    sh_coeffs = data.sh_coeffs.detach().clone() if data.sh_coeffs is not None else None
    vis = torch.clamp(vis_count.detach(), min=1.0)
    grad_norms = torch.nan_to_num(grad_accum.detach() / vis, nan=0.0, posinf=0.0, neginf=0.0)
    if grad_norms.numel() != n:
        grad_norms = torch.zeros(n, device=device)
    abs_grad_norms = None
    if absgrad_accum is not None and absgrad_accum.numel() == n:
        abs_grad_norms = torch.nan_to_num(
            absgrad_accum.detach() / vis, nan=0.0, posinf=0.0, neginf=0.0
        )
    grad3d_norms = None
    if grad3d_accum is not None and grad3d_accum.numel() == n:
        grad3d_norms = torch.nan_to_num(grad3d_accum.detach() / vis, nan=0.0, posinf=0.0, neginf=0.0)
    radii = torch.zeros(n, device=device)
    if max_radii2d is not None and max_radii2d.numel() == n:
        radii = max_radii2d.detach().to(device=device, dtype=means.dtype)

    k = min(cfg.k_views, len(frames))
    view_idx = torch.randperm(len(frames))[:k].tolist()
    counts_list = []
    photo_list = []
    n_session_ok = 0
    for vi in view_idx:
        frame = frames[vi]
        target, intrinsics, height, width = load_frame_image(
            frame, device=str(device), max_resolution=max_resolution
        )
        rendered = data.render(
            intrinsics=intrinsics,
            camera_to_world=frame.camera_to_world,
            height=height,
            width=width,
        ).detach()
        mask = high_error_mask(rendered, target, cfg.error_tau)
        hits = count_session_hits(mask, n, height, width)
        if hits is None:
            hits = torch.zeros(n, dtype=torch.int64)
        else:
            n_session_ok += 1
            hits = hits.to(device)
        counts_list.append(hits)
        photo_list.append(
            photometric_loss_value(rendered, target, ssim_fn, cfg.ssim_lambda)
        )

    vcd = accumulate_vcd_scores(counts_list)
    vcp = accumulate_vcp_scores(counts_list, photo_list)
    clone_mask = torch.zeros(n, dtype=torch.bool, device=device)
    split_mask = torch.zeros(n, dtype=torch.bool, device=device)
    if do_densify:
        clone_mask, split_mask = densify_mask_vcd(
            vcd,
            grad_norms,
            log_scales,
            cfg,
            scene_scale=scene_scale,
            abs_grad_norms=abs_grad_norms,
        )

    opa_sig = torch.sigmoid(opacities)
    max_scale = torch.exp(log_scales).max(dim=-1).values
    n_clone = int(clone_mask.sum().item())
    n_split = int(split_mask.sum().item())
    vcd_p50, vcd_p90, vcd_max = _tensor_pstats(vcd)
    vcd_frac = float((vcd > cfg.densify_score_thresh).float().mean().item())
    g2d_p50, g2d_p90, g2d_max = _tensor_pstats(grad_norms)
    abs_p50, abs_p90, abs_max = _tensor_pstats(
        abs_grad_norms if abs_grad_norms is not None else torch.zeros(0, device=device)
    )
    g3d_p50, g3d_p90, g3d_max = _tensor_pstats(
        grad3d_norms if grad3d_norms is not None else torch.zeros(0, device=device)
    )
    sc_p50, sc_p90, sc_max = _tensor_pstats(max_scale)
    vis_mean = float(vis.mean().item())
    vis_frac = float((vis_count.detach() > 0).float().mean().item())
    tqdm.write(
        f"[FastGS step={step}] N={n} VCD p50/p90/max={vcd_p50:.2f}/{vcd_p90:.2f}/{vcd_max:.1f} "
        f"frac>τd={vcd_frac:.4f} session_hits={n_session_ok}/{len(view_idx)} "
        f"grad2d p50/p90/max={g2d_p50:.4g}/{g2d_p90:.4g}/{g2d_max:.4g} "
        f"absgrad p50/p90/max={abs_p50:.4g}/{abs_p90:.4g}/{abs_max:.4g} "
        f"grad3d p50/p90/max={g3d_p50:.4g}/{g3d_p90:.4g}/{g3d_max:.4g} "
        f"n_clone={n_clone} n_split={n_split} "
        f"vis_mean={vis_mean:.1f} vis_frac={vis_frac:.3f} "
        f"scale p50/p90/max={sc_p50:.4f}/{sc_p90:.4f}/{sc_max:.4f} "
        f"split_shrink={cfg.split_scale_shrink:g} size_th={size_threshold} "
        f"grow={cfg.grow_scale3d:g}*extent error_tau={cfg.error_tau:g}"
    )

    n_before = means.shape[0]
    capacity = max(0, max_gaussians - n_before)
    extra = int(clone_mask.sum().item()) + int(split_mask.sum().item())
    if extra > capacity:
        scores = grad_norms
        split_idx = torch.where(split_mask)[0]
        clone_idx = torch.where(clone_mask)[0]
        chosen_split = torch.zeros_like(split_mask)
        chosen_clone = torch.zeros_like(clone_mask)
        rem = capacity
        if split_idx.numel() and rem > 0:
            ks = min(split_idx.numel(), rem)
            top = torch.topk(scores[split_idx], k=ks).indices
            chosen_split[split_idx[top]] = True
            rem -= ks
        if clone_idx.numel() and rem > 0:
            kc = min(clone_idx.numel(), rem)
            top = torch.topk(scores[clone_idx], k=kc).indices
            chosen_clone[clone_idx[top]] = True
        split_mask, clone_mask = chosen_split, chosen_clone

    new_means, new_log_scales, new_rots, new_colors, new_opa, new_sh = [], [], [], [], [], []
    new_radii = []
    n_new = 0
    if split_mask.any():
        n_split = int(split_mask.sum().item())
        sel_scales = torch.exp(log_scales[split_mask])
        sel_quats = F.normalize(rotations[split_mask], dim=-1)
        w, x, y, z = sel_quats[:, 0], sel_quats[:, 1], sel_quats[:, 2], sel_quats[:, 3]
        rotmats = torch.stack(
            [
                torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=1),
                torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=1),
                torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=1),
            ],
            dim=1,
        )
        noise = torch.randn(2, n_split, 3, device=means.device)
        samples = torch.einsum("nij,nj,bnj->bni", rotmats, sel_scales, noise)
        new_means.append((means[split_mask].unsqueeze(0) + samples).reshape(-1, 3))
        new_log_scales.append(
            split_child_log_scales(log_scales[split_mask], cfg.split_scale_shrink).repeat(2, 1)
        )
        new_rots.append(rotations[split_mask].repeat(2, 1))
        new_colors.append(colors[split_mask].repeat(2, 1))
        new_opa.append(opacities[split_mask].repeat(2))
        new_radii.append(radii[split_mask].repeat(2))
        if sh_coeffs is not None:
            new_sh.append(sh_coeffs[split_mask].repeat(2, 1, 1))
        n_new += 2 * n_split

    if clone_mask.any():
        n_clone_kept = int(clone_mask.sum().item())
        new_means.append(means[clone_mask])
        new_log_scales.append(log_scales[clone_mask])
        new_rots.append(rotations[clone_mask])
        new_colors.append(colors[clone_mask])
        new_opa.append(opacities[clone_mask])
        new_radii.append(radii[clone_mask])
        if sh_coeffs is not None:
            new_sh.append(sh_coeffs[clone_mask])
        n_new += n_clone_kept

    keep_src = ~split_mask
    keep_idx = orig_idx[keep_src]
    vcp_kept = vcp[keep_src]
    radii_kept = radii[keep_src]
    if split_mask.any() or clone_mask.any():
        means = torch.cat([means[keep_src]] + new_means, dim=0)
        log_scales = torch.cat([log_scales[keep_src]] + new_log_scales, dim=0)
        rotations = F.normalize(torch.cat([rotations[keep_src]] + new_rots, dim=0), dim=-1)
        colors = torch.cat([colors[keep_src]] + new_colors, dim=0)
        opacities = torch.cat([opacities[keep_src]] + new_opa, dim=0)
        radii_all = torch.cat([radii_kept] + new_radii, dim=0)
        if sh_coeffs is not None:
            sh_coeffs = torch.cat([sh_coeffs[keep_src]] + new_sh, dim=0)
    else:
        radii_all = radii_kept

    n_keep = int(keep_idx.numel())
    prune = torch.zeros(means.shape[0], dtype=torch.bool, device=device)
    if do_prune:
        if do_densify:
            # Official densification_postfix zeros max_radii2D before prune, so the
            # 20px test never fires in the same densify call. Applying it on the
            # accumulated radii deletes the splits we just added.
            prune = prune_mask_fastgs_densify(
                opacities,
                log_scales,
                vcp_kept,
                scene_scale,
                cfg,
                max_radii2d=None,
                size_threshold=size_threshold,
            )
        else:
            prune = prune_mask_vcp(
                vcp_kept if vcp_kept.numel() == means.shape[0] else torch.cat(
                    [
                        vcp_kept,
                        torch.zeros(means.shape[0] - vcp_kept.numel(), device=device),
                    ]
                ),
                opacities,
                step,
                cfg,
                log_scales=log_scales,
                scene_scale=scene_scale,
            )
            if prune.numel() != means.shape[0]:
                pad = torch.zeros(means.shape[0], dtype=torch.bool, device=device)
                pad[: prune.numel()] = prune
                prune = pad
    n_prune = int(prune.sum().item())
    tqdm.write(
        f"[FastGS step={step}] n_prune={n_prune} n_keep={n_keep} n_new={n_new} "
        f"opa_p50={float(torch.quantile(opa_sig, 0.5)):.4f}"
    )
    if prune.any():
        if prune.all():
            prune[int(torch.argmax(torch.sigmoid(opacities)).item())] = False
        surv = ~prune
        orig_surv = surv[:n_keep]
        new_surv = surv[n_keep:] if n_new > 0 else surv[:0]
        means = torch.cat([means[:n_keep][orig_surv], means[n_keep:][new_surv]], dim=0)
        log_scales = torch.cat(
            [log_scales[:n_keep][orig_surv], log_scales[n_keep:][new_surv]], dim=0
        )
        rotations = torch.cat(
            [rotations[:n_keep][orig_surv], rotations[n_keep:][new_surv]], dim=0
        )
        colors = torch.cat([colors[:n_keep][orig_surv], colors[n_keep:][new_surv]], dim=0)
        opacities = torch.cat(
            [opacities[:n_keep][orig_surv], opacities[n_keep:][new_surv]], dim=0
        )
        if sh_coeffs is not None:
            sh_coeffs = torch.cat(
                [sh_coeffs[:n_keep][orig_surv], sh_coeffs[n_keep:][new_surv]], dim=0
            )
        keep_idx = keep_idx[orig_surv]
        n_new = int(new_surv.sum().item()) if new_surv.numel() else 0

    # Official FastGS densify_and_prune_fastgs: cap opacity at 0.8 (inverse-sigmoid).
    clamped_opacity = False
    if do_densify:
        opa_prob = torch.sigmoid(opacities)
        opacities = torch.logit(torch.clamp(opa_prob, min=1e-6, max=0.8))
        clamped_opacity = True

    changed = means.shape[0] != n or n_new > 0 or bool(prune.any().item()) or clamped_opacity
    if changed:
        payload = {
            "means": means,
            "log_scales": log_scales,
            "rotations": rotations,
            "colors": colors,
            "opacities": opacities,
        }
        if sh_coeffs is not None:
            payload["sh_coeffs"] = sh_coeffs
        data.replace(payload)
        return True, keep_idx, n_new
    return False, orig_idx, 0


def densify_and_prune(
    data: GaussianData,
    grad_accum: torch.Tensor,
    vis_count: torch.Tensor,
    grad_thresh: float,
    prune_opacity_thresh: float,
    max_gaussians: int,
    split_scale_shrink: float,
    grow_scale3d: float = 0.01,
    prune_scale3d: float = 0.1,
    scene_scale: float = 1.0,
    cull_screen_size: float = 0.0,
    split_screen_size: float = 0.0,
    intrinsics: torch.Tensor = None,
    camera_to_world: torch.Tensor = None,
    height: int = 0,
    width: int = 0,
) -> bool:
    """Densify and prune following gsplat's DefaultStrategy."""
    with torch.no_grad():
        n = data.num_gaussians
        means = data.means.detach().clone()[:n]
        log_scales = data.log_scales.detach().clone()[:n]
        rotations = data.rotations.detach().clone()[:n]
        colors = data.colors.detach().clone()[:n]
        opacities = data.opacities.detach().clone()[:n]
        grad_accum = grad_accum.detach().clone()[:n]
        vis_count = vis_count.detach().clone()[:n]
        grad_norms = grad_accum / torch.clamp(vis_count, min=1.0)

        sizes = [
            means.shape[0],
            log_scales.shape[0],
            rotations.shape[0],
            colors.shape[0],
            opacities.shape[0],
            grad_norms.shape[0],
        ]
        if len(set(sizes)) != 1:
            print(
                f"SIZE MISMATCH in densify: means={sizes[0]} log_scales={sizes[1]} rotations={sizes[2]} colors={sizes[3]} opacities={sizes[4]} grad_norms={sizes[5]} n={n}"
            )

        assert means.shape[0] == n, f"means {means.shape[0]} != {n}"
        assert log_scales.shape[0] == n, f"log_scales {log_scales.shape[0]} != {n}"
        assert grad_norms.shape[0] == n, f"grad_norms {grad_norms.shape[0]} != {n}"
        n_before = means.shape[0]

        # --- Prune low-opacity Gaussians ---
        visible_opacities = torch.sigmoid(opacities)
        is_low_opa = visible_opacities < prune_opacity_thresh

        # --- Prune overly large Gaussians (3D scale) ---
        max_scale = torch.exp(log_scales).max(dim=-1).values
        is_too_big = max_scale > prune_scale3d * scene_scale

        # --- Screen-size pruning (nerfstudio: cull > 15% of screen) ---
        is_too_big_screen = torch.zeros(n, dtype=torch.bool, device=means.device)
        is_big_screen = torch.zeros(n, dtype=torch.bool, device=means.device)
        if (
            cull_screen_size > 0
            and intrinsics is not None
            and camera_to_world is not None
            and height > 0
        ):
            cov_matrices = data.covariance_matrices()[:n]
            proj_means, proj_covs, _, proj_mask = project_gaussians_3d_to_2d(
                means,
                cov_matrices,
                intrinsics,
                camera_to_world,
                near_plane=1e-4,
                min_covariance=1e-4,
            )
            if proj_mask.any():
                cov_xx = proj_covs[:, 0, 0]
                cov_xy = proj_covs[:, 0, 1]
                cov_yy = proj_covs[:, 1, 1]
                trace = cov_xx + cov_yy
                disc = torch.sqrt(
                    torch.clamp((cov_xx - cov_yy) ** 2 + 4.0 * cov_xy * cov_xy, min=0.0)
                )
                lambda_max = 0.5 * (trace + disc)
                screen_radius = 3.0 * torch.sqrt(torch.clamp(lambda_max, min=0.0))
                screen_frac = screen_radius / float(max(height, width))
                is_too_big_screen = (screen_frac > cull_screen_size) & proj_mask
                if split_screen_size > 0:
                    is_big_screen = (
                        (screen_frac > split_screen_size) & proj_mask & ~is_too_big_screen
                    )

        prune_mask = is_low_opa | is_too_big | is_too_big_screen
        # Always keep at least one
        if prune_mask.all():
            prune_mask[torch.argmax(visible_opacities)] = False
        keep_mask = ~prune_mask

        means = means[keep_mask]
        log_scales = log_scales[keep_mask]
        rotations = rotations[keep_mask]
        colors = colors[keep_mask]
        opacities = opacities[keep_mask]
        grad_norms = grad_norms[keep_mask]

        pruned_sizes = [
            means.shape[0],
            log_scales.shape[0],
            rotations.shape[0],
            colors.shape[0],
            opacities.shape[0],
            grad_norms.shape[0],
        ]
        if len(set(pruned_sizes)) != 1:
            min_n = min(pruned_sizes)
            means = means[:min_n]
            log_scales = log_scales[:min_n]
            rotations = rotations[:min_n]
            colors = colors[:min_n]
            opacities = opacities[:min_n]
            grad_norms = grad_norms[:min_n]

        assert log_scales.shape[0] == grad_norms.shape[0], (
            f"After pruning fix: log_scales={log_scales.shape[0]} grad_norms={grad_norms.shape[0]}"
        )

        # --- Grow: duplicate small, split large ---
        is_grad_high = grad_norms > grad_thresh
        max_scale_kept = torch.exp(log_scales).max(dim=-1).values
        is_small = max_scale_kept <= grow_scale3d * scene_scale
        is_large = ~is_small

        # Also split big screen-size gaussians (nerfstudio: split_screen_size=0.05)
        if split_screen_size > 0 and is_big_screen.numel() == is_grad_high.numel():
            is_grad_high = is_grad_high | is_big_screen

        dupli_mask = is_grad_high & is_small
        split_mask = is_grad_high & is_large

        capacity = max(0, max_gaussians - means.shape[0])
        n_new = (dupli_mask.sum() + split_mask.sum()).item()
        if n_new > capacity and n_new > 0:
            # Prefer high-gradient candidates under capacity constraints.
            split_idx_all = torch.where(split_mask)[0]
            dupli_idx_all = torch.where(dupli_mask)[0]

            split_scores = grad_norms[split_idx_all] if split_idx_all.numel() > 0 else None
            dupli_scores = grad_norms[dupli_idx_all] if dupli_idx_all.numel() > 0 else None

            chosen_split = torch.zeros_like(split_mask)
            chosen_dupli = torch.zeros_like(dupli_mask)

            # Each clone or split adds one extra Gaussian.
            remaining = int(capacity)
            if split_idx_all.numel() > 0 and remaining > 0:
                max_splits = min(split_idx_all.numel(), remaining)
                topk = torch.topk(split_scores, k=max_splits, largest=True).indices
                chosen_split[split_idx_all[topk]] = True
                remaining -= int(max_splits)

            if dupli_idx_all.numel() > 0 and remaining > 0:
                max_duplis = min(dupli_idx_all.numel(), remaining)
                topk = torch.topk(dupli_scores, k=max_duplis, largest=True).indices
                chosen_dupli[dupli_idx_all[topk]] = True

            split_mask = chosen_split
            dupli_mask = chosen_dupli

        n_dupli = dupli_mask.sum().item()
        n_split = split_mask.sum().item()

        # Collect new Gaussians (from split and duplicate)
        new_means = []
        new_log_scales = []
        new_rotations = []
        new_colors = []
        new_opacities = []

        # Split: create 2 smaller copies
        if n_split > 0:
            sel_scales = torch.exp(log_scales[split_mask])
            sel_quats = F.normalize(rotations[split_mask], dim=-1)
            w, x, y, z = sel_quats[:, 0], sel_quats[:, 1], sel_quats[:, 2], sel_quats[:, 3]
            rotmats = torch.stack(
                [
                    torch.stack(
                        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=1
                    ),
                    torch.stack(
                        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=1
                    ),
                    torch.stack(
                        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=1
                    ),
                ],
                dim=1,
            )
            noise = torch.randn(2, n_split, 3, device=means.device)
            samples = torch.einsum("nij,nj,bnj->bni", rotmats, sel_scales, noise)

            new_means.append((means[split_mask].unsqueeze(0) + samples).reshape(-1, 3))
            new_log_scales.append(split_child_log_scales(log_scales[split_mask], split_scale_shrink).repeat(2, 1))
            new_rotations.append(rotations[split_mask].repeat(2, 1))
            new_colors.append(colors[split_mask].repeat(2, 1))
            new_opacities.append(opacities[split_mask].repeat(2))

        # Duplicate: copy as-is
        if n_dupli > 0:
            new_means.append(means[dupli_mask])
            new_log_scales.append(log_scales[dupli_mask])
            new_rotations.append(rotations[dupli_mask])
            new_colors.append(colors[dupli_mask])
            new_opacities.append(opacities[dupli_mask])

        # Keep clone parents; replace split parents with two children.
        if n_split > 0 or n_dupli > 0:
            keep_mask = ~split_mask
            means = torch.cat([means[keep_mask]] + new_means, dim=0)
            log_scales = torch.cat([log_scales[keep_mask]] + new_log_scales, dim=0)
            rotations = torch.cat([rotations[keep_mask]] + new_rotations, dim=0)
            rotations = F.normalize(rotations, dim=-1)
            colors = torch.cat([colors[keep_mask]] + new_colors, dim=0)
            opacities = torch.cat([opacities[keep_mask]] + new_opacities, dim=0)

        changed = means.shape[0] != n_before
        if changed:
            payload = {
                "means": means,
                "log_scales": log_scales,
                "rotations": rotations,
                "colors": colors,
                "opacities": opacities,
            }
            if data.sh_coeffs is not None and _HAS_FASTGS:
                # Keep SH in sync with RGB densify by re-initting from colors.
                payload["sh_coeffs"] = init_sh_from_rgb(colors)
            data.replace(payload)
        return changed


def reset_opacities(data: GaussianData, value: float = 0.01):
    """Reset all opacities to the provided alpha value (in probability space)."""
    with torch.no_grad():
        value = max(min(float(value), 1.0 - 1e-6), 1e-6)
        reset_logit = torch.logit(
            torch.tensor(value, device=data.opacities.device, dtype=data.opacities.dtype)
        ).item()
        data.opacities.data.fill_(reset_logit)


def main():
    args = parse_args()
    if args.densify_grad_thresh is None:
        args.densify_grad_thresh = 0.0002 if args.fastgs else 8e-4
    if args.densify_grad_abs_thresh is None:
        args.densify_grad_abs_thresh = 0.0009
    if args.fastgs and args.split_scale_shrink == 0.8:
        args.split_scale_shrink = 1.6
    if args.fastgs and args.densify_every == 100:
        args.densify_every = 500
    if not _HAS_SSIM:
        raise RuntimeError("pytorch_msssim is required: pip install pytorch-msssim")
    device = resolve_device(args.device)
    if args.cache_images is None:
        args.cache_images = device == "mps"
    set_seed(args.seed)
    configure_torch_threads(args.torch_num_threads, args.torch_num_inter_op_threads)
    effective_viser_update_every = args.viser_update_every
    if device == "mps" and effective_viser_update_every == 10:
        effective_viser_update_every = 100

    scene_dir, all_frames, points3d = load_dataset_frames(args.dataset_json.resolve(), device)
    if not all_frames:
        raise ValueError("Dataset does not contain any frames.")
    if args.limit_frames > 0:
        all_frames = all_frames[: args.limit_frames]

    frames, eval_frames = split_train_eval(all_frames, args.eval_hold)
    scene_scale = scene_scale_from_frames(frames)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(
            tempfile.mkdtemp(prefix="tinysplat_json_train_", dir=args.dataset_json.resolve().parent)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.no_viser:
        visualizer = _NullVisualizer()
    else:
        visualizer = ViserVisualizer(port=args.viser_port)
        visualizer.update_status("**Status:** server started, initializing gaussians")

    prepared_frames: Optional[List[PreparedFrame]] = None
    if args.cache_images:
        prepared_frames = prepare_dataset_frames(
            frames,
            device=device,
            max_resolution=args.max_resolution,
        )

    if prepared_frames is not None:
        first_prepared = prepared_frames[0]
        first_target = first_prepared.image
        first_intrinsics = first_prepared.intrinsics
        first_height = first_prepared.height
        first_width = first_prepared.width
    else:
        first_target, first_intrinsics, first_height, first_width = load_frame_image(
            frames[0],
            device=device,
            max_resolution=args.max_resolution,
        )

    if points3d:
        gaussians = build_sparse_points_gaussians_3d(
            points3d=points3d,
            device=first_target.device,
            dtype=first_target.dtype,
        )
        init_voxel_size = None
    else:
        init_voxel_size = 0.1
        gaussians = build_pose_bbox_gaussians_3d(
            target=first_target,
            frames=frames,
            voxel_size=init_voxel_size,
        )
    gauss_data = GaussianData(gaussians, device, sh_degree=args.sh_degree)
    with torch.no_grad():
        init_max_scale = torch.exp(gauss_data.log_scales).max(dim=-1).values
        print(
            f"Init Gaussians: N={gauss_data.num_gaussians} "
            f"scale p50/p90/max="
            f"{float(torch.quantile(init_max_scale, 0.5)):.4f}/"
            f"{float(torch.quantile(init_max_scale, 0.9)):.4f}/"
            f"{float(init_max_scale.max()):.4f}"
        )

    def make_optimizers(step: int):
        return build_optimizers(
            gauss_data,
            scene_scale=scene_scale,
            step=step,
            max_steps=args.iterations,
            lr_scale=args.lr,
        )

    optimizers = make_optimizers(0)
    print(
        f"3DGS LRs (x{args.lr:g}): means={means_lr_at(0, scene_scale, args.iterations, args.lr):.3e} "
        f"-> {means_lr_at(args.iterations, scene_scale, args.iterations, args.lr):.3e}, "
        f"scales={_SCALES_LR * args.lr:g}, quats={_QUATS_LR * args.lr:g}, "
        f"opacity={_OPACITY_LR * args.lr:g}, sh={_SH_LR * args.lr:g}"
    )
    strategy = None
    strategy_state = None
    strategy_params = None
    use_fastgs = bool(args.fastgs)
    if use_fastgs and not _HAS_FASTGS:
        raise RuntimeError("--fastgs requires tinysplat.fastgs / metal_backend (install legacy package).")
    if use_fastgs and not metal_available():
        print("Warning: Metal dylib unavailable; FastGS footprint counts may fall back / fail.")
    fastgs_cfg = None
    if use_fastgs:
        fastgs_cfg = FastGSConfig(
            densify_every=args.densify_every if args.densify_every > 0 else 500,
            densify_until=args.densify_until if args.densify_until > 0 else 15000,
            grad_thresh=args.densify_grad_thresh,
            grad_abs_thresh=args.densify_grad_abs_thresh,
            split_scale_shrink=args.split_scale_shrink,
            grow_scale3d=0.001,
            error_tau=0.1,
            cull_screen_size=0.0,
            ssim_lambda=args.ssim_lambda,
        )
        # FastGS replaces gsplat DefaultStrategy.
        strategy = None
        print(
            f"FastGS enabled: densify_every={fastgs_cfg.densify_every}, "
            f"K={fastgs_cfg.k_views}, τ_d={fastgs_cfg.densify_score_thresh}, "
            f"error_tau={fastgs_cfg.error_tau}, grow_scale3d={fastgs_cfg.grow_scale3d}, "
            f"grad_thresh={fastgs_cfg.grad_thresh}, grad_abs_thresh={fastgs_cfg.grad_abs_thresh}, "
            f"split_shrink={fastgs_cfg.split_scale_shrink}, "
            f"scene_scale={scene_scale:.4f}, metal={metal_available()}, sh_degree={args.sh_degree}"
        )
    elif _HAS_GSPLAT_STRATEGY:
        strategy = DefaultStrategy(
            prune_opa=args.prune_opacity_thresh,
            grow_grad2d=args.densify_grad_thresh,
            grow_scale3d=0.01,
            grow_scale2d=args.split_screen_size,
            prune_scale3d=0.1,
            prune_scale2d=args.cull_screen_size,
            refine_scale2d_stop_iter=args.densify_until if args.densify_until > 0 else 0,
            refine_start_iter=args.densify_from,
            refine_stop_iter=args.densify_until if args.densify_until > 0 else max(args.iterations, 1 << 30),
            reset_every=args.reset_opacity_every if args.reset_opacity_every > 0 else (1 << 30),
            refine_every=args.densify_every if args.densify_every > 0 else (1 << 30),
            pause_refine_after_reset=len(frames) + (args.densify_every if args.densify_every > 0 else 0),
            absgrad=False,
            revised_opacity=True,
            verbose=False,
            key_for_gradient="means2d",
        )
        strategy_params = {
            "means": gauss_data.means,
            "scales": gauss_data.log_scales,
            "quats": gauss_data.rotations,
            "colors": gauss_data.colors,
            "opacities": gauss_data.opacities,
        }
        strategy.check_sanity(strategy_params, optimizers)
        strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    # Fallback densify stats when gsplat.DefaultStrategy is unavailable.
    grad_accum = torch.zeros(gauss_data.num_gaussians, device=gauss_data.device)
    absgrad_accum = torch.zeros(gauss_data.num_gaussians, device=gauss_data.device)
    grad3d_accum = torch.zeros(gauss_data.num_gaussians, device=gauss_data.device)
    vis_count = torch.zeros(gauss_data.num_gaussians, device=gauss_data.device)
    max_radii2d = torch.zeros(gauss_data.num_gaussians, device=gauss_data.device)

    visualizer.set_cameras(frames)
    with torch.no_grad():
        initial_render = gauss_data.render(
            intrinsics=first_intrinsics,
            camera_to_world=frames[0].camera_to_world,
            height=first_height,
            width=first_width,
        ).detach()
        snap = gauss_data.snapshot_for_visualizer()
        visualizer.update_gaussians(
            snap["means"],
            snap["colors"],
            snap["opacities"],
            snap["covariances"],
        )
    visualizer.update_gaussian_stats(gauss_data.num_gaussians)
    visualizer.update_status(f"**Status:** training live at http://localhost:{args.viser_port}")
    visualizer.update_frame_preview(0, frames[0], first_target, initial_render)

    save_image(first_target, output_dir / "target_frame0.png")
    save_image(initial_render, output_dir / "render_frame0_initial.png")

    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset_json.resolve()}")
    print(f"Scene dir: {scene_dir}")
    print(f"Train frames: {len(frames)}")
    print(f"Eval frames: {len(eval_frames)} (eval_hold={args.eval_hold})")
    print(f"Resolution: {first_width}x{first_height}")
    print(f"Initial gaussians: {gauss_data.num_gaussians}")
    if init_voxel_size is None:
        print(f"Initialization: sparse COLMAP points3D ({len(points3d)} points)")
    else:
        print(f"Initial bbox voxel size (m): {init_voxel_size}")
    print(f"Max resolution: {args.max_resolution or 'original'}")
    print(f"Output directory: {output_dir}")
    if not args.no_viser:
        print(f"Viser: http://localhost:{args.viser_port}")
    else:
        print("Viser: disabled")
    print(f"Torch threads: {torch.get_num_threads()}")
    print(f"Cache images: {args.cache_images}")
    print(f"Viser update every: {effective_viser_update_every}")
    print(f"FastGS: {use_fastgs}")
    print(f"SH max degree: {args.sh_degree}")
    print(f"Metal available: {metal_available()}")

    progress = tqdm(range(args.iterations), desc="Training", unit="iter")
    last_loss = None
    for step in progress:
        visualizer.wait_if_paused()
        gauss_data.set_sh_degree_for_step(step + 1)

        # Resolution schedule: nerfstudio starts at 1/2^d and doubles every resolution_schedule steps
        downscale = max(0, args.num_downscales - step // args.resolution_schedule)
        schedule_max_res = (args.max_resolution or max(first_height, first_width)) // (2**downscale)

        if prepared_frames is not None:
            prepared = prepared_frames[random.randrange(len(prepared_frames))]
            frame = prepared.frame
            target_raw = prepared.image
            intrinsics_raw = prepared.intrinsics
            height_raw = prepared.height
            width_raw = prepared.width
            if downscale > 0:
                target_raw = (
                    F.interpolate(
                        target_raw.permute(2, 0, 1).unsqueeze(0),
                        scale_factor=1.0 / (2**downscale),
                        mode="area",
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                )
                intrinsics_raw = intrinsics_raw.clone()
                intrinsics_raw[:2, :3] /= 2**downscale
                height_raw, width_raw = target_raw.shape[:2]
            target, intrinsics, height, width = target_raw, intrinsics_raw, height_raw, width_raw
        else:
            frame = frames[random.randrange(len(frames))]
            target, intrinsics, height, width = load_frame_image(
                frame, device=device, max_resolution=schedule_max_res
            )

        zero_optimizers(optimizers)
        rendered = gauss_data.render(
            intrinsics=intrinsics,
            camera_to_world=frame.camera_to_world,
            height=height,
            width=width,
        )

        # Combined photometric loss: (1-lambda)*L1 + lambda*(1-SSIM).
        l1_loss = F.l1_loss(rendered, target)
        mse_loss = F.mse_loss(rendered, target)
        ssim_loss = 1.0 - _compute_ssim(
            rendered.permute(2, 0, 1).unsqueeze(0),
            target.permute(2, 0, 1).unsqueeze(0),
            data_range=1.0,
            size_average=True,
        )
        loss = (1.0 - args.ssim_lambda) * l1_loss + args.ssim_lambda * ssim_loss

        info = get_last_render_info()
        if strategy is not None and isinstance(info, dict):
            strategy.step_pre_backward(strategy_params, optimizers, strategy_state, step + 1, info)

        loss.backward()

        prev_n = gauss_data.num_gaussians
        if strategy is not None and isinstance(info, dict):
            strategy.step_post_backward(strategy_params, optimizers, strategy_state, step + 1, info, packed=False)
            gauss_data.sync_from_strategy_params(strategy_params)
        elif strategy is None and (args.densify_every > 0 or use_fastgs):
            n = gauss_data.num_gaussians
            if grad_accum.numel() != n:
                grad_accum = torch.zeros(n, device=gauss_data.device)
                absgrad_accum = torch.zeros(n, device=gauss_data.device)
                grad3d_accum = torch.zeros(n, device=gauss_data.device)
                vis_count = torch.zeros(n, device=gauss_data.device)
                max_radii2d = torch.zeros(n, device=gauss_data.device)
            radii = last_radii2d() if use_fastgs else None
            vis = None
            if radii is not None and radii.shape[0] == n:
                radii = radii.to(device=gauss_data.device, dtype=max_radii2d.dtype)
                vis = radii > 0
                max_radii2d = torch.where(
                    vis, torch.maximum(max_radii2d, radii), max_radii2d
                )
            g2d = last_grad_means2d() if use_fastgs else None
            g2d_abs = last_grad_means2d_abs() if use_fastgs else None
            if g2d is not None and g2d.shape[0] == n:
                g2d = g2d.to(device=gauss_data.device, dtype=gauss_data.means.dtype)
                g2d = torch.nan_to_num(g2d, nan=0.0, posinf=0.0, neginf=0.0)
                # Inria/FastGS store dL/dmean2D in NDC (pixel grad × 0.5*{W,H}).
                ndc = g2d.new_tensor([0.5 * float(width), 0.5 * float(height)])
                g2d = g2d * ndc
                g_norm = g2d.norm(dim=-1)
                if g2d_abs is not None and g2d_abs.shape[0] == n:
                    g2d_abs = g2d_abs.to(device=gauss_data.device, dtype=gauss_data.means.dtype)
                    g2d_abs = torch.nan_to_num(g2d_abs, nan=0.0, posinf=0.0, neginf=0.0)
                    g_abs = (g2d_abs * ndc).norm(dim=-1)
                else:
                    g_abs = g2d.abs().sum(dim=-1)
                # Official visibility_filter is radii>0, not "any nonzero grad".
                if vis is None:
                    vis = g_norm > 0
                vis_f = vis.to(dtype=g_norm.dtype)
                grad_accum = torch.where(vis, grad_accum + g_norm, grad_accum)
                absgrad_accum = torch.where(vis, absgrad_accum + g_abs, absgrad_accum)
                vis_count = vis_count + vis_f
            elif gauss_data.means.grad is not None:
                g_norm = gauss_data.means.grad.detach().norm(dim=-1)
                vis_m = g_norm > 0
                grad_accum = torch.where(vis_m, grad_accum + g_norm, grad_accum)
                vis_count = vis_count + vis_m.to(dtype=vis_count.dtype)
            if gauss_data.means.grad is not None:
                grad3d_accum += torch.nan_to_num(
                    gauss_data.means.grad.detach().norm(dim=-1), nan=0.0, posinf=0.0, neginf=0.0
                )

        # Official FastGS: densify/prune in no_grad before the Adam step.
        step_idx = step + 1
        if use_fastgs and fastgs_cfg is not None:
            # Official FastGS: densify while iteration < densify_until (last event 14500).
            densify_now = (
                step_idx >= args.densify_from
                and step_idx < fastgs_cfg.densify_until
                and step_idx % fastgs_cfg.densify_every == 0
            )
            prune_now = (
                step_idx > fastgs_cfg.densify_until
                and step_idx < args.iterations
                and step_idx % fastgs_cfg.prune_every_late == 0
            )
            if densify_now or prune_now:
                size_th = None
                if (
                    args.reset_opacity_every > 0
                    and step_idx > args.reset_opacity_every
                ):
                    size_th = fastgs_cfg.max_screen_size
                changed, keep_idx, n_new = densify_and_prune_fastgs(
                    gauss_data,
                    frames,
                    grad_accum=grad_accum,
                    vis_count=vis_count,
                    cfg=fastgs_cfg,
                    step=step_idx,
                    max_gaussians=args.max_gaussians,
                    max_resolution=schedule_max_res,
                    ssim_fn=_compute_ssim if _HAS_SSIM else None,
                    do_densify=densify_now,
                    do_prune=True,
                    scene_scale=scene_scale,
                    absgrad_accum=absgrad_accum,
                    grad3d_accum=grad3d_accum,
                    max_radii2d=max_radii2d,
                    size_threshold=size_th,
                )
                if changed:
                    optimizers = remap_optimizers_after_densify(
                        optimizers,
                        gauss_data,
                        keep_idx,
                        n_new,
                        step_idx,
                        scene_scale,
                        args.iterations,
                        args.lr,
                    )
                    if densify_now:
                        # Official replace_tensor_to_optimizer after the 0.8 opacity cap.
                        reset_opacity_adam(optimizers)
                n = gauss_data.num_gaussians
                grad_accum = torch.zeros(n, device=gauss_data.device)
                absgrad_accum = torch.zeros(n, device=gauss_data.device)
                grad3d_accum = torch.zeros(n, device=gauss_data.device)
                vis_count = torch.zeros(n, device=gauss_data.device)
                max_radii2d = torch.zeros(n, device=gauss_data.device)
            # Official resets opacity only inside the densify phase (not after 15k).
            if (
                args.reset_opacity_every > 0
                and step_idx >= args.densify_from
                and step_idx < fastgs_cfg.densify_until
                and step_idx % args.reset_opacity_every == 0
            ):
                reset_opacities(gauss_data, value=0.01)
                reset_opacity_adam(optimizers)

        do_opt = True
        if use_fastgs and fastgs_cfg is not None:
            do_opt = should_step_optimizer(step_idx, fastgs_cfg)
        if do_opt:
            scale_sh_rest_grads(gauss_data)
            set_optimizer_lr(
                optimizers["means"],
                means_lr_at(step_idx, scene_scale, args.iterations, args.lr),
            )
            step_optimizers(optimizers)
            sanitize_gaussians(gauss_data, scene_scale)

        if (not use_fastgs) and strategy is None and args.densify_every > 0:
            if (
                step_idx >= args.densify_from
                and (args.densify_until <= 0 or step_idx <= args.densify_until)
                and step_idx % args.densify_every == 0
            ):
                changed = densify_and_prune(
                    gauss_data,
                    grad_accum=grad_accum,
                    vis_count=vis_count,
                    grad_thresh=args.densify_grad_thresh,
                    prune_opacity_thresh=args.prune_opacity_thresh,
                    max_gaussians=args.max_gaussians,
                    split_scale_shrink=args.split_scale_shrink,
                    scene_scale=scene_scale,
                    cull_screen_size=args.cull_screen_size,
                    split_screen_size=args.split_screen_size,
                    intrinsics=intrinsics,
                    camera_to_world=frame.camera_to_world,
                    height=height,
                    width=width,
                )
                if changed:
                    optimizers = make_optimizers(step_idx)
                n = gauss_data.num_gaussians
                grad_accum = torch.zeros(n, device=gauss_data.device)
                vis_count = torch.zeros(n, device=gauss_data.device)
            if (
                args.reset_opacity_every > 0
                and step_idx >= args.densify_from
                and step_idx % args.reset_opacity_every == 0
            ):
                reset_opacities(gauss_data, value=0.01)

        if strategy is not None and gauss_data.num_gaussians > args.max_gaussians:
            keep = args.max_gaussians
            idx = torch.randperm(gauss_data.num_gaussians, device=gauss_data.device)[:keep]
            strategy_params["means"] = torch.nn.Parameter(strategy_params["means"].detach()[idx].contiguous(), requires_grad=True)
            strategy_params["scales"] = torch.nn.Parameter(strategy_params["scales"].detach()[idx].contiguous(), requires_grad=True)
            strategy_params["quats"] = torch.nn.Parameter(strategy_params["quats"].detach()[idx].contiguous(), requires_grad=True)
            strategy_params["colors"] = torch.nn.Parameter(strategy_params["colors"].detach()[idx].contiguous(), requires_grad=True)
            strategy_params["opacities"] = torch.nn.Parameter(strategy_params["opacities"].detach()[idx].contiguous(), requires_grad=True)
            gauss_data.sync_from_strategy_params(strategy_params)
            optimizers = make_optimizers(step + 1)
            if strategy is not None:
                strategy_state = strategy.initialize_state(scene_scale=scene_scale)
            print(f"Capped gaussians at step {step + 1}: {gauss_data.num_gaussians} -> {keep}")

        if gauss_data.num_gaussians != prev_n:
            print(f"Densified at step {step + 1}: {gauss_data.num_gaussians} gaussians")
            with torch.no_grad():
                snap = gauss_data.snapshot_for_visualizer()
                visualizer.update_gaussians(
                    snap["means"],
                    snap["colors"],
                    snap["opacities"],
                    snap["covariances"],
                )
            visualizer.update_gaussian_stats(gauss_data.num_gaussians)
            visualizer.update_status(
                f"**Status:** training live at http://localhost:{args.viser_port} (densified/pruned)"
            )

        last_loss = loss.detach()
        psnr = -10.0 * torch.log10(mse_loss.detach() + 1e-10)

        ssim_val = 0.0
        if _HAS_SSIM:
            ssim_val = float((1.0 - ssim_loss).detach().item())

        progress.set_postfix(
            frame=frame.image_id,
            loss=f"{last_loss.item():.6f}",
            psnr=f"{psnr.item():.2f}",
            ssim=f"{ssim_val:.4f}",
        )

        visualizer.update_step(step + 1, last_loss.item(), psnr.item(), frame.image_id)
        if effective_viser_update_every and (step + 1) % effective_viser_update_every == 0:
            with torch.no_grad():
                snap = gauss_data.snapshot_for_visualizer()
                visualizer.update_gaussians(
                    snap["means"],
                    snap["colors"],
                    snap["opacities"],
                    snap["covariances"],
                )
            visualizer.update_gaussian_stats(gauss_data.num_gaussians)

        # Densify/prune/reset are handled by gsplat.DefaultStrategy in step_post_backward.
        if visualizer.should_render_selected_frame(step + 1, effective_viser_update_every):
            selected_idx = min(visualizer.selected_frame_idx, len(frames) - 1)
            if prepared_frames is not None:
                selected_prepared = prepared_frames[selected_idx]
                selected_frame = selected_prepared.frame
                selected_target = selected_prepared.image
                selected_intrinsics = selected_prepared.intrinsics
                selected_height = selected_prepared.height
                selected_width = selected_prepared.width
            else:
                selected_frame = frames[selected_idx]
                selected_target, selected_intrinsics, selected_height, selected_width = (
                    load_frame_image(
                        selected_frame, device=device, max_resolution=args.max_resolution
                    )
                )
            with torch.no_grad():
                selected_render = gauss_data.render(
                    intrinsics=selected_intrinsics,
                    camera_to_world=selected_frame.camera_to_world,
                    height=selected_height,
                    width=selected_width,
                ).detach()
            visualizer.update_frame_preview(
                selected_idx,
                selected_frame,
                selected_target,
                selected_render,
            )

        if args.eval_every and (step + 1) % args.eval_every == 0:
            with torch.no_grad():
                eval_render = gauss_data.render(
                    intrinsics=first_intrinsics,
                    camera_to_world=frames[0].camera_to_world,
                    height=first_height,
                    width=first_width,
                ).detach()
            save_image(eval_render, output_dir / f"render_frame0_step_{step + 1:05d}.png")

    with torch.no_grad():
        final_render = gauss_data.render(
            intrinsics=first_intrinsics,
            camera_to_world=frames[0].camera_to_world,
            height=first_height,
            width=first_width,
        ).detach()
    final_loss = F.mse_loss(final_render, first_target)
    final_psnr = -10.0 * torch.log10(final_loss + 1e-10)

    save_image(final_render, output_dir / "render_frame0_final.png")
    save_image(torch.cat([first_target, final_render], dim=1), output_dir / "comparison_frame0.png")
    save_checkpoint(gauss_data, output_dir / "gaussians.pt")
    save_ply(gauss_data, output_dir / "point_cloud.ply")
    print(f"Exported PLY: {output_dir / 'point_cloud.ply'} ({gauss_data.num_gaussians} gaussians)")

    if eval_frames:
        metrics = evaluate_heldout(
            gauss_data,
            eval_frames,
            device=device,
            max_resolution=args.max_resolution,
        )
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(
            f"Held-out ({int(metrics['num_views'])} views): "
            f"PSNR={metrics['psnr']:.3f}  SSIM={metrics['ssim']:.4f}  "
            f"LPIPS={metrics['lpips']:.4f}"
        )
        print(f"Wrote {metrics_path}")

    with torch.no_grad():
        snap = gauss_data.snapshot_for_visualizer()
        visualizer.update_gaussians(
            snap["means"],
            snap["colors"],
            snap["opacities"],
            snap["covariances"],
        )
    visualizer.update_gaussian_stats(gauss_data.num_gaussians)
    visualizer.update_status("**Status:** training complete")
    visualizer.update_frame_preview(0, frames[0], first_target, final_render)

    print("Training complete.")
    print(
        f"Last sampled-frame loss: {last_loss.item():.6f}"
        if last_loss is not None
        else "No steps run."
    )
    print(f"Frame-0 loss: {final_loss.item():.6f}")
    print(f"Frame-0 PSNR: {final_psnr.item():.2f} dB")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
