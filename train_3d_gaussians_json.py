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
import torch.nn as nn
import torch.nn.functional as F

import viser
from tqdm.auto import tqdm

from tinysplat import gaussian_splat_3d
from tinysplat.gaussian_splat_3d_core import project_gaussians_3d_to_2d


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
        self.server = viser.ViserServer(port=port)
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
    parser.add_argument("--lr", type=float, default=5e-2, help="Adam learning rate.")
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
        default=64,
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
        help="Run densify/prune every N steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--densify-from", type=int, default=50, help="Start densification after this many steps."
    )
    parser.add_argument(
        "--densify-until",
        type=int,
        default=-1,
        help="Stop densification after this many steps. Use -1 to run until the end.",
    )
    parser.add_argument(
        "--densify-grad-thresh",
        type=float,
        default=1e-6,
        help="Split/duplicate gaussians whose mean gradient norm exceeds this value.",
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
        default=50000,
        help="Maximum number of gaussians after densification.",
    )
    parser.add_argument(
        "--split-scale-shrink",
        type=float,
        default=0.8,
        help="Scale shrink factor applied to split gaussians.",
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
        action="store_true",
        help="Preload and cache original-resolution training images in memory for faster CPU training.",
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
    depth: float,
    intrinsics: torch.Tensor,
    camera_to_world: torch.Tensor,
) -> torch.Tensor:
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    x_cam = (pixel_centers[:, 0] - cx) * depth / fx
    y_cam = (pixel_centers[:, 1] - cy) * depth / fy
    z_cam = torch.full_like(x_cam, depth)
    points_camera = torch.stack([x_cam, y_cam, z_cam], dim=1)

    rotation = camera_to_world[:3, :3]
    translation = camera_to_world[:3, 3]
    return points_camera @ rotation.transpose(0, 1) + translation


def build_pixel_gaussians_3d(
    target: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_to_world: torch.Tensor,
    init_grid_long_side: int,
) -> Dict[str, torch.Tensor]:
    height, width, channels = target.shape
    grid_height, grid_width = compute_initial_grid_shape(
        height=height,
        width=width,
        long_side_limit=init_grid_long_side,
    )
    dtype = target.dtype
    device = target.device

    y_coords = (
        (torch.arange(grid_height, device=device, dtype=dtype) + 0.5) * (height / grid_height) - 0.5
    ).clamp(0, height - 1)
    x_coords = (
        (torch.arange(grid_width, device=device, dtype=dtype) + 0.5) * (width / grid_width) - 0.5
    ).clamp(0, width - 1)

    pixel_centers = torch.stack(
        [
            x_coords.repeat(grid_height),
            y_coords.repeat_interleave(grid_width),
        ],
        dim=1,
    )

    pooled_input = target.permute(2, 0, 1).unsqueeze(0)
    if target.device.type == "mps":
        pooled = F.adaptive_avg_pool2d(
            pooled_input.cpu(),
            output_size=(grid_height, grid_width),
        ).to(device)
    else:
        pooled = F.adaptive_avg_pool2d(
            pooled_input,
            output_size=(grid_height, grid_width),
        )
    colors = pooled.squeeze(0).permute(1, 2, 0).reshape(-1, channels).contiguous()

    initial_depth = 3.0
    means = backproject_pixels_to_world(
        pixel_centers=pixel_centers,
        depth=initial_depth,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
    )

    pixel_size_x = initial_depth / intrinsics[0, 0]
    pixel_size_y = initial_depth / intrinsics[1, 1]
    patch_scale_x = pixel_size_x * max(width / grid_width, 1.0) * 0.5
    patch_scale_y = pixel_size_y * max(height / grid_height, 1.0) * 0.5
    patch_scale_z = 0.05

    log_scales = torch.stack(
        [
            torch.full(
                (grid_height * grid_width,),
                math.log(float(patch_scale_x)),
                device=device,
                dtype=dtype,
            ),
            torch.full(
                (grid_height * grid_width,),
                math.log(float(patch_scale_y)),
                device=device,
                dtype=dtype,
            ),
            torch.full(
                (grid_height * grid_width,),
                math.log(float(patch_scale_z)),
                device=device,
                dtype=dtype,
            ),
        ],
        dim=1,
    )

    initial_alpha = 0.9
    initial_opacity_logit = torch.logit(
        torch.tensor(initial_alpha, device=device, dtype=dtype)
    ).item()
    opacity_logits = torch.full(
        (grid_height * grid_width,),
        initial_opacity_logit,
        device=device,
        dtype=dtype,
    )

    num_gaussians = grid_height * grid_width
    rotations = torch.zeros(num_gaussians, 4, device=device, dtype=dtype)
    rotations[:, 0] = 1.0

    return {
        "means": means,
        "log_scales": log_scales,
        "rotations": rotations,
        "colors": colors,
        "opacities": opacity_logits,
    }


def load_dataset_frames(dataset_json: Path, device: str) -> Tuple[Path, List[FrameSample]]:
    dataset = json.loads(dataset_json.read_text(encoding="utf-8"))
    scene_dir = Path(dataset["scene_dir"])
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
    return scene_dir, frames


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


class MultiViewGaussianSplatTrainer(nn.Module):
    def __init__(self, gaussians: Dict[str, torch.Tensor], device: str):
        super().__init__()
        self.device_obj = torch.device(device)
        self.register_parameter("means", nn.Parameter(gaussians["means"].to(self.device_obj)))
        self.register_parameter(
            "log_scales", nn.Parameter(gaussians["log_scales"].to(self.device_obj))
        )
        self.register_parameter(
            "rotations", nn.Parameter(gaussians["rotations"].to(self.device_obj))
        )
        self.register_parameter("colors", nn.Parameter(gaussians["colors"].to(self.device_obj)))
        self.register_parameter(
            "opacities", nn.Parameter(gaussians["opacities"].to(self.device_obj))
        )

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
        epsilon = torch.eye(3, device=self.device_obj).unsqueeze(0) * 1e-6
        return covariance + epsilon

    def num_gaussians(self) -> int:
        return int(self.means.shape[0])

    def replace_gaussians(self, gaussians: Dict[str, torch.Tensor]):
        self.means = nn.Parameter(gaussians["means"].to(self.device_obj))
        self.log_scales = nn.Parameter(gaussians["log_scales"].to(self.device_obj))
        self.rotations = nn.Parameter(gaussians["rotations"].to(self.device_obj))
        self.colors = nn.Parameter(gaussians["colors"].to(self.device_obj))
        self.opacities = nn.Parameter(gaussians["opacities"].to(self.device_obj))

    def export_gaussians(self) -> Dict[str, torch.Tensor]:
        return {
            "means": self.means.detach(),
            "log_scales": self.log_scales.detach(),
            "rotations": self.rotations.detach(),
            "colors": self.colors.detach(),
            "opacities": self.opacities.detach(),
        }

    def render(
        self,
        intrinsics: torch.Tensor,
        camera_to_world: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        return gaussian_splat_3d(
            means=self.means,
            covariances=self.covariance_matrices(),
            colors=torch.clamp(self.colors, 0.0, 1.0),
            opacities=torch.sigmoid(self.opacities),
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            height=height,
            width=width,
            device=self.device_obj.type,
        )


def save_checkpoint(
    model: MultiViewGaussianSplatTrainer,
    output_path: Path,
):
    torch.save(
        {
            "means": model.means.detach().cpu(),
            "log_scales": model.log_scales.detach().cpu(),
            "rotations": model.rotations.detach().cpu(),
            "colors": model.colors.detach().cpu(),
            "opacities": model.opacities.detach().cpu(),
        },
        output_path,
    )


def rebuild_optimizer(model: MultiViewGaussianSplatTrainer, lr: float):
    return torch.optim.Adam(model.parameters(), lr=lr)


def densify_and_prune(
    model: MultiViewGaussianSplatTrainer,
    grad_norms: torch.Tensor,
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
        n = model.num_gaussians()
        means = model.means.detach().clone()[:n]
        log_scales = model.log_scales.detach().clone()[:n]
        rotations = model.rotations.detach().clone()[:n]
        colors = model.colors.detach().clone()[:n]
        opacities = model.opacities.detach().clone()[:n]
        grad_norms = grad_norms.detach().clone()[:n]

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
                f"SIZE MISMATCH in densify: means={sizes[0]} log_scales={sizes[1]} rotations={sizes[2]} colors={sizes[3]} opacities={sizes[4]} grad_norms={sizes[5]} model.n={n}"
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
            cov_matrices = model.covariance_matrices()[:n]
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
        n_new = (dupli_mask.sum() * 1 + split_mask.sum() * 2).item()
        if n_new > capacity and n_new > 0:
            ratio = capacity / n_new
            n_dupli = int(dupli_mask.sum().item() * ratio)
            n_split = int(split_mask.sum().item() * ratio)
            dupli_idx = torch.where(dupli_mask)[0][:n_dupli]
            split_idx = torch.where(split_mask)[0][:n_split]
            dupli_mask = torch.zeros_like(dupli_mask)
            split_mask = torch.zeros_like(split_mask)
            if n_dupli > 0:
                dupli_mask[dupli_idx] = True
            if n_split > 0:
                split_mask[split_idx] = True

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
            new_log_scales.append(torch.log(torch.exp(log_scales[split_mask]) / 1.6).repeat(2, 1))
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

        # Remove split/dupli Gaussians, add all new ones
        if n_split > 0 or n_dupli > 0:
            keep_mask = ~(dupli_mask | split_mask)
            means = torch.cat([means[keep_mask]] + new_means, dim=0)
            log_scales = torch.cat([log_scales[keep_mask]] + new_log_scales, dim=0)
            rotations = torch.cat([rotations[keep_mask]] + new_rotations, dim=0)
            colors = torch.cat([colors[keep_mask]] + new_colors, dim=0)
            opacities = torch.cat([opacities[keep_mask]] + new_opacities, dim=0)

        changed = means.shape[0] != n_before
        if changed:
            model.replace_gaussians(
                {
                    "means": means,
                    "log_scales": log_scales,
                    "rotations": rotations,
                    "colors": colors,
                    "opacities": opacities,
                }
            )
        return changed


def reset_opacities(model: MultiViewGaussianSplatTrainer, value: float = 0.01):
    """Reset opacities to prevent Gaussians from becoming permanently opaque/transparent."""
    with torch.no_grad():
        new_opacities = torch.clamp(
            model.opacities.data,
            max=torch.logit(torch.tensor(value)).item(),
        )
        model.opacities.data.copy_(new_opacities)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)
    configure_torch_threads(args.torch_num_threads, args.torch_num_inter_op_threads)
    effective_viser_update_every = args.viser_update_every
    if device == "mps" and effective_viser_update_every == 10:
        effective_viser_update_every = 100

    scene_dir, frames = load_dataset_frames(args.dataset_json.resolve(), device)
    if not frames:
        raise ValueError("Dataset does not contain any frames.")
    if args.limit_frames > 0:
        frames = frames[: args.limit_frames]

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(
            tempfile.mkdtemp(prefix="tinysplat_json_train_", dir=args.dataset_json.resolve().parent)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

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

    gaussians = build_pixel_gaussians_3d(
        target=first_target,
        intrinsics=first_intrinsics,
        camera_to_world=frames[0].camera_to_world,
        init_grid_long_side=args.init_grid_long_side,
    )
    model = MultiViewGaussianSplatTrainer(gaussians=gaussians, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    visualizer.set_cameras(frames)
    with torch.no_grad():
        initial_render = model.render(
            intrinsics=first_intrinsics,
            camera_to_world=frames[0].camera_to_world,
            height=first_height,
            width=first_width,
        ).detach()
        visualizer.update_gaussians(
            model.means,
            torch.clamp(model.colors, 0.0, 1.0),
            torch.sigmoid(model.opacities),
            model.covariance_matrices(),
        )
    visualizer.update_gaussian_stats(model.num_gaussians())
    visualizer.update_status(f"**Status:** training live at http://localhost:{args.viser_port}")
    visualizer.update_frame_preview(0, frames[0], first_target, initial_render)

    save_image(first_target, output_dir / "target_frame0.png")
    save_image(initial_render, output_dir / "render_frame0_initial.png")

    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset_json.resolve()}")
    print(f"Scene dir: {scene_dir}")
    print(f"Frames: {len(frames)}")
    print(f"Resolution: {first_width}x{first_height}")
    print(f"Initial gaussians: {model.means.shape[0]}")
    print(f"Initial grid long side cap: {args.init_grid_long_side}")
    print(f"Max resolution: {args.max_resolution or 'original'}")
    print(f"Output directory: {output_dir}")
    print(f"Viser: http://localhost:{args.viser_port}")
    print(f"Torch threads: {torch.get_num_threads()}")
    print(f"Cache images: {args.cache_images}")
    print(f"Viser update every: {effective_viser_update_every}")

    progress = tqdm(range(args.iterations), desc="Training", unit="iter")
    last_loss = None
    for step in progress:
        visualizer.wait_if_paused()

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

        optimizer.zero_grad()
        rendered = model.render(
            intrinsics=intrinsics,
            camera_to_world=frame.camera_to_world,
            height=height,
            width=width,
        )

        # L1 loss (nerfstudio uses L1 instead of MSE)
        Ll1 = torch.abs(rendered - target).mean()

        # SSIM loss if available and enabled
        simloss = torch.tensor(0.0, device=device)
        if args.ssim_lambda > 0:
            try:
                from pytorch_msssim import ssim as compute_ssim

                ssim_val = compute_ssim(
                    rendered.permute(2, 0, 1).unsqueeze(0),
                    target.permute(2, 0, 1).unsqueeze(0),
                    data_range=1.0,
                    size_average=True,
                )
                simloss = 1.0 - ssim_val
            except ImportError:
                pass

        # nerfstudio default: (1 - ssim_lambda) * L1 + ssim_lambda * SSIM
        loss = (1.0 - args.ssim_lambda) * Ll1 + args.ssim_lambda * simloss

        # Scale regularization (PhysGaussian): penalize huge spikey gaussians
        if args.use_scale_regularization and step % 10 == 0:
            scale_exp = torch.exp(model.log_scales)
            max_min_ratio = scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1).clamp(min=1e-8)
            scale_reg = torch.clamp(max_min_ratio - args.max_gauss_ratio, min=0.0).mean()
            loss = loss + 0.1 * scale_reg

        loss.backward()
        grad_norms = model.means.grad.detach().norm(dim=1)
        optimizer.step()

        last_loss = loss.detach()
        psnr = -10.0 * torch.log10(last_loss + 1e-10)
        progress.set_postfix(
            frame=frame.image_id,
            loss=f"{last_loss.item():.6f}",
            psnr=f"{psnr.item():.2f} dB",
        )

        visualizer.update_step(step + 1, last_loss.item(), psnr.item(), frame.image_id)
        if effective_viser_update_every and (step + 1) % effective_viser_update_every == 0:
            with torch.no_grad():
                visualizer.update_gaussians(
                    model.means,
                    torch.clamp(model.colors, 0.0, 1.0),
                    torch.sigmoid(model.opacities),
                    model.covariance_matrices(),
                )
            visualizer.update_gaussian_stats(model.num_gaussians())

        if (
            args.densify_every
            and (step + 1) % args.densify_every == 0
            and args.densify_from <= (step + 1)
            and (args.densify_until < 0 or (step + 1) <= args.densify_until)
        ):
            changed = densify_and_prune(
                model=model,
                grad_norms=grad_norms,
                grad_thresh=args.densify_grad_thresh,
                prune_opacity_thresh=args.prune_opacity_thresh,
                max_gaussians=args.max_gaussians,
                split_scale_shrink=args.split_scale_shrink,
                cull_screen_size=args.cull_screen_size,
                split_screen_size=args.split_screen_size,
                intrinsics=intrinsics,
                camera_to_world=frame.camera_to_world,
                height=height,
                width=width,
            )
            if changed:
                print(f"Densified at step {step + 1}: {model.num_gaussians()} gaussians")
                optimizer = rebuild_optimizer(model, args.lr)
                with torch.no_grad():
                    visualizer.update_gaussians(
                        model.means,
                        torch.clamp(model.colors, 0.0, 1.0),
                        torch.sigmoid(model.opacities),
                        model.covariance_matrices(),
                    )
                visualizer.update_gaussian_stats(model.num_gaussians())
                visualizer.update_status(
                    f"**Status:** training live at http://localhost:{args.viser_port} (densified/pruned)"
                )

        if args.reset_opacity_every > 0 and (step + 1) % args.reset_opacity_every == 0 and step > 0:
            reset_opacities(model)
            if args.densify_every:
                print(f"Reset opacities at step {step + 1}")
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
                selected_render = model.render(
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
                eval_render = model.render(
                    intrinsics=first_intrinsics,
                    camera_to_world=frames[0].camera_to_world,
                    height=first_height,
                    width=first_width,
                ).detach()
            save_image(eval_render, output_dir / f"render_frame0_step_{step + 1:05d}.png")

    with torch.no_grad():
        final_render = model.render(
            intrinsics=first_intrinsics,
            camera_to_world=frames[0].camera_to_world,
            height=first_height,
            width=first_width,
        ).detach()
    final_loss = F.mse_loss(final_render, first_target)
    final_psnr = -10.0 * torch.log10(final_loss + 1e-10)

    save_image(final_render, output_dir / "render_frame0_final.png")
    save_image(torch.cat([first_target, final_render], dim=1), output_dir / "comparison_frame0.png")
    save_checkpoint(model, output_dir / "gaussians.pt")

    with torch.no_grad():
        visualizer.update_gaussians(
            model.means,
            torch.clamp(model.colors, 0.0, 1.0),
            torch.sigmoid(model.opacities),
            model.covariance_matrices(),
        )
    visualizer.update_gaussian_stats(model.num_gaussians())
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
