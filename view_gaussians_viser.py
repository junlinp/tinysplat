#!/usr/bin/env python3
"""Visualize a trained 3DGS checkpoint with viser.

Two rendering modes, switchable live from the GUI:

``webgl``     — push the Gaussians to the browser once and let viser's
                client-side splat rasterizer draw them. Smooth and
                interactive, but it is *viser's* renderer, not TinySplat's.
``tinysplat`` — rasterize every frame on the server with TinySplat (Metal on
                MPS, CUDA, or CPU) and stream the result as the background
                image. Slower, but shows exactly what this repo's rasterizer
                produces.

Having both in one viewer makes it easy to A/B the TinySplat rasterizer
against a known-good reference on the same camera.

Usage::

    python view_gaussians_viser.py outputs/<run>/gaussians.pt \\
        --dataset data/tandt/truck/dataset.json
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import viser
import viser.transforms as vtf

_REPO_ROOT = Path(__file__).resolve().parent
_LEGACY = _REPO_ROOT / "legacy"
if _LEGACY.is_dir():
    sys.path.insert(0, str(_LEGACY))

_TRAIN_PATH = _REPO_ROOT / "train_3d_gaussians_json.py"
_spec = importlib.util.spec_from_file_location("train_3d_gaussians_json", _TRAIN_PATH)
_train = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_train)
load_dataset_frames = _train.load_dataset_frames
resolve_device = _train.resolve_device
tensor_image_to_uint8 = _train.tensor_image_to_uint8
GaussianData = _train.GaussianData

SH_C0 = 0.28209479177387814


def load_checkpoint(path: Path, device: str) -> Any:
    """Load ``gaussians.pt`` into a frozen :class:`GaussianData`."""
    params = torch.load(path, map_location=device, weights_only=False)
    sh_degree = 0
    if "sh_degree" in params:
        sh_val = params["sh_degree"]
        sh_degree = int(sh_val.item() if hasattr(sh_val, "item") else sh_val)
    elif params.get("sh_coeffs") is not None:
        sh_degree = 3
    gauss_data = GaussianData(params, device, sh_degree=sh_degree)
    if gauss_data.sh_coeffs is not None:
        gauss_data.active_sh_degree = sh_degree if sh_degree > 0 else gauss_data.max_sh_degree
    for tensor in gauss_data.parameters():
        tensor.requires_grad_(False)
    return gauss_data


@torch.no_grad()
def base_colors(gauss_data: Any) -> torch.Tensor:
    """View-independent RGB in [0, 1] (SH DC term when SH is active)."""
    if gauss_data.sh_coeffs is not None:
        return torch.clamp(SH_C0 * gauss_data.sh_coeffs[:, 0, :] + 0.5, 0.0, 1.0)
    return torch.clamp(gauss_data.colors, 0.0, 1.0)


@torch.no_grad()
def splat_arrays(
    gauss_data: Any,
    opacity_floor: float,
    max_splats: int,
) -> Dict[str, np.ndarray]:
    """Numpy arrays for ``scene.add_gaussian_splats``, culled and subsampled."""
    opacities = gauss_data.visible_opacities().reshape(-1)
    keep = opacities >= opacity_floor
    idx = torch.nonzero(keep, as_tuple=False).reshape(-1)
    if max_splats > 0 and idx.numel() > max_splats:
        # Keep the most opaque ones; they carry most of the rendered signal.
        order = torch.argsort(opacities[idx], descending=True)[:max_splats]
        idx = idx[order]

    to_np = lambda t: t.detach().cpu().numpy()  # noqa: E731
    return {
        "centers": to_np(gauss_data.means[idx]).astype(np.float32),
        "covariances": to_np(gauss_data.covariance_matrices()[idx]).astype(np.float32),
        "rgbs": to_np(base_colors(gauss_data)[idx]).astype(np.float32),
        "opacities": to_np(opacities[idx]).astype(np.float32).reshape(-1, 1),
        "kept": idx.numel(),
    }


def scene_bounds(means: np.ndarray) -> Tuple[np.ndarray, float]:
    """Robust center and radius, ignoring the usual far-flung outliers."""
    center = np.median(means, axis=0).astype(np.float64)
    radii = np.linalg.norm(means - center[None, :], axis=1)
    extent = float(np.quantile(radii, 0.90)) if radii.size else 1.0
    return center, max(extent, 0.5) * 1.6


class Renderer:
    """Server-side TinySplat rasterization for the ``tinysplat`` mode."""

    def __init__(self, gauss_data: Any):
        self.gauss_data = gauss_data
        self.lock = threading.Lock()

    @torch.no_grad()
    def render(
        self,
        c2w: np.ndarray,
        fov_y: float,
        width: int,
        height: int,
    ) -> Tuple[np.ndarray, float]:
        device = self.gauss_data.device
        fy = 0.5 * height / np.tan(fov_y * 0.5)
        fx = fy  # viser's camera has square pixels.
        intrinsics = torch.tensor(
            [[fx, 0.0, width * 0.5], [0.0, fy, height * 0.5], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        camera_to_world = torch.tensor(c2w, dtype=torch.float32, device=device)
        start = time.perf_counter()
        with self.lock:
            image = self.gauss_data.render(
                intrinsics=intrinsics,
                camera_to_world=camera_to_world,
                height=height,
                width=width,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return tensor_image_to_uint8(image), elapsed_ms


def add_training_cameras(
    server: viser.ViserServer,
    frames: List[Any],
    scale: float,
) -> List[Any]:
    """Draw the training camera poses as frustums."""
    handles = []
    for i, frame in enumerate(frames):
        c2w = frame.camera_to_world.detach().cpu().numpy().astype(np.float64)
        k = frame.intrinsics.detach().cpu().numpy()
        fov_y = 2.0 * float(np.arctan2(0.5 * float(frame.height), float(k[1, 1])))
        handles.append(
            server.scene.add_camera_frustum(
                f"/cameras/{i:04d}",
                fov=fov_y,
                aspect=float(frame.width) / float(frame.height),
                scale=scale,
                color=(110, 160, 255),
                wxyz=vtf.SO3.from_matrix(c2w[:3, :3]).wxyz,
                position=c2w[:3, 3],
                visible=False,
            )
        )
    return handles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to gaussians.pt")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional dataset.json; draws the training cameras as frustums.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--mode",
        choices=["webgl", "tinysplat"],
        default="webgl",
        help="Initial render mode (switchable in the GUI).",
    )
    parser.add_argument(
        "--opacity-floor",
        type=float,
        default=0.01,
        help="Drop Gaussians below this opacity from the WebGL view.",
    )
    parser.add_argument(
        "--max-splats",
        type=int,
        default=0,
        help="Cap the WebGL splat count (0 = no cap); keeps the most opaque.",
    )
    parser.add_argument(
        "--render-height",
        type=int,
        default=720,
        help="Initial server-side render height for tinysplat mode.",
    )
    return parser.parse_args()


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    device = resolve_device(args.device)
    ckpt = args.checkpoint.resolve()
    if not ckpt.is_file():
        print(f"Missing checkpoint: {ckpt}", file=sys.stderr)
        return 1

    print(f"Loading {ckpt} on {device} ...")
    gauss_data = load_checkpoint(ckpt, device)
    print(f"Loaded {gauss_data.num_gaussians} gaussians (SH {gauss_data.active_sh_degree})")

    frames: List[Any] = []
    if args.dataset is not None:
        _, frames, _ = load_dataset_frames(args.dataset.resolve(), device)
        print(f"Loaded {len(frames)} cameras from {args.dataset}")

    means_np = gauss_data.means.detach().cpu().numpy()
    center, radius = scene_bounds(means_np)
    renderer = Renderer(gauss_data)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")

    print("Uploading splats to the browser ...")
    arrays = splat_arrays(gauss_data, args.opacity_floor, args.max_splats)
    kept = arrays.pop("kept")
    splat_handle = server.scene.add_gaussian_splats("/splats", **arrays)
    print(f"WebGL splats: {kept}/{gauss_data.num_gaussians}")

    frustums = add_training_cameras(server, frames, scale=radius * 0.04) if frames else []

    # ---- GUI ----------------------------------------------------------------
    with server.gui.add_folder("Model"):
        server.gui.add_text("checkpoint", initial_value=ckpt.name, disabled=True)
        server.gui.add_text(
            "gaussians", initial_value=f"{gauss_data.num_gaussians}", disabled=True
        )
        server.gui.add_text("device", initial_value=device, disabled=True)

    with server.gui.add_folder("Render"):
        gui_mode = server.gui.add_dropdown(
            "mode", options=("webgl", "tinysplat"), initial_value=args.mode
        )
        gui_height = server.gui.add_slider(
            "height", min=180, max=1440, step=60, initial_value=args.render_height
        )
        gui_stats = server.gui.add_text("frame", initial_value="—", disabled=True)

    if frustums:
        with server.gui.add_folder("Cameras"):
            gui_show_cams = server.gui.add_checkbox("show frustums", initial_value=False)
            gui_jump = server.gui.add_dropdown(
                "jump to",
                options=tuple(f"{i}: {fr.file_path.name}" for i, fr in enumerate(frames)),
                initial_value=f"0: {frames[0].file_path.name}",
            )
            gui_jump_btn = server.gui.add_button("go")

        @gui_show_cams.on_update
        def _(_event: Any) -> None:
            for handle in frustums:
                handle.visible = gui_show_cams.value

        @gui_jump_btn.on_click
        def _(event: Any) -> None:
            index = int(gui_jump.value.split(":", 1)[0])
            c2w = frames[index].camera_to_world.detach().cpu().numpy().astype(np.float64)
            forward = c2w[:3, 2]  # OpenCV convention: +Z is the viewing direction.
            client = event.client
            client.camera.position = c2w[:3, 3]
            client.camera.wxyz = vtf.SO3.from_matrix(c2w[:3, :3]).wxyz
            client.camera.look_at = c2w[:3, 3] + forward * radius

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = center + np.array([0.0, -radius, radius * 0.35])
        client.camera.look_at = center
        client.camera.up_direction = np.array([0.0, 0.0, 1.0])

    # ---- Server-side render loop -------------------------------------------
    def render_loop() -> None:
        last_key: Dict[int, Any] = {}
        while True:
            if gui_mode.value != "tinysplat":
                splat_handle.visible = True
                for client in server.get_clients().values():
                    if last_key.pop(client.client_id, None) is not None:
                        client.scene.set_background_image(None)
                time.sleep(0.05)
                continue

            splat_handle.visible = False
            for client in server.get_clients().values():
                camera = client.camera
                height = int(gui_height.value)
                width = max(64, int(round(height * float(camera.aspect))))
                key = (
                    tuple(np.asarray(camera.wxyz).tolist()),
                    tuple(np.asarray(camera.position).tolist()),
                    float(camera.fov),
                    width,
                    height,
                )
                if last_key.get(client.client_id) == key:
                    continue

                c2w = np.eye(4, dtype=np.float32)
                c2w[:3, :3] = vtf.SO3(np.asarray(camera.wxyz)).as_matrix()
                c2w[:3, 3] = np.asarray(camera.position)
                try:
                    rgb, elapsed_ms = renderer.render(c2w, float(camera.fov), width, height)
                except Exception as exc:  # keep the viewer alive on a bad frame
                    gui_stats.value = f"error: {exc}"
                    time.sleep(0.5)
                    continue

                client.scene.set_background_image(rgb, format="jpeg", jpeg_quality=85)
                last_key[client.client_id] = key
                gui_stats.value = f"{width}x{height}  {elapsed_ms:.1f} ms"
            time.sleep(0.01)

    threading.Thread(target=render_loop, daemon=True).start()

    print(f"Viewer: http://{args.host}:{args.port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
