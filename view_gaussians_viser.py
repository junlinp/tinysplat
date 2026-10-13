#!/usr/bin/env python3
"""Visualize a trained 3DGS checkpoint with viser.

Two rendering modes, switchable live from the GUI:

``webgl``     — push the Gaussians and update SH colors per camera; let viser's
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
        "indices": to_np(idx),
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

    @torch.no_grad()
    def render_frame(self, frame: Any) -> Tuple[np.ndarray, float]:
        """Render at the recorded resolution with the exact dataset K and pose."""
        start = time.perf_counter()
        with self.lock:
            image = self.gauss_data.render(
                intrinsics=frame.intrinsics,
                camera_to_world=frame.camera_to_world,
                height=frame.height,
                width=frame.width,
            )
        return tensor_image_to_uint8(image), (time.perf_counter() - start) * 1000.0


def letterbox_image(rgb: np.ndarray, aspect: float) -> np.ndarray:
    """Pad to the browser aspect without stretching or cropping calibrated pixels."""
    height, width = rgb.shape[:2]
    aspect = max(float(aspect), 0.01)
    canvas_width = max(width, int(round(height * aspect)))
    canvas_height = max(height, int(round(width / aspect)))
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    x, y = (canvas_width - width) // 2, (canvas_height - height) // 2
    canvas[y:y + height, x:x + width] = rgb
    return canvas


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


def set_camera_to_frame(client: Any, frame: Any, look_distance: float = 1.0) -> None:
    """Match a dataset camera's optical pose and vertical field of view."""
    c2w = frame.camera_to_world.detach().cpu().numpy().astype(np.float64)
    k = frame.intrinsics.detach().cpu().numpy().astype(np.float64)
    client.camera.up_direction = -c2w[:3, 1]  # OpenCV +Y points down.
    client.camera.position = c2w[:3, 3]
    client.camera.look_at = c2w[:3, 3] + c2w[:3, 2] * look_distance
    client.camera.fov = 2.0 * float(np.arctan2(frame.height * 0.5, k[1, 1]))


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
        help="Initial render mode: webgl uses view-dependent SH colors; tinysplat includes "
             "learned view-dependent SH appearance (switchable in the GUI).",
    )
    parser.add_argument(
        "--opacity-floor",
        type=float,
        default=0.0,
        help="Drop Gaussians below this opacity from WebGL (default: keep all).",
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
    parser.add_argument(
        "--gpu-memory-limit-gb", type=float, default=0.0,
        help="Cap the CUDA PyTorch allocator in GiB (0 leaves it unrestricted).",
    )
    parser.add_argument(
        "--torch-num-threads", type=int, default=0,
        help="Set CPU intra-op threads (0 keeps the runtime default).",
    )
    args = parser.parse_args()
    if (not np.isfinite(args.gpu_memory_limit_gb)
            or args.gpu_memory_limit_gb < 0 or args.torch_num_threads < 0):
        parser.error("memory limit must be finite and limits must be nonnegative")
    return args


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    device = resolve_device(args.device)
    ckpt = args.checkpoint.resolve()
    if not ckpt.is_file():
        print(f"Missing checkpoint: {ckpt}", file=sys.stderr)
        return 1

    print(f"Loading {ckpt} on {device} ...")
    if device == "cuda" and args.gpu_memory_limit_gb > 0:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        fraction = args.gpu_memory_limit_gb * 1024**3 / total_memory
        if fraction > 1.0:
            raise ValueError("--gpu-memory-limit-gb exceeds the selected GPU's total memory")
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
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
    splat_indices = torch.as_tensor(arrays.pop("indices"), device=gauss_data.device)
    splat_handle = server.scene.add_gaussian_splats("/splats", **arrays, visible=False)
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
        gui_calibrated = server.gui.add_checkbox(
            "use dataset calibration", initial_value=bool(frames) and args.mode == "tinysplat",
            hint="Exact recorded pose, intrinsics and original resolution. Uncheck to orbit.",
        )
        gui_stats = server.gui.add_text("frame", initial_value="—", disabled=True)

    @gui_mode.on_update
    def _(_event: Any) -> None:
        # WebGL must always expose the interactive browser camera.
        if gui_mode.value == "webgl":
            gui_calibrated.value = False

    @gui_calibrated.on_update
    def _(_event: Any) -> None:
        if gui_calibrated.value:
            gui_mode.value = "tinysplat"
        gui_height.disabled = bool(frames) and gui_calibrated.value

    gui_height.disabled = bool(frames) and gui_calibrated.value

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
            gui_calibrated.value = True
            gui_mode.value = "tinysplat"
            set_camera_to_frame(event.client, frames[index], radius)

    client_splats: Dict[int, Any] = {}

    @server.on_client_disconnect
    def _(client: viser.ClientHandle) -> None:
        client_splats.pop(client.client_id, None)

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client_splats[client.client_id] = client.scene.add_gaussian_splats(
            "/view_dependent_splats", **arrays, visible=False,
        )
        if frames:
            set_camera_to_frame(client, frames[0])
        else:
            client.camera.position = center + np.array([0.0, -radius, radius * 0.35])
            client.camera.look_at = center
            client.camera.up_direction = np.array([0.0, 0.0, 1.0])

    # ---- Server-side render loop -------------------------------------------
    def render_loop() -> None:
        last_key: Dict[int, Any] = {}
        color_keys: Dict[int, Any] = {}
        webgl_background: set[int] = set()
        while True:
            if gui_mode.value != "tinysplat":
                splat_handle.visible = False
                for client in server.get_clients().values():
                    handle = client_splats.get(client.client_id)
                    if handle is None:
                        continue
                    handle.visible = True
                    position_key = tuple(np.asarray(client.camera.position).tolist())
                    if color_keys.get(client.client_id) != position_key:
                        pose = torch.eye(4, dtype=torch.float32, device=gauss_data.device)
                        pose[:3, 3] = torch.as_tensor(position_key, device=gauss_data.device)
                        with torch.no_grad(), renderer.lock:
                            colors = gauss_data.visible_colors(pose)[splat_indices]
                            handle.rgbs = colors.detach().cpu().numpy()
                        color_keys[client.client_id] = position_key
                    gui_stats.value = f"WebGL: {kept} splats, SH {gauss_data.active_sh_degree}"
                    if last_key.pop(client.client_id, None) is not None or client.client_id not in webgl_background:
                        # Match TinySplat's black background; white washed out translucent edges.
                        client.scene.set_background_image(np.zeros((1, 1, 3), dtype=np.uint8), format="png")
                        webgl_background.add(client.client_id)
                time.sleep(0.05)
                continue

            splat_handle.visible = False
            for client in server.get_clients().values():
                webgl_background.discard(client.client_id)
                handle = client_splats.get(client.client_id)
                if handle is not None:
                    handle.visible = False
                camera = client.camera
                calibrated_index = int(gui_jump.value.split(":", 1)[0]) if frames and gui_calibrated.value else None
                height = int(gui_height.value)
                width = max(64, int(round(height * float(camera.aspect))))
                key = ("calibrated", calibrated_index, float(camera.aspect)) if calibrated_index is not None else (
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
                    if calibrated_index is not None:
                        rgb, elapsed_ms = renderer.render_frame(frames[calibrated_index])
                        height, width = rgb.shape[:2]
                        rgb = letterbox_image(rgb, camera.aspect)
                    else:
                        rgb, elapsed_ms = renderer.render(c2w, float(camera.fov), width, height)
                except Exception as exc:  # keep the viewer alive on a bad frame
                    gui_stats.value = f"error: {exc}"
                    time.sleep(0.5)
                    continue

                client.scene.set_background_image(rgb, format="png")
                last_key[client.client_id] = key
                gui_stats.value = f"{width}x{height}  {elapsed_ms:.1f} ms" + ("  exact dataset camera" if calibrated_index is not None else "  free camera")
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
