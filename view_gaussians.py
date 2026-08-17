#!/usr/bin/env python3
"""Serve a trained 3DGS checkpoint and render it in the browser.

Loads ``gaussians.pt``, rasterizes with TinySplat (Metal on MPS), and streams
JPEG frames to an orbit-camera web UI.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent
_LEGACY = _REPO_ROOT / "legacy"
if _LEGACY.is_dir():
    sys.path.insert(0, str(_LEGACY))

_TRAIN_PATH = _REPO_ROOT / "train_3d_gaussians_json.py"
_spec = importlib.util.spec_from_file_location("train_3d_gaussians_json", _TRAIN_PATH)
_train = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_train)
GaussianData = _train.GaussianData
load_dataset_frames = _train.load_dataset_frames
resolve_device = _train.resolve_device
tensor_image_to_uint8 = _train.tensor_image_to_uint8


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TinySplat viewer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0d0f12;
      --panel: rgba(14, 17, 22, 0.82);
      --line: #2a3140;
      --text: #e8edf5;
      --muted: #8b95a8;
      --accent: #6ea8ff;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      height: 100%;
      background: var(--bg);
      color: var(--text);
      font: 13px/1.4 ui-sans-serif, system-ui, -apple-system, sans-serif;
      overflow: hidden;
    }
    canvas { display: block; width: 100%; height: 100%; }
    #hud {
      position: absolute;
      top: 12px;
      left: 12px;
      right: 12px;
      display: flex;
      gap: 12px;
      align-items: flex-start;
      pointer-events: none;
    }
    .panel {
      pointer-events: auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      backdrop-filter: blur(10px);
    }
    .panel h1 {
      margin: 0 0 6px;
      font-size: 13px;
      font-weight: 600;
    }
    .row { color: var(--muted); }
    .row b { color: var(--text); font-weight: 600; }
    select, button {
      background: #1a2030;
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 4px 8px;
      font: inherit;
    }
    button { cursor: pointer; }
    button:hover, select:hover { border-color: var(--accent); }
    #help { margin-left: auto; max-width: 280px; color: var(--muted); }
  </style>
</head>
<body>
  <canvas id="view"></canvas>
  <div id="hud">
    <div class="panel">
      <h1>TinySplat 3DGS</h1>
      <div class="row">Gaussians <b id="n-gauss">—</b> · SH <b id="sh">—</b> · <b id="device">—</b></div>
      <div class="row">Render <b id="ms">—</b> ms · <span id="size">—</span></div>
      <div class="row" style="margin-top:8px">
        <label>Camera
          <select id="cameras"><option value="-1">Orbit</option></select>
        </label>
        <button id="reset" type="button">Reset</button>
      </div>
    </div>
    <div class="panel" id="help">
      Drag orbit · scroll zoom · right-drag pan<br />
      Metal rasterizer on the server, shown here.
    </div>
  </div>
  <script>
    const canvas = document.getElementById("view");
    const ctx = canvas.getContext("2d");
    const camerasSel = document.getElementById("cameras");

    let info = null;
    let target = [0, 0, 0];
    let radius = 4;
    let theta = 0;
    let phi = Math.PI / 3;
    let dragging = false;
    let panning = false;
    let lastX = 0;
    let lastY = 0;
    let inFlight = false;
    let pending = false;
    let moving = false;
    let moveTimer = null;

    function sub(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
    function add(a, b) { return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
    function scale(a, s) { return [a[0]*s, a[1]*s, a[2]*s]; }
    function dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
    function cross(a, b) {
      return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
    }
    function length(a) { return Math.hypot(a[0], a[1], a[2]); }
    function normalize(a) {
      const n = length(a) || 1;
      return scale(a, 1 / n);
    }

    function sphericalEye() {
      const sp = Math.sin(phi), cp = Math.cos(phi);
      const st = Math.sin(theta), ct = Math.cos(theta);
      return add(target, [radius * sp * ct, radius * sp * st, radius * cp]);
    }

    function lookAt(eye, tgt, up) {
      const z = normalize(sub(tgt, eye));
      let x = cross(z, up);
      if (length(x) < 1e-6) {
        const alt = Math.abs(up[2]) < 0.9 ? [0, 0, 1] : [0, 1, 0];
        x = cross(z, alt);
      }
      x = normalize(x);
      const y = cross(z, x);
      return [
        x[0], x[1], x[2], 0,
        y[0], y[1], y[2], 0,
        z[0], z[1], z[2], 0,
        eye[0], eye[1], eye[2], 1,
      ];
    }

    function setOrbitFromC2W(c2w) {
      const eye = [c2w[12], c2w[13], c2w[14]];
      const fwd = [c2w[8], c2w[9], c2w[10]];
      target = info.scene_center.slice();
      const toCenter = sub(target, eye);
      const depth = Math.max(dot(toCenter, fwd), 0.2);
      target = add(eye, scale(fwd, depth));
      const offset = sub(eye, target);
      radius = Math.max(length(offset), 0.05);
      theta = Math.atan2(offset[1], offset[0]);
      phi = Math.acos(Math.min(1, Math.max(-1, offset[2] / radius)));
      phi = Math.min(Math.max(phi, 0.05), Math.PI - 0.05);
    }

    function currentC2W() {
      const idx = Number(camerasSel.value);
      if (idx >= 0 && info.cameras[idx]) return info.cameras[idx].c2w;
      return lookAt(sphericalEye(), target, info.up);
    }

    function viewSize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const scale = moving ? 0.5 : 1.0;
      let w = Math.max(64, Math.round(canvas.clientWidth * dpr * scale));
      let h = Math.max(64, Math.round(canvas.clientHeight * dpr * scale));
      const cap = info.max_resolution || 1280;
      const long = Math.max(w, h);
      if (long > cap) {
        const s = cap / long;
        w = Math.max(64, Math.round(w * s));
        h = Math.max(64, Math.round(h * s));
      }
      return [w, h];
    }

    async function render() {
      if (!info) return;
      if (inFlight) { pending = true; return; }
      inFlight = true;
      pending = false;
      const [width, height] = viewSize();
      const body = {
        c2w: currentC2W(),
        width,
        height,
        quality: moving ? 70 : 88,
      };
      try {
        const res = await fetch("/render", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!res.ok) throw new Error(await res.text());
        const ms = res.headers.get("X-Render-Ms");
        document.getElementById("ms").textContent = ms || "—";
        document.getElementById("size").textContent = width + "×" + height;
        const blob = await res.blob();
        const bmp = await createImageBitmap(blob);
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight;
        ctx.fillStyle = "#0d0f12";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(bmp, 0, 0, canvas.width, canvas.height);
        bmp.close();
      } catch (err) {
        console.error(err);
        document.getElementById("ms").textContent = "err";
      } finally {
        inFlight = false;
        if (pending) render();
      }
    }

    function markMoving() {
      moving = true;
      clearTimeout(moveTimer);
      moveTimer = setTimeout(() => { moving = false; render(); }, 140);
    }

    canvas.addEventListener("pointerdown", (e) => {
      camerasSel.value = "-1";
      canvas.setPointerCapture(e.pointerId);
      lastX = e.clientX;
      lastY = e.clientY;
      panning = e.button === 2 || e.shiftKey;
      dragging = !panning;
    });
    canvas.addEventListener("pointerup", () => { dragging = false; panning = false; });
    canvas.addEventListener("pointermove", (e) => {
      if (!dragging && !panning) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      markMoving();
      if (panning) {
        const eye = sphericalEye();
        const z = normalize(sub(target, eye));
        let x = normalize(cross(z, info.up));
        const y = normalize(cross(z, x));
        const pan = radius * 0.002;
        target = add(target, add(scale(x, -dx * pan), scale(y, dy * pan)));
      } else {
        theta -= dx * 0.005;
        phi -= dy * 0.005;
        phi = Math.min(Math.max(phi, 0.05), Math.PI - 0.05);
      }
      render();
    });
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      camerasSel.value = "-1";
      markMoving();
      radius *= Math.exp(e.deltaY * 0.001);
      radius = Math.min(Math.max(radius, 0.05), info.max_radius || 1e4);
      render();
    }, { passive: false });
    camerasSel.addEventListener("change", () => {
      const idx = Number(camerasSel.value);
      if (idx >= 0) setOrbitFromC2W(info.cameras[idx].c2w);
      render();
    });
    document.getElementById("reset").addEventListener("click", () => {
      camerasSel.value = info.cameras.length ? "0" : "-1";
      if (info.cameras.length) setOrbitFromC2W(info.cameras[0].c2w);
      else {
        target = info.scene_center.slice();
        radius = info.radius;
        theta = info.theta;
        phi = info.phi;
      }
      render();
    });
    window.addEventListener("resize", () => render());

    async function boot() {
      info = await (await fetch("/info")).json();
      document.getElementById("n-gauss").textContent = info.num_gaussians.toLocaleString();
      document.getElementById("sh").textContent = String(info.sh_degree);
      document.getElementById("device").textContent = info.device;
      target = info.scene_center.slice();
      radius = info.radius;
      theta = info.theta;
      phi = info.phi;
      for (let i = 0; i < info.cameras.length; i++) {
        const cam = info.cameras[i];
        const opt = document.createElement("option");
        opt.value = String(i);
        opt.textContent = cam.name;
        camerasSel.appendChild(opt);
      }
      if (info.cameras.length) {
        camerasSel.value = "0";
        setOrbitFromC2W(info.cameras[0].c2w);
      }
      render();
    }
    boot();
  </script>
</body>
</html>
"""


def _mat4_col_major(c2w: np.ndarray) -> List[float]:
    return [float(v) for v in c2w.T.reshape(-1)]


def _mat4_from_col_major(values: List[float]) -> np.ndarray:
    arr = np.array(values, dtype=np.float32).reshape(4, 4).T
    return arr


class ViewerState:
    def __init__(
        self,
        gauss_data: Any,
        device: str,
        frames: Optional[List[Any]],
        max_resolution: int,
    ):
        self.gauss_data = gauss_data
        self.device = device
        self.frames = frames or []
        self.max_resolution = max(64, int(max_resolution) if max_resolution else 1280)
        self.lock = threading.Lock()
        means = gauss_data.means.detach().cpu().numpy()
        center = np.median(means, axis=0).astype(np.float64)
        radii = np.linalg.norm(means - center[None, :], axis=1)
        extent = float(np.quantile(radii, 0.90)) if radii.size else 1.0
        extent = max(extent, 0.5)
        self.scene_center = center
        self.up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.radius = extent * 1.6
        eye = center + np.array([0.0, -self.radius, self.radius * 0.35], dtype=np.float64)
        offset = eye - center
        self.theta = float(math.atan2(offset[1], offset[0]))
        self.phi = float(math.acos(max(-1.0, min(1.0, offset[2] / (np.linalg.norm(offset) + 1e-8)))))
        if self.frames:
            fr = self.frames[0]
            self.width = int(fr.width)
            self.height = int(fr.height)
            k = fr.intrinsics.detach().cpu().numpy()
            self.fx = float(k[0, 0])
            self.fy = float(k[1, 1])
            self.cx = float(k[0, 2])
            self.cy = float(k[1, 2])
        else:
            self.width = 1280
            self.height = 720
            fov_y = math.radians(50.0)
            self.fy = 0.5 * self.height / math.tan(fov_y * 0.5)
            self.fx = self.fy
            self.cx = self.width * 0.5
            self.cy = self.height * 0.5

    def info(self) -> Dict[str, Any]:
        cameras = []
        for i, fr in enumerate(self.frames):
            c2w = fr.camera_to_world.detach().cpu().numpy().astype(np.float32)
            cameras.append(
                {
                    "name": f"{i}: {fr.file_path.name}",
                    "c2w": _mat4_col_major(c2w),
                }
            )
        return {
            "num_gaussians": int(self.gauss_data.num_gaussians),
            "sh_degree": int(self.gauss_data.active_sh_degree),
            "device": self.device,
            "scene_center": [float(v) for v in self.scene_center],
            "up": [float(v) for v in self.up],
            "radius": float(self.radius),
            "max_radius": float(self.radius * 20.0),
            "theta": self.theta,
            "phi": self.phi,
            "max_resolution": self.max_resolution,
            "cameras": cameras,
        }

    def render(self, c2w: np.ndarray, width: int, height: int, quality: int) -> Tuple[bytes, float]:
        width = int(max(64, min(width, self.max_resolution * 2)))
        height = int(max(64, min(height, self.max_resolution * 2)))
        long_side = max(width, height)
        if long_side > self.max_resolution:
            scale = self.max_resolution / float(long_side)
            width = max(64, int(round(width * scale)))
            height = max(64, int(round(height * scale)))
        sx = width / float(self.width)
        sy = height / float(self.height)
        intrinsics = torch.tensor(
            [
                [self.fx * sx, 0.0, self.cx * sx],
                [0.0, self.fy * sy, self.cy * sy],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=self.gauss_data.device,
        )
        camera_to_world = torch.tensor(c2w, dtype=torch.float32, device=self.gauss_data.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            image = self.gauss_data.render(
                intrinsics=intrinsics,
                camera_to_world=camera_to_world,
                height=height,
                width=width,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        rgb = tensor_image_to_uint8(image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(
            ".jpg",
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(max(40, min(quality, 95)))],
        )
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return buf.tobytes(), elapsed_ms


def load_checkpoint(path: Path, device: str) -> Any:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to gaussians.pt")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional dataset.json for training cameras and intrinsics.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=1280,
        help="Cap the longer render side (default 1280).",
    )
    parser.add_argument("--open", action="store_true", help="Open the viewer in a browser.")
    return parser.parse_args()


def make_handler(state: ViewerState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            if self.path in ("/info", "/"):
                super().log_message(fmt, *args)

        def _send(self, code: int, body: bytes, content_type: str, extra: Optional[Dict[str, str]] = None):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if extra:
                for key, value in extra.items():
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/info":
                payload = json.dumps(state.info()).encode("utf-8")
                self._send(200, payload, "application/json")
                return
            self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path != "/render":
                self._send(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                req = json.loads(raw.decode("utf-8"))
                c2w = _mat4_from_col_major(req["c2w"])
                width = int(req["width"])
                height = int(req["height"])
                quality = int(req.get("quality", 85))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._send(400, str(exc).encode("utf-8"), "text/plain")
                return
            try:
                with state.lock:
                    jpeg, ms = state.render(c2w, width, height, quality)
            except Exception as exc:
                self._send(500, str(exc).encode("utf-8"), "text/plain")
                return
            self._send(
                200,
                jpeg,
                "image/jpeg",
                extra={"X-Render-Ms": f"{ms:.1f}"},
            )

    return Handler


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

    frames = None
    if args.dataset is not None:
        _, frames, _ = load_dataset_frames(args.dataset.resolve(), device)
        print(f"Loaded {len(frames)} cameras from {args.dataset}")

    state = ViewerState(gauss_data, device, frames, args.max_resolution)
    handler = make_handler(state)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}"
    if state.frames:
        warmup_c2w = state.frames[0].camera_to_world.detach().cpu().numpy()
    else:
        warmup_c2w = np.eye(4, dtype=np.float32)
        warmup_c2w[:3, 3] = (state.scene_center + np.array([0.0, -state.radius, state.radius * 0.35])).astype(
            np.float32
        )
    print("Warming up renderer ...")
    with state.lock:
        _, ms = state.render(
            warmup_c2w,
            min(state.width, 640),
            min(state.height, 360),
            70,
        )
    print(f"Warmup {ms:.1f} ms")
    print(f"Viewer: {url}")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
