#!/usr/bin/env python3
"""Train/render raw calibrated fisheye images without undistorting pixels.

Run with PYTHONPATH=legacy. Fixed calibration and optical camera-to-world poses;
fixed Gaussian count for predictable memory. GPU defaults to native CUDA/Metal.
"""
import argparse
import collections
import json
import importlib.util
from pathlib import Path
import sys

# The repository also has a root helpers package named tinysplat. Prefer the
# training package when executing this file directly from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent / "legacy"))

import cv2
import numpy as np
import torch

from tinysplat.cameras import Camera
from tinysplat.render_3dgut import render_3dgut


def load_raw_frame(scene_dir, frame, device):
    camera = Camera.from_frame(frame)
    image = cv2.imread(str(scene_dir / frame["file_path"]), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (camera.height, camera.width):
        raise ValueError(f"Image missing or dimensions disagree with calibration: {frame['file_path']}")
    rgb = torch.tensor(image[..., ::-1].copy(), device=device, dtype=torch.float32)/255
    _, valid = camera.rays(device=device)
    if frame.get("mask_path"):
        mask = cv2.imread(str(scene_dir / frame["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != image.shape[:2]:
            raise ValueError("Mask missing or dimensions disagree with raw image")
        valid &= torch.tensor(mask > 0, device=device)
    if not valid.any():
        raise ValueError("Frame has no valid calibrated pixels")
    pose = torch.tensor(frame["transform_matrix"], device=device, dtype=torch.float32)
    return camera, rgb, valid, pose


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    p.add_argument("--backend", choices=["auto", "native", "gsplat", "reference"], default="auto")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--max-gaussians", type=int, default=100000)
    p.add_argument("--gpu-memory-limit-gb", type=float, default=6)
    p.add_argument("--min-free-gb", type=float, default=4)
    p.add_argument("--eval-views", type=int, default=8)
    args = p.parse_args()
    if args.steps < 0 or args.max_gaussians < 1 or args.eval_views < 1:
        p.error("steps >= 0, max-gaussians >= 1 and eval-views >= 1 required")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            p.error("CUDA requested but unavailable")
        total = torch.cuda.get_device_properties(0).total_memory
        if not 0 < args.gpu_memory_limit_gb*1024**3 <= total:
            p.error("GPU memory limit must be positive and fit the device")
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_limit_gb*1024**3/total)
    args.output.mkdir(parents=True, exist_ok=False)
    raw_json = args.dataset.read_text()
    dataset = json.loads(raw_json)
    scene = Path(dataset["scene_dir"])
    frames = dataset["frames"]
    if not frames:
        raise ValueError("Dataset has no frames")
    cameras = [Camera.from_frame(f) for f in frames]
    by_id = {f["image_id"]: i for i, f in enumerate(frames)}
    if len(by_id) != len(frames):
        raise ValueError("Duplicate image_id")
    train_ids = dataset.get("train_ids", list(by_id))
    val_ids = dataset.get("val_ids", [])
    if not train_ids or set(train_ids) & set(val_ids):
        raise ValueError("Training split must be nonempty and disjoint from validation")
    train_indices = [by_id[i] for i in train_ids]
    val_indices = [by_id[i] for i in val_ids]
    # Reuse the existing checkpoint format, SH and initialization implementation.
    spec = importlib.util.spec_from_file_location("_tinysplat_training", Path(__file__).with_name("train_3d_gaussians_json.py"))
    training = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = training
    spec.loader.exec_module(training)
    GaussianData = training.GaussianData
    build_sparse_points_gaussians_3d = training.build_sparse_points_gaussians_3d
    torch.manual_seed(42)
    if args.checkpoint:
        params = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if len(params["means"]) > args.max_gaussians:
            raise ValueError("Checkpoint exceeds --max-gaussians; increase the explicit limit")
    else:
        points = dataset.get("points3d", [])
        if not points:
            raise ValueError("Supply points3d in the BA/SLAM world frame or --checkpoint")
        indices = np.random.default_rng(42).choice(len(points), min(len(points), args.max_gaussians), replace=False)
        params = build_sparse_points_gaussians_3d([points[i] for i in indices], torch.device(args.device), torch.float32)
    model = GaussianData(params, args.device, sh_degree=3)
    model.active_sh_degree = 3
    optimizer = torch.optim.Adam([
        {"params": [model.means], "lr": .0001},
        {"params": [model.log_scales], "lr": .001},
        {"params": [model.rotations], "lr": .0002},
        {"params": [model.opacities], "lr": .003},
        {"params": [model.sh_coeffs if model.sh_coeffs is not None else model.colors], "lr": .001},
    ])
    cache = collections.OrderedDict()
    def frame(i):
        if i not in cache:
            cache[i] = load_raw_frame(scene, frames[i], "cpu")
            while len(cache) > 2:
                cache.popitem(last=False)
        c, target, mask, pose = cache[i]
        return c, target.to(args.device), mask.to(args.device), pose.to(args.device)
    def render(camera, pose):
        return render_3dgut(model.means, model.rotations, model.log_scales.exp(),
                             model.visible_opacities(), model.visible_colors(pose), camera, pose,
                             backend=args.backend)[0]
    (args.output/"dataset.json").write_text(raw_json)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(intrinsics_fixed=True, distortion_fixed=True, poses_fixed=True,
                  raw_pixels=True, undistorted=False, densification=False)
    (args.output/"renderer.json").write_text(json.dumps(config, indent=2))
    rng = np.random.default_rng(42)
    for step in range(1, args.steps+1):
        if args.device == "cuda" and torch.cuda.mem_get_info()[0] < args.min_free_gb*1024**3:
            raise RuntimeError("Free GPU memory below safety guard")
        if (step-1) % len(train_indices) == 0:
            order = rng.permutation(train_indices)
        c, target, valid, pose = frame(int(order[(step-1)%len(order)]))
        optimizer.zero_grad(set_to_none=True)
        pred = render(c, pose)
        loss = (pred[valid]-target[valid]).abs().mean()
        loss.backward()
        if not torch.isfinite(loss) or any(t.grad is not None and not torch.isfinite(t.grad).all() for t in model.parameters()):
            raise RuntimeError("Nonfinite loss/gradient")
        optimizer.step()
        with torch.no_grad():
            model.log_scales.clamp_(-12, 3)
            model.opacities.clamp_(-12, 12)
        if step % 50 == 0 or step == args.steps:
            status = dict(step=step, steps=args.steps, loss=float(loss.detach()), gaussians=model.num_gaussians,
                          peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2 if args.device=="cuda" else None)
            print(json.dumps(status), flush=True)
            (args.output/"status.json").write_text(json.dumps(status))
    torch.save(model.export_params(), args.output/"gaussians.pt")
    pool = val_indices or train_indices
    selected = np.linspace(0, len(pool)-1, min(args.eval_views,len(pool))).astype(int)
    metrics = []
    with torch.no_grad():
        for j in selected:
            i = pool[int(j)]
            c, target, valid, pose = frame(i)
            pred = render(c, pose)
            mse = (pred[valid]-target[valid]).square().mean().clamp_min(1e-12)
            metrics.append(dict(image_id=frames[i]["image_id"], psnr=float(-10*torch.log10(mse))))
            comparison = torch.cat([target, pred],1).clamp(0,1).cpu().numpy()
            cv2.imwrite(str(args.output/f"comparison_{frames[i]['image_id']}.png"), (comparison[...,::-1]*255).round().astype(np.uint8))
    assert [Camera.from_frame(f) for f in frames] == cameras
    (args.output/"evaluation.json").write_text(json.dumps(dict(views=metrics,
        mean_psnr=float(np.mean([v["psnr"] for v in metrics])), held_out=bool(val_indices),
        intrinsics_fixed=True, distortion_fixed=True, poses_fixed=True), indent=2))
    (args.output/"status.json").write_text(json.dumps(dict(status="complete",steps=args.steps)))


if __name__ == "__main__":
    main()
