"""Train/render raw calibrated fisheye images without undistorting pixels.

Run with PYTHONPATH=python. Fixed calibration and optical camera-to-world poses;
fixed Gaussian count for predictable memory. GPU defaults to native CUDA/Metal.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tinysplat.fisheye_evaluation import evaluate_fisheye_views
from tinysplat.fisheye_dataset import FisheyeDataset, load_raw_frame
from tinysplat.gaussian_model import GaussianData, build_sparse_points_gaussians_3d
from tinysplat.render_3dgut import render_3dgut


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    p.add_argument("--backend", choices=["auto", "native", "reference"], default="auto")
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
    metadata = json.loads(raw_json)
    dataset = FisheyeDataset(metadata)
    train_indices = dataset.train_indices
    torch.manual_seed(42)
    if args.checkpoint:
        params = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if len(params["means"]) > args.max_gaussians:
            raise ValueError("Checkpoint exceeds --max-gaussians; increase the explicit limit")
    else:
        points = metadata.get("points3d", [])
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
        c, target, valid, pose = dataset.frame(int(order[(step-1)%len(order)]), args.device)
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
    evaluate_fisheye_views(dataset, render, args.device, args.eval_views, args.output)
    (args.output/"status.json").write_text(json.dumps(dict(status="complete",steps=args.steps)))


if __name__ == "__main__":
    main()
