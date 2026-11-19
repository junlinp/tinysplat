"""Evaluation and image export for calibrated fisheye scenes."""

import json

import cv2
import numpy as np
import torch

from .cameras import Camera


@torch.no_grad()
def evaluate_fisheye_views(dataset, render, device, max_views, output_dir):
    """Evaluate evenly spaced validation views, using training views if needed."""
    pool = dataset.val_indices or dataset.train_indices
    positions = np.linspace(0, len(pool) - 1, min(max_views, len(pool))).astype(int)
    metrics = []
    for position in positions:
        index = pool[position]
        camera, target, valid, pose = dataset.frame(index, device)
        prediction = render(camera, pose)
        mse = (prediction[valid] - target[valid]).square().mean().clamp_min(1e-12)
        image_id = dataset.frames[index]["image_id"]
        metrics.append({"image_id": image_id, "psnr": float(-10 * torch.log10(mse))})

        comparison = torch.cat([target, prediction], dim=1).clamp(0, 1).cpu().numpy()
        bgr = (comparison[..., ::-1] * 255).round().astype(np.uint8)
        cv2.imwrite(str(output_dir / f"comparison_{image_id}.png"), bgr)

    assert [Camera.from_frame(frame) for frame in dataset.frames] == dataset.cameras
    report = {
        "views": metrics,
        "mean_psnr": float(np.mean([view["psnr"] for view in metrics])),
        "held_out": bool(dataset.val_indices),
        "intrinsics_fixed": True,
        "distortion_fixed": True,
        "poses_fixed": True,
    }
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2))
