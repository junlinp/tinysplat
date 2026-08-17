#!/usr/bin/env python3
"""Evaluate a saved Gaussian checkpoint on held-out views (PSNR / SSIM / LPIPS)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

# Prefer complete legacy package for `tinysplat`. Load the repo-root trainer by path
# so `legacy/train_3d_gaussians_json.py` cannot shadow it.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LEGACY = _REPO_ROOT / "legacy"
if _LEGACY.is_dir():
    sys.path.insert(0, str(_LEGACY))

import importlib.util

_TRAIN_PATH = _REPO_ROOT / "train_3d_gaussians_json.py"
_spec = importlib.util.spec_from_file_location("train_3d_gaussians_json", _TRAIN_PATH)
_train = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_train)
GaussianData = _train.GaussianData
evaluate_heldout = _train.evaluate_heldout
load_dataset_frames = _train.load_dataset_frames
resolve_device = _train.resolve_device
split_train_eval = _train.split_train_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_json", type=Path)
    parser.add_argument("checkpoint", type=Path, help="Path to gaussians.pt")
    parser.add_argument("--eval-hold", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--max-resolution", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write metrics JSON here (default: <checkpoint_dir>/metrics.json).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    _, frames, _ = load_dataset_frames(args.dataset_json.resolve(), device)
    _, eval_frames = split_train_eval(frames, args.eval_hold)
    if not eval_frames:
        print("No held-out frames (check --eval-hold).", file=sys.stderr)
        return 1

    params = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sh_degree = 0
    if "sh_degree" in params:
        sh_val = params["sh_degree"]
        sh_degree = int(sh_val.item() if hasattr(sh_val, "item") else sh_val)
    elif params.get("sh_coeffs") is not None:
        sh_degree = 3
    gauss_data = GaussianData(params, device, sh_degree=sh_degree)
    if gauss_data.sh_coeffs is not None:
        gauss_data.active_sh_degree = (
            sh_degree if sh_degree > 0 else gauss_data.max_sh_degree
        )
    metrics = evaluate_heldout(
        gauss_data,
        eval_frames,
        device=device,
        max_resolution=args.max_resolution,
    )
    out = args.output or (args.checkpoint.resolve().parent / "metrics.json")
    out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
