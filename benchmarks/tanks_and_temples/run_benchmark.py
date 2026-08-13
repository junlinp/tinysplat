#!/usr/bin/env python3
"""Run the Tanks & Temples quality benchmark (train + truck) with this repo."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

DEFAULT_SCENES = ("train", "truck")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tandt_benchmark"))
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument(
        "--iterations",
        type=int,
        default=30000,
        help="Training iterations per scene (3DGS paper default: 30000).",
    )
    parser.add_argument("--eval-hold", type=int, default=8, help="Hold every N-th view for eval.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--max-resolution", type=int, default=0)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-train", action="store_true", help="Only eval existing checkpoints.")
    parser.add_argument(
        "--extra-train-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args forwarded to train_3d_gaussians_json.py (prefix with --).",
    )
    return parser.parse_args()


def _run(cmd: List[str], env: Dict[str, str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    bench_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir if args.data_dir.is_absolute() else (repo_root / args.data_dir)
    output_root = (
        args.output_dir if args.output_dir.is_absolute() else (repo_root / args.output_dir)
    )
    output_root.mkdir(parents=True, exist_ok=True)

    # Prefer complete legacy package when importing tinysplat.
    env = os.environ.copy()
    legacy = str(repo_root / "legacy")
    env["PYTHONPATH"] = legacy + os.pathsep + str(repo_root) + os.pathsep + env.get(
        "PYTHONPATH", ""
    )
    py = sys.executable

    if not args.skip_download:
        _run(
            [
                py,
                str(bench_dir / "download_hf.py"),
                "--data-dir",
                str(data_dir),
                "--scenes",
                *args.scenes,
            ],
            env,
        )

    if not args.skip_prepare:
        _run(
            [
                py,
                str(bench_dir / "prepare_scenes.py"),
                "--data-dir",
                str(data_dir),
                "--scenes",
                *args.scenes,
            ],
            env,
        )

    summary: Dict[str, object] = {"scenes": {}, "protocol": {
        "eval_hold": args.eval_hold,
        "iterations": args.iterations,
        "dataset": "Tanks & Temples (train, truck) via Hugging Face",
        "metrics": ["psnr", "ssim", "lpips"],
    }}

    for scene in args.scenes:
        scene_dir = data_dir / "tandt" / scene
        dataset_json = scene_dir / "dataset.json"
        scene_out = output_root / scene
        scene_out.mkdir(parents=True, exist_ok=True)
        ckpt = scene_out / "gaussians.pt"

        if not args.skip_train:
            train_cmd = [
                py,
                str(repo_root / "train_3d_gaussians_json.py"),
                str(dataset_json),
                "--iterations",
                str(args.iterations),
                "--eval-hold",
                str(args.eval_hold),
                "--device",
                args.device,
                "--max-resolution",
                str(args.max_resolution),
                "--output-dir",
                str(scene_out),
                "--no-viser",
                # T&T images are ~1MP; avoid aggressive resolution curriculum.
                "--num-downscales",
                "0",
            ]
            extra = args.extra_train_args
            if extra and extra[0] == "--":
                extra = extra[1:]
            train_cmd.extend(extra)
            _run(train_cmd, env)

        if not ckpt.is_file():
            print(f"Missing checkpoint for {scene}: {ckpt}", file=sys.stderr)
            return 1

        metrics_path = scene_out / "metrics.json"
        _run(
            [
                py,
                str(bench_dir / "eval_heldout.py"),
                str(dataset_json),
                str(ckpt),
                "--eval-hold",
                str(args.eval_hold),
                "--device",
                args.device,
                "--max-resolution",
                str(args.max_resolution),
                "--output",
                str(metrics_path),
            ],
            env,
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        summary["scenes"][scene] = metrics
        print(
            f"[{scene}] PSNR={metrics['psnr']:.3f}  "
            f"SSIM={metrics['ssim']:.4f}  LPIPS={metrics['lpips']:.4f}  "
            f"({metrics['num_views']} views)"
        )

    # Mean over scenes (standard reporting style).
    scene_metrics = list(summary["scenes"].values())
    if scene_metrics:
        summary["mean"] = {
            "psnr": sum(m["psnr"] for m in scene_metrics) / len(scene_metrics),
            "ssim": sum(m["ssim"] for m in scene_metrics) / len(scene_metrics),
            "lpips": sum(m["lpips"] for m in scene_metrics) / len(scene_metrics),
        }

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")
    if "mean" in summary:
        m = summary["mean"]
        print(
            f"[mean] PSNR={m['psnr']:.3f}  SSIM={m['ssim']:.4f}  LPIPS={m['lpips']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
