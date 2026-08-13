#!/usr/bin/env python3
"""Download Tanks & Temples (train, truck) COLMAP scenes from Hugging Face."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ID = "alexmkwizu/gaussian_training_datasets"
DEFAULT_SCENES = ("train", "truck")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Local directory for the HF snapshot (default: ./data).",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=list(DEFAULT_SCENES),
        help=f"Scene names under tandt/ (default: {' '.join(DEFAULT_SCENES)}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is required: pip install huggingface_hub",
            file=sys.stderr,
        )
        return 1

    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    allow_patterns = [f"tandt/{scene}/**" for scene in args.scenes]

    print(f"Downloading {REPO_ID} → {data_dir}")
    print(f"Patterns: {allow_patterns}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(data_dir),
        allow_patterns=allow_patterns,
    )

    missing = []
    for scene in args.scenes:
        scene_dir = data_dir / "tandt" / scene
        images = scene_dir / "images"
        sparse = scene_dir / "sparse" / "0"
        if not images.is_dir() or not sparse.is_dir():
            missing.append(scene)
        else:
            print(f"OK  {scene_dir}")

    if missing:
        print(f"Missing expected layout for: {', '.join(missing)}", file=sys.stderr)
        return 1

    print("Download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
