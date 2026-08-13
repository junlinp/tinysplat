#!/usr/bin/env python3
"""Convert downloaded T&T COLMAP scenes into tinysplat dataset JSON files."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_SCENES = ("train", "truck")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory that contains tandt/<scene>/ (default: ./data).",
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
    repo_root = Path(__file__).resolve().parents[2]
    convert = repo_root / "convert_colmap_to_json.py"
    if not convert.is_file():
        print(f"Missing converter: {convert}", file=sys.stderr)
        return 1

    data_dir = args.data_dir.resolve()
    for scene in args.scenes:
        scene_dir = data_dir / "tandt" / scene
        if not scene_dir.is_dir():
            print(f"Scene not found: {scene_dir}", file=sys.stderr)
            return 1
        out = scene_dir / "dataset.json"
        cmd = [
            sys.executable,
            str(convert),
            str(scene_dir),
            "--output",
            str(out),
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"Wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
