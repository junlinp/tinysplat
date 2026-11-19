#!/usr/bin/env python3
"""Convert a COLMAP sparse reconstruction into a JSON dataset description."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict


# Keep both standalone converters usable from outside the repository.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from colmap_io import (
    CAMERA_MODELS,
    Camera,
    ImageRecord,
    colmap_image_to_c2w,
    matmul3x3_vec3,
    parse_intrinsics,
    qvec_to_rotmat,
    read_cameras_binary,
    read_images_binary,
    read_next_bytes,
    transpose3x3,
)


def build_dataset_json(
    scene_dir: Path,
    images_dir: Path,
    sparse_dir: Path,
) -> Dict[str, object]:
    cameras = read_cameras_binary(sparse_dir / "cameras.bin")
    images = read_images_binary(sparse_dir / "images.bin")

    frames = []
    for image in sorted(images.values(), key=lambda item: item.name):
        camera = cameras[image.camera_id]
        intrinsics = parse_intrinsics(camera)
        image_path = images_dir / image.name
        if not image_path.exists():
            raise FileNotFoundError(f"Referenced image does not exist: {image_path}")

        frames.append(
            {
                "image_id": image.image_id,
                "camera_id": image.camera_id,
                "file_path": str(image_path.relative_to(scene_dir)),
                "camera_model": camera.model_name,
                "width": camera.width,
                "height": camera.height,
                "intrinsics": intrinsics,
                "transform_matrix": colmap_image_to_c2w(image),
                "colmap": {
                    "qvec": list(image.qvec),
                    "tvec": list(image.tvec),
                },
            }
        )

    return {
        "format": "tinysplat-colmap-json",
        "scene_dir": str(scene_dir),
        "images_dir": str(images_dir.relative_to(scene_dir)),
        "sparse_dir": str(sparse_dir.relative_to(scene_dir)),
        "num_frames": len(frames),
        "frames": frames,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scene_dir",
        type=Path,
        help="Scene root containing `images/` and `sparse/0/`.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Override image directory. Defaults to <scene_dir>/images.",
    )
    parser.add_argument(
        "--sparse-dir",
        type=Path,
        default=None,
        help="Override sparse model directory. Defaults to <scene_dir>/sparse/0.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <scene_dir>/dataset.json.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    images_dir = (args.images_dir or (scene_dir / "images")).resolve()
    sparse_dir = (args.sparse_dir or (scene_dir / "sparse" / "0")).resolve()
    output_path = (args.output or (scene_dir / "dataset.json")).resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
    if not sparse_dir.exists():
        raise FileNotFoundError(f"Sparse directory does not exist: {sparse_dir}")

    dataset = build_dataset_json(
        scene_dir=scene_dir,
        images_dir=images_dir,
        sparse_dir=sparse_dir,
    )
    output_path.write_text(json.dumps(dataset, indent=args.indent) + "\n", encoding="utf-8")
    print(f"Wrote {dataset['num_frames']} frames to {output_path}")


if __name__ == "__main__":
    main()
