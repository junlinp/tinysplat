#!/usr/bin/env python3
"""Convert a COLMAP sparse reconstruction into a JSON dataset description."""

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass
class Camera:
    camera_id: int
    model_id: int
    model_name: str
    width: int
    height: int
    params: List[float]


@dataclass
class ImageRecord:
    image_id: int
    qvec: Tuple[float, float, float, float]
    tvec: Tuple[float, float, float]
    camera_id: int
    name: str


def read_next_bytes(fid, num_bytes: int, fmt: str):
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError(f"Unexpected EOF while reading {num_bytes} bytes")
    return struct.unpack("<" + fmt, data)


def read_cameras_binary(path: Path) -> Dict[int, Camera]:
    cameras: Dict[int, Camera] = {}
    with path.open("rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_next_bytes(fid, 24, "iiQQ")
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"Unsupported COLMAP camera model id: {model_id}")
            model_name, num_params = CAMERA_MODELS[model_id]
            params = list(read_next_bytes(fid, 8 * num_params, "d" * num_params))
            cameras[camera_id] = Camera(
                camera_id=camera_id,
                model_id=model_id,
                model_name=model_name,
                width=width,
                height=height,
                params=params,
            )
    return cameras


def read_images_binary(path: Path) -> Dict[int, ImageRecord]:
    images: Dict[int, ImageRecord] = {}
    with path.open("rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            image_id = read_next_bytes(fid, 4, "i")[0]
            qvec = read_next_bytes(fid, 32, "dddd")
            tvec = read_next_bytes(fid, 24, "ddd")
            camera_id = read_next_bytes(fid, 4, "i")[0]

            name_bytes = bytearray()
            while True:
                ch = fid.read(1)
                if ch == b"":
                    raise EOFError("Unexpected EOF while reading image name")
                if ch == b"\x00":
                    break
                name_bytes.extend(ch)
            name = name_bytes.decode("utf-8")

            num_points2d = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(num_points2d * 24, 1)

            images[image_id] = ImageRecord(
                image_id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name,
            )
    return images


def read_points3d_binary(path: Path) -> List[Dict[str, object]]:
    points: List[Dict[str, object]] = []
    with path.open("rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            point3d_id = read_next_bytes(fid, 8, "Q")[0]
            x, y, z = read_next_bytes(fid, 24, "ddd")
            r, g, b = read_next_bytes(fid, 3, "BBB")
            error = read_next_bytes(fid, 8, "d")[0]
            track_len = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(track_len * 8, 1)  # (image_id, point2D_idx) pairs
            points.append(
                {
                    "id": int(point3d_id),
                    "xyz": [float(x), float(y), float(z)],
                    "rgb": [int(r), int(g), int(b)],
                    "error": float(error),
                }
            )
    return points


def read_points3d_text(path: Path) -> List[Dict[str, object]]:
    points: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        point3d_id = int(parts[0])
        x, y, z = map(float, parts[1:4])
        r, g, b = map(int, parts[4:7])
        error = float(parts[7])
        points.append(
            {
                "id": point3d_id,
                "xyz": [x, y, z],
                "rgb": [r, g, b],
                "error": error,
            }
        )
    return points


def qvec_to_rotmat(qvec: Tuple[float, float, float, float]) -> List[List[float]]:
    qw, qx, qy, qz = qvec
    return [
        [
            1.0 - 2.0 * qy * qy - 2.0 * qz * qz,
            2.0 * qx * qy - 2.0 * qw * qz,
            2.0 * qx * qz + 2.0 * qw * qy,
        ],
        [
            2.0 * qx * qy + 2.0 * qw * qz,
            1.0 - 2.0 * qx * qx - 2.0 * qz * qz,
            2.0 * qy * qz - 2.0 * qw * qx,
        ],
        [
            2.0 * qx * qz - 2.0 * qw * qy,
            2.0 * qy * qz + 2.0 * qw * qx,
            1.0 - 2.0 * qx * qx - 2.0 * qy * qy,
        ],
    ]


def transpose3x3(mat: List[List[float]]) -> List[List[float]]:
    return [[mat[j][i] for j in range(3)] for i in range(3)]


def matmul3x3_vec3(mat: List[List[float]], vec: Tuple[float, float, float]) -> List[float]:
    return [
        mat[0][0] * vec[0] + mat[0][1] * vec[1] + mat[0][2] * vec[2],
        mat[1][0] * vec[0] + mat[1][1] * vec[1] + mat[1][2] * vec[2],
        mat[2][0] * vec[0] + mat[2][1] * vec[1] + mat[2][2] * vec[2],
    ]


def colmap_image_to_c2w(image: ImageRecord) -> List[List[float]]:
    rot_w2c = qvec_to_rotmat(image.qvec)
    rot_c2w = transpose3x3(rot_w2c)
    rotated_t = matmul3x3_vec3(rot_c2w, image.tvec)
    translation = [-rotated_t[0], -rotated_t[1], -rotated_t[2]]
    return [
        [rot_c2w[0][0], rot_c2w[0][1], rot_c2w[0][2], translation[0]],
        [rot_c2w[1][0], rot_c2w[1][1], rot_c2w[1][2], translation[1]],
        [rot_c2w[2][0], rot_c2w[2][1], rot_c2w[2][2], translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def image_width_height(path: Path) -> Tuple[int, int]:
    """Return (width, height) for JPEG/PNG without decoding pixels."""
    with path.open("rb") as fid:
        sig = fid.read(8)
        if sig.startswith(b"\xff\xd8"):
            fid.seek(2)
            while True:
                marker = fid.read(2)
                if len(marker) < 2 or marker[0] != 0xFF:
                    raise ValueError(f"Could not parse JPEG size: {path}")
                code = marker[1]
                if code in (0xD8, 0xD9) or 0xD0 <= code <= 0xD7:
                    continue
                seglen_bytes = fid.read(2)
                if len(seglen_bytes) < 2:
                    raise ValueError(f"Could not parse JPEG size: {path}")
                seglen = struct.unpack(">H", seglen_bytes)[0]
                if code in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    sof = fid.read(5)
                    if len(sof) < 5:
                        raise ValueError(f"Could not parse JPEG size: {path}")
                    _prec, height, width = struct.unpack(">BHH", sof)
                    return int(width), int(height)
                fid.seek(seglen - 2, 1)
        if sig == b"\x89PNG\r\n\x1a\n":
            fid.seek(16)
            wh = fid.read(8)
            if len(wh) < 8:
                raise ValueError(f"Could not parse PNG size: {path}")
            width, height = struct.unpack(">II", wh)
            return int(width), int(height)
    raise ValueError(f"Unsupported image type for size probe: {path}")


def scale_intrinsics_to_image(
    intrinsics: Dict[str, object], cam_width: int, cam_height: int, img_width: int, img_height: int
) -> Dict[str, object]:
    if cam_width <= 0 or cam_height <= 0:
        return intrinsics
    if img_width == cam_width and img_height == cam_height:
        return intrinsics
    sx = img_width / float(cam_width)
    sy = img_height / float(cam_height)
    scaled = dict(intrinsics)
    scaled["fx"] = float(intrinsics["fx"]) * sx
    scaled["fy"] = float(intrinsics["fy"]) * sy
    scaled["cx"] = float(intrinsics["cx"]) * sx
    scaled["cy"] = float(intrinsics["cy"]) * sy
    return scaled


def parse_intrinsics(camera: Camera) -> Dict[str, object]:
    params = camera.params
    model = camera.model_name

    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        return {"fx": f, "fy": f, "cx": cx, "cy": cy, "distortion": []}
    if model == "PINHOLE":
        fx, fy, cx, cy = params
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "distortion": []}
    if model in {"SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"}:
        f, cx, cy, k1 = params
        return {"fx": f, "fy": f, "cx": cx, "cy": cy, "distortion": [k1]}
    if model in {"RADIAL", "RADIAL_FISHEYE"}:
        f, cx, cy, k1, k2 = params
        return {"fx": f, "fy": f, "cx": cx, "cy": cy, "distortion": [k1, k2]}
    if model in {"OPENCV", "OPENCV_FISHEYE"}:
        fx, fy, cx, cy, k1, k2, p1, p2 = params
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "distortion": [k1, k2, p1, p2]}
    if model == "FULL_OPENCV":
        fx, fy, cx, cy, *distortion = params
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "distortion": distortion}
    if model == "FOV":
        fx, fy, cx, cy, omega = params
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "distortion": [omega]}
    if model == "THIN_PRISM_FISHEYE":
        fx, fy, cx, cy, *distortion = params
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "distortion": distortion}

    raise ValueError(f"Unsupported COLMAP camera model: {model}")


def build_dataset_json(
    scene_dir: Path,
    images_dir: Path,
    sparse_dir: Path,
) -> Dict[str, object]:
    cameras = read_cameras_binary(sparse_dir / "cameras.bin")
    images = read_images_binary(sparse_dir / "images.bin")

    points_bin = sparse_dir / "points3D.bin"
    points_txt = sparse_dir / "points3D.txt"
    if points_bin.exists():
        points3d = read_points3d_binary(points_bin)
    elif points_txt.exists():
        points3d = read_points3d_text(points_txt)
    else:
        points3d = []

    scaled_from = None
    frames = []
    for image in sorted(images.values(), key=lambda item: item.name):
        camera = cameras[image.camera_id]
        intrinsics = parse_intrinsics(camera)
        image_path = images_dir / image.name
        if not image_path.exists():
            raise FileNotFoundError(f"Referenced image does not exist: {image_path}")
        img_w, img_h = image_width_height(image_path)
        if img_w != camera.width or img_h != camera.height:
            scaled_from = (camera.width, camera.height, img_w, img_h)
        intrinsics = scale_intrinsics_to_image(
            intrinsics, camera.width, camera.height, img_w, img_h
        )

        frames.append(
            {
                "image_id": image.image_id,
                "camera_id": image.camera_id,
                "file_path": str(image_path.relative_to(scene_dir)),
                "camera_model": camera.model_name,
                "width": img_w,
                "height": img_h,
                "intrinsics": intrinsics,
                "transform_matrix": colmap_image_to_c2w(image),
                "colmap": {
                    "qvec": list(image.qvec),
                    "tvec": list(image.tvec),
                },
            }
        )

    if scaled_from is not None:
        cam_w, cam_h, img_w, img_h = scaled_from
        print(
            f"Scaled COLMAP cameras {cam_w}x{cam_h} to JPEG size {img_w}x{img_h}"
        )

    return {
        "format": "tinysplat-colmap-json",
        "scene_dir": str(scene_dir),
        "images_dir": str(images_dir.relative_to(scene_dir)),
        "sparse_dir": str(sparse_dir.relative_to(scene_dir)),
        "num_frames": len(frames),
        "num_points3d": len(points3d),
        "points3d": points3d,
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
