"""Read COLMAP sparse camera and image records and convert their calibration and poses."""

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
