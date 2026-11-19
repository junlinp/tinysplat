"""Raw fisheye dataset validation, image decoding, and frame access."""

from collections import OrderedDict
from pathlib import Path

import cv2
import torch

from .cameras import Camera


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


class FisheyeDataset:
    """Index calibrated frames and keep up to two decoded frames on the CPU."""

    def __init__(self, metadata):
        self.scene_dir = Path(metadata["scene_dir"])
        self.frames = metadata["frames"]
        if not self.frames:
            raise ValueError("Dataset has no frames")
        self.cameras = [Camera.from_frame(frame) for frame in self.frames]
        by_id = {frame["image_id"]: index for index, frame in enumerate(self.frames)}
        if len(by_id) != len(self.frames):
            raise ValueError("Duplicate image_id")
        train_ids = metadata.get("train_ids", list(by_id))
        val_ids = metadata.get("val_ids", [])
        if not train_ids or set(train_ids) & set(val_ids):
            raise ValueError("Training split must be nonempty and disjoint from validation")
        self.train_indices = [by_id[image_id] for image_id in train_ids]
        self.val_indices = [by_id[image_id] for image_id in val_ids]
        self._cache = OrderedDict()

    def frame(self, index, device):
        if index not in self._cache:
            self._cache[index] = load_raw_frame(self.scene_dir, self.frames[index], "cpu")
            if len(self._cache) > 2:
                self._cache.popitem(last=False)
        camera, target, mask, pose = self._cache[index]
        return camera, target.to(device), mask.to(device), pose.to(device)
