"""Fixed, explicit calibration for raw images (OpenCV optical coordinates)."""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str = "PINHOLE"
    distortion: tuple = ()

    def __post_init__(self):
        if self.model not in ("PINHOLE", "OPENCV_FISHEYE"):
            raise ValueError(f"Unsupported camera model: {self.model}")
        if self.width <= 0 or self.height <= 0 or self.fx <= 0 or self.fy <= 0:
            raise ValueError("Camera dimensions and focal lengths must be positive")
        if not all(math.isfinite(v) for v in (self.fx, self.fy, self.cx, self.cy, *self.distortion)):
            raise ValueError("Calibration must be finite")
        if len(self.distortion) != (4 if self.model == "OPENCV_FISHEYE" else 0):
            raise ValueError("OPENCV_FISHEYE requires exactly [k1,k2,k3,k4]; PINHOLE has no distortion")
        if self.distortion:
            k1, k2, k3, k4 = self.distortion
            for i in range(257):
                t2 = (math.pi/2*i/256)**2
                if 1+t2*(3*k1+t2*(5*k2+t2*(7*k3+t2*9*k4))) <= 0:
                    raise ValueError("Fisheye calibration must be invertible over the forward hemisphere")

    @classmethod
    def from_frame(cls, frame):
        return cls(frame["width"], frame["height"], **frame["intrinsics"],
                   model=frame.get("camera_model", "PINHOLE"),
                   distortion=tuple(frame.get("distortion", ())))

    def matrix(self, *, device="cpu", dtype=torch.float32):
        return torch.tensor([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
                            device=device, dtype=dtype)

    def project(self, points):
        """Project camera-space points; no pinhole substitution for fisheye."""
        xy = points[..., :2]
        if self.model == "PINHOLE":
            uv = xy / points[..., 2:].clamp_min(1e-12)
        else:
            radius = xy.square().sum(-1).clamp_min(1e-24).sqrt()
            theta = torch.atan2(radius, points[..., 2])
            t2 = theta.square()
            k1, k2, k3, k4 = self.distortion
            distorted = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
            uv = xy * (distorted / radius)[..., None]
        return uv * uv.new_tensor([self.fx, self.fy]) + uv.new_tensor([self.cx, self.cy])

    def rays(self, *, device="cpu", dtype=torch.float32):
        """Unit rays at pixel centers and an invertible forward-hemisphere mask.

        This OpenCV model supports fields of view up to 180 degrees. Models for
        beyond-180-degree cameras must be represented explicitly in a future API.
        """
        y, x = torch.meshgrid(torch.arange(self.height, device=device, dtype=dtype) + .5,
                              torch.arange(self.width, device=device, dtype=dtype) + .5, indexing="ij")
        xy = torch.stack([(x-self.cx)/self.fx, (y-self.cy)/self.fy], -1)
        if self.model == "PINHOLE":
            rays = torch.cat([xy, torch.ones_like(x[..., None])], -1)
            return torch.nn.functional.normalize(rays, dim=-1), torch.ones_like(x, dtype=torch.bool)
        radius = xy.square().sum(-1).clamp_min(1e-24).sqrt()
        theta = radius.clamp(max=math.pi/2-1e-4)
        k1, k2, k3, k4 = self.distortion
        for _ in range(20):
            t2 = theta.square()
            value = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2*k4))))
            derivative = 1 + t2 * (3*k1 + t2*(5*k2 + t2*(7*k3 + t2*9*k4)))
            theta = (theta - (value-radius)/derivative.clamp_min(1e-8)).clamp(0, math.pi/2-1e-4)
        t2 = theta.square()
        residual = theta*(1+t2*(k1+t2*(k2+t2*(k3+t2*k4))))-radius
        derivative = 1+t2*(3*k1+t2*(5*k2+t2*(7*k3+t2*9*k4)))
        valid = (residual.abs() < 1e-5) & (derivative > 0) & (theta < math.pi/2-2e-4)
        rays = torch.cat([xy*(theta.sin()/radius)[..., None], theta.cos()[..., None]], -1)
        return rays, valid
