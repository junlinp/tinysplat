# Raw fisheye images and 3DGUT

This feature accepts original, distorted images. Calibration and optical
camera-to-world poses remain fixed during training. No undistortion or pinhole
approximation is applied. Supply the calibration for these exact pixels; a
rectified camera matrix cannot describe raw fisheye images.

Each frame in the existing JSON dataset adds explicit camera metadata:

```json
{
  "image_id": 0,
  "file_path": "images/raw_000.png",
  "width": 1280,
  "height": 1024,
  "camera_model": "OPENCV_FISHEYE",
  "intrinsics": {"fx": 420.0, "fy": 420.0, "cx": 640.0, "cy": 512.0},
  "distortion": [0.01, -0.002, 0.0, 0.0],
  "transform_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
}
```

The numbers above illustrate the schema; use measured calibration. Coefficients
are OpenCV fisheye `[k1,k2,k3,k4]`, with
`theta_d = theta * (1+k1*theta²+k2*theta⁴+k3*theta⁶+k4*theta⁸)`.
Zero coefficients still mean an equidistant fisheye, not pinhole. `PINHOLE`
frames omit distortion. Other lens models and fields of view beyond 180 degrees
are not supported in this initial feature. An optional `mask_path` points to a
same-size grayscale image; nonzero pixels participate in the loss/evaluation.
Noninvertible rays are masked automatically. Lens coefficients must belong to
the frame's calibration, not be guessed from field of view.

The top-level dataset retains `scene_dir`, `frames`, `points3d`, `train_ids`,
and `val_ids`. Seed points must be in the same world frame as the poses. For
fisheye SfM, triangulate/BA with the actual camera model before training. This
feature does not run SfM or optimize calibration/poses.

## CUDA training

```sh
python -m pip install -r requirements-3dgut.txt
PYTHONPATH=legacy python train_3dgut.py \
  --dataset data/fisheye/dataset.json --output outputs/fisheye_3dgut \
  --device cuda --backend gsplat --steps 3000 --max-gaussians 100000 \
  --gpu-memory-limit-gb 6 --min-free-gb 4
```

The optional, pinned gsplat backend uses **both** `with_ut=True` and
`with_eval3d=True`: nonlinear unscented projection plus Gaussian response
evaluation in 3D. It is not a 2D image warp. It uses a single camera per step,
a two-image CPU cache, fixed Gaussian count and an explicit allocator cap/free
memory guard. `--checkpoint` accepts a TinySplat checkpoint in the same world
frame; it must fit the chosen Gaussian limit. Output directories must be new.

Outputs include `gaussians.pt`, unchanged `dataset.json`, `renderer.json`,
`evaluation.json`, and original/render comparison PNGs. With `--steps 0` and
`--checkpoint`, the CLI renders/evaluates an existing checkpoint. Evaluation
uses validation IDs when provided, otherwise labels results as training views.

## CPU and Apple

`--device cpu` or `--device mps --backend reference` uses a differentiable
PyTorch reference that evaluates 3D Gaussian responses along calibrated rays.
It is for small-scene correctness checks and has a two-million Gaussian/pixel
pair limit. It does not implement UT tile acceleration and is not intended for
full-scene training. CUDA gsplat is the accelerated 3DGUT implementation.
Existing TinySplat pinhole CPU/CUDA/Apple backends remain available unchanged.
Missing gsplat or unsupported camera models fail explicitly.

The current WebGL viewer is pinhole-only. Its free-orbit view can display a
Gaussian checkpoint, but is not a calibrated raw-fisheye comparison. Use this
CLI's native raw-camera renders to judge calibration and training quality.

## Validation

```sh
PYTHONPATH=legacy python -m pytest tests/test_fisheye_3dgut.py -q
```

Tests compare projection against OpenCV, invert pixel rays, check geometry and
appearance gradients (including finite differences), exercise raw-image
training/export, and run real gsplat CUDA forward/backward when installed.
Apple tests skip on hosts without an Apple GPU.

References: [3DGUT project](https://research.nvidia.com/labs/toronto-ai/3DGUT/)
and [gsplat rasterization API](https://docs.gsplat.studio/main/apis/rasterization.html).
