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
PYTHONPATH=legacy python train_3dgut.py \
  --dataset data/fisheye/dataset.json --output outputs/fisheye_3dgut \
  --device cuda --backend native --steps 3000 --max-gaussians 100000 \
  --gpu-memory-limit-gb 6 --min-free-gb 4
```

The default GPU path uses TinySplat-owned CUDA/Metal kernels. CUDA compiles its
extension on first use and needs the CUDA toolkit and ninja. The kernels perform
nonlinear unscented projection, build depth-sorted tile candidate lists, and
evaluate 3D Gaussian responses at each calibrated pixel ray. Native backward
propagates gradients to means, covariance/scale/rotation, opacity and color/SH.
Intrinsics and distortion are fixed constants, never optimizer parameters.

The current UT uses alpha=1, beta=0, kappa=0 (six noncentral sigma samples with
weight 1/6 and zero center weight). It adds a conservative pixel margin and falls
back to full-frame bounds for large or near-plane-crossing particles. Tile
intersection construction has an eight-million-pair safety limit. Projection
bounds and depth ordering are discrete; their derivatives are not propagated.
Images use front-to-back alpha compositing and 3-sigma response truncation.

For an optional comparison backend, install `requirements-3dgut.txt` and select
`--backend gsplat`. The pinned gsplat backend uses **both** `with_ut=True` and
`with_eval3d=True`: nonlinear unscented projection plus Gaussian response
evaluation in 3D. It is not a 2D image warp. It uses a single camera per step,
a two-image CPU cache, fixed Gaussian count and an explicit allocator cap/free
memory guard. `--checkpoint` accepts a TinySplat checkpoint in the same world
frame; it must fit the chosen Gaussian limit. Output directories must be new.

Outputs include `gaussians.pt`, unchanged `dataset.json`, `renderer.json`,
`evaluation.json`, and original/render comparison PNGs. With `--steps 0` and
`--checkpoint`, the CLI renders/evaluates an existing checkpoint. Evaluation
uses validation IDs when provided, otherwise labels results as training views.

## Apple Metal

```sh
python metal/build_native_3dgut.py
PYTHONPATH=legacy python train_3dgut.py \
  --dataset data/fisheye/dataset.json --output outputs/fisheye_metal \
  --device mps --backend native --steps 3000 --max-gaussians 100000
```

The Metal kernels share camera projection, UT, ray-response and backward math
with CUDA. The current ctypes Metal bridge stages tensors and tile lists through
CPU/shared buffers; it is functional but has transfer overhead and is not yet a
zero-copy MPS implementation. GPU parity and small-scene fitting are tested;
full-scene performance and real raw-fisheye dataset quality require separate
benchmarking. CUDA allocator limits apply to PyTorch CUDA allocations; Metal's
temporary shared buffers are outside that allocator.

## CPU reference

`--device cpu` or `--device mps --backend reference` uses a differentiable
PyTorch reference that evaluates 3D Gaussian responses along calibrated rays.
It is for small-scene correctness checks and has a two-million Gaussian/pixel
pair limit. It does not implement UT tile acceleration and is not intended for
full-scene training. `auto` selects native kernels on CUDA/Apple and reference
on CPU. Native GPU rendering does not require gsplat.
Existing TinySplat pinhole CPU/CUDA/Apple backends remain available unchanged.
Missing native libraries, missing explicitly selected gsplat, or unsupported
camera models fail explicitly.

The current WebGL viewer is pinhole-only. Its free-orbit view can display a
Gaussian checkpoint, but is not a calibrated raw-fisheye comparison. Use this
CLI's native raw-camera renders to judge calibration and training quality.

## Validation

```sh
PYTHONPATH=legacy python -m pytest tests/test_fisheye_3dgut.py tests/test_native_3dgut.py -q
```

Tests compare projection against OpenCV, invert pixel rays, check geometry and
appearance gradients (including finite differences), exercise raw-image
training/export, and run real gsplat CUDA forward/backward when installed.
Apple tests skip on hosts without an Apple GPU.
Native tests compare RGB/alpha and all Gaussian parameter gradients against the
reference, check finite differences, behind-camera culling, large particles,
tile-memory limits, and a synthetic fisheye fit with fixed calibration.

References: [3DGUT project](https://research.nvidia.com/labs/toronto-ai/3DGUT/)
and [gsplat rasterization API](https://docs.gsplat.studio/main/apis/rasterization.html).
