# TinySplat (C++ / Bazel)

A lightweight **2D and 3D Gaussian splatting** library implemented in C++ and built with [Bazel](https://bazel.build). The Python training commands use the PyTorch package under `python/tinysplat/`.

## Python quick start

Use `./splat --help` to list commands and `./splat <command> --help` for options.
Run these commands from the repository root for a COLMAP scene with PINHOLE calibration.

```sh
pip install -e ./python
./splat convert data/scene
./splat train data/scene/dataset.json --output-dir outputs/scene --no-viser
./splat view outputs/scene/gaussians.pt --dataset data/scene/dataset.json
```

For raw fisheye images, use `./splat train-fisheye` with the
[calibrated dataset format](docs/raw_fisheye_3dgut.md).

## Features

- **2D Gaussian splatting** — tiled CPU forward and backward (weight-normalized compositing)
- **3D Gaussian splatting** — world-space projection + front-to-back alpha compositing
- **CUDA 2D/3D** — tiled GPU rasterizer (`--define=cuda=1`): weight-normalized 2D, alpha 3DGS-style 3D
- **OpenMP** — parallel CPU kernels when `-fopenmp` is available
- **No PyTorch** — standalone C++ library suitable for embedding in games, vision pipelines, or custom trainers

## Requirements

- Bazel 7+ (or [Bazelisk](https://github.com/bazelbuild/bazelisk))
- C++17 compiler (GCC or Clang)
- Optional: NVIDIA CUDA toolkit (`nvcc`, `libcudart`) for GPU targets

## Quick start

```bash
# Run unit tests
bazel test //tests:gaussian_2d_test

# Render a demo image (PPM)
bazel run //examples:render_2d

# GPU render (when CUDA is installed)
bazel build --define=cuda=1 //examples:render_2d_cuda
bazel run --define=cuda=1 //examples:render_2d_cuda -- --cuda

# 3D benchmark (CPU vs CUDA)
bazel run --define=cuda=1 //examples:bench_3d_cuda -- --n 100000 --h 1080 --w 1920
```

### Compositing modes

| API | Compositing |
|-----|-------------|
| `gaussian_splat_2d_*` | Weight-normalized: `Σ wᵢ cᵢ / Σ wᵢ` |
| `gaussian_splat_3d_*` | Front-to-back alpha + transmittance (3DGS-style) |

CUDA 2D supports `CompositingMode::Weighted` (default) and `Alpha` (used internally for projected 3D splats).

### CUDA build flag

Enable GPU code paths with Bazel:

```bash
bazel build --define=cuda=1 //src/tinysplat:tinysplat
bazel test --define=cuda=1 //tests/...
```

Requires `nvcc` and a visible CUDA device for GPU tests and benchmarks.

## API overview

Headers live under `src/tinysplat/include/tinysplat/`:

| Header | Purpose |
|--------|---------|
| `types.h` | `Gaussians2D`, `Gaussians3D`, `Gradients2D`, camera types |
| `image.h` | Row-major `Image` buffer |
| `gaussian_2d.h` | `gaussian_splat_2d_forward` / `gaussian_splat_2d_backward` |
| `gaussian_3d.h` | 3D projection and splatting |
| `gaussian_2d_cuda.h` | CUDA forward (`tinysplat::cuda`) |

### Minimal 2D example

```cpp
#include "tinysplat/gaussian_2d.h"

tinysplat::Gaussians2D g;
g.means = {{100.f, 100.f}};
g.covariances = {{{50.f, 0.f, 0.f, 50.f}}};
g.colors = {{1.f, 0.f, 0.f}};
g.opacities = {0.9f};

tinysplat::Image img = tinysplat::gaussian_splat_2d_forward(g, 256, 256);
```

## Project layout

| Location | Purpose |
|---|---|
| `src/tinysplat/` | C++ library and public headers |
| `cuda/`, `metal/` | Native CUDA and Metal backends |
| `python/tinysplat/` | Installable PyTorch package, including the shared Gaussian model in `gaussian_model.py` and PyTorch extensions in `cpp/` |
| `python/tinysplat/pinhole_training.py`, `fisheye_training.py` in the same directory | Training loops for pinhole and raw fisheye images |
| `splat`, `scripts/` | Checkout launcher and command implementations |
| `convert_colmap_to_json.py`, `legacy/convert_colmap_to_json.py` | Current and legacy dataset converters. Both use the COLMAP readers in `scripts/colmap_io.py` |
| [`experiments/halide/`](experiments/halide/SPEC.md) | Experimental 2D Halide kernels, standalone build, and Python bridge |
| `examples/`, `benchmarks/` | C++ demos and dataset benchmarks |
| `tests/` | C++ tests and Python regression tests, including the synthetic fisheye scene |
| `legacy/` | Additional Python examples and backend tests |

## Build targets

| Target | Description |
|--------|-------------|
| `//src/tinysplat` | Core library |
| `//tests:gaussian_2d_test` | CPU tests |
| `//tests:gaussian_2d_cuda_test` | CPU vs CUDA diff (`--define=cuda=1`) |
| `//examples:render_2d` | CPU demo → `render_2d.ppm` |
| `//examples:render_2d_cuda` | CUDA demo (`--cuda`, `--define=cuda=1`) |
| `//examples:bench_3d` | 3D forward timing (CPU) |
| `//examples:bench_3d_cuda` | 3D forward timing (CPU + CUDA, `--define=cuda=1`) |
| `//cuda:tinysplat_cuda` | Static CUDA library |

### Known limits (C++ core)

- No spherical harmonics, densification, or COLMAP loader in C++
- The standalone `cuda/` Bazel library still has no tile binning in its 3D
  compositing kernel; the PyTorch extension under `python/tinysplat/cpp/` (which
  Python training uses) does
- Run pinhole training with `./splat train` for built-in densification and pruning.

### Python training backends

Both Metal and CUDA support the full pipeline, including FastGS:

| | Metal (Apple Silicon) | CUDA |
|---|---|---|
| rasterizer | fused `qs` kernels | tiled forward + per-pixel backward |
| projection | fused in Metal | fused CUDA kernel with hand-written VJP |
| FastGS VCD / AbsGS | yes | yes (`cuda_backend`, dispatched by `fastgs_stats`) |

The CUDA backward computes gradients in projected 2D space and recovers 3D
gradients by re-running the projection under autograd, where Metal computes them
directly in its kernels. Both agree numerically -- the CUDA path matches the CPU
reference on every parameter -- so this is an implementation difference rather
than a capability gap.

## Python package

Install the training package from the repository root with `pip install -e ./python`. See the [raw fisheye training guide](docs/raw_fisheye_3dgut.md) for dataset calibration, training commands, and evaluation.

### Tanks & Temples quality benchmark

Novel-view PSNR / SSIM / LPIPS on the standard **train** / **truck** scenes (Hugging Face download, every-8th holdout):

```bash
pip install -e ./python
pip install huggingface_hub lpips pytorch-msssim

python benchmarks/tanks_and_temples/run_benchmark.py --device cuda
# smoke: add --iterations 200
```

See [`benchmarks/tanks_and_temples/README.md`](benchmarks/tanks_and_temples/README.md).

## License

MIT

## Inspecting quality and preserving detail

The Viser viewer defaults to browser WebGL rendering. It retains all Gaussians
by default and updates their SH colors for each client camera position. Geometry
is rasterized in the browser; color updates require the server connection.
WebGL uses a black background to match TinySplat. Viser still differs in its
screen-space filtering, culling, covariance precision, and alpha compositing.
To inspect TinySplat's own CUDA renderer:

```bash
./splat view outputs/run/gaussians.pt \
  --dataset data/scene/dataset.json --device cuda --mode tinysplat \
  --render-height 540 --gpu-memory-limit-gb 4 --torch-num-threads 4
```

With a dataset, the viewer starts at its first camera. In TinySplat mode,
`use dataset calibration` uses the exact recorded intrinsics, pose, and original
resolution, with letterboxing and lossless PNG streaming. Camera jumps enable
this mode. Uncheck it to orbit with the server renderer; selecting WebGL also
releases the calibration lock. Without a dataset, the viewer uses an orbit
overview. The optional memory limit caps the selected
GPU's PyTorch allocator; it is not a reservation or a limit on other processes.

FastGS late pruning remains enabled by default. For scenes where pruning removes
useful detail, compare a run with it disabled while retaining early pruning and
densification. Intermediate checkpoints help inspect the model before pruning:

```bash
./splat train data/scene/dataset.json \
  --device cuda --fastgs --no-fastgs-late-prune --checkpoint-every 5000 \
  --iterations 30000 --no-viser --output-dir outputs/no_late_prune
```

`--checkpoint-every 0` (default) saves only the final `gaussians.pt`. Positive
intervals also save `gaussians_step_NNNNN.pt` before the final step. Disabling
late pruning can retain more Gaussians and increase memory and rendering cost;
compare held-out metrics and use `--max-gaussians` to bound growth.
