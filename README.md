# TinySplat (C++ / Bazel)

A lightweight **2D and 3D Gaussian splatting** library implemented in C++ and built with [Bazel](https://bazel.build). This is a rewrite of the original [Python/PyTorch TinySplat](https://github.com/junlinp/tinysplat); the previous implementation lives under `legacy/`.

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
bazel test --define=cuda=1 //tests:...
```

Requires `nvcc` and a visible CUDA device for GPU tests and benchmarks.
```

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

```
tinysplat/
├── MODULE.bazel          # Bazel module (bzlmod)
├── BUILD.bazel           # Root aliases
├── src/tinysplat/        # Core C++ library
├── cuda/                 # CUDA kernels (nvcc genrule)
├── examples/             # `render_2d`, `render_2d_cuda`, `debug`
├── tests/                # C++ tests
└── legacy/               # Original Python/PyTorch code (reference)
```

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
- 3D CUDA backward is in **projected 2D space** only (full 3D Jacobian chain not implemented)
- Python training remains in `train_3d_gaussians_json.py` (gsplat backend)

## Legacy Python version

The original package (PyTorch autograd, JIT C++/CUDA extensions, training scripts) is preserved in `legacy/` for reference. It is **not** built by Bazel.

### Tanks & Temples quality benchmark

Novel-view PSNR / SSIM / LPIPS on the standard **train** / **truck** scenes (Hugging Face download, every-8th holdout):

```bash
pip install -e legacy
pip install gsplat huggingface_hub lpips pytorch-msssim

python benchmarks/tanks_and_temples/run_benchmark.py --device cuda
# smoke: add --iterations 200
```

See [`benchmarks/tanks_and_temples/README.md`](benchmarks/tanks_and_temples/README.md).

## License

MIT
