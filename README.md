# TinySplat (C++ / Bazel)

A lightweight **2D and 3D Gaussian splatting** library implemented in C++ and built with [Bazel](https://bazel.build). This is a rewrite of the original [Python/PyTorch TinySplat](https://github.com/junlinp/tinysplat); the previous implementation lives under `legacy/`.

## Features

- **2D Gaussian splatting** — tiled CPU forward and backward (weight-normalized compositing)
- **3D Gaussian splatting** — world-space projection + front-to-back alpha compositing
- **CUDA 2D forward** — optional GPU path via `nvcc` (`//examples:render_2d_cuda`)
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
bazel build //examples:render_2d_cuda
bazel run //examples:render_2d_cuda -- --cuda
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
| `//examples:render_2d` | CPU demo → `render_2d.ppm` |
| `//examples:render_2d_cuda` | CUDA demo (`--cuda`) |
| `//cuda:gaussian_2d_cuda` | Static CUDA library |

## Legacy Python version

The original package (PyTorch autograd, JIT C++/CUDA extensions, training scripts) is preserved in `legacy/` for reference. It is **not** built by Bazel.

## License

MIT
