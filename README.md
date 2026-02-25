# TinySplat

A lightweight 2D Gaussian splatting implementation for PyTorch with support for CPU, CUDA, and MPS backends.

## Features

- **2D Gaussian Splatting**: Efficient rendering of 2D Gaussian distributions
- **Multi-Backend Support**: Automatic detection and support for CPU, CUDA, and MPS devices
- **PyTorch Integration**: Both functional and module APIs for easy integration
- **Differentiable**: Fully differentiable for gradient-based optimization
- **Custom Kernel Ready**: Uses `torch.autograd.Function` for easy extension with custom C++/CUDA/MPS kernels

## Installation

```bash
pip install -e .
```

For development with additional tools:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Functional API

```python
import torch
from tinysplat import gaussian_splat_2d

# Define Gaussian parameters
means = torch.tensor([[100.0, 100.0], [150.0, 150.0]])  # (N, 2)
covariances = torch.eye(2).unsqueeze(0).repeat(2, 1, 1) * 100.0  # (N, 2, 2)
colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # (N, 3)
opacities = torch.tensor([0.8, 0.9])  # (N,)

# Render
image = gaussian_splat_2d(
    means=means,
    covariances=covariances,
    colors=colors,
    opacities=opacities,
    height=256,
    width=256,
    device="cuda"  # or "cpu", "mps", or None for auto-detect
)
```

### Module API

```python
import torch
from tinysplat import GaussianSplat2D

# Create a model
model = GaussianSplat2D(
    num_gaussians=10,
    num_channels=3,
    height=256,
    width=256,
    device="cuda"  # optional
)

# Render
image = model()  # (256, 256, 3)

# Optimize parameters
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
target = torch.zeros(256, 256, 3)
target[100:150, 100:150] = torch.tensor([1.0, 0.0, 0.0])

for _ in range(100):
    optimizer.zero_grad()
    output = model()
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()
```

## Examples

Run the example script to see various use cases:

```bash
python example.py
```

This will generate example images demonstrating:
- Functional API usage
- Module API usage with optimization
- Backend comparison (CPU/CUDA/MPS)

## Testing

First, download the Lena test image:

```bash
python tests/download_lena.py
```

Then run the test suite:

```bash
pytest tests/
```

The tests use the Lena image to verify:
- Image loading and preprocessing
- Reconstruction using Gaussian splats
- Optimization to match the target image
- Gradient computation
- Different resolutions
- Backend consistency (CPU/CUDA/MPS)

Test outputs are saved to `tests/test_outputs/` for visual inspection.

## API Reference

### `gaussian_splat_2d`

Renders 2D Gaussian splats onto a canvas.

**Parameters:**
- `means` (Tensor): Shape `(N, 2)` - 2D mean positions
- `covariances` (Tensor): Shape `(N, 2, 2)` - 2x2 covariance matrices
- `colors` (Tensor): Shape `(N, C)` - Colors (C can be 1, 3, or 4)
- `opacities` (Tensor): Shape `(N,)` - Opacity values in [0, 1]
- `height` (int): Output image height
- `width` (int): Output image width
- `device` (str, optional): Device to use ('cpu', 'cuda', 'mps', or None for auto-detect)

**Returns:**
- `Tensor`: Shape `(height, width, C)` - Rendered image

### `GaussianSplat2D`

PyTorch module for 2D Gaussian splatting with learnable parameters.

**Parameters:**
- `num_gaussians` (int): Number of Gaussian splats
- `num_channels` (int): Number of color channels (1, 3, or 4)
- `height` (int): Output image height
- `width` (int): Output image width
- `device` (str, optional): Device to use

**Methods:**
- `forward()`: Render the Gaussian splats
- `get_parameters_dict()`: Get all parameters as a dictionary

## Extending with Custom Kernels

The implementation uses `torch.autograd.Function`, making it easy to add custom C++/CUDA/MPS kernels:

1. **CUDA**: Implement `_forward_cuda` and `_backward_cuda` in `GaussianSplat2DFunction`
2. **MPS**: Implement `_forward_mps` and `_backward_mps` in `GaussianSplat2DFunction`
3. **C++**: Create a C++ extension and call it from `_forward_cpu`/`_backward_cpu`

The forward pass saves intermediate values that can be reused in custom backward implementations for efficiency.

## Project Structure

```
tinysplat/
├── pyproject.toml          # Build configuration
├── README.md               # This file
├── example.py              # Example usage
├── tests/                  # Test suite
│   ├── __init__.py
│   ├── data/
│   │   └── lena.png        # Test image
│   └── test_gaussian_splat_2d.py
└── tinysplat/              # Package directory
    ├── __init__.py         # Package initialization
    └── gaussian_splat_2d.py # Core implementation with autograd Function
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.20.0

## License

MIT
