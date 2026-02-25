"""
Example usage of TinySplat 2D Gaussian splatting.
"""

import torch
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Install with: pip install matplotlib")
    print("Or install dev dependencies: pip install -e '.[dev]'")

from tinysplat import GaussianSplat2D, gaussian_splat_2d


def example_functional():
    """Example using the functional API."""
    if not HAS_MATPLOTLIB:
        print("Skipping functional example (matplotlib not available)")
        return
    
    print("Running functional API example...")
    
    # Create some Gaussian splats
    num_gaussians = 5
    height, width = 256, 256
    
    # Means: random positions
    means = torch.rand(num_gaussians, 2) * torch.tensor([width, height])
    
    # Covariances: create some elliptical Gaussians
    covariances = []
    for i in range(num_gaussians):
        # Create a 2x2 covariance matrix
        angle = i * torch.pi / num_gaussians
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        R = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]])
        S = torch.diag(torch.tensor([20.0 + i * 5, 10.0 + i * 3]))
        cov = R @ S @ S.T @ R.T
        covariances.append(cov)
    covariances = torch.stack(covariances)
    
    # Colors: random RGB colors
    colors = torch.rand(num_gaussians, 3)
    
    # Opacities: vary between 0.3 and 1.0
    opacities = torch.linspace(0.3, 1.0, num_gaussians)
    
    # Render
    image = gaussian_splat_2d(
        means=means,
        covariances=covariances,
        colors=colors,
        opacities=opacities,
        height=height,
        width=width,
    )
    
    # Display
    plt.figure(figsize=(8, 8))
    plt.imshow(image.detach().cpu().numpy())
    plt.title("Functional API Example")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("example_functional.png", dpi=150, bbox_inches="tight")
    print("Saved example_functional.png")
    plt.close()


def example_module():
    """Example using the module API."""
    if not HAS_MATPLOTLIB:
        print("Skipping module example (matplotlib not available)")
        return
    
    print("Running module API example...")
    
    # Create module
    model = GaussianSplat2D(
        num_gaussians=10,
        num_channels=3,
        height=256,
        width=256,
    )
    
    # Render
    image = model()
    
    # Display
    plt.figure(figsize=(8, 8))
    plt.imshow(image.detach().cpu().numpy())
    plt.title("Module API Example (Initial State)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("example_module_initial.png", dpi=150, bbox_inches="tight")
    print("Saved example_module_initial.png")
    plt.close()
    
    # Optimize to create a target pattern (simple example)
    print("Optimizing to create a pattern...")
    target = torch.zeros(256, 256, 3)
    target[100:150, 100:150] = torch.tensor([1.0, 0.0, 0.0])  # Red square
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for i in range(100):
        optimizer.zero_grad()
        output = model()
        loss = F.mse_loss(output, target.to(output.device))
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 20 == 0:
            print(f"Iteration {i+1}, Loss: {loss.item():.6f}")
    
    # Render final result
    final_image = model()
    plt.figure(figsize=(8, 8))
    plt.imshow(final_image.detach().cpu().numpy())
    plt.title("Module API Example (After Optimization)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("example_module_optimized.png", dpi=150, bbox_inches="tight")
    print("Saved example_module_optimized.png")
    plt.close()


def example_backend_comparison():
    """Compare different backends."""
    if not HAS_MATPLOTLIB:
        print("Skipping backend comparison example (matplotlib not available)")
        return
    
    print("Running backend comparison example...")
    
    num_gaussians = 5
    height, width = 128, 128
    
    means = torch.rand(num_gaussians, 2) * torch.tensor([width, height])
    covariances = torch.eye(2).unsqueeze(0).repeat(num_gaussians, 1, 1) * 100.0
    colors = torch.rand(num_gaussians, 3)
    opacities = torch.ones(num_gaussians) * 0.8
    
    backends = ["cpu"]
    if torch.cuda.is_available():
        backends.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        backends.append("mps")
    
    fig, axes = plt.subplots(1, len(backends), figsize=(5 * len(backends), 5))
    if len(backends) == 1:
        axes = [axes]
    
    for idx, backend in enumerate(backends):
        print(f"  Rendering on {backend}...")
        image = gaussian_splat_2d(
            means=means,
            covariances=covariances,
            colors=colors,
            opacities=opacities,
            height=height,
            width=width,
            device=backend,
        )
        
        axes[idx].imshow(image.detach().cpu().numpy())
        axes[idx].set_title(f"Backend: {backend.upper()}")
        axes[idx].axis("off")
    
    plt.tight_layout()
    plt.savefig("example_backends.png", dpi=150, bbox_inches="tight")
    print("Saved example_backends.png")
    plt.close()


if __name__ == "__main__":
    print("TinySplat 2D Gaussian Splatting Examples")
    print("=" * 50)
    
    example_functional()
    example_module()
    example_backend_comparison()
    
    print("\nAll examples completed!")

