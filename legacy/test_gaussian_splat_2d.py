"""
Tests for 2D Gaussian splatting functionality using Lena test image.
"""

import torch
import pytest
import numpy as np
from pathlib import Path

from tinysplat import gaussian_splat_2d, GaussianSplat2D


# Path to Lena image
LENA_PATH = Path(__file__).parent / "data" / "lena.png"
TEST_OUTPUT_DIR = Path(__file__).parent / "test_outputs"
TEST_OUTPUT_DIR.mkdir(exist_ok=True)


def load_lena_image(device="cpu"):
    """
    Load the Lena test image.
    
    Args:
        device: Device to load image on.
    
    Returns:
        Tensor of shape (H, W, 3) with values in [0, 1].
    """
    if not LENA_PATH.exists():
        pytest.skip(f"Lena image not found at {LENA_PATH}. Run tests/download_lena.py first.")
    
    try:
        from PIL import Image
        img = Image.open(LENA_PATH)
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Convert to numpy and normalize to [0, 1]
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Convert to tensor and move to device
        img_tensor = torch.from_numpy(img_array).to(device)
        
        return img_tensor
    except ImportError:
        try:
            import matplotlib.image as mpimg
            img_array = mpimg.imread(str(LENA_PATH))
            # Handle different formats
            if img_array.max() > 1.0:
                img_array = img_array / 255.0
            # Convert RGBA to RGB if needed
            if img_array.shape[-1] == 4:
                img_array = img_array[..., :3]
            
            img_tensor = torch.from_numpy(img_array.astype(np.float32)).to(device)
            return img_tensor
        except ImportError:
            pytest.skip("PIL or matplotlib required to load images. Install with: pip install Pillow matplotlib")


def save_test_image(image: torch.Tensor, filename: str):
    """Save a test image for visual inspection."""
    try:
        import matplotlib.pyplot as plt
        
        # Convert to numpy and handle different channel counts
        img_np = image.detach().cpu().numpy()
        
        # Clamp values to [0, 1]
        img_np = np.clip(img_np, 0.0, 1.0)
        
        if img_np.shape[-1] == 1:
            # Grayscale
            plt.imsave(str(TEST_OUTPUT_DIR / filename), img_np.squeeze(-1), cmap="gray")
        elif img_np.shape[-1] == 3:
            # RGB
            plt.imsave(str(TEST_OUTPUT_DIR / filename), img_np)
        elif img_np.shape[-1] == 4:
            # RGBA - save as RGB
            plt.imsave(str(TEST_OUTPUT_DIR / filename), img_np[..., :3])
        else:
            # Save first channel
            plt.imsave(str(TEST_OUTPUT_DIR / filename), img_np[..., 0], cmap="viridis")
        
        print(f"Saved test image: {TEST_OUTPUT_DIR / filename}")
    except ImportError:
        # If matplotlib not available, just skip saving
        pass


class TestGaussianSplat2D:
    """Test suite for Gaussian splatting using Lena image."""
    
    @pytest.fixture
    def device(self):
        """Get available device for testing."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    @pytest.fixture
    def lena_image(self, device):
        """Load Lena image."""
        return load_lena_image(device)
    
    def test_load_lena_image(self, lena_image):
        """Test that Lena image can be loaded."""
        assert lena_image is not None
        assert len(lena_image.shape) == 3
        assert lena_image.shape[-1] == 3  # RGB
        assert torch.all(lena_image >= 0) and torch.all(lena_image <= 1)
        
        height, width = lena_image.shape[:2]
        assert height > 0 and width > 0
        print(f"Loaded Lena image: {height}x{width}")
    
    def test_reconstruct_lena_with_gaussians(self, lena_image, device):
        """Test reconstructing Lena image using Gaussian splats."""
        height, width = lena_image.shape[:2]
        
        # Downsample for faster testing
        if height > 256 or width > 256:
            scale = min(256 / height, 256 / width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            lena_image = torch.nn.functional.interpolate(
                lena_image.permute(2, 0, 1).unsqueeze(0),
                size=(new_height, new_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)
            height, width = new_height, new_width
        
        # Sample points from the image
        num_gaussians = 100
        step_y = height // int(np.sqrt(num_gaussians))
        step_x = width // int(np.sqrt(num_gaussians))
        
        means = []
        colors = []
        
        for y in range(step_y // 2, height, step_y):
            for x in range(step_x // 2, width, step_x):
                if len(means) >= num_gaussians:
                    break
                means.append([float(x), float(y)])
                colors.append(lena_image[y, x].cpu().numpy())
            if len(means) >= num_gaussians:
                break
        
        means = torch.tensor(means, device=device)
        colors = torch.tensor(colors, device=device)
        
        # Create small circular Gaussians
        covariances = torch.eye(2, device=device).unsqueeze(0).repeat(len(means), 1, 1) * 100.0
        opacities = torch.ones(len(means), device=device) * 0.8
        
        # Render
        reconstructed = gaussian_splat_2d(
            means=means,
            covariances=covariances,
            colors=colors,
            opacities=opacities,
            height=height,
            width=width,
            device=device,
        )
        
        assert reconstructed.shape == (height, width, 3)
        assert torch.all(reconstructed >= 0) and torch.all(reconstructed <= 1)
        
        # Compute reconstruction error
        mse = torch.nn.functional.mse_loss(reconstructed, lena_image)
        print(f"Reconstruction MSE: {mse.item():.6f}")
        
        save_test_image(reconstructed, f"reconstructed_lena_{device}.png")
        save_test_image(lena_image, f"lena_original_{device}.png")
    
    def test_optimize_to_match_lena(self, lena_image, device):
        """Test optimizing Gaussian splats to match Lena image."""
        height, width = lena_image.shape[:2]
        
        # Downsample for faster testing
        if height > 128 or width > 128:
            scale = min(128 / height, 128 / width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            lena_image = torch.nn.functional.interpolate(
                lena_image.permute(2, 0, 1).unsqueeze(0),
                size=(new_height, new_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)
            height, width = new_height, new_width
        
        # Create model
        model = GaussianSplat2D(
            num_gaussians=50,
            num_channels=3,
            height=height,
            width=width,
            device=device,
        )
        
        # Move target to device
        target = lena_image.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        initial_loss = None
        losses = []
        
        print("Optimizing to match Lena image...")
        for i in range(100):
            optimizer.zero_grad()
            output = model()
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            if i == 0:
                initial_loss = loss.item()
            
            if (i + 1) % 20 == 0:
                print(f"  Iteration {i+1}, Loss: {loss.item():.6f}")
        
        final_loss = loss.item()
        assert final_loss < initial_loss, "Loss should decrease during optimization"
        assert final_loss < initial_loss * 0.5, "Loss should decrease significantly"
        
        final_output = model()
        save_test_image(final_output.detach(), f"optimized_lena_{device}.png")
        save_test_image(target, f"lena_target_{device}.png")
        
        # Compute PSNR
        mse = torch.nn.functional.mse_loss(final_output, target)
        psnr = -10 * torch.log10(mse + 1e-10)
        print(f"Final PSNR: {psnr.item():.2f} dB")
    
    def test_gradient_computation_with_lena(self, lena_image, device):
        """Test that gradients can be computed when optimizing to match Lena."""
        height, width = lena_image.shape[:2]
        
        # Downsample for faster testing
        if height > 64 or width > 64:
            scale = min(64 / height, 64 / width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            lena_image = torch.nn.functional.interpolate(
                lena_image.permute(2, 0, 1).unsqueeze(0),
                size=(new_height, new_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)
            height, width = new_height, new_width
        
        # Create learnable parameters
        num_gaussians = 20
        means = torch.randn(num_gaussians, 2, device=device, requires_grad=True) * min(width, height) * 0.1 + torch.tensor([width / 2, height / 2], device=device)
        covariances = torch.eye(2, device=device).unsqueeze(0).repeat(num_gaussians, 1, 1) * 200.0
        covariances.requires_grad_(True)
        colors = torch.rand(num_gaussians, 3, device=device, requires_grad=True)
        opacities = torch.ones(num_gaussians, device=device, requires_grad=True) * 0.5
        
        target = lena_image.to(device)
        
        # Forward and backward
        output = gaussian_splat_2d(
            means=means,
            covariances=covariances,
            colors=colors,
            opacities=opacities,
            height=height,
            width=width,
            device=device,
        )
        
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        
        # Check that gradients exist
        assert means.grad is not None
        assert covariances.grad is not None
        assert colors.grad is not None
        assert opacities.grad is not None
        
        # Check that gradients are not all zero
        assert torch.any(means.grad != 0)
        assert torch.any(colors.grad != 0)
        assert torch.any(opacities.grad != 0)
    
    def test_different_resolutions(self, lena_image, device):
        """Test rendering at different resolutions."""
        original_height, original_width = lena_image.shape[:2]
        
        # Test at different scales
        scales = [0.25, 0.5, 1.0]
        
        for scale in scales:
            height = int(original_height * scale)
            width = int(original_width * scale)
            
            # Sample a few points
            num_gaussians = 10
            means = torch.rand(num_gaussians, 2, device=device) * torch.tensor([width, height], device=device)
            covariances = torch.eye(2, device=device).unsqueeze(0).repeat(num_gaussians, 1, 1) * 100.0
            colors = torch.rand(num_gaussians, 3, device=device)
            opacities = torch.ones(num_gaussians, device=device) * 0.8
            
            image = gaussian_splat_2d(
                means=means,
                covariances=covariances,
                colors=colors,
                opacities=opacities,
                height=height,
                width=width,
                device=device,
            )
            
            assert image.shape == (height, width, 3)
            assert torch.all(image >= 0) and torch.all(image <= 1)
    
    def test_backend_consistency_with_lena(self, lena_image):
        """Test that different backends produce similar results when reconstructing Lena."""
        height, width = lena_image.shape[:2]
        
        # Downsample for faster testing
        if height > 128 or width > 128:
            scale = min(128 / height, 128 / width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            lena_image = torch.nn.functional.interpolate(
                lena_image.permute(2, 0, 1).unsqueeze(0),
                size=(new_height, new_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)
            height, width = new_height, new_width
        
        # Create test data
        torch.manual_seed(42)
        num_gaussians = 20
        means = torch.rand(num_gaussians, 2) * torch.tensor([width, height])
        covariances = torch.eye(2).unsqueeze(0).repeat(num_gaussians, 1, 1) * 200.0
        colors = torch.rand(num_gaussians, 3)
        opacities = torch.ones(num_gaussians) * 0.8
        
        results = {}
        
        # Test CPU
        results["cpu"] = gaussian_splat_2d(
            means=means,
            covariances=covariances,
            colors=colors,
            opacities=opacities,
            height=height,
            width=width,
            device="cpu",
        )
        
        # Test CUDA if available
        if torch.cuda.is_available():
            results["cuda"] = gaussian_splat_2d(
                means=means,
                covariances=covariances,
                colors=colors,
                opacities=opacities,
                height=height,
                width=width,
                device="cuda",
            )
            # Compare with CPU (allowing for small numerical differences)
            diff = torch.abs(results["cpu"] - results["cuda"].cpu())
            max_diff = diff.max().item()
            assert max_diff < 1e-4, f"CPU and CUDA results differ by {max_diff}"
        
        # Test MPS if available
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            results["mps"] = gaussian_splat_2d(
                means=means,
                covariances=covariances,
                colors=colors,
                opacities=opacities,
                height=height,
                width=width,
                device="mps",
            )
            # Compare with CPU (allowing for small numerical differences)
            diff = torch.abs(results["cpu"] - results["mps"].cpu())
            max_diff = diff.max().item()
            assert max_diff < 2e-3, f"CPU and MPS results differ by {max_diff}"

    def test_mps_compiled_backward_matches_cpu_gradients(self):
        """Test that the compiled MPS backward path matches CPU gradients on a small case."""
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("MPS is not available")

        torch.manual_seed(0)
        height, width = 8, 8
        num_gaussians = 4

        means_cpu = (
            torch.rand(num_gaussians, 2, dtype=torch.float32) * torch.tensor([width, height], dtype=torch.float32)
        ).requires_grad_()
        covariances_cpu = (torch.eye(2, dtype=torch.float32).unsqueeze(0).repeat(num_gaussians, 1, 1) * 10.0).requires_grad_()
        colors_cpu = torch.rand(num_gaussians, 3, dtype=torch.float32, requires_grad=True)
        opacities_cpu = torch.rand(num_gaussians, dtype=torch.float32, requires_grad=True)

        means_mps = means_cpu.detach().clone().to("mps").requires_grad_()
        covariances_mps = covariances_cpu.detach().clone().to("mps").requires_grad_()
        colors_mps = colors_cpu.detach().clone().to("mps").requires_grad_()
        opacities_mps = opacities_cpu.detach().clone().to("mps").requires_grad_()

        target = torch.rand(height, width, 3, dtype=torch.float32)

        output_cpu = gaussian_splat_2d(
            means_cpu,
            covariances_cpu,
            colors_cpu,
            opacities_cpu,
            height=height,
            width=width,
            device="cpu",
        )
        loss_cpu = torch.nn.functional.mse_loss(output_cpu, target)
        loss_cpu.backward()

        output_mps = gaussian_splat_2d(
            means_mps,
            covariances_mps,
            colors_mps,
            opacities_mps,
            height=height,
            width=width,
            device="mps",
        )
        loss_mps = torch.nn.functional.mse_loss(output_mps, target.to("mps"))
        loss_mps.backward()

        assert torch.allclose(means_cpu.grad, means_mps.grad.cpu(), atol=5e-3, rtol=5e-2)
        assert torch.allclose(covariances_cpu.grad, covariances_mps.grad.cpu(), atol=5e-3, rtol=5e-2)
        assert torch.allclose(colors_cpu.grad, colors_mps.grad.cpu(), atol=5e-3, rtol=5e-2)
        assert torch.allclose(opacities_cpu.grad, opacities_mps.grad.cpu(), atol=5e-3, rtol=5e-2)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
