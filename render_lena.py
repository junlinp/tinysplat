"""
Render Lena image using 2D Gaussian splatting.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import cv2

from tinysplat import GaussianSplat2D


def load_lena_image(image_path, device="cpu"):
    """
    Load the Lena test image using OpenCV.
    
    Args:
        image_path: Path to the image file.
        device: Device to load image on.
    
    Returns:
        Tensor of shape (H, W, 3) with values in [0, 1].
    """
    # Read image using OpenCV (reads in BGR format)
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img_array = img_rgb.astype(np.float32) / 255.0
    
    # Convert to tensor and move to device
    img_tensor = torch.from_numpy(img_array).to(device)
    
    return img_tensor


def save_image(image: torch.Tensor, filename: str):
    """Save an image using OpenCV."""
    # Convert to numpy and handle different channel counts
    img_np = image.detach().cpu().numpy()
    
    # Clamp values to [0, 1]
    img_np = np.clip(img_np, 0.0, 1.0)
    
    # Convert to uint8 [0, 255]
    img_np = (img_np * 255.0).astype(np.uint8)
    
    if img_np.shape[-1] == 1:
        # Grayscale - cv2.imwrite expects 2D array for grayscale
        cv2.imwrite(str(filename), img_np.squeeze(-1))
    elif img_np.shape[-1] == 3:
        # RGB - convert to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(filename), img_bgr)
    elif img_np.shape[-1] == 4:
        # RGBA - convert to RGB then BGR
        img_rgb = img_np[..., :3]
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(filename), img_bgr)
    else:
        # Save first channel as grayscale
        cv2.imwrite(str(filename), img_np[..., 0])
    
    print(f"Saved image: {filename}")


def main():
    """Main function to render Lena image with Gaussian splatting."""
    # Paths
    lena_path = Path(__file__).parent / "tests" / "data" / "lena.png"
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    if not lena_path.exists():
        print(f"Error: Lena image not found at {lena_path}")
        print("Please ensure the image exists or run tests/download_lena.py first.")
        return
    
    # Auto-detect device
    # Note: MPS backend may have issues with backward pass, so prefer CPU/CUDA
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Try MPS, but fall back to CPU if backward fails
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    print(f"Loading Lena image from {lena_path}...")
    
    # Load Lena image
    lena_image = load_lena_image(lena_path, device=device)
    height, width = lena_image.shape[:2]
    print(f"Loaded image: {height}x{width}")
    
    # Optionally downsample for faster optimization
    # Uncomment the following lines to downsample:
    # if height > 256 or width > 256:
    #     scale = min(256 / height, 256 / width)
    #     new_height = int(height * scale)
    #     new_width = int(width * scale)
    #     lena_image = F.interpolate(
    #         lena_image.permute(2, 0, 1).unsqueeze(0),
    #         size=(new_height, new_width),
    #         mode="bilinear",
    #         align_corners=False,
    #     ).squeeze(0).permute(1, 2, 0)
    #     height, width = new_height, new_width
    #     print(f"Downsampled to: {height}x{width}")
    
    # Save original image
    save_image(lena_image, str(output_dir / "lena_original.png"))
    
    # Create Gaussian splatting model
    num_gaussians = 500  # Increase for better quality (but slower)
    print(f"\nCreating GaussianSplat2D model with {num_gaussians} Gaussians...")
    
    # Try to create model, fall back to CPU if MPS fails
    try:
        model = GaussianSplat2D(
            num_gaussians=num_gaussians,
            num_channels=3,
            height=height,
            width=width,
            device=device,
        )
    except Exception as e:
        print(f"Warning: Failed to create model on {device}: {e}")
        print("Falling back to CPU...")
        device = "cpu"
        model = GaussianSplat2D(
            num_gaussians=num_gaussians,
            num_channels=3,
            height=height,
            width=width,
            device=device,
        )
    
    # Move target to device
    target = lena_image.to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    num_iterations = 500
    print(f"\nOptimizing for {num_iterations} iterations...")
    print("This may take a while depending on image size and number of Gaussians.")
    
    initial_loss = None
    best_loss = float('inf')
    best_iteration = 0
    
    for i in range(num_iterations):
        optimizer.zero_grad()
        output = model()
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        
        loss_value = loss.item()
        if initial_loss is None:
            initial_loss = loss_value
        
        if loss_value < best_loss:
            best_loss = loss_value
            best_iteration = i
        
        # Print progress
        if (i + 1) % 50 == 0:
            psnr = -10 * torch.log10(loss + 1e-10)
            print(f"Iteration {i+1}/{num_iterations}, Loss: {loss_value:.6f}, PSNR: {psnr.item():.2f} dB")
            
            # Save intermediate result
            save_image(output.detach(), str(output_dir / f"lena_iter_{i+1}.png"))
    
    # Final results
    final_output = model()
    final_loss = F.mse_loss(final_output, target)
    final_psnr = -10 * torch.log10(final_loss + 1e-10)
    
    print(f"\nOptimization complete!")
    print(f"Initial Loss: {initial_loss:.6f}")
    print(f"Final Loss: {final_loss.item():.6f}")
    print(f"Final PSNR: {final_psnr.item():.2f} dB")
    print(f"Best Loss: {best_loss:.6f} (at iteration {best_iteration + 1})")
    
    # Save final result
    save_image(final_output.detach(), str(output_dir / "lena_reconstructed.png"))
    
    # Create comparison image using OpenCV (side-by-side)
    original_np = lena_image.detach().cpu().numpy()
    reconstructed_np = final_output.detach().cpu().numpy()
    
    # Clamp and convert to uint8
    original_np = np.clip(original_np, 0.0, 1.0)
    reconstructed_np = np.clip(reconstructed_np, 0.0, 1.0)
    
    original_uint8 = (original_np * 255.0).astype(np.uint8)
    reconstructed_uint8 = (reconstructed_np * 255.0).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    original_bgr = cv2.cvtColor(original_uint8, cv2.COLOR_RGB2BGR)
    reconstructed_bgr = cv2.cvtColor(reconstructed_uint8, cv2.COLOR_RGB2BGR)
    
    # Concatenate side by side
    comparison = np.hstack([original_bgr, reconstructed_bgr])
    
    # Add text labels
    cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(comparison, f"Reconstructed (PSNR: {final_psnr.item():.2f} dB)", 
                (original_bgr.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite(str(output_dir / "lena_comparison.png"), comparison)
    print(f"\nSaved comparison image: {output_dir / 'lena_comparison.png'}")
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
