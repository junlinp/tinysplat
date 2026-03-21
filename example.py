"""
Train 2D Gaussian splats to reconstruct the Lena image.

The script:
1. Loads ``tests/data/lena.png``
2. Builds one Gaussian per pixel of the working image
3. Optimizes those Gaussians with the ``GaussianSplat2D`` PyTorch module
4. Writes outputs into a temporary directory under the repository root

The reference implementation in ``tinysplat/gaussian_splat_2d.py`` scales as
O(num_gaussians * height * width), so the script downsamples Lena by default to
keep the example runnable.
"""

import argparse
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from tinysplat import GaussianSplat2D


REPO_ROOT = Path(__file__).resolve().parent
LENA_PATH = REPO_ROOT / "tests" / "data" / "lena.png"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Resize Lena to a square working resolution before training. Defaults to full resolution.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of optimization steps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-2,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use for training and rendering.",
    )
    parser.add_argument(
        "--save-initial-render",
        action="store_true",
        help="Render and save the initial image before training. Disabled by default because it is expensive.",
    )
    return parser.parse_args()


def choose_device():
    """Pick the best available torch device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(device_arg: str) -> str:
    """Resolve the requested device, validating availability when needed."""
    if device_arg == "auto":
        return choose_device()
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if device_arg == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available.")
    return device_arg


def load_lena_image(image_size: int, device: str) -> torch.Tensor:
    """Load and resize the Lena image from the repo's test data directory."""
    if not LENA_PATH.exists():
        raise FileNotFoundError(
            f"Expected Lena image at {LENA_PATH}. Run tests/download_lena.py first."
        )

    image_bgr = cv2.imread(str(LENA_PATH), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image from {LENA_PATH}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if image_size is not None:
        image_rgb = cv2.resize(
            image_rgb,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )

    image = torch.from_numpy(image_rgb.astype(np.float32) / 255.0).to(device)
    return image


def save_image(image: torch.Tensor, output_path: Path):
    """Save a tensor image to disk."""
    image_np = image.detach().cpu().clamp(0.0, 1.0).numpy()
    image_np = (image_np * 255.0).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), image_bgr)


def compute_grid_size(length: int) -> int:
    """Compute the Gaussian grid resolution for one image axis."""
    return max(1, length // 4)


def build_pixel_gaussians(target: torch.Tensor) -> dict:
    """
    Create one learnable Gaussian per 4x4 image region.

    The Gaussians start on a regular coarse grid with colors initialized from
    pooled image patches.
    """
    height, width, _ = target.shape
    grid_height = compute_grid_size(height)
    grid_width = compute_grid_size(width)
    dtype = target.dtype
    device = target.device

    y_coords = (
        (torch.arange(grid_height, device=device, dtype=dtype) + 0.5)
        * (height / grid_height)
        - 0.5
    ).clamp(0, height - 1)
    x_coords = (
        (torch.arange(grid_width, device=device, dtype=dtype) + 0.5)
        * (width / grid_width)
        - 0.5
    ).clamp(0, width - 1)

    means = torch.stack(
        [
            x_coords.repeat(grid_height),
            y_coords.repeat_interleave(grid_width),
        ],
        dim=1,
    )

    pooled = F.adaptive_avg_pool2d(
        target.permute(2, 0, 1).unsqueeze(0),
        output_size=(grid_height, grid_width),
    )
    colors = pooled.squeeze(0).permute(1, 2, 0).reshape(-1, target.shape[-1]).contiguous()

    # Start each Gaussian with a footprint roughly aligned to a 4x4 region.
    initial_scale = 2.0
    log_scales = torch.full(
        (grid_height * grid_width, 2),
        math.log(initial_scale),
        device=device,
        dtype=dtype,
    )
    rotations = torch.zeros(grid_height * grid_width, device=device, dtype=dtype)

    # GaussianSplat2D applies sigmoid() to opacities in forward().
    initial_alpha = 0.9
    initial_opacity_logit = torch.logit(
        torch.tensor(initial_alpha, device=device, dtype=dtype)
    ).item()
    opacity_logits = torch.full(
        (grid_height * grid_width,),
        initial_opacity_logit,
        device=device,
        dtype=dtype,
    )

    return {
        "means": means,
        "log_scales": log_scales,
        "rotations": rotations,
        "colors": colors,
        "opacities": opacity_logits,
    }


def main():
    """Run the Lena reconstruction example."""
    args = parse_args()
    device = resolve_device(args.device)

    target = load_lena_image(args.image_size, device)
    height, width, _ = target.shape
    num_gaussians = compute_grid_size(height) * compute_grid_size(width)

    output_dir = Path(
        tempfile.mkdtemp(prefix="tinysplat_example_", dir=REPO_ROOT)
    )

    print(f"Using device: {device}")
    print(f"Loaded Lena from: {LENA_PATH}")
    print(f"Working resolution: {width}x{height}")
    print(f"Generating {num_gaussians} Gaussian splats ((H/4) * (W/4))")
    print(f"Output directory: {output_dir}")

    gaussians = build_pixel_gaussians(target)
    model = GaussianSplat2D(
        gaussians=gaussians,
        height=height,
        width=width,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_image(target, output_dir / "target.png")

    initial_render = None
    if args.save_initial_render:
        print("Rendering initial image...")
        initial_render = model().detach()
        save_image(initial_render, output_dir / "render_initial.png")

    print(f"Training for {args.iterations} iterations...")
    progress = tqdm(range(args.iterations), desc="Training", unit="iter")

    for step in progress:
        optimizer.zero_grad()
        rendered = model()
        loss = F.mse_loss(rendered, target)
        loss.backward()
        optimizer.step()

        psnr = -10.0 * torch.log10(loss.detach() + 1e-10)
        progress.set_postfix(loss=f"{loss.item():.6f}", psnr=f"{psnr.item():.2f} dB")

    final_render = model().detach()
    final_loss = F.mse_loss(final_render, target)
    final_psnr = -10.0 * torch.log10(final_loss + 1e-10)

    save_image(final_render, output_dir / "render_final.png")

    if initial_render is None:
        comparison = torch.cat([target, final_render], dim=1)
    else:
        comparison = torch.cat([target, initial_render, final_render], dim=1)
    save_image(comparison, output_dir / "comparison.png")

    print("Training complete.")
    print(f"Final loss: {final_loss.item():.6f}")
    print(f"Final PSNR: {final_psnr.item():.2f} dB")
    if initial_render is None:
        print(f"Saved target, final render, and comparison to: {output_dir}")
    else:
        print(f"Saved target, initial render, final render, and comparison to: {output_dir}")


if __name__ == "__main__":
    main()
