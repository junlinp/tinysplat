"""Pure PyTorch fallback backend implementations."""

import torch


DEFAULT_GAUSSIAN_CHUNK_SIZE = 64


def forward_pytorch(
    means: torch.Tensor,
    covariances: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    height: int,
    width: int,
) -> tuple:
    """Chunked PyTorch forward pass used as the default fallback backend."""
    num_gaussians = means.shape[0]
    num_channels = colors.shape[1]
    device = means.device

    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    coords = torch.stack([x_coords, y_coords], dim=-1)

    image_numerator = torch.zeros(height, width, num_channels, dtype=colors.dtype, device=device)
    total_weight = torch.zeros(height, width, dtype=means.dtype, device=device)

    for start_idx in range(0, num_gaussians, DEFAULT_GAUSSIAN_CHUNK_SIZE):
        end_idx = min(start_idx + DEFAULT_GAUSSIAN_CHUNK_SIZE, num_gaussians)

        means_chunk = means[start_idx:end_idx]
        covariances_chunk = covariances[start_idx:end_idx]
        colors_chunk = colors[start_idx:end_idx]
        opacities_chunk = opacities[start_idx:end_idx]

        diff = coords.unsqueeze(0) - means_chunk.unsqueeze(1).unsqueeze(1)

        inv_covariances = torch.linalg.inv(covariances_chunk)
        diff_expanded = diff.unsqueeze(-1)
        inv_cov_expanded = inv_covariances.unsqueeze(1).unsqueeze(1)

        quad_form = torch.matmul(
            torch.matmul(diff.unsqueeze(-2), inv_cov_expanded),
            diff_expanded,
        )
        quad_form = quad_form.squeeze(-1).squeeze(-1)

        gaussian_values = torch.exp(-0.5 * quad_form)

        det_covariances = torch.linalg.det(covariances_chunk)
        normalization = 1.0 / (2 * torch.pi * torch.sqrt(det_covariances + 1e-8))
        gaussian_values = gaussian_values * normalization.unsqueeze(1).unsqueeze(2)

        weighted_gaussians = gaussian_values * opacities_chunk.unsqueeze(1).unsqueeze(2)
        total_weight = total_weight + weighted_gaussians.sum(dim=0)
        image_numerator = image_numerator + (
            weighted_gaussians.unsqueeze(-1)
            * colors_chunk.unsqueeze(1).unsqueeze(1)
        ).sum(dim=0)

    image = image_numerator / torch.clamp(total_weight.unsqueeze(-1), min=1e-8)

    if num_channels == 4:
        image[..., :3] = image[..., :3] * image[..., 3:4]

    return image, []


def backward_pytorch(
    grad_output: torch.Tensor,
    means: torch.Tensor,
    covariances: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    height: int,
    width: int,
    intermediates: list,
    needs_input_grad: tuple,
):
    """Fallback backward pass that replays the PyTorch forward graph."""
    del intermediates

    means_requires_grad = needs_input_grad[0] if len(needs_input_grad) > 0 else False
    cov_requires_grad = needs_input_grad[1] if len(needs_input_grad) > 1 else False
    colors_requires_grad = needs_input_grad[2] if len(needs_input_grad) > 2 else False
    opacities_requires_grad = needs_input_grad[3] if len(needs_input_grad) > 3 else False

    means_new = means.detach().clone().requires_grad_(means_requires_grad)
    cov_new = covariances.detach().clone().requires_grad_(cov_requires_grad)
    colors_new = colors.detach().clone().requires_grad_(colors_requires_grad)
    opacities_new = opacities.detach().clone().requires_grad_(opacities_requires_grad)

    inputs = []
    if means_requires_grad:
        inputs.append(means_new)
    if cov_requires_grad:
        inputs.append(cov_new)
    if colors_requires_grad:
        inputs.append(colors_new)
    if opacities_requires_grad:
        inputs.append(opacities_new)

    if inputs:
        try:
            output, _ = forward_pytorch(
                means_new,
                cov_new,
                colors_new,
                opacities_new,
                height,
                width,
            )
            grads = torch.autograd.grad(
                outputs=output,
                inputs=inputs,
                grad_outputs=grad_output,
                retain_graph=False,
                only_inputs=True,
                allow_unused=True,
            )
        except RuntimeError:
            grads = [None] * len(inputs)
    else:
        grads = []

    grad_idx = 0
    grad_means = grads[grad_idx] if means_requires_grad else None
    if means_requires_grad:
        grad_idx += 1
    grad_cov = grads[grad_idx] if cov_requires_grad else None
    if cov_requires_grad:
        grad_idx += 1
    grad_colors = grads[grad_idx] if colors_requires_grad else None
    if colors_requires_grad:
        grad_idx += 1
    grad_opacities = grads[grad_idx] if opacities_requires_grad else None

    return grad_means, grad_cov, grad_colors, grad_opacities
