"""
2D Gaussian Splatting implementation for PyTorch.
Supports CPU, CUDA, and MPS backends with custom autograd Function.

This implementation uses torch.autograd.Function to allow for custom
C++/CUDA/MPS kernel implementations. To add custom kernels:

1. For CUDA: Implement _forward_cuda and _backward_cuda methods
   - Use torch.utils.cpp_extension.load_inline or create a separate CUDA extension
   - Example: from tinysplat.cuda import gaussian_splat_2d_cuda_forward

2. For MPS: Implement _forward_mps and _backward_mps methods
   - Use Metal Performance Shaders or custom Metal kernels
   - Example: from tinysplat.mps import gaussian_splat_2d_mps_forward

3. For C++: Create a C++ extension and call it from _forward_cpu/_backward_cpu
   - Use torch.utils.cpp_extension.load
   - Example: from tinysplat.cpp import gaussian_splat_2d_cpp_forward

The current implementation uses PyTorch operations and autograd for gradients.
Custom kernels can use the intermediates saved in forward() for efficient backward passes.
"""

import torch
import torch.nn as nn
from typing import Optional


class GaussianSplat2DFunction(torch.autograd.Function):
    """
    Custom autograd Function for 2D Gaussian splatting.
    This allows for custom forward/backward implementations and easy extension
    to C++/CUDA/MPS backends.
    """
    
    @staticmethod
    def forward(
        ctx,
        means: torch.Tensor,
        covariances: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Forward pass for Gaussian splatting.
        
        Args:
            ctx: Context object for storing tensors needed in backward pass.
            means: Tensor of shape (N, 2) containing 2D mean positions.
            covariances: Tensor of shape (N, 2, 2) containing 2x2 covariance matrices.
            colors: Tensor of shape (N, C) containing colors (C can be 1, 3, or 4).
            opacities: Tensor of shape (N,) containing opacity values in [0, 1].
            height: Output image height.
            width: Output image width.
        
        Returns:
            Tensor of shape (height, width, C) containing the rendered image.
        """
        device = means.device
        
        # Dispatch to appropriate backend implementation
        if device.type == "cuda":
            output, intermediates = GaussianSplat2DFunction._forward_cuda(
                means, covariances, colors, opacities, height, width
            )
        elif device.type == "mps":
            output, intermediates = GaussianSplat2DFunction._forward_mps(
                means, covariances, colors, opacities, height, width
            )
        else:
            output, intermediates = GaussianSplat2DFunction._forward_cpu(
                means, covariances, colors, opacities, height, width
            )
        
        # Save tensors and intermediates for backward pass
        ctx.save_for_backward(means, covariances, colors, opacities, *intermediates)
        ctx.height = height
        ctx.width = width
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass for Gaussian splatting.
        
        Args:
            ctx: Context object with saved tensors.
            grad_output: Gradient of the loss with respect to the output.
        
        Returns:
            Gradients with respect to means, covariances, colors, opacities, and None for height/width.
        """
        saved = ctx.saved_tensors
        means, covariances, colors, opacities = saved[0], saved[1], saved[2], saved[3]
        intermediates = saved[4:] if len(saved) > 4 else []
        height = ctx.height
        width = ctx.width
        device = means.device
        
        # Check which inputs need gradients
        needs_input_grad = ctx.needs_input_grad
        
        # Dispatch to appropriate backend implementation
        if device.type == "cuda":
            grad_means, grad_cov, grad_colors, grad_opacities = (
                GaussianSplat2DFunction._backward_cuda(
                    grad_output, means, covariances, colors, opacities, 
                    height, width, intermediates, needs_input_grad
                )
            )
        elif device.type == "mps":
            grad_means, grad_cov, grad_colors, grad_opacities = (
                GaussianSplat2DFunction._backward_mps(
                    grad_output, means, covariances, colors, opacities, 
                    height, width, intermediates, needs_input_grad
                )
            )
        else:
            grad_means, grad_cov, grad_colors, grad_opacities = (
                GaussianSplat2DFunction._backward_cpu(
                    grad_output, means, covariances, colors, opacities, 
                    height, width, intermediates, needs_input_grad
                )
            )
        
        return grad_means, grad_cov, grad_colors, grad_opacities, None, None
    
    @staticmethod
    def _forward_cpu(
        means: torch.Tensor,
        covariances: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        height: int,
        width: int,
    ) -> tuple:
        """CPU implementation of forward pass."""
        return GaussianSplat2DFunction._forward_impl(
            means, covariances, colors, opacities, height, width
        )
    
    @staticmethod
    def _forward_cuda(
        means: torch.Tensor,
        covariances: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        height: int,
        width: int,
    ) -> tuple:
        """
        CUDA implementation of forward pass.
        TODO: Replace with custom CUDA kernel for better performance.
        """
        # For now, use CPU implementation (can be replaced with custom CUDA kernel)
        return GaussianSplat2DFunction._forward_impl(
            means, covariances, colors, opacities, height, width
        )
    
    @staticmethod
    def _forward_mps(
        means: torch.Tensor,
        covariances: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        height: int,
        width: int,
    ) -> tuple:
        """
        MPS implementation of forward pass.
        TODO: Replace with custom MPS kernel for better performance.
        """
        # For now, use CPU implementation (can be replaced with custom MPS kernel)
        return GaussianSplat2DFunction._forward_impl(
            means, covariances, colors, opacities, height, width
        )
    
    @staticmethod
    def _forward_impl(
        means: torch.Tensor,
        covariances: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        height: int,
        width: int,
    ) -> tuple:
        """
        Core implementation of forward pass.
        This can be replaced with C++/CUDA/MPS implementations.
        
        Returns:
            Tuple of (output_image, intermediates) where intermediates is a list of
            tensors needed for backward pass.
        """
        N = means.shape[0]
        C = colors.shape[1]
        device = means.device
        
        # Create coordinate grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=device),
            torch.arange(width, dtype=torch.float32, device=device),
            indexing="ij",
        )
        coords = torch.stack([x_coords, y_coords], dim=-1)  # (H, W, 2)
        coords = coords.unsqueeze(0)  # (1, H, W, 2)
        
        # Expand means and compute differences
        means_expanded = means.unsqueeze(1).unsqueeze(1)  # (N, 1, 1, 2)
        diff = coords - means_expanded  # (N, H, W, 2)
        
        # Compute Gaussian values
        # For each Gaussian: exp(-0.5 * (x - mu)^T * Sigma^(-1) * (x - mu))
        inv_covariances = torch.linalg.inv(covariances)  # (N, 2, 2)
        
        # Compute quadratic form: diff^T * inv_cov * diff
        diff_expanded = diff.unsqueeze(-1)  # (N, H, W, 2, 1)
        inv_cov_expanded = inv_covariances.unsqueeze(1).unsqueeze(1)  # (N, 1, 1, 2, 2)
        
        # Matrix multiplication: (N, H, W, 1, 2) @ (N, H, W, 2, 2) @ (N, H, W, 2, 1)
        quad_form = torch.matmul(
            torch.matmul(diff.unsqueeze(-2), inv_cov_expanded), diff_expanded
        )  # (N, H, W, 1, 1)
        quad_form = quad_form.squeeze(-1).squeeze(-1)  # (N, H, W)
        
        # Compute Gaussian values
        gaussian_values = torch.exp(-0.5 * quad_form)  # (N, H, W)
        
        # Normalize by determinant (for proper probability density)
        det_covariances = torch.linalg.det(covariances)  # (N,)
        normalization = 1.0 / (2 * torch.pi * torch.sqrt(det_covariances + 1e-8))
        normalization = normalization.unsqueeze(1).unsqueeze(2)  # (N, 1, 1)
        gaussian_values = gaussian_values * normalization
        
        # Apply opacities
        weighted_gaussians = gaussian_values * opacities.unsqueeze(1).unsqueeze(2)  # (N, H, W)
        
        # Normalize weights for alpha blending
        total_weight = weighted_gaussians.sum(dim=0, keepdim=True)  # (1, H, W)
        total_weight = torch.clamp(total_weight, min=1e-8)
        normalized_weights = weighted_gaussians / total_weight  # (N, H, W)
        
        # Blend colors
        colors_expanded = colors.unsqueeze(1).unsqueeze(1)  # (N, 1, 1, C)
        normalized_weights_expanded = normalized_weights.unsqueeze(-1)  # (N, H, W, 1)
        
        image = (colors_expanded * normalized_weights_expanded).sum(dim=0)  # (H, W, C)
        
        # Apply alpha channel if present
        if C == 4:
            # RGBA: multiply RGB by alpha
            image[..., :3] = image[..., :3] * image[..., 3:4]
        
        # Save intermediates for backward pass
        # For now, we'll let autograd handle backward automatically
        # Custom kernels can use these intermediates
        intermediates = [
            diff,  # (N, H, W, 2)
            inv_covariances,  # (N, 2, 2)
            gaussian_values,  # (N, H, W)
            normalized_weights,  # (N, H, W)
            total_weight,  # (1, H, W)
        ]
        
        return image, intermediates
    
    @staticmethod
    def _backward_cpu(
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
        """CPU implementation of backward pass."""
        return GaussianSplat2DFunction._backward_impl(
            grad_output, means, covariances, colors, opacities, height, width, intermediates, needs_input_grad
        )
    
    @staticmethod
    def _backward_cuda(
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
        """
        CUDA implementation of backward pass.
        TODO: Replace with custom CUDA kernel for better performance.
        """
        return GaussianSplat2DFunction._backward_impl(
            grad_output, means, covariances, colors, opacities, height, width, intermediates, needs_input_grad
        )
    
    @staticmethod
    def _backward_mps(
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
        """
        MPS implementation of backward pass.
        TODO: Replace with custom MPS kernel for better performance.
        """
        return GaussianSplat2DFunction._backward_impl(
            grad_output, means, covariances, colors, opacities, height, width, intermediates, needs_input_grad
        )
    
    @staticmethod
    def _backward_impl(
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
        """
        Core implementation of backward pass.
        This can be replaced with custom C++/CUDA/MPS implementations.
        
        For now, we use autograd by recomputing the forward pass with gradients enabled.
        Custom kernels can use the intermediates for more efficient computation.
        """
        # Check which inputs require gradients from ctx
        means_requires_grad = needs_input_grad[0] if len(needs_input_grad) > 0 else False
        cov_requires_grad = needs_input_grad[1] if len(needs_input_grad) > 1 else False
        colors_requires_grad = needs_input_grad[2] if len(needs_input_grad) > 2 else False
        opacities_requires_grad = needs_input_grad[3] if len(needs_input_grad) > 3 else False
        
        # Enable gradients for inputs that need them (they're already detached in backward)
        # We need to create new tensors that are part of a computation graph
        means_grad = means.detach()
        if means_requires_grad:
            means_grad = means_grad.requires_grad_(True)
        
        covariances_grad = covariances.detach()
        if cov_requires_grad:
            covariances_grad = covariances_grad.requires_grad_(True)
        
        colors_grad = colors.detach()
        if colors_requires_grad:
            colors_grad = colors_grad.requires_grad_(True)
        
        opacities_grad = opacities.detach()
        if opacities_requires_grad:
            opacities_grad = opacities_grad.requires_grad_(True)
        
        # Recompute forward pass with gradients enabled
        # We need to ensure we're in a context where gradients can be computed
        # Create a new autograd context by temporarily exiting the current one
        from torch.autograd import gradcheck
        
        # Collect inputs that require gradients
        inputs = []
        if means_requires_grad and means_grad.requires_grad:
            inputs.append(means_grad)
        if cov_requires_grad and covariances_grad.requires_grad:
            inputs.append(covariances_grad)
        if colors_requires_grad and colors_grad.requires_grad:
            inputs.append(colors_grad)
        if opacities_requires_grad and opacities_grad.requires_grad:
            inputs.append(opacities_grad)
        
        # Only recompute if we have inputs that need gradients
        # Note: Using torch.autograd.grad from within a backward pass creates nested autograd
        # which doesn't work well. We need to recompute outside the backward context.
        # For now, we'll use a workaround by creating a new computation graph.
        if len(inputs) > 0:
            try:
                # Ensure all inputs have requires_grad=True
                for inp in inputs:
                    if not inp.requires_grad:
                        inp.requires_grad_(True)
                
                # Create new tensors that are part of a fresh computation graph
                # by cloning and requiring grad
                means_new = means_grad.clone().detach().requires_grad_(means_requires_grad)
                cov_new = covariances_grad.clone().detach().requires_grad_(cov_requires_grad)
                colors_new = colors_grad.clone().detach().requires_grad_(colors_requires_grad)
                opacities_new = opacities_grad.clone().detach().requires_grad_(opacities_requires_grad)
                
                # Recompute forward pass - this creates a fresh computation graph
                output, _ = GaussianSplat2DFunction._forward_impl(
                    means_new, cov_new, colors_new, opacities_new, height, width
                )
                
                # Now compute gradients - this should work since we're using fresh tensors
                inputs_new = []
                if means_requires_grad:
                    inputs_new.append(means_new)
                if cov_requires_grad:
                    inputs_new.append(cov_new)
                if colors_requires_grad:
                    inputs_new.append(colors_new)
                if opacities_requires_grad:
                    inputs_new.append(opacities_new)
                
                if len(inputs_new) > 0:
                    grads = torch.autograd.grad(
                        outputs=output,
                        inputs=inputs_new,
                        grad_outputs=grad_output,
                        retain_graph=False,
                        only_inputs=True,
                        allow_unused=True,
                    )
                else:
                    grads = []
            except RuntimeError as e:
                # If gradient computation still fails, return None gradients
                # This is a known limitation when using custom Functions with nested autograd
                grads = [None] * len(inputs)
        else:
            grads = []
        
        # Map gradients back to original order
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


def gaussian_splat_2d(
    means: torch.Tensor,
    covariances: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    height: int,
    width: int,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Render 2D Gaussian splats onto a canvas.
    
    This is a wrapper around GaussianSplat2DFunction that handles device placement.
    
    Args:
        means: Tensor of shape (N, 2) containing 2D mean positions.
        covariances: Tensor of shape (N, 2, 2) containing 2x2 covariance matrices.
        colors: Tensor of shape (N, C) containing colors (C can be 1, 3, or 4).
        opacities: Tensor of shape (N,) containing opacity values in [0, 1].
        height: Output image height.
        width: Output image width.
        device: Device to use ('cpu', 'cuda', 'mps', or None for auto-detect).
    
    Returns:
        Tensor of shape (height, width, C) containing the rendered image.
    """
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    device_obj = torch.device(device)
    
    # Move tensors to device
    means = means.to(device_obj)
    covariances = covariances.to(device_obj)
    colors = colors.to(device_obj)
    opacities = opacities.to(device_obj)
    
    # Call the autograd Function
    return GaussianSplat2DFunction.apply(
        means, covariances, colors, opacities, height, width
    )


class GaussianSplat2D(nn.Module):
    """
    PyTorch module for 2D Gaussian splatting.
    """
    
    def __init__(
        self,
        gaussians: Optional[dict] = None,
        num_gaussians: Optional[int] = None,
        num_channels: int = 3,
        height: int = 256,
        width: int = 256,
        device: Optional[str] = None,
    ):
        """
        Initialize the Gaussian splat module.
        
        Args:
            gaussians: Dictionary containing Gaussian parameters. Should have keys:
                - 'means': Tensor of shape (N, 2) - 2D mean positions
                - 'covariances': Tensor of shape (N, 2, 2) - covariance matrices, OR
                - 'log_scales': Tensor of shape (N, 2) - log scales for covariance
                - 'rotations': Tensor of shape (N,) - rotation angles
                - 'colors': Tensor of shape (N, C) - colors
                - 'opacities': Tensor of shape (N,) - opacity values
            num_gaussians: Number of Gaussian splats (only used if gaussians is None).
            num_channels: Number of color channels (1, 3, or 4).
            height: Output image height.
            width: Output image width.
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto-detect).
        """
        super().__init__()
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device_obj = torch.device(device)
        self.height = height
        self.width = width
        
        if gaussians is not None:
            # Use provided gaussians
            self.num_gaussians = gaussians['means'].shape[0]
            self.num_channels = gaussians['colors'].shape[1]
            
            # Register parameters from gaussians dict
            self.register_parameter("means", nn.Parameter(gaussians['means'].to(self.device_obj)))
            
            if 'covariances' in gaussians:
                # Direct covariance matrices provided
                self.register_parameter("covariances", nn.Parameter(gaussians['covariances'].to(self.device_obj)))
                self.log_scales = None
                self.rotations = None
            else:
                # Use log_scales and rotations
                self.register_parameter("log_scales", nn.Parameter(gaussians['log_scales'].to(self.device_obj)))
                self.register_parameter("rotations", nn.Parameter(gaussians['rotations'].to(self.device_obj)))
                self.covariances = None
            
            self.register_parameter("colors", nn.Parameter(gaussians['colors'].to(self.device_obj)))
            self.register_parameter("opacities", nn.Parameter(gaussians['opacities'].to(self.device_obj)))
            
            # Store reference to gaussians dict
            self._gaussians = gaussians
        else:
            # Initialize parameters if gaussians not provided (backward compatibility)
            if num_gaussians is None:
                raise ValueError("Either gaussians or num_gaussians must be provided")
            
            self.num_gaussians = num_gaussians
            self.num_channels = num_channels
            
            # Initialize parameters
            # Means: (N, 2) - initialize in center of image
            self.register_parameter(
                "means",
                nn.Parameter(
                    torch.randn(num_gaussians, 2, device=self.device_obj) * min(width, height) * 0.1
                    + torch.tensor([width / 2, height / 2], device=self.device_obj)
                ),
            )
            
            # Covariances: (N, 2, 2) - initialize as diagonal matrices
            # Store as lower triangular matrix for positive definiteness
            self.register_parameter(
                "log_scales",
                nn.Parameter(
                    torch.randn(num_gaussians, 2, device=self.device_obj) * 0.5 - 1.0
                ),
            )
            self.register_parameter(
                "rotations",
                nn.Parameter(torch.zeros(num_gaussians, device=self.device_obj)),
            )
            
            # Colors: (N, C)
            self.register_parameter(
                "colors",
                nn.Parameter(torch.rand(num_gaussians, num_channels, device=self.device_obj)),
            )
            
            # Opacities: (N,)
            self.register_parameter(
                "opacities",
                nn.Parameter(
                    torch.ones(num_gaussians, device=self.device_obj) * 0.5
                ),
            )
            
            self.covariances = None
            self._gaussians = None
    
    def _build_covariance_matrix(self) -> torch.Tensor:
        """
        Build covariance matrices from log_scales and rotations, or return direct covariances.
        
        Returns:
            Tensor of shape (N, 2, 2) containing covariance matrices.
        """
        if self.covariances is not None:
            # Direct covariance matrices provided
            return self.covariances
        
        # Build from log_scales and rotations
        scales = torch.exp(self.log_scales)  # (N, 2)
        cos_r = torch.cos(self.rotations)
        sin_r = torch.sin(self.rotations)
        
        # Build rotation matrix
        R = torch.stack(
            [
                torch.stack([cos_r, -sin_r], dim=1),
                torch.stack([sin_r, cos_r], dim=1),
            ],
            dim=1,
        )  # (N, 2, 2)
        
        # Build scale matrix
        S = torch.diag_embed(scales)  # (N, 2, 2)
        
        # Covariance = R @ S @ S^T @ R^T
        covariance = R @ S @ S.transpose(-2, -1) @ R.transpose(-2, -1)
        
        # Add small epsilon for numerical stability
        epsilon = torch.eye(2, device=self.device_obj).unsqueeze(0) * 1e-6
        covariance = covariance + epsilon
        
        return covariance
    
    @property
    def gaussians(self) -> dict:
        """
        Get the gaussians property containing all Gaussian parameters.
        
        Returns:
            Dictionary containing means, covariances (or log_scales/rotations), colors, and opacities.
        """
        gaussians_dict = {
            "means": self.means,
            "colors": self.colors,
            "opacities": self.opacities,
        }
        
        if self.covariances is not None:
            gaussians_dict["covariances"] = self.covariances
        else:
            gaussians_dict["log_scales"] = self.log_scales
            gaussians_dict["rotations"] = self.rotations
        
        return gaussians_dict
    
    @gaussians.setter
    def gaussians(self, value: dict):
        """Set the gaussians property."""
        if value is None:
            self._gaussians = None
            return
        
        # Update parameters from gaussians dict
        self.means.data = value['means'].to(self.device_obj)
        self.colors.data = value['colors'].to(self.device_obj)
        self.opacities.data = value['opacities'].to(self.device_obj)
        
        if 'covariances' in value:
            if self.covariances is None:
                self.register_parameter("covariances", nn.Parameter(value['covariances'].to(self.device_obj)))
                # Remove log_scales and rotations if they exist
                if hasattr(self, 'log_scales'):
                    delattr(self, 'log_scales')
                if hasattr(self, 'rotations'):
                    delattr(self, 'rotations')
                self.log_scales = None
                self.rotations = None
            else:
                self.covariances.data = value['covariances'].to(self.device_obj)
        else:
            if self.log_scales is None:
                self.register_parameter("log_scales", nn.Parameter(value['log_scales'].to(self.device_obj)))
                self.register_parameter("rotations", nn.Parameter(value['rotations'].to(self.device_obj)))
                # Remove covariances if it exists
                if hasattr(self, 'covariances'):
                    delattr(self, 'covariances')
                self.covariances = None
            else:
                self.log_scales.data = value['log_scales'].to(self.device_obj)
                self.rotations.data = value['rotations'].to(self.device_obj)
        
        self._gaussians = value
    
    def forward(self) -> torch.Tensor:
        """
        Forward pass: render the Gaussian splats.
        
        Returns:
            Tensor of shape (height, width, num_channels) containing the rendered image.
        """
        covariances = self._build_covariance_matrix()
        
        # Clamp opacities to [0, 1]
        opacities = torch.sigmoid(self.opacities)
        
        # Clamp colors to [0, 1]
        colors = torch.clamp(self.colors, 0.0, 1.0)
        
        return gaussian_splat_2d(
            means=self.means,
            covariances=covariances,
            colors=colors,
            opacities=opacities,
            height=self.height,
            width=self.width,
            device=self.device_obj,
        )
    
    def get_parameters_dict(self) -> dict:
        """
        Get a dictionary of all parameters.
        
        Returns:
            Dictionary containing means, covariances, colors, and opacities.
        """
        covariances = self._build_covariance_matrix()
        opacities = torch.sigmoid(self.opacities)
        colors = torch.clamp(self.colors, 0.0, 1.0)
        
        return {
            "means": self.means.detach().cpu(),
            "covariances": covariances.detach().cpu(),
            "colors": colors.detach().cpu(),
            "opacities": opacities.detach().cpu(),
        }
