"""Public interface and autograd dispatch for 2D Gaussian splatting."""

import torch
import torch.nn as nn
from typing import Optional

from .backends import get_backend


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
        backend = get_backend(means.device.type)
        output, intermediates = backend.forward(
            means, covariances, colors, opacities, height, width
        )

        # Save tensors and intermediates for backward pass
        ctx.save_for_backward(means, covariances, colors, opacities, *intermediates)
        ctx.height = height
        ctx.width = width
        ctx.backend_name = backend.name
        
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
        needs_input_grad = ctx.needs_input_grad

        backend = get_backend(ctx.backend_name)
        grad_means, grad_cov, grad_colors, grad_opacities = backend.backward(
            grad_output,
            means,
            covariances,
            colors,
            opacities,
            height,
            width,
            intermediates,
            needs_input_grad,
        )
        
        return grad_means, grad_cov, grad_colors, grad_opacities, None, None


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
