"""
Example: Using torch.autograd.Function to create custom functions with automatic gradients.

torch.autograd.Function allows you to:
1. Define custom forward and backward passes
2. Save intermediate values for efficient backward computation
3. Control which inputs need gradients
4. Integrate custom C++/CUDA kernels
"""

import torch


class MyCustomFunction(torch.autograd.Function):
    """
    Custom autograd Function example.
    
    This function computes: f(x, y) = x^2 * y + sin(x)
    """
    
    @staticmethod
    def forward(ctx, x, y):
        """
        Forward pass: compute the output.
        
        Args:
            ctx: Context object to store values needed for backward pass
            x: Input tensor (can have requires_grad=True)
            y: Input tensor (can have requires_grad=True)
        
        Returns:
            Output tensor
        """
        # Compute forward pass
        x_squared = x ** 2
        output = x_squared * y + torch.sin(x)
        
        # Save tensors needed for backward pass
        # These will be available in backward() via ctx.saved_tensors
        ctx.save_for_backward(x, y, x_squared)
        
        # You can also save non-tensor values
        ctx.x_value = x.item() if x.numel() == 1 else None
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradients w.r.t. inputs.
        
        Args:
            ctx: Context object with saved tensors from forward()
            grad_output: Gradient of loss w.r.t. the output of forward()
                        Has the same shape as the output of forward()
        
        Returns:
            Tuple of gradients w.r.t. each input (in same order as forward inputs)
            Return None for inputs that don't need gradients
        """
        # Retrieve saved tensors
        x, y, x_squared = ctx.saved_tensors
        
        # Check which inputs need gradients (optional optimization)
        # ctx.needs_input_grad is a tuple of booleans, one per input
        needs_x_grad, needs_y_grad = ctx.needs_input_grad
        
        # Compute gradients using chain rule
        # f(x, y) = x^2 * y + sin(x)
        # ∂f/∂x = 2*x*y + cos(x)
        # ∂f/∂y = x^2
        
        grad_x = None
        grad_y = None
        
        if needs_x_grad:
            # Multiply by grad_output to apply chain rule
            grad_x = grad_output * (2 * x * y + torch.cos(x))
        
        if needs_y_grad:
            grad_y = grad_output * x_squared
        
        # Return gradients in the same order as forward() inputs
        # Return None for inputs that don't need gradients
        return grad_x, grad_y


# Example 1: Basic usage
print("=" * 70)
print("Example 1: Basic usage of torch.autograd.Function")
print("=" * 70)

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Call the function using .apply()
output = MyCustomFunction.apply(x, y)

print(f"Input: x = {x.item()}, y = {y.item()}")
print(f"Output: f(x, y) = x^2 * y + sin(x) = {output.item():.4f}")

# Compute gradients
output.backward()

print(f"Gradient w.r.t. x: {x.grad.item():.4f}")  # Should be 2*x*y + cos(x) = 12 + cos(2)
print(f"Gradient w.r.t. y: {y.grad.item():.4f}")  # Should be x^2 = 4
print()


# Example 2: Using in a computation graph
print("=" * 70)
print("Example 2: Custom function in a larger computation graph")
print("=" * 70)

x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)

# Use custom function in a chain
z1 = MyCustomFunction.apply(x, y)
z2 = z1 ** 2
loss = z2 + x

print(f"x = {x.item()}, y = {y.item()}")
print(f"z1 = MyCustomFunction(x, y) = {z1.item():.4f}")
print(f"z2 = z1^2 = {z2.item():.4f}")
print(f"loss = z2 + x = {loss.item():.4f}")

# Compute gradients through the entire graph
loss.backward()
print(f"∂loss/∂x = {x.grad.item():.4f}")
print(f"∂loss/∂y = {y.grad.item():.4f}")
print()


# Example 3: Handling inputs that don't need gradients
print("=" * 70)
print("Example 3: Some inputs don't need gradients")
print("=" * 70)

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=False)  # y doesn't need gradients

output = MyCustomFunction.apply(x, y)
print(f"x requires_grad={x.requires_grad}, y requires_grad={y.requires_grad}")

output.backward()
print(f"Gradient w.r.t. x: {x.grad.item():.4f}")
print(f"y.grad is None: {y.grad is None}")  # y.grad will be None
print()


# Example 4: Batch processing
print("=" * 70)
print("Example 4: Processing batches of tensors")
print("=" * 70)

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)

output = MyCustomFunction.apply(x, y)
print(f"Input x: {x.tolist()}")
print(f"Input y: {y.tolist()}")
print(f"Output: {output.tolist()}")

# Sum to get a scalar for backward
loss = output.sum()
loss.backward()

print(f"Gradient w.r.t. x: {x.grad.tolist()}")
print(f"Gradient w.r.t. y: {y.grad.tolist()}")
print()


# Example 5: Advanced - saving intermediate values for efficiency
print("=" * 70)
print("Example 5: Saving intermediate values for efficient backward")
print("=" * 70)


class EfficientCustomFunction(torch.autograd.Function):
    """
    Example showing how to save intermediate computations
    to avoid recomputing them in backward().
    """
    
    @staticmethod
    def forward(ctx, x, y):
        # Compute expensive intermediate values
        x_squared = x ** 2
        x_cubed = x ** 3
        sin_x = torch.sin(x)
        cos_x = torch.cos(x)
        
        # Save all intermediate values we'll need in backward
        ctx.save_for_backward(x, y, x_squared, x_cubed, sin_x, cos_x)
        
        # Compute output
        output = x_squared * y + x_cubed * sin_x
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y, x_squared, x_cubed, sin_x, cos_x = ctx.saved_tensors
        
        # Use precomputed values instead of recomputing
        # This is more efficient, especially for complex operations
        grad_x = grad_output * (2 * x * y + 3 * x_squared * sin_x + x_cubed * cos_x)
        grad_y = grad_output * x_squared
        
        return grad_x, grad_y


x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

output = EfficientCustomFunction.apply(x, y)
print(f"Output: {output.item():.4f}")

output.backward()
print(f"Gradients computed efficiently using saved intermediates")
print(f"∂output/∂x = {x.grad.item():.4f}")
print(f"∂output/∂y = {y.grad.item():.4f}")
