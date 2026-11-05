import torch.nn as nn
import torch

'''
TODO: Implement this Module.

RMSNorm: Root Mean Square Layer Normalization.

Specification:
- Normalizes over the last (feature) dimension only, so each vector is
  normalized independently of every other vector
- y = x / sqrt(mean(x^2) + eps) * weight
- Has a learnable per-feature scale, initialized to ones
- Has no bias, and does not subtract the mean
- Computes the statistic in float32 and casts the result back to the input
  dtype. We train under mixed precision, so x arrives as float16, where x^2
  overflows above |x| = 256 and drives the output silently to zero instead of
  to NaN
- Reused as QK-Norm inside the attention layer, where it is applied to the
  queries and keys over head_dim
'''

class RMSNorm(nn.Module):
    '''
    Root Mean Square Layer Normalization.
    Normalizes over the last dimension and applies a learnable per-feature scale.
    '''
    def __init__(self, d_model: int, eps: float = 1e-6):
        '''
        Initialize the RMSNorm.
        Args:
            d_model (int): The dimension being normalized over.
            eps   (float): Added inside the square root for numerical stability.
        '''
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the RMSNorm.
        Args:
            x (torch.Tensor): Input tensor, shape (..., d_model)
        Returns:
            x (torch.Tensor): Normalized tensor of the same shape
        '''
        input_dtype = x.dtype
        x_float = x.float()
        rms = torch.sqrt(torch.mean(x_float * x_float, dim=-1, keepdim=True) + self.eps)
        normalized = (x_float / rms).to(dtype=input_dtype)
        return normalized * self.weight.to(dtype=input_dtype)

## -------------------------------------------------------------------------------------------------

def create_norm(norm_type: str, d_model: int) -> nn.Module:
    '''
    Create a normalization layer by name. (DO NOT MODIFY)
    Args:
        norm_type (str): The normalization to use, 'rmsnorm' or 'layernorm'.
        d_model   (int): The dimension being normalized over.
    Returns:
        nn.Module: The normalization layer.
    '''
    if norm_type not in ['rmsnorm', 'layernorm']:
        raise ValueError("norm_type must be 'rmsnorm' or 'layernorm'")

    if norm_type == 'rmsnorm':
        return RMSNorm(d_model)
    elif norm_type == 'layernorm':
        return nn.LayerNorm(d_model)
