"""
RMSNorm: Root Mean Square Layer Normalization.

Used in Atlas paper for normalizing Q and K projections.
More efficient than LayerNorm (no mean subtraction needed).

Reference: https://arxiv.org/abs/1910.07467
"""

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm doesn't subtract the mean - it only
    normalizes by the root mean square of the input. This makes it
    slightly faster while maintaining similar effectiveness.

    Args:
        dim: The dimension to normalize over (last dim by default)
        eps: Small constant for numerical stability (default: 1e-6)

    Shape:
        - Input: (*, dim)
        - Output: (*, dim) - same as input
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim
        # Learnable scale parameter (like gamma in LayerNorm)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        """Compute the RMS normalization."""
        # Compute root mean square: sqrt(mean(x^2))
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply RMSNorm to input tensor.

        Args:
            x: Input tensor of shape (*, dim)

        Returns:
            Normalized tensor of same shape, scaled by learnable weight
        """
        # Normalize and apply learnable scale
        return self._norm(x.float()).type_as(x) * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"
