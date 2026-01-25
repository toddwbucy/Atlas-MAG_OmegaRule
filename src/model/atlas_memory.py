"""
Atlas Memory Module: Gated MLP for Test-Time Memorization.

The Atlas memory learns to store and retrieve information through a
gated MLP architecture. Unlike attention, it can update its internal
state at inference time, enabling true test-time learning.

Architecture:
    M(x) = x + W1 · (σ(W2·x) ⊙ W3·x)

Where:
    - σ is SiLU (Swish) activation
    - ⊙ is element-wise multiplication (gating)
    - The residual connection enables additive updates

Reference: Atlas paper (arXiv:2505.23735)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import cast
from src.config import MEMORY_EXPANSION, POLY_DEGREE


class AtlasMemory(nn.Module):
    """
    Atlas++ Memory: Gated MLP with residual connection.

    This is the core memory module that enables test-time learning.
    The gating mechanism controls what information gets stored.

    Args:
        dim: Model dimension
        expansion: Hidden dimension multiplier (default: 4)
        bias: Whether to use bias in linear layers (default: False)

    Shape:
        - Input: (batch, seq_len, dim)
        - Output: (batch, seq_len, dim)
    """

    def __init__(
        self,
        dim: int,
        expansion: int = MEMORY_EXPANSION,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = dim * expansion

        # Gate projection (SiLU applied)
        self.w1 = nn.Linear(self.hidden_dim, dim, bias=bias)
        # Up projection for gate
        self.w2 = nn.Linear(dim, self.hidden_dim, bias=bias)
        # Up projection for value
        self.w3 = nn.Linear(dim, self.hidden_dim, bias=bias)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable start."""
        for module in [self.w1, self.w2, self.w3]:
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply gated memory update.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)

        Returns:
            Updated tensor with memory contribution added
        """
        # Gate: σ(W2·x)
        gate = F.silu(self.w2(x))
        # Value: W3·x
        value = self.w3(x)
        # Gated value: gate ⊙ value
        gated = gate * value
        # Project down and add residual: x + W1·(gate ⊙ value)
        out: Tensor = x + self.w1(gated)
        return out

    def extra_repr(self) -> str:
        return f"dim={self.dim}, hidden_dim={self.hidden_dim}"


class AtlasMemoryPoly(nn.Module):
    """
    Atlas Memory with Polynomial Features.

    Extends the basic Atlas memory by first expanding the input
    with polynomial features (degree 2 by default), increasing
    the representational power of the memory.

    Polynomial features for input x create:
        [x, x_i * x_j for i <= j]

    This increases the input dimension from d to d + d*(d+1)/2.

    Args:
        dim: Model dimension
        expansion: Hidden dimension multiplier (default: 4)
        poly_degree: Polynomial degree (default: 2, only 2 is implemented)
        bias: Whether to use bias (default: False)
    """

    def __init__(
        self,
        dim: int,
        expansion: int = MEMORY_EXPANSION,
        poly_degree: int = POLY_DEGREE,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.poly_degree = poly_degree

        if poly_degree != 2:
            raise NotImplementedError("Only polynomial degree 2 is implemented")

        # Compute expanded dimension: d + d*(d+1)/2
        self.poly_dim = dim + (dim * (dim + 1)) // 2
        self.hidden_dim = dim * expansion

        # Projections now take poly_dim as input
        self.w1 = nn.Linear(self.hidden_dim, dim, bias=bias)
        self.w2 = nn.Linear(self.poly_dim, self.hidden_dim, bias=bias)
        self.w3 = nn.Linear(self.poly_dim, self.hidden_dim, bias=bias)

        # Precompute indices for upper triangular (including diagonal)
        self.register_buffer(
            "triu_indices",
            torch.triu_indices(dim, dim),
            persistent=False,
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize with smaller weights due to larger input dim."""
        scale = 0.02 / math.sqrt(self.poly_dim / self.dim)
        for module in [self.w1, self.w2, self.w3]:
            nn.init.normal_(module.weight, std=scale)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _polynomial_features(self, x: Tensor) -> Tensor:
        """
        Compute degree-2 polynomial features.

        Args:
            x: Input of shape (batch, seq_len, dim)

        Returns:
            Expanded tensor of shape (batch, seq_len, poly_dim)
        """
        batch, seq_len, d = x.shape

        # Compute outer product: x_i * x_j
        # Shape: (batch, seq_len, dim, dim)
        outer = x.unsqueeze(-1) * x.unsqueeze(-2)

        # Extract upper triangular (including diagonal)
        # Shape: (batch, seq_len, d*(d+1)/2)
        triu_idx = cast(Tensor, self.triu_indices)
        triu = outer[..., triu_idx[0], triu_idx[1]]

        # Concatenate: [x, polynomial features]
        return torch.cat([x, triu], dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply polynomial-enhanced gated memory update.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)

        Returns:
            Updated tensor with memory contribution added
        """
        # Expand with polynomial features
        x_poly = self._polynomial_features(x)

        # Gate and value use polynomial features
        gate = F.silu(self.w2(x_poly))
        value = self.w3(x_poly)
        gated = gate * value

        # Residual connection with original x (not poly)
        out: Tensor = x + self.w1(gated)
        return out

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, poly_dim={self.poly_dim}, "
            f"hidden_dim={self.hidden_dim}, poly_degree={self.poly_degree}"
        )
