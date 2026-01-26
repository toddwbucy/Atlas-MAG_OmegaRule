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
        """
        Initialize Atlas memory module.

        Args:
            dim: Model dimension
            expansion: Hidden dimension multiplier for the gated MLP
            bias: Whether to use bias in linear layers
        """
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

    def forward(self, x: Tensor, return_contribution: bool = False) -> Tensor:
        """
        Apply gated memory update.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            return_contribution: If True, return only the memory contribution
                without the residual connection. Used for output-level
                combination where the caller handles the residual.

        Returns:
            If return_contribution=False: x + memory_contribution (default)
            If return_contribution=True: memory_contribution only
        """
        # Gate: σ(W2·x)
        gate = F.silu(self.w2(x))
        # Value: W3·x
        value = self.w3(x)
        # Gated value: gate ⊙ value
        gated = gate * value
        # Memory contribution: W1·(gate ⊙ value)
        contribution: Tensor = self.w1(gated)

        if return_contribution:
            return contribution

        # Default: add residual for standalone use
        out: Tensor = x + contribution
        return out

    def extra_repr(self) -> str:
        """Return a string with extra module information for repr()."""
        return f"dim={self.dim}, hidden_dim={self.hidden_dim}"


class AtlasMemoryPoly(nn.Module):
    """
    Atlas Memory with Polynomial Features on Key-Sized Projections.

    Per Atlas paper Section 3.1, memory capacity is O(d_k) without polynomial
    features and O(d_k^p) with them. To keep parameters tractable, we:
    1. Project input from dim to key_dim (default: 64)
    2. Apply polynomial features on key_dim → poly_dim ≈ 2144
    3. Process through gated MLP
    4. Project back to dim

    This increases memory capacity from O(64) to O(4096) associations
    while keeping parameter count reasonable (~10M per layer vs 1B+).

    Memory capacity (Atlas Propositions 1 & 2):
        - Without φ: O(d_k) ≈ 64 associations
        - With φ_2: O(d_k²) ≈ 4,096 associations

    Args:
        dim: Model dimension (e.g., 768)
        key_dim: Dimension for polynomial features (default: 64, like head_dim)
        expansion: Hidden dimension multiplier (default: 4)
        poly_degree: Polynomial degree (default: 2, only 2 is implemented)
        bias: Whether to use bias (default: False)
    """

    def __init__(
        self,
        dim: int,
        key_dim: int = 64,  # Same as head_dim for capacity theorem
        expansion: int = MEMORY_EXPANSION,
        poly_degree: int = POLY_DEGREE,
        bias: bool = False,
    ):
        """
        Initialize polynomial-enhanced Atlas memory module.

        Args:
            dim: Model dimension
            key_dim: Dimension for polynomial feature expansion (capacity = O(key_dim²))
            expansion: Hidden dimension multiplier
            poly_degree: Polynomial degree for feature expansion (only 2 supported)
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        self.dim = dim
        self.key_dim = key_dim
        self.poly_degree = poly_degree

        if poly_degree != 2:
            raise NotImplementedError("Only polynomial degree 2 is implemented")

        # Compute expanded dimension: key_dim + key_dim*(key_dim+1)/2
        # For key_dim=64: 64 + 2080 = 2144
        self.poly_dim = key_dim + (key_dim * (key_dim + 1)) // 2
        self.hidden_dim = key_dim * expansion  # Scale hidden with key_dim, not full dim

        # Project dim → key_dim for polynomial features
        self.proj_down = nn.Linear(dim, key_dim, bias=bias)

        # Normalize polynomial features to maintain signal magnitude
        self.poly_norm = nn.LayerNorm(self.poly_dim)

        # Gated MLP operates on poly_dim
        self.w1 = nn.Linear(self.hidden_dim, key_dim, bias=bias)  # Output key_dim
        self.w2 = nn.Linear(self.poly_dim, self.hidden_dim, bias=bias)
        self.w3 = nn.Linear(self.poly_dim, self.hidden_dim, bias=bias)

        # Project key_dim → dim for output
        self.proj_up = nn.Linear(key_dim, dim, bias=bias)

        # Precompute indices for upper triangular (including diagonal)
        self.register_buffer(
            "triu_indices",
            torch.triu_indices(key_dim, key_dim),
            persistent=False,
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training with meaningful initial output.

        After LayerNorm, the polynomial features have unit variance. We use
        standard initialization (0.02) for all layers since the normalization
        handles the input scale.
        """
        # Standard initialization for all layers (LayerNorm handles input scaling)
        nn.init.normal_(self.w1.weight, std=0.02)
        nn.init.normal_(self.w2.weight, std=0.02)
        nn.init.normal_(self.w3.weight, std=0.02)

        for module in [self.w1, self.w2, self.w3]:
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        # Standard init for projection layers
        nn.init.normal_(self.proj_down.weight, std=0.02)
        nn.init.normal_(self.proj_up.weight, std=0.02)

    def _polynomial_features(self, x: Tensor) -> Tensor:
        """
        Compute degree-2 polynomial features.

        Args:
            x: Input of shape (batch, seq_len, key_dim)

        Returns:
            Expanded tensor of shape (batch, seq_len, poly_dim)
        """
        # Compute outer product: x_i * x_j
        # Shape: (batch, seq_len, key_dim, key_dim)
        outer = x.unsqueeze(-1) * x.unsqueeze(-2)

        # Extract upper triangular (including diagonal)
        # Shape: (batch, seq_len, key_dim*(key_dim+1)/2)
        triu_idx = cast(Tensor, self.triu_indices)
        triu = outer[..., triu_idx[0], triu_idx[1]]

        # Concatenate: [x, polynomial features]
        return torch.cat([x, triu], dim=-1)

    def forward(self, x: Tensor, return_contribution: bool = False) -> Tensor:
        """
        Apply polynomial-enhanced gated memory update.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            return_contribution: If True, return only the memory contribution
                without the residual connection. Used for output-level
                combination where the caller handles the residual.

        Returns:
            If return_contribution=False: x + memory_contribution (default)
            If return_contribution=True: memory_contribution only
        """
        # Project to key dimension for polynomial features
        x_key = self.proj_down(x)  # (batch, seq_len, key_dim)

        # Expand with polynomial features and normalize
        # Normalization maintains signal magnitude despite varying term scales
        x_poly = self._polynomial_features(x_key)  # (batch, seq_len, poly_dim)
        x_poly = self.poly_norm(x_poly)

        # Gate and value use normalized polynomial features
        gate = F.silu(self.w2(x_poly))
        value = self.w3(x_poly)
        gated = gate * value

        # Memory output in key dimension
        mem_key: Tensor = self.w1(gated)  # (batch, seq_len, key_dim)

        # Project back to full dimension
        contribution: Tensor = self.proj_up(mem_key)  # (batch, seq_len, dim)

        if return_contribution:
            return contribution

        # Default: residual connection with original x
        out: Tensor = x + contribution
        return out

    def extra_repr(self) -> str:
        """Return a string with extra module information for repr()."""
        return (
            f"dim={self.dim}, key_dim={self.key_dim}, poly_dim={self.poly_dim}, "
            f"hidden_dim={self.hidden_dim}, poly_degree={self.poly_degree}"
        )
