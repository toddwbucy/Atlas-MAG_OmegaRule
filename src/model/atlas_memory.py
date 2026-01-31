"""
Atlas Memory Module: Deep Neural Memory with Polynomial Features.

Paper Reference:
    Atlas: Learning to Optimally Memorize the Context at Test Time
    arXiv:2505.23735, Section 3.1 "Associative Memory with Super Linear Capacity"

Theoretical Foundation:
    The paper establishes memory capacity bounds for different architectures:

    Proposition 1 (Matrix Memory Capacity):
        Matrix-valued memory M ∈ R^(d_v × d_k) with ℓ₂ attentional bias
        can store at most O(d_k) key-value pairs with linearly independent keys.

    Theorem 1 (Deep Memory Capacity):
        An MLP with L_M ≥ 2 layers, d_k input dim, and d_h hidden dim
        can store O(d_k * d_v) to O(d_k * d_v * Π min{d_h^(j)}) pairs.

    Proposition 2 (Polynomial Feature Enhancement):
        Using polynomial kernel φ_p on keys increases capacity to O(d_k^p).
        For degree p=2: capacity increases from O(64) to O(4096) associations.

Architecture (Section 4 "DeepTransformers"):
    1. Project input to key_dim for polynomial expansion
    2. Apply polynomial features: φ(x) = [x, x_i*x_j] for O(d_k²) capacity
    3. Gated MLP: W1 · (σ(W2·φ(x)) ⊙ W3·φ(x))
    4. Project back to full dimension

    The gated MLP acts as the "deep non-linear neural memory that encodes
    past abstractions into its parameters" (Section 2).

Test-Time Learning (TTL):
    Unlike attention, this memory module supports parameter updates at inference
    time via gradient descent with momentum (the "inner loop" optimization).
    This enables "test time memorization" - storing and retrieving information
    strictly within the context without modifying core learned parameters.

Implementation Notes:
    - Momentum buffers are registered as persistent buffers for TTL updates
    - freeze_static_weights() allows testing TTL contribution in isolation
    - poly_rank enables optional low-rank compression of polynomial features
"""

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import MEMORY_EXPANSION, POLY_DEGREE, POLY_RANK


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
        poly_rank: int = POLY_RANK,
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

        if poly_degree not in (1, 2):
            raise NotImplementedError(f"Polynomial degree {poly_degree} not implemented (use 1 or 2)")

        # Compute expanded dimension based on polynomial degree
        if poly_degree == 1:
            # Degree 1: identity features only (no polynomial expansion)
            # poly_dim = key_dim (e.g., 64 for key_dim=64)
            self.poly_dim = key_dim
        else:
            # Degree 2: key_dim + key_dim*(key_dim+1)/2
            # For key_dim=64: 64 + 2080 = 2144
            self.poly_dim = key_dim + (key_dim * (key_dim + 1)) // 2
        self.poly_rank = poly_rank
        self.hidden_dim = key_dim * expansion  # Scale hidden with key_dim, not full dim

        # Project dim → key_dim for polynomial features
        self.proj_down = nn.Linear(dim, key_dim, bias=bias)

        # Normalize polynomial features to maintain signal magnitude
        self.poly_norm = nn.LayerNorm(self.poly_dim)

        # Optional low-rank compression of polynomial features
        # Reduces w2/w3 input from poly_dim (e.g. 2144) to poly_rank (e.g. 512)
        self.poly_compress: nn.Linear | None = None
        mlp_input_dim = self.poly_dim
        if poly_rank > 0 and poly_rank < self.poly_dim:
            self.poly_compress = nn.Linear(self.poly_dim, poly_rank, bias=bias)
            mlp_input_dim = poly_rank

        # Gated MLP operates on (compressed) poly features
        self.w1 = nn.Linear(self.hidden_dim, key_dim, bias=bias)  # Output key_dim
        self.w2 = nn.Linear(mlp_input_dim, self.hidden_dim, bias=bias)
        self.w3 = nn.Linear(mlp_input_dim, self.hidden_dim, bias=bias)

        # Project key_dim → dim for output
        self.proj_up = nn.Linear(key_dim, dim, bias=bias)

        # Normalize output to match attention scale (Atlas paper: "layer norm at end of chunk")
        # This ensures memory contribution has similar magnitude to attention output,
        # giving the learned gate a fair comparison between the two branches.
        self.output_norm = nn.LayerNorm(dim)

        # Learnable output scale initialized to match typical attention output magnitude.
        # LayerNorm gives unit variance (~0.8-1.0 RMS), but attention output is ~0.15 RMS.
        # This scale factor brings memory output to the same ballpark as attention,
        # preventing either branch from dominating purely due to magnitude mismatch.
        self.output_scale = nn.Parameter(torch.tensor(0.15))

        # Precompute indices for upper triangular (only needed for degree 2)
        if poly_degree == 2:
            self.register_buffer(
                "triu_indices",
                torch.triu_indices(key_dim, key_dim),
                persistent=False,
            )

        self._init_weights()
        self._init_momentum_buffers()

    def _init_momentum_buffers(self) -> None:
        """
        Initialize momentum buffers for TTL (Test-Time Learning).

        Creates a zero-initialized buffer for each parameter, used to accumulate
        momentum during TTL updates per Eq. 33: S_t = theta * S_{t-1} + grad.

        Buffers are registered with persistent=True so they survive serialization.
        """
        for name, param in self.named_parameters():
            # Replace dots with underscores for valid attribute names
            buffer_name = f"momentum_{name.replace('.', '_')}"
            self.register_buffer(buffer_name, torch.zeros_like(param), persistent=True)

    def get_momentum(self, name: str) -> Tensor:
        """
        Get the momentum buffer for a named parameter.

        Args:
            name: Parameter name (e.g., 'w1.weight', 'proj_down.weight')

        Returns:
            Momentum tensor (same shape as the parameter)
        """
        buffer_name = f"momentum_{name.replace('.', '_')}"
        momentum = getattr(self, buffer_name)
        return cast(Tensor, momentum)

    def reset_momentum(self) -> None:
        """
        Reset all momentum buffers to zero.

        Call this at sequence/batch boundaries depending on TTL_RESET_MODE.
        """
        for name, param in self.named_parameters():
            buffer_name = f"momentum_{name.replace('.', '_')}"
            getattr(self, buffer_name).zero_()

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
        if self.poly_compress is not None:
            nn.init.normal_(self.poly_compress.weight, std=0.02)

    def _polynomial_features(self, x: Tensor) -> Tensor:
        """
        Compute polynomial features of the specified degree.

        Degree 1: Identity (returns x unchanged)
        Degree 2: Concatenates x with upper-triangular outer products [x, x_i*x_j]

        Args:
            x: Input of shape (batch, seq_len, key_dim)

        Returns:
            Expanded tensor of shape (batch, seq_len, poly_dim)
        """
        if self.poly_degree == 1:
            return x

        # Degree 2: compute outer product x_i * x_j
        outer = x.unsqueeze(-1) * x.unsqueeze(-2)

        # Extract upper triangular (including diagonal)
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

        # Optional low-rank compression before MLP
        x_mlp_in = x_poly
        if self.poly_compress is not None:
            x_mlp_in = self.poly_compress(x_poly)

        # Gate and value use (compressed) polynomial features
        gate = F.gelu(self.w2(x_mlp_in))
        value = self.w3(x_mlp_in)
        gated = gate * value

        # Memory output in key dimension
        mem_key: Tensor = self.w1(gated)  # (batch, seq_len, key_dim)

        # Project back to full dimension, normalize, and scale to match attention magnitude
        # 1. proj_up: key_dim → dim
        # 2. output_norm: normalize to unit variance (stable gradients)
        # 3. output_scale: learnable scale to match attention output magnitude (~0.15)
        contribution: Tensor = self.output_scale * self.output_norm(self.proj_up(mem_key))

        if return_contribution:
            return contribution

        # Default: residual connection with original x
        out: Tensor = x + contribution
        return out

    def freeze_static_weights(self) -> None:
        """Freeze all non-TTL (static) parameters so only TTL (ΔM) can reduce loss.

        This forces the inner loop (TTL) to carry the load, preventing the model
        from relying solely on static weight optimization.
        """
        for param in self.parameters():
            param.requires_grad_(False)

    def unfreeze_static_weights(self) -> None:
        """Unfreeze all parameters to resume normal training."""
        for param in self.parameters():
            param.requires_grad_(True)

    def extra_repr(self) -> str:
        """Return a string with extra module information for repr()."""
        rank_str = f", poly_rank={self.poly_rank}" if self.poly_rank > 0 else ""
        return (
            f"dim={self.dim}, key_dim={self.key_dim}, poly_dim={self.poly_dim}, "
            f"hidden_dim={self.hidden_dim}, poly_degree={self.poly_degree}{rank_str}"
        )
