"""
SwiGLU: Swish-Gated Linear Unit.

Paper Reference:
    GLU Variants Improve Transformer
    arXiv:2002.05202 (Shazeer, 2020)

Usage in Atlas-MAG:
    SwiGLU is used for the feedforward network (FFN) after the attention/memory
    gating stage in each MAGBlock. This is a standard architectural choice used
    in modern transformers (LLaMA, PaLM, Mistral).

    Note: The Atlas paper (arXiv:2505.23735) specifies GELU for the *memory MLP*
    (which we follow in atlas_memory.py), but does not prescribe an activation
    for the main transformer FFN. SwiGLU is used here as a performant default.

Mathematical Formulation:
    SwiGLU(x) = W3(SiLU(W1(x)) ⊙ W2(x))

    Where:
    - SiLU(x) = x * sigmoid(x) (also called Swish)
    - ⊙ denotes element-wise multiplication
    - W1, W2 project to hidden_dim, W3 projects back to dim

    The hidden dimension is scaled by 2/3 to maintain parameter parity
    with a standard 2-layer FFN when using 3 weight matrices.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SwiGLU(nn.Module):
    """
    SwiGLU Feedforward block.

    Implements: output = W3(SiLU(W1(x)) ⊙ W2(x))

    The hidden dimension is scaled by 2/3 to maintain parameter parity
    with a standard 2-layer FFN when using 3 weight matrices.

    Args:
        dim: Input/output dimension
        hidden_dim: Optional hidden dimension (default: int(dim * 4 * 2/3))
        bias: Whether to use bias in linear layers (default: False)

    Shape:
        - Input: (batch, seq_len, dim)
        - Output: (batch, seq_len, dim)
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        bias: bool = False,
    ):
        """
        Initialize SwiGLU feedforward block.

        Args:
            dim: Input and output dimension of the block
            hidden_dim: Hidden layer dimension. If None, computed as int(dim * 4 * 2/3)
                       rounded to multiple of 64 for GPU efficiency
            bias: Whether to include bias terms in linear layers
        """
        super().__init__()
        # Default hidden dim maintains parameter parity with standard FFN
        # Standard FFN: 2 * dim * (4*dim) = 8 * dim^2
        # SwiGLU: 3 * dim * hidden = 8 * dim^2 → hidden = 8/3 * dim ≈ 2.67 * dim
        if hidden_dim is None:
            hidden_dim = int(dim * 4 * 2 / 3)
            # Round to multiple of 64 for GPU efficiency
            hidden_dim = ((hidden_dim + 63) // 64) * 64

        self.dim = dim
        self.hidden_dim = hidden_dim

        # Gate projection (applies SiLU)
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        # Up projection (element-wise multiplied with gate)
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)
        # Down projection
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply SwiGLU transformation.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # SiLU (Swish) gating
        gate = F.silu(self.w1(x))
        # Element-wise product with up projection
        hidden = gate * self.w2(x)
        # Project back to model dimension
        out: Tensor = self.w3(hidden)
        return out

    def extra_repr(self) -> str:
        """Return a string with extra module information for repr()."""
        return f"dim={self.dim}, hidden_dim={self.hidden_dim}"


class SwiGLUFused(nn.Module):
    """
    Fused SwiGLU for better memory efficiency.

    Combines W1 and W2 into a single projection, then splits.
    Reduces memory bandwidth at the cost of slightly more compute.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        bias: bool = False,
    ):
        """
        Initialize fused SwiGLU feedforward block.

        The fused variant combines W1 and W2 into a single projection matrix,
        reducing memory bandwidth at the cost of slightly more compute.

        Args:
            dim: Input and output dimension of the block
            hidden_dim: Hidden layer dimension. If None, computed as int(dim * 4 * 2/3)
                       rounded to multiple of 64 for GPU efficiency
            bias: Whether to include bias terms in linear layers
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(dim * 4 * 2 / 3)
            hidden_dim = ((hidden_dim + 63) // 64) * 64

        self.dim = dim
        self.hidden_dim = hidden_dim

        # Fused up projection (gate + value combined)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        # Down projection
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply fused SwiGLU transformation."""
        # Single matmul for both gate and value
        gate_and_value = self.w12(x)
        # Split into gate and value
        gate, value = gate_and_value.chunk(2, dim=-1)
        # Apply SiLU to gate, multiply with value
        hidden = F.silu(gate) * value
        out: Tensor = self.w3(hidden)
        return out

    def extra_repr(self) -> str:
        """Return a string with extra module information for repr()."""
        return f"dim={self.dim}, hidden_dim={self.hidden_dim}, fused=True"
