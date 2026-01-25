"""
Q, K, V Projections and Causal Conv1d.

These modules handle the input transformations for attention and memory.
The paper uses RMSNorm on Q and K before attention for stability.

Reference: Atlas paper (arXiv:2505.23735)
"""

import math
from typing import Union, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import D, N_HEADS, HEAD_DIM
from src.nn.rmsnorm import RMSNorm


class QKVProjection(nn.Module):
    """
    Query, Key, Value projections with optional normalization.

    Features:
        - Separate Q, K, V projections (not fused for flexibility)
        - RMSNorm on Q and K (per Atlas paper recommendation)
        - Optional multi-head reshaping

    Args:
        dim: Model dimension
        n_heads: Number of attention heads (for reshaping)
        qk_norm: Whether to apply RMSNorm to Q and K (default: True)
        bias: Whether to use bias in projections (default: False)
    """

    def __init__(
        self,
        dim: int = D,
        n_heads: int = N_HEADS,
        qk_norm: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        # Validate dim is divisible by n_heads to prevent cryptic .view() errors
        if dim % n_heads != 0:
            raise ValueError(
                f"dim must be divisible by n_heads: got dim={dim}, n_heads={n_heads}"
            )
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qk_norm = qk_norm

        # Projections
        self.w_q = nn.Linear(dim, dim, bias=bias)
        self.w_k = nn.Linear(dim, dim, bias=bias)
        self.w_v = nn.Linear(dim, dim, bias=bias)

        # Optional normalization (highly recommended for stability)
        self.q_norm: Union[RMSNorm, nn.Identity]
        self.k_norm: Union[RMSNorm, nn.Identity]
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """Initialize with scaled normal distribution."""
        for module in [self.w_q, self.w_k, self.w_v]:
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: Tensor,
        reshape_heads: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Project input to Q, K, V.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            reshape_heads: If True, reshape to (batch, n_heads, seq_len, head_dim)

        Returns:
            Tuple of (Q, K, V) tensors
        """
        batch, seq_len, dim = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        if reshape_heads:
            # Reshape: (batch, seq_len, dim) -> (batch, n_heads, seq_len, head_dim)
            q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

            # Apply per-head normalization
            q = self.q_norm(q)
            k = self.k_norm(k)
        else:
            # Apply normalization to full dimension
            # This path is used when heads aren't split (e.g., for memory)
            if self.qk_norm:
                # Temporarily reshape for normalization
                q = q.view(batch, seq_len, self.n_heads, self.head_dim)
                k = k.view(batch, seq_len, self.n_heads, self.head_dim)
                q = self.q_norm(q)
                k = self.k_norm(k)
                q = q.view(batch, seq_len, dim)
                k = k.view(batch, seq_len, dim)

        return q, k, v

    def extra_repr(self) -> str:
        """Return a string with extra module information for repr()."""
        return f"dim={self.dim}, n_heads={self.n_heads}, qk_norm={self.qk_norm}"


class CausalConv1d(nn.Module):
    """
    Causal 1D Convolution for local context modeling.

    Used before attention to inject local positional information.
    The convolution is causal (no future information leakage).

    Args:
        dim: Number of channels (model dimension)
        kernel_size: Convolution kernel size (default: 4)
        groups: Number of groups for depthwise conv (default: dim for depthwise)
    """

    def __init__(
        self,
        dim: int = D,
        kernel_size: int = 4,
        groups: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        # Default to depthwise convolution (groups = dim)
        self.groups = groups if groups is not None else dim

        # Padding for causal conv: (kernel_size - 1) on the left only
        self.padding = kernel_size - 1

        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=0,  # We'll pad manually for causality
            groups=self.groups,
            bias=True,
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize with small values."""
        nn.init.normal_(self.conv.weight, std=0.02 / math.sqrt(self.kernel_size))
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply causal convolution.

        Args:
            x: Input of shape (batch, seq_len, dim)

        Returns:
            Output of shape (batch, seq_len, dim)
        """
        # Transpose to (batch, dim, seq_len) for Conv1d
        x = x.transpose(1, 2)

        # Left-pad for causality
        x = F.pad(x, (self.padding, 0))

        # Apply convolution
        x = self.conv(x)

        # Transpose back to (batch, seq_len, dim)
        return x.transpose(1, 2)

    def extra_repr(self) -> str:
        """Return a string with extra module information for repr()."""
        return f"dim={self.dim}, kernel_size={self.kernel_size}, groups={self.groups}"


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Encodes relative position information by rotating Q and K vectors.
    More effective than absolute positional embeddings for long contexts.

    Args:
        dim: Per-head dimension (must be even)
        max_seq_len: Maximum sequence length to cache (default: 8192)
        base: Base for frequency computation (default: 10000)
    """

    def __init__(
        self,
        dim: int = HEAD_DIM,
        max_seq_len: int = 8192,
        base: float = 10000.0,
    ):
        super().__init__()
        assert dim % 2 == 0, f"dim must be even, got {dim}"

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache for cos/sin values
        self._cache_built = False
        self._cached_seq_len = 0

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Build or extend the sin/cos cache."""
        # Rebuild if: (1) cache not built, or (2) need longer sequence than cached
        if self._cache_built and seq_len <= self._cached_seq_len:
            return

        # Position indices
        t = torch.arange(seq_len, device=device, dtype=dtype)

        # Frequencies: (seq_len, dim/2)
        inv_freq = cast(Tensor, self.inv_freq).to(device)
        freqs = torch.outer(t, inv_freq)

        # Duplicate for complex rotation: (seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)

        # Cache cos and sin
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

        self._cache_built = True
        self._cached_seq_len = seq_len

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        seq_offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """
        Apply rotary embeddings to Q and K.

        Args:
            q: Query tensor of shape (batch, n_heads, seq_len, head_dim)
            k: Key tensor of shape (batch, n_heads, seq_len, head_dim)
            seq_offset: Starting position for incremental decoding

        Returns:
            Tuple of rotated (Q, K) tensors
        """
        seq_len = q.shape[2]
        self._build_cache(seq_offset + seq_len, q.device, q.dtype)

        cos_cached = cast(Tensor, self.cos_cached)
        sin_cached = cast(Tensor, self.sin_cached)
        cos = cos_cached[seq_offset : seq_offset + seq_len]
        sin = sin_cached[seq_offset : seq_offset + seq_len]

        # Apply rotation
        q_embed = self._apply_rotary(q, cos, sin)
        k_embed = self._apply_rotary(k, cos, sin)

        return q_embed, k_embed

    def _apply_rotary(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply rotary embedding to a single tensor."""
        # Handle the case where head_dim > self.dim (only rotate first self.dim dims)
        # Split into rotated part and unchanged tail
        rotated_part = x[..., : self.dim]
        tail = x[..., self.dim :]  # May be empty if head_dim == self.dim

        # Split rotated part into even and odd dimensions
        half_dim = self.dim // 2
        x1, x2 = rotated_part[..., :half_dim], rotated_part[..., half_dim:]

        # Rotate
        # cos and sin have shape (seq_len, dim)
        # x has shape (batch, n_heads, seq_len, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Split cos/sin for the two halves
        cos1, cos2 = cos[..., :half_dim], cos[..., half_dim:self.dim]
        sin1, sin2 = sin[..., :half_dim], sin[..., half_dim:self.dim]

        # Apply rotation: [x1, x2] * [cos, -sin; sin, cos]
        rotated = torch.cat(
            [x1 * cos1 - x2 * sin1, x1 * sin2 + x2 * cos2],
            dim=-1,
        )

        # Concatenate with unchanged tail (if any)
        if tail.size(-1) > 0:
            return torch.cat([rotated, tail], dim=-1)
        return rotated

    def extra_repr(self) -> str:
        """Return a string with extra module information for repr()."""
        return f"dim={self.dim}, max_seq_len={self.max_seq_len}, base={self.base}"
