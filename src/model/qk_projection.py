"""
Q-K Projection with proper normalization for memory retrieval.

REQ-P2-001: Q-K Projection with Normalization

Key equations (Omega Rule from Atlas paper Eq. 9):
    M_t = M_persistent + Σ(i=t-c+1 to t) γ_i^(t) * (k_i ⊗ k_i)
    norm_sum = norm_persistent + Σ(i=t-c+1 to t) γ_i^(t) * ||k_i||²
    q'_t = M_t @ q_t / norm_sum

Where:
    c = context window size (OMEGA_CONTEXT_WINDOW)
    γ_i^(t) = decay_base^(t-i) gives exponential decay for older tokens

Reference: Atlas paper (arXiv:2505.23735) Eq. 9 - Omega Rule
"""

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.config import OMEGA_CONTEXT_WINDOW, OMEGA_DECAY_BASE

logger = logging.getLogger(__name__)


class CausalQKMemoryProjection(nn.Module):
    """
    Causal Q-K memory projection with Omega Rule (Atlas paper Eq. 9).

    Implements a sliding context window with exponential decay:
        M_t = M_persistent + Σ(i=t-c+1 to t) γ^(t-i) * (k_i ⊗ k_i)

    Where:
        c = context_window (only last c keys contribute)
        γ = decay_base (exponential decay for older keys)

    This integrates with attention by providing memory-enhanced queries:
        q_original: standard queries (pure attention)
        q_projected: memory-enhanced queries (retrieval from past)

    Args:
        dim: Model dimension
        n_heads: Number of attention heads
        persistent_memory: Optional PersistentMemory module
        context_window: Size of sliding context window (default: OMEGA_CONTEXT_WINDOW)
        decay_base: Base decay rate for exponential weighting (default: OMEGA_DECAY_BASE)
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        persistent_memory: Optional[Any] = None,
        context_window: int = OMEGA_CONTEXT_WINDOW,
        decay_base: float = OMEGA_DECAY_BASE,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.context_window = context_window
        self.decay_base = decay_base

        # Precompute decay weights for positions 0..context_window-1
        # decay_weights[i] = decay_base^i (i=0 is most recent, highest weight)
        decay_weights = torch.tensor(
            [decay_base ** i for i in range(context_window)],
            dtype=torch.float32
        )
        self.register_buffer("decay_weights", decay_weights)

        # Get persistent memory values
        if persistent_memory is not None:
            self.register_buffer(
                "m_persistent",
                persistent_memory.m_persistent.clone()
            )
            self.register_buffer(
                "norm_persistent",
                torch.tensor(persistent_memory.norm_persistent)
            )
        else:
            # Identity-like initialization if no persistent memory
            self.register_buffer(
                "m_persistent",
                torch.eye(dim) * 0.1
            )
            self.register_buffer(
                "norm_persistent",
                torch.tensor(dim * 0.01)
            )

        logger.info(
            f"CausalQKMemoryProjection: context_window={context_window}, "
            f"decay_base={decay_base}"
        )

    def forward(self, q: Tensor, k: Tensor, gamma_gates: Optional[Tensor] = None) -> Tensor:
        """
        Project queries through causally accumulated key memory with Omega Rule.

        Dispatches to the chunked parallel implementation for performance.

        Args:
            q: Query tensor (batch, n_heads, seq_len, head_dim)
            k: Key tensor (batch, n_heads, seq_len, head_dim)
            gamma_gates: Optional input-dependent gates (batch, seq_len, 1)
                        If provided, multiplied with base decay weights

        Returns:
            Projected queries (batch, n_heads, seq_len, head_dim)
        """
        return self.forward_chunked(q, k, gamma_gates)

    def forward_sequential(self, q: Tensor, k: Tensor, gamma_gates: Optional[Tensor] = None) -> Tensor:
        """
        Sequential (reference) implementation - O(seq_len) Python loop.

        Kept for testing numerical equivalence with the parallel version.

        Args:
            q: Query tensor (batch, n_heads, seq_len, head_dim)
            k: Key tensor (batch, n_heads, seq_len, head_dim)
            gamma_gates: Optional input-dependent gates (batch, seq_len, 1)
                        If provided, multiplied with base decay weights

        Returns:
            Projected queries (batch, n_heads, seq_len, head_dim)
        """
        batch, n_heads, seq_len, head_dim = q.shape
        device = q.device
        dtype = q.dtype

        # Reshape to combine heads: (batch, seq_len, dim)
        dim = n_heads * head_dim
        q_flat = q.transpose(1, 2).reshape(batch, seq_len, dim)
        k_flat = k.transpose(1, 2).reshape(batch, seq_len, dim)

        # Get persistent memory on correct device/dtype
        m_persistent = self.m_persistent.to(device=device, dtype=dtype)
        norm_persistent = float(self.norm_persistent.item())  # type: ignore[union-attr]
        decay_weights = self.decay_weights.to(device=device, dtype=dtype)  # type: ignore[union-attr]

        # Store keys in a ring buffer for context window
        # We'll process position by position but only look back context_window
        q_projected_list = []

        for t in range(seq_len):
            q_t = q_flat[:, t, :]  # (batch, dim)

            # Determine context range: max(0, t - context_window) to t
            start_idx = max(0, t - self.context_window)
            context_len = t - start_idx  # Number of past keys to consider

            if context_len == 0:
                # No past keys, use only persistent memory
                M_t = m_persistent.unsqueeze(0).expand(batch, -1, -1)  # type: ignore[union-attr]
                norm_sum = torch.full((batch,), norm_persistent, device=device, dtype=dtype)
            else:
                # Get keys in context window: (batch, context_len, dim)
                k_context = k_flat[:, start_idx:t, :]

                # Compute decay weights for this context
                # Most recent key (position t-1) gets weight decay^0 = 1
                # Oldest key gets weight decay^(context_len-1)
                weights = decay_weights[:context_len].flip(0)  # (context_len,)

                # Apply input-dependent gamma gates if provided
                if gamma_gates is not None:
                    # gamma_gates: (batch, seq_len, 1)
                    # Get gates for positions in context: (batch, context_len)
                    context_gamma = gamma_gates[:, start_idx:t, 0]
                    # Combine with base decay: element-wise multiplication
                    weights = weights.unsqueeze(0) * context_gamma  # (batch, context_len)
                else:
                    weights = weights.unsqueeze(0).expand(batch, -1)  # (batch, context_len)

                # Compute weighted sum of outer products
                # k_context: (batch, context_len, dim)
                # weights: (batch, context_len)
                # Result: (batch, dim, dim)
                weighted_keys = k_context * weights.unsqueeze(-1).sqrt()  # Apply sqrt to both k's
                M_context = torch.einsum('bci,bcj->bij', weighted_keys, weighted_keys)

                # Add persistent memory
                M_t = m_persistent.unsqueeze(0) + M_context  # type: ignore[union-attr]

                # Compute weighted norm sum
                # ||k_i||² weighted by decay
                k_norms_sq = (k_context.norm(dim=-1) ** 2)  # (batch, context_len)
                weighted_norms = (k_norms_sq * weights).sum(dim=1)  # (batch,)
                norm_sum = norm_persistent + weighted_norms

            # Project query: q' = M_t @ q / norm_sum
            q_prime = torch.bmm(M_t, q_t.unsqueeze(-1)).squeeze(-1)  # (batch, dim)
            q_prime = q_prime / (norm_sum.unsqueeze(-1) + 1e-6)

            q_projected_list.append(q_prime)

        # Stack: (batch, seq_len, dim)
        q_proj = torch.stack(q_projected_list, dim=1)

        # Reshape to heads: (batch, n_heads, seq_len, head_dim)
        q_proj = q_proj.reshape(batch, seq_len, n_heads, head_dim).transpose(1, 2)

        return q_proj

    def _build_chunk_decay_weights(
        self,
        chunk_start: int,
        chunk_end: int,
        context_start: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """
        Build (chunk_len, context_len) decay weight matrix for parallel processing.

        For each position t in [chunk_start, chunk_end), computes weights for
        all positions j in [context_start, chunk_end) where j < t.

        Args:
            chunk_start: First position in the chunk (inclusive)
            chunk_end: Last position in the chunk (exclusive)
            context_start: First position in the context window
            device: Target device
            dtype: Target dtype

        Returns:
            Tensor of shape (chunk_len, max_context_len) with causal decay weights
        """
        # Create position indices
        t_indices = torch.arange(chunk_start, chunk_end, device=device, dtype=dtype)
        j_indices = torch.arange(context_start, chunk_end, device=device, dtype=dtype)

        # Compute distances: t - j for all pairs
        distances = t_indices.unsqueeze(1) - j_indices.unsqueeze(0)  # (chunk_len, max_context_len)

        # Causal mask: only attend to past positions (j < t, so distance > 0)
        causal_mask = distances > 0

        # Context window mask: only attend to positions within context_window
        window_mask = distances <= self.context_window

        # Compute decay weights: decay_base^(distance - 1) for distance > 0
        decay_exponents = torch.clamp(distances - 1, min=0)
        decay_weights = torch.pow(
            torch.tensor(self.decay_base, device=device, dtype=dtype),
            decay_exponents
        )

        # Apply masks
        combined_mask = causal_mask & window_mask
        decay_weights = decay_weights * combined_mask.to(dtype)

        return decay_weights

    def forward_chunked(
        self,
        q: Tensor,
        k: Tensor,
        gamma_gates: Optional[Tensor] = None,
        chunk_size: int = 64,
    ) -> Tensor:
        """
        Chunked parallel implementation - avoids O(seq_len) Python loop.

        Key insight: Instead of materializing M_t (which is dim × dim), we use
        the algebraic identity:
            q'_t = M_t @ q_t
                 = M_persistent @ q_t + Σ_j w_j * k_j * (k_j · q_t)

        This transforms O(dim²) per position into O(context_len × dim) via
        batched einsum operations.

        Args:
            q: Query tensor (batch, n_heads, seq_len, head_dim)
            k: Key tensor (batch, n_heads, seq_len, head_dim)
            gamma_gates: Optional input-dependent gates (batch, seq_len, 1)
            chunk_size: Number of positions to process in parallel

        Returns:
            Projected queries (batch, n_heads, seq_len, head_dim)
        """
        batch, n_heads, seq_len, head_dim = q.shape
        device = q.device
        dtype = q.dtype
        dim = n_heads * head_dim

        # Reshape to combine heads: (batch, seq_len, dim)
        q_flat = q.transpose(1, 2).reshape(batch, seq_len, dim)
        k_flat = k.transpose(1, 2).reshape(batch, seq_len, dim)

        # Get persistent memory on correct device/dtype
        m_persistent = self.m_persistent.to(device=device, dtype=dtype)
        norm_persistent = float(self.norm_persistent.item())  # type: ignore[union-attr]

        # Output buffer
        q_projected = torch.zeros_like(q_flat)

        # Process in chunks
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_len = chunk_end - chunk_start

            # Determine context window start for this chunk
            context_start = max(0, chunk_start - self.context_window)

            # Get chunk queries: (batch, chunk_len, dim)
            q_chunk = q_flat[:, chunk_start:chunk_end, :]

            # Compute contribution from persistent memory
            q_from_persistent = torch.einsum('btd,de->bte', q_chunk, m_persistent)

            # Handle the context keys
            if chunk_start == 0:
                # First chunk: position 0 has no context (only persistent memory)
                q_projected[:, 0, :] = q_from_persistent[:, 0, :] / (norm_persistent + 1e-6)

                if chunk_len > 1:
                    # Process positions 1 to chunk_end-1
                    k_context = k_flat[:, :chunk_end, :]  # (batch, chunk_end, dim)

                    # Build decay weights
                    decay_weights = self._build_chunk_decay_weights(
                        chunk_start=1,
                        chunk_end=chunk_end,
                        context_start=0,
                        device=device,
                        dtype=dtype,
                    )  # (chunk_len-1, chunk_end)

                    # Apply gamma gates if provided
                    if gamma_gates is not None:
                        context_gamma = gamma_gates[:, :chunk_end, 0]  # (batch, chunk_end)
                        decay_weights = decay_weights.unsqueeze(0) * context_gamma.unsqueeze(1)
                    else:
                        decay_weights = decay_weights.unsqueeze(0).expand(batch, -1, -1)

                    # Compute q · k for all pairs
                    kq_dots = torch.einsum('btd,bjd->btj', q_chunk[:, 1:, :], k_context)

                    # Apply decay weights
                    weighted_dots = kq_dots * decay_weights  # (batch, chunk_len-1, chunk_end)

                    # Compute contribution from context
                    q_from_context = torch.einsum('btj,bjd->btd', weighted_dots, k_context)

                    # Compute norm_sum for each position
                    k_norms_sq = (k_context ** 2).sum(dim=-1)  # (batch, chunk_end)
                    weighted_norms = torch.einsum('btj,bj->bt', decay_weights, k_norms_sq)
                    norm_sum = norm_persistent + weighted_norms  # (batch, chunk_len-1)

                    # Combine and normalize
                    q_total = q_from_persistent[:, 1:, :] + q_from_context
                    q_projected[:, 1:chunk_end, :] = q_total / (norm_sum.unsqueeze(-1) + 1e-6)

            else:
                # Non-first chunk: all positions have context
                k_context = k_flat[:, context_start:chunk_end, :]  # (batch, context_len, dim)

                # Build decay weights
                decay_weights = self._build_chunk_decay_weights(
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    context_start=context_start,
                    device=device,
                    dtype=dtype,
                )  # (chunk_len, context_len)

                # Apply gamma gates if provided
                if gamma_gates is not None:
                    context_gamma = gamma_gates[:, context_start:chunk_end, 0]  # (batch, context_len)
                    decay_weights = decay_weights.unsqueeze(0) * context_gamma.unsqueeze(1)
                else:
                    decay_weights = decay_weights.unsqueeze(0).expand(batch, -1, -1)

                # Compute q · k for all pairs
                kq_dots = torch.einsum('btd,bjd->btj', q_chunk, k_context)

                # Apply decay weights
                weighted_dots = kq_dots * decay_weights

                # Compute contribution from context
                q_from_context = torch.einsum('btj,bjd->btd', weighted_dots, k_context)

                # Compute norm_sum for each position
                k_norms_sq = (k_context ** 2).sum(dim=-1)
                weighted_norms = torch.einsum('btj,bj->bt', decay_weights, k_norms_sq)
                norm_sum = norm_persistent + weighted_norms

                # Combine and normalize
                q_total = q_from_persistent + q_from_context
                q_projected[:, chunk_start:chunk_end, :] = q_total / (norm_sum.unsqueeze(-1) + 1e-6)

        # Reshape to heads: (batch, n_heads, seq_len, head_dim)
        q_proj = q_projected.reshape(batch, seq_len, n_heads, head_dim).transpose(1, 2)

        return q_proj

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, n_heads={self.n_heads}, "
            f"context_window={self.context_window}, decay_base={self.decay_base}"
        )
