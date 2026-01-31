"""
Q-K Projection with proper normalization for memory retrieval.

REQ-P2-001: Q-K Projection with Normalization

This fixes "Silent Killer #2" from the PRD: M_persistent alone causes
volume mismatch. We need norm_persistent as the denominator for proper scaling.

Key equations (Omega Rule from Atlas paper Eq. 9):
    M_t = M_persistent + Σ(i=t-c+1 to t) γ_i^(t) * (k_i ⊗ k_i)
    norm_sum = norm_persistent + Σ(i=t-c+1 to t) γ_i^(t) * ||k_i||²
    q'_t = M_t @ q_t / norm_sum

Where:
    c = context window size (OMEGA_CONTEXT_WINDOW)
    γ_i^(t) = decay_base^(t-i) gives exponential decay for older tokens

At shard boundaries, both M_t and norm_sum are reset to their persistent
values, ensuring memory is accessible from the first token.

Reference: PRD Section 5.4 "Silent Killer #2 Fix: Normalization"
Reference: Atlas paper (arXiv:2505.23735) Eq. 9 - Omega Rule
"""

import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch import Tensor

from src_clean.config import OMEGA_CONTEXT_WINDOW, OMEGA_DECAY_BASE

logger = logging.getLogger(__name__)


class QKProjection(nn.Module):
    """
    Q-K Projection with running M_t accumulator.

    At shard boundaries, M_t is reset to M_persistent and norm_sum is
    reset to norm_persistent. This ensures persistent memory is accessible
    from the first token of each shard.

    The key insight: without the denominator (norm_sum), the projection
    has incorrect volume scaling, causing persistent memory to either
    be too quiet (drowned out) or too loud (overwhelming).

    Args:
        dim: Model dimension
        m_persistent: Precomputed M_persistent matrix of shape (dim, dim)
        norm_persistent: Precomputed norm_persistent scalar
    """

    def __init__(
        self,
        dim: int,
        m_persistent: Tensor,
        norm_persistent: float,
    ):
        super().__init__()
        self.dim = dim

        # Store persistent values (not parameters, just buffers)
        self.register_buffer("m_persistent", m_persistent.clone())
        self.norm_persistent = norm_persistent

        # Running accumulators
        self.register_buffer("M_t", torch.zeros(dim, dim))
        self.norm_sum: float = 0.0

        # Track initialization state
        self._initialized = False

        # Storage for NIAH testing (key -> value mapping)
        self._stored_values: Dict[int, Tensor] = {}

        logger.debug(
            f"QKProjection initialized: dim={dim}, "
            f"norm_persistent={norm_persistent:.4f}"
        )

    @classmethod
    def from_persistent_memory(cls, persistent_memory: Any) -> "QKProjection":
        """
        Create QKProjection from a PersistentMemory module.

        Args:
            persistent_memory: PersistentMemory instance with m_persistent
                              and norm_persistent attributes

        Returns:
            QKProjection instance
        """
        return cls(
            dim=persistent_memory.dim,
            m_persistent=persistent_memory.m_persistent,
            norm_persistent=persistent_memory.norm_persistent,
        )

    def reset_at_shard_boundary(self) -> None:
        """
        Reset M_t and norm_sum at shard boundary.

        CRITICAL: Must inject BOTH numerator (M_persistent) AND
        denominator (norm_persistent) for correct volume scaling.

        This is the key fix for Silent Killer #2.
        """
        self.M_t = self.m_persistent.clone()  # type: ignore[operator]
        self.norm_sum = self.norm_persistent
        self._initialized = True

        logger.debug(
            f"QKProjection reset at shard boundary: "
            f"norm_sum={self.norm_sum:.4f}"
        )

    def update(self, k_t: Tensor) -> None:
        """
        Update running accumulators with new key(s).

        Args:
            k_t: Key tensor of shape (dim,) for single key,
                 or (batch, dim) for batch of keys
        """
        if not self._initialized:
            self.reset_at_shard_boundary()

        if k_t.dim() == 1:
            # Single key: outer product k @ k^T
            self.M_t = self.M_t + torch.outer(k_t, k_t)
            self.norm_sum = self.norm_sum + (k_t.norm() ** 2).item()
        elif k_t.dim() == 2:
            # Batch of keys: sum of outer products
            # k_t: (batch, dim)
            # outer: (batch, dim, dim) via k[:, :, None] @ k[:, None, :]
            batch_outer = k_t.unsqueeze(-1) @ k_t.unsqueeze(-2)
            self.M_t = self.M_t + batch_outer.sum(dim=0)
            self.norm_sum = self.norm_sum + (k_t.norm(dim=-1) ** 2).sum().item()
        else:
            raise ValueError(f"Expected 1D or 2D tensor, got {k_t.dim()}D")

    def project(self, q_t: Tensor) -> Tensor:
        """
        Project query through accumulated key space.

        Computes: q' = M_t @ q / norm_sum

        Args:
            q_t: Query tensor of shape (dim,) for single query,
                 or (batch, dim) for batch of queries

        Returns:
            Projected query of same shape as input
        """
        if not self._initialized:
            self.reset_at_shard_boundary()

        # Avoid division by zero
        norm = max(self.norm_sum, 1e-8)

        if q_t.dim() == 1:
            # Single query: M_t @ q / norm
            result: Tensor = (self.M_t @ q_t) / norm
            return result
        elif q_t.dim() == 2:
            # Batch: q @ M_t^T / norm (equivalent to M_t @ q for each)
            result = (q_t @ self.M_t.T) / norm
            return result
        else:
            raise ValueError(f"Expected 1D or 2D tensor, got {q_t.dim()}D")

    def inject_memory(self, key: Tensor, value: Tensor) -> None:
        """
        Inject a key-value pair for NIAH testing.

        This simulates storing information that should be retrievable.
        The key is added to M_t, and the value is stored for later
        comparison during retrieval testing.

        Args:
            key: Key tensor of shape (dim,)
            value: Value tensor of shape (dim,)
        """
        # Update M_t with the key
        self.update(key)

        # Store value for retrieval testing
        # Use a simple hash of key for identification
        key_hash = self._compute_key_hash(key)
        self._stored_values[key_hash] = value.clone().detach()

        logger.debug(f"Injected memory: key_hash={key_hash}")

    def query_memory(self, key: Tensor) -> Tensor:
        """
        Query memory with a key (for NIAH testing).

        Returns the projected key, which should be similar to the
        stored value if memory is working correctly.

        Args:
            key: Key tensor of shape (dim,)

        Returns:
            Projected key of shape (dim,)
        """
        return self.project(key)

    def _compute_key_hash(self, key: Tensor) -> int:
        """Compute a hash for a key tensor."""
        # Use first 10 elements for efficiency
        key_slice = key.detach().cpu().numpy().flatten()[:10]
        return hash(tuple(float(x) for x in key_slice))

    def get_stored_value(self, key: Tensor) -> Optional[Tensor]:
        """
        Get the stored value for a key (for NIAH testing).

        Args:
            key: Key tensor

        Returns:
            Stored value if found, None otherwise
        """
        key_hash = self._compute_key_hash(key)
        return self._stored_values.get(key_hash)

    def clear_stored_values(self) -> None:
        """Clear stored values (for testing reset)."""
        self._stored_values.clear()

    def get_state(self) -> dict:
        """Get current state for debugging."""
        return {
            "initialized": self._initialized,
            "norm_sum": self.norm_sum,
            "norm_persistent": self.norm_persistent,
            "M_t_norm": self.M_t.norm().item(),
            "m_persistent_norm": self.m_persistent.norm().item(),  # type: ignore[operator]
            "num_stored_values": len(self._stored_values),
        }

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, "
            f"norm_persistent={self.norm_persistent:.4f}, "
            f"initialized={self._initialized}"
        )


def create_qk_projection_for_model(model: Any) -> Optional[QKProjection]:
    """
    Create a QKProjection from a model's persistent memory.

    Searches for PersistentMemory in the model and creates
    a QKProjection from it.

    Args:
        model: Model with persistent_memory attribute

    Returns:
        QKProjection instance, or None if no persistent memory found
    """
    if hasattr(model, "persistent_memory"):
        return QKProjection.from_persistent_memory(model.persistent_memory)

    # Search in common locations
    for attr in ["memory", "persistent", "pm"]:
        if hasattr(model, attr):
            pm = getattr(model, attr)
            if hasattr(pm, "m_persistent") and hasattr(pm, "norm_persistent"):
                return QKProjection.from_persistent_memory(pm)

    logger.warning("No persistent memory found in model")
    return None


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
        norm_persistent = float(self.norm_persistent.item())  # type: ignore[operator]
        decay_weights = self.decay_weights.to(device=device, dtype=dtype)  # type: ignore[operator]

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
                M_t = m_persistent.unsqueeze(0).expand(batch, -1, -1)  # type: ignore[operator]
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
                M_t = m_persistent.unsqueeze(0) + M_context  # type: ignore[operator]

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
        # Note: chunk_len and max_context_len are implicit in the tensor shapes
        # chunk_len = chunk_end - chunk_start
        # max_context_len = chunk_end - context_start

        # Create position indices
        # t_indices[i] = chunk_start + i (positions being queried)
        t_indices = torch.arange(chunk_start, chunk_end, device=device, dtype=dtype)
        # j_indices[j] = context_start + j (positions being attended to)
        j_indices = torch.arange(context_start, chunk_end, device=device, dtype=dtype)

        # Compute distances: t - j for all pairs
        # distances[i, j] = t_indices[i] - j_indices[j]
        distances = t_indices.unsqueeze(1) - j_indices.unsqueeze(0)  # (chunk_len, max_context_len)

        # Causal mask: only attend to past positions (j < t, so distance > 0)
        causal_mask = distances > 0

        # Context window mask: only attend to positions within context_window
        window_mask = distances <= self.context_window

        # Compute decay weights: decay_base^(distance - 1) for distance > 0
        # distance=1 → weight=1, distance=2 → weight=decay_base, etc.
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
        norm_persistent = float(self.norm_persistent.item())  # type: ignore[operator]

        # Output buffer
        q_projected = torch.zeros_like(q_flat)

        # Process in chunks
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_len = chunk_end - chunk_start

            # Determine context window start for this chunk
            # The first position in chunk needs context from max(0, chunk_start - context_window)
            context_start = max(0, chunk_start - self.context_window)

            # Get chunk queries: (batch, chunk_len, dim)
            q_chunk = q_flat[:, chunk_start:chunk_end, :]

            # Compute contribution from persistent memory
            # q_from_persistent[b, t, d] = Σ_e m_persistent[d, e] * q_chunk[b, t, e]
            # = (q_chunk @ m_persistent.T)
            q_from_persistent = torch.einsum('btd,de->bte', q_chunk, m_persistent)

            # Handle the context keys
            if chunk_start == 0:
                # First chunk: positions 0..chunk_end
                # Position 0 has no context (only persistent memory)
                # Position t has context from 0 to t-1

                # For simplicity, we handle position 0 separately
                q_projected[:, 0, :] = q_from_persistent[:, 0, :] / (norm_persistent + 1e-6)

                if chunk_len > 1:
                    # Process positions 1 to chunk_end-1
                    # Context is from 0 to chunk_end-1 (all keys before each position)
                    k_context = k_flat[:, :chunk_end, :]  # (batch, chunk_end, dim)

                    # Build decay weights for positions 1..chunk_end-1
                    # Each row i corresponds to position (1 + i) in the chunk
                    # Each column j corresponds to key position j
                    decay_weights = self._build_chunk_decay_weights(
                        chunk_start=1,
                        chunk_end=chunk_end,
                        context_start=0,
                        device=device,
                        dtype=dtype,
                    )  # (chunk_len-1, chunk_end)

                    # Apply gamma gates if provided
                    if gamma_gates is not None:
                        # gamma_gates: (batch, seq_len, 1)
                        # Get gates for context positions: (batch, chunk_end)
                        context_gamma = gamma_gates[:, :chunk_end, 0]  # (batch, chunk_end)
                        # Broadcast multiply: (1, chunk_len-1, chunk_end) * (batch, 1, chunk_end)
                        decay_weights = decay_weights.unsqueeze(0) * context_gamma.unsqueeze(1)
                    else:
                        decay_weights = decay_weights.unsqueeze(0).expand(batch, -1, -1)

                    # Compute q · k for all pairs
                    # q_chunk[:, 1:, :]: (batch, chunk_len-1, dim)
                    # k_context: (batch, chunk_end, dim)
                    # kq_dots[b, t, j] = q_chunk[b, 1+t, :] · k_context[b, j, :]
                    kq_dots = torch.einsum('btd,bjd->btj', q_chunk[:, 1:, :], k_context)

                    # Apply decay weights
                    weighted_dots = kq_dots * decay_weights  # (batch, chunk_len-1, chunk_end)

                    # Compute contribution from context: Σ_j w_j * k_j * (k_j · q)
                    # q_from_context[b, t, d] = Σ_j weighted_dots[b, t, j] * k_context[b, j, d]
                    q_from_context = torch.einsum('btj,bjd->btd', weighted_dots, k_context)

                    # Compute norm_sum for each position
                    # k_norms_sq[b, j] = ||k_context[b, j]||²
                    k_norms_sq = (k_context ** 2).sum(dim=-1)  # (batch, chunk_end)
                    # norm_sum[b, t] = norm_persistent + Σ_j decay_weights[b, t, j] * k_norms_sq[b, j]
                    weighted_norms = torch.einsum('btj,bj->bt', decay_weights, k_norms_sq)
                    norm_sum = norm_persistent + weighted_norms  # (batch, chunk_len-1)

                    # Combine and normalize
                    q_total = q_from_persistent[:, 1:, :] + q_from_context
                    q_projected[:, 1:chunk_end, :] = q_total / (norm_sum.unsqueeze(-1) + 1e-6)

            else:
                # Non-first chunk: all positions have context
                # Context is from context_start to chunk_end
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
                    # Get gates for context positions
                    context_gamma = gamma_gates[:, context_start:chunk_end, 0]  # (batch, context_len)
                    decay_weights = decay_weights.unsqueeze(0) * context_gamma.unsqueeze(1)
                else:
                    decay_weights = decay_weights.unsqueeze(0).expand(batch, -1, -1)

                # Compute q · k for all pairs
                # kq_dots[b, t, j] = q_chunk[b, t, :] · k_context[b, j, :]
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
