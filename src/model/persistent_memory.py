"""
Persistent Memory: M_persistent, norm_persistent, and W_init computation.

These are the foundation primitives that all subsequent phases depend on.
They are computed once at startup and shared across all shards.

Phase 0 Requirements:
    - P0-T1: W_init via steady-state calibration
    - P0-T2: M_persistent from 64 persistent keys
    - P0-T3: norm_persistent scalar
    - P0-T4: Hash verification for multi-GPU consistency

Reference: Atlas paper (arXiv:2505.23735)
Note: TNT hierarchical memory is NOT implemented; only Atlas-MAG with Omega Rule.
"""

import hashlib
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.config import D, N_PERSISTENT

logger = logging.getLogger(__name__)


def compute_m_persistent(persistent_keys: Tensor) -> Tensor:
    """
    Compute M_persistent: the outer product sum of all persistent keys.

    M_persistent = Σ k_p @ k_p^T for all persistent keys k_p

    This matrix is used in the Q-K projection to provide a stable
    "background" for the memory that doesn't reset at shard boundaries.

    Args:
        persistent_keys: Tensor of shape (N_persistent, dim)

    Returns:
        M_persistent matrix of shape (dim, dim)
    """
    # Validate input shape
    assert persistent_keys.ndim == 2, f"Expected 2D tensor, got {persistent_keys.ndim}D"

    # Compute outer product sum using einsum
    # 'pd,pe->de' means: sum over p (persistent tokens),
    # multiply d-th and e-th components
    m_persistent: Tensor = torch.einsum("pd,pe->de", persistent_keys, persistent_keys)

    logger.debug(
        f"Computed M_persistent: shape={m_persistent.shape}, "
        f"norm={torch.linalg.norm(m_persistent).item():.4f}"
    )

    return m_persistent


def compute_norm_persistent(persistent_keys: Tensor) -> float:
    """
    Compute norm_persistent: the sum of squared norms of persistent keys.

    norm_persistent = Σ ||k_p||² for all persistent keys k_p

    This scalar is used to normalize the Q-K projection denominator,
    ensuring numerical stability as memory accumulates.

    Args:
        persistent_keys: Tensor of shape (N_persistent, dim)

    Returns:
        norm_persistent as a Python float
    """
    # Compute squared L2 norm of each key, then sum
    norms_squared = persistent_keys.norm(dim=-1).pow(2)
    norm_persistent: float = float(norms_squared.sum().item())

    logger.debug(f"Computed norm_persistent: {norm_persistent:.6f}")

    return norm_persistent


def compute_hash(tensor: Tensor) -> str:
    """
    Compute a deterministic hash of a tensor for verification.

    Used to verify that M_persistent is identical across all GPUs.

    Args:
        tensor: Any tensor

    Returns:
        SHA-256 hash as hex string
    """
    # Move to CPU and convert to bytes deterministically
    data = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


class PersistentMemory(nn.Module):
    """
    Persistent Memory Module.

    Manages the persistent memory tokens and computes M_persistent
    and norm_persistent. These values are cached and only recomputed
    when the persistent keys change.

    Args:
        dim: Model dimension (default: 768)
        n_persistent: Number of persistent tokens (default: 64)

    Attributes:
        persistent_keys: Learnable persistent memory tokens
        m_persistent: Cached outer product sum (read-only buffer)
        norm_persistent: Cached norm sum (Python float)
    """

    # Type annotations for buffers
    m_persistent: Tensor
    _norm_persistent: Tensor

    def __init__(
        self,
        dim: int = D,
        n_persistent: int = N_PERSISTENT,
    ):
        super().__init__()
        self.dim = dim
        self.n_persistent = n_persistent

        # Learnable persistent memory tokens
        # Initialize with small random values for stable training start
        self.persistent_keys = nn.Parameter(
            torch.randn(n_persistent, dim) * 0.02
        )

        # Cached values (registered as buffers so they're saved with model)
        self.register_buffer("m_persistent", torch.zeros(dim, dim))
        self.register_buffer("_norm_persistent", torch.tensor(0.0))

        # Hash for consistency verification
        self._hash: Optional[str] = None

        # Compute initial values
        self.recompute()

    @property
    def norm_persistent(self) -> float:
        """Get norm_persistent as Python float."""
        return float(self._norm_persistent.item())

    def recompute(self) -> None:
        """
        Recompute M_persistent and norm_persistent from current keys.

        Should be called after persistent_keys are updated.
        """
        with torch.no_grad():
            self.m_persistent.copy_(compute_m_persistent(self.persistent_keys))
            self._norm_persistent.fill_(compute_norm_persistent(self.persistent_keys))
            self._hash = compute_hash(self.m_persistent)

        logger.info(
            f"Recomputed persistent memory: "
            f"M_persistent norm={torch.linalg.norm(self.m_persistent).item():.4f}, "
            f"norm_persistent={self.norm_persistent:.6f}"
        )

    def get_hash(self) -> str:
        """Get the current M_persistent hash for verification."""
        if self._hash is None:
            self._hash = compute_hash(self.m_persistent)
        return self._hash

    def verify_consistency(self, expected_hash: str) -> bool:
        """
        Verify M_persistent matches expected hash.

        Used for multi-GPU consistency checks.

        Args:
            expected_hash: Expected SHA-256 hash

        Returns:
            True if hashes match, False otherwise
        """
        current_hash = self.get_hash()
        matches = current_hash == expected_hash

        if not matches:
            logger.warning(
                f"Persistent memory hash mismatch! "
                f"Expected: {expected_hash[:16]}..., Got: {current_hash[:16]}..."
            )

        return matches

    def forward(self, keys: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute Q-K projection with persistent memory.

        The Q-K projection formula:
            q'_t = M_t @ q_t / norm_sum_t
            M_t = M_persistent + Σ(k_i @ k_i^T)
            norm_sum_t = norm_persistent + Σ||k_i||²

        Args:
            keys: Key vectors of shape (batch, seq_len, dim)

        Returns:
            Tuple of:
                - M_accumulated: M_persistent + cumulative key outer products
                - norm_accumulated: norm_persistent + cumulative key norms
        """
        batch, _seq_len, _dim = keys.shape
        device = keys.device
        dtype = keys.dtype

        # Start with persistent memory
        m_accumulated = self.m_persistent.clone().unsqueeze(0).expand(batch, -1, -1)
        norm_accumulated = torch.full(
            (batch, 1), self.norm_persistent, device=device, dtype=dtype
        )

        # Compute cumulative key contributions
        # For each position, add all previous keys' outer products
        # This is O(seq_len) outer products - can be optimized later

        # For now, return just the persistent component
        # Full accumulation will be added in Phase 1 with attention

        return m_accumulated, norm_accumulated

    def extra_repr(self) -> str:
        return f"dim={self.dim}, n_persistent={self.n_persistent}"


def broadcast_persistent_memory(
    persistent_memory: PersistentMemory,
    rank: int,
    world_size: int,
) -> str:
    """
    Broadcast M_persistent from rank 0 to all GPUs via NCCL.

    This ensures all GPUs have identical persistent memory for
    deterministic training across shards.

    Args:
        persistent_memory: The PersistentMemory module
        rank: Current process rank
        world_size: Total number of processes

    Returns:
        The verified hash string
    """
    import torch.distributed as dist

    if world_size == 1:
        # Single GPU mode - no broadcast needed
        return persistent_memory.get_hash()

    # Guard against uninitialized distributed backend
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized; call init_process_group() first"
        )

    # Broadcast M_persistent tensor
    if rank == 0:
        # Rank 0 has the authoritative values
        m_tensor = persistent_memory.m_persistent.contiguous()
        norm_tensor = persistent_memory._norm_persistent.clone()
    else:
        # Other ranks receive
        m_tensor = torch.zeros_like(persistent_memory.m_persistent)
        norm_tensor = torch.zeros_like(persistent_memory._norm_persistent)

    dist.broadcast(m_tensor, src=0)
    dist.broadcast(norm_tensor, src=0)

    # Update non-rank-0 processes
    if rank != 0:
        persistent_memory.m_persistent.copy_(m_tensor)
        persistent_memory._norm_persistent.copy_(norm_tensor)
        persistent_memory._hash = compute_hash(m_tensor)

    # Verify all ranks have same hash
    local_hash = persistent_memory.get_hash()
    all_hashes: list[Optional[str]] = [None] * world_size
    dist.all_gather_object(all_hashes, local_hash)

    if len(set(all_hashes)) != 1:
        logger.error(f"Persistent memory divergence detected: {all_hashes}")
        raise RuntimeError("M_persistent not consistent across GPUs!")

    logger.info(f"Persistent memory broadcast verified (hash: {local_hash[:16]}...)")

    return local_hash
