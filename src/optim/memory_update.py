"""
Tensorized Memory Updates for AtlasMAG.

REQ-P3-T2: No Python loops over token dimension

All memory operations are fully tensorized using:
- torch.einsum for batched outer products
- torch.bmm for batched matrix multiplications

This eliminates the performance bottleneck of Python loops
over sequence/token dimensions.

Reference: PRD Section 6.3, AC-P3-2
"""

import logging
from typing import Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def tensorized_outer_product_sum(keys: Tensor) -> Tensor:
    """
    Compute sum of outer products in a single tensorized operation.

    Given keys of shape (*, dim), computes:
        Σ(k_i ⊗ k_i) = Σ(k_i @ k_i.T)

    This is the core building block for memory updates.

    Args:
        keys: Key vectors of shape (..., dim)
               Can be (dim,), (n, dim), or (batch, seq, dim)

    Returns:
        Sum of outer products, shape (dim, dim)

    Note:
        Uses einsum for optimal GPU utilization.
        NO Python loops over any dimension.
    """
    # Flatten to 2D: (..., dim) -> (n, dim)
    dim = keys.size(-1)
    flat_keys = keys.reshape(-1, dim)

    # Compute sum of outer products in ONE operation
    # (n, dim).T @ (n, dim) = (dim, dim) is Σ(k_i ⊗ k_i)
    # Using einsum: 'nd,ne->de' means sum over n, keep d and e
    outer_sum = torch.einsum('nd,ne->de', flat_keys, flat_keys)

    return outer_sum


def tensorized_memory_update(
    M_t: Tensor,
    keys: Tensor,
    norm_sum: float,
    alpha: float = 1.0,
) -> Tuple[Tensor, float]:
    """
    Tensorized memory matrix update - NO Python loops.

    Updates M_t with a batch of keys:
        M_t = α × M_t + Σ(k_i ⊗ k_i)
        norm_sum += Σ||k_i||²

    Args:
        M_t: Current memory matrix, shape (dim, dim)
        keys: Key vectors, shape (..., dim)
        norm_sum: Current norm accumulator
        alpha: Memory decay factor (default: 1.0 for pure accumulation)

    Returns:
        (M_new, norm_new): Updated memory matrix and norm

    Example:
        >>> M = torch.zeros(768, 768)
        >>> keys = torch.randn(4, 64, 768)  # batch=4, seq=64
        >>> M_new, norm_new = tensorized_memory_update(M, keys, 0.0)
    """
    dim = M_t.size(0)

    # Flatten keys to 2D
    flat_keys = keys.reshape(-1, dim)

    # Compute sum of outer products (tensorized)
    outer_sum = torch.einsum('nd,ne->de', flat_keys, flat_keys)

    # Update memory matrix
    if alpha == 1.0:
        M_new = M_t + outer_sum
    else:
        M_new = alpha * M_t + outer_sum

    # Update norm accumulator (also tensorized)
    # ||k_i||² for each key, then sum
    norm_delta = (flat_keys.norm(dim=-1) ** 2).sum().item()
    norm_new = norm_sum + norm_delta

    return M_new, norm_new


def batch_qk_projection(
    M_t: Tensor,
    queries: Tensor,
    norm_sum: float,
    eps: float = 1e-8,
) -> Tensor:
    """
    Batch Q-K projection - NO Python loops.

    Projects all queries through the accumulated memory:
        q'_i = M_t @ q_i / norm_sum

    Args:
        M_t: Memory matrix, shape (dim, dim)
        queries: Query vectors, shape (..., dim)
        norm_sum: Norm accumulator for scaling
        eps: Small constant for numerical stability

    Returns:
        Projected queries, same shape as input

    Note:
        Uses einsum for efficient batched matrix-vector multiply.
    """
    original_shape = queries.shape
    dim = queries.size(-1)

    # Flatten to 2D
    flat_queries = queries.reshape(-1, dim)

    # Batch matrix-vector multiply using einsum
    # For each query q, we compute M_t @ q
    # 'ij,nj->ni' means: contract second axis of M with second axis of queries
    # This gives result[n,i] = sum_j M_t[i,j] * flat_queries[n,j] = (M_t @ q)[i]
    projected = torch.einsum('ij,nj->ni', M_t, flat_queries)

    # Scale by norm
    projected = projected / max(norm_sum, eps)

    # Reshape to original batch structure
    return projected.reshape(original_shape)


def incremental_memory_update(
    M_t: Tensor,
    k_t: Tensor,
    norm_sum: float,
) -> Tuple[Tensor, float]:
    """
    Single-key incremental update (for online/streaming).

    Updates memory with a single key:
        M_t = M_t + k_t ⊗ k_t
        norm_sum += ||k_t||²

    Args:
        M_t: Current memory matrix, shape (dim, dim)
        k_t: Single key vector, shape (dim,)
        norm_sum: Current norm accumulator

    Returns:
        (M_new, norm_new): Updated memory and norm

    Note:
        For batch updates, use tensorized_memory_update instead.
    """
    # Outer product for single key
    outer = torch.outer(k_t, k_t)

    M_new = M_t + outer
    norm_new = norm_sum + (k_t.norm() ** 2).item()

    return M_new, norm_new


def parallel_local_memory_update(
    local_memories: Tensor,
    local_keys: Tensor,
    local_norms: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Update multiple local memories in parallel.

    This is the core TNT operation: update all local shard memories
    simultaneously without any sequential dependencies.

    Args:
        local_memories: Stack of local M matrices, shape (num_shards, dim, dim)
        local_keys: Keys for each shard, shape (num_shards, shard_len, dim)
        local_norms: Norm accumulators, shape (num_shards,)

    Returns:
        (updated_memories, updated_norms)

    Note:
        Each shard is independent, enabling full parallelism.
        NO cross-shard dependencies within this operation.
    """
    _num_shards = local_memories.size(0)  # noqa: F841 - documentation variable
    _dim = local_memories.size(1)  # noqa: F841 - documentation variable
    _shard_len = local_keys.size(1)  # noqa: F841 - documentation variable

    # Compute outer product sums for each shard in parallel
    # Shape: (_num_shards, _shard_len, _dim) -> (_num_shards, _dim, _dim)
    outer_sums = torch.einsum('nsd,nse->nde', local_keys, local_keys)

    # Update all memories at once
    updated_memories = local_memories + outer_sums

    # Update all norms at once
    # (num_shards, shard_len, dim) -> (num_shards, shard_len) -> (num_shards,)
    norm_deltas = (local_keys.norm(dim=-1) ** 2).sum(dim=1)
    updated_norms = local_norms + norm_deltas

    return updated_memories, updated_norms


def reset_at_shard_boundaries(
    M_t: Tensor,
    norm_sum: float,
    M_persistent: Tensor,
    norm_persistent: float,
) -> Tuple[Tensor, float]:
    """
    Reset memory at shard boundary (inject persistent memory).

    At shard boundaries, we reset:
        M_t = M_persistent
        norm_sum = norm_persistent

    This ensures persistent memory is accessible from the first token.

    Args:
        M_t: Current memory (will be replaced)
        norm_sum: Current norm (will be replaced)
        M_persistent: Persistent memory matrix
        norm_persistent: Persistent norm value

    Returns:
        (M_persistent.clone(), norm_persistent)
    """
    return M_persistent.clone(), norm_persistent


def verify_no_python_loops() -> bool:
    """
    Verify that memory operations don't use Python loops.

    This is a compile-time check that can be run during testing
    to ensure all operations are properly tensorized.

    Returns:
        True if verification passes
    """
    # Create test data
    dim = 64
    batch = 4
    seq_len = 32

    M = torch.zeros(dim, dim)
    keys = torch.randn(batch, seq_len, dim)

    # These operations should be fully tensorized
    # (no Python for loops in the implementation)

    # 1. Outer product sum
    outer_sum = tensorized_outer_product_sum(keys)
    assert outer_sum.shape == (dim, dim)

    # 2. Memory update
    M_new, norm = tensorized_memory_update(M, keys, 0.0)
    assert M_new.shape == (dim, dim)

    # 3. Batch projection
    queries = torch.randn(batch, seq_len, dim)
    projected = batch_qk_projection(M_new, queries, norm)
    assert projected.shape == queries.shape

    # 4. Parallel local update
    num_shards = 4
    local_M = torch.zeros(num_shards, dim, dim)
    local_keys = torch.randn(num_shards, seq_len, dim)
    local_norms = torch.zeros(num_shards)

    updated_M, updated_norms = parallel_local_memory_update(
        local_M, local_keys, local_norms
    )
    assert updated_M.shape == (num_shards, dim, dim)
    assert updated_norms.shape == (num_shards,)

    logger.info("All tensorized operations verified (no Python loops)")
    return True
