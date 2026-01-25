"""
Hash Verification for Multi-GPU Consistency.

Ensures M_persistent is identical across all GPUs before training starts.
This is critical for deterministic training with sharded data.

PRD Requirement P0-T4:
    Hash verification infrastructure.
    ACCEPTANCE: All GPUs report identical hash.

Reference: TNT paper (arXiv:2511.07343)
"""

import hashlib
import logging
from typing import List, Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def compute_tensor_hash(tensor: Tensor) -> str:
    """
    Compute deterministic SHA-256 hash of a tensor.

    Args:
        tensor: Any tensor

    Returns:
        SHA-256 hash as hex string (64 characters)
    """
    # Ensure deterministic byte representation
    data = tensor.detach().cpu().contiguous().float().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def all_gather_hashes(
    local_hash: str,
    world_size: int,
) -> List[str]:
    """
    Gather hashes from all processes.

    Args:
        local_hash: This process's hash
        world_size: Total number of processes

    Returns:
        List of all hashes (length = world_size)
    """
    try:
        import torch.distributed as dist

        if not dist.is_initialized():
            return [local_hash]

        all_hashes: List[Optional[str]] = [None] * world_size
        dist.all_gather_object(all_hashes, local_hash)
        # Filter out None values (should not happen in practice)
        return [h for h in all_hashes if h is not None]

    except ImportError:
        return [local_hash]


def verify_m_persistent_consistency(
    m_persistent: Tensor,
    world_size: int = 1,
    rank: int = 0,
) -> bool:
    """
    Verify M_persistent is identical across all GPUs.

    PRD Requirement: All GPUs must report identical hash before
    training can proceed.

    Args:
        m_persistent: The persistent memory matrix
        world_size: Number of processes (1 for single-GPU)
        rank: Current process rank

    Returns:
        True if all hashes match, False otherwise
    """
    local_hash = compute_tensor_hash(m_persistent)

    if world_size == 1:
        logger.info(f"Single-GPU mode: M_persistent hash = {local_hash[:16]}...")
        return True

    # Gather all hashes
    all_hashes = all_gather_hashes(local_hash, world_size)

    # Check for consistency
    unique_hashes = set(all_hashes)

    if len(unique_hashes) == 1:
        logger.info(
            f"M_persistent consistency verified across {world_size} GPUs "
            f"(hash: {local_hash[:16]}...)"
        )
        return True
    else:
        logger.error(
            f"M_persistent DIVERGENCE detected! "
            f"Found {len(unique_hashes)} different hashes across {world_size} GPUs:"
        )
        for i, h in enumerate(all_hashes):
            logger.error(f"  Rank {i}: {h[:16]}...")
        return False


def verify_tensor_consistency(
    tensor: Tensor,
    name: str,
    world_size: int = 1,
    rank: int = 0,
) -> bool:
    """
    Generic tensor consistency verification.

    Useful for verifying any shared state across GPUs.

    Args:
        tensor: Tensor to verify
        name: Name for logging
        world_size: Number of processes
        rank: Current process rank

    Returns:
        True if consistent, False otherwise
    """
    local_hash = compute_tensor_hash(tensor)

    if world_size == 1:
        logger.debug(f"{name} hash (single-GPU): {local_hash[:16]}...")
        return True

    all_hashes = all_gather_hashes(local_hash, world_size)
    unique_hashes = set(all_hashes)

    if len(unique_hashes) == 1:
        logger.debug(f"{name} consistent across {world_size} GPUs")
        return True
    else:
        logger.warning(f"{name} INCONSISTENT across GPUs!")
        return False


def run_phase0_verification(
    model,
    world_size: int = 1,
    rank: int = 0,
) -> dict:
    """
    Run all Phase 0 verifications.

    Args:
        model: The AtlasMAG model
        world_size: Number of processes
        rank: Current process rank

    Returns:
        Dict with verification results
    """
    results = {}

    # Verify M_persistent
    m_persistent = model.persistent_memory.m_persistent
    results["m_persistent"] = verify_m_persistent_consistency(
        m_persistent, world_size, rank
    )

    # Verify norm_persistent
    norm_tensor = torch.tensor([model.persistent_memory.norm_persistent])
    results["norm_persistent"] = verify_tensor_consistency(
        norm_tensor, "norm_persistent", world_size, rank
    )

    # Verify persistent_keys
    results["persistent_keys"] = verify_tensor_consistency(
        model.persistent_memory.persistent_keys,
        "persistent_keys",
        world_size,
        rank,
    )

    # Verify W_init
    results["w_init"] = verify_tensor_consistency(
        model.w_init, "w_init", world_size, rank
    )

    # Summary
    all_passed = all(results.values())
    if all_passed:
        logger.info("Phase 0 verification: ALL CHECKS PASSED")
    else:
        failed = [k for k, v in results.items() if not v]
        logger.error(f"Phase 0 verification: FAILED checks: {failed}")

    return results


class ConsistencyMonitor:
    """
    Monitor for ongoing consistency checks during training.

    Periodically verifies that shared state remains consistent
    across GPUs throughout training.
    """

    def __init__(
        self,
        model,
        world_size: int = 1,
        rank: int = 0,
        check_interval: int = 1000,
    ):
        self.model = model
        self.world_size = world_size
        self.rank = rank
        self.check_interval = check_interval
        self.step = 0
        self.history: List[tuple[int, dict]] = []

    def step_check(self) -> Optional[dict]:
        """
        Check consistency if at interval.

        Returns:
            Verification results if check was run, None otherwise
        """
        self.step += 1

        if self.step % self.check_interval != 0:
            return None

        results = run_phase0_verification(
            self.model, self.world_size, self.rank
        )
        self.history.append((self.step, results))

        return results

    def get_history(self) -> List[tuple]:
        """Get history of all checks."""
        return self.history
