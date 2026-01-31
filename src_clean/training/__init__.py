"""Training utilities for Atlas-MAG with Omega Rule."""

from src_clean.training.omega_loss import compute_omega_loss, compute_omega_loss_with_stats
from src_clean.training.ttl_update import ttl_step, ttl_step_with_grad_clip, TTLUpdater
from src_clean.training.validation import run_validation
from src_clean.training.niah_probe import NIAHProbe, NIAHResult
from src_clean.training.checkpoint import CheckpointManager, CheckpointMetadata, verify_rollback_trigger

__all__ = [
    # Omega Loss
    "compute_omega_loss",
    "compute_omega_loss_with_stats",
    # TTL (Test-Time Learning)
    "ttl_step",
    "ttl_step_with_grad_clip",
    "TTLUpdater",
    # Validation
    "run_validation",
    # NIAH Probes
    "NIAHProbe",
    "NIAHResult",
    # Checkpointing
    "CheckpointManager",
    "CheckpointMetadata",
    "verify_rollback_trigger",
]
