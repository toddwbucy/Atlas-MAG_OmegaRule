"""Training utilities for Atlas-MAG with Omega Rule."""

from src.training.checkpoint import CheckpointManager, CheckpointMetadata, verify_rollback_trigger
from src.training.niah_probe import NIAHProbe, NIAHResult
from src.training.omega_loss import compute_omega_loss, compute_omega_loss_with_stats
from src.training.ttl_update import TTLUpdater, ttl_step, ttl_step_with_grad_clip
from src.training.validation import run_validation

__all__ = [  # noqa: RUF022 - grouped by feature on purpose
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
