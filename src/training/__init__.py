"""
Training utilities for Atlas-MAG with Omega Rule.

Phase 1: Architecture Validation
- Gate polarization loss with annealing
- Fast-fail monitoring for early abort
- Training loop orchestration

Phase 2: Training Infrastructure
- NIAH retrieval probes for memory validation
- PPL delta telemetry for spike detection
- Checkpoint management with auto-rollback
- Full Phase 2 training loop

Note: TNT (Titans-in-Titans) Stage 2 fine-tuning is NOT implemented.
"""

from src.training.polarization import (
    get_lambda_polar,
    gate_polarization_loss,
    compute_gate_statistics,
)
from src.training.gate_monitor import (
    FastFailError,
    GateMonitor,
)
from src.training.trainer import Phase1Trainer, verify_multiplicative_fusion
from src.training.niah_probe import NIAHProbe, NIAHResult, measure_niah_overhead
from src.training.telemetry import TelemetryLogger, PPLDeltaTracker, StepMetrics
from src.training.checkpoint import CheckpointManager, CheckpointMetadata, verify_rollback_trigger
from src.training.phase2_trainer import Phase2Trainer, Phase2StepResult, run_phase2_validation

__all__ = [
    # Polarization
    "get_lambda_polar",
    "gate_polarization_loss",
    "compute_gate_statistics",
    # Fast-fail
    "FastFailError",
    "GateMonitor",
    # Phase 1 Training
    "Phase1Trainer",
    "verify_multiplicative_fusion",
    # Phase 2: NIAH Probes
    "NIAHProbe",
    "NIAHResult",
    "measure_niah_overhead",
    # Phase 2: Telemetry
    "TelemetryLogger",
    "PPLDeltaTracker",
    "StepMetrics",
    # Phase 2: Checkpointing
    "CheckpointManager",
    "CheckpointMetadata",
    "verify_rollback_trigger",
    # Phase 2: Training Loop
    "Phase2Trainer",
    "Phase2StepResult",
    "run_phase2_validation",
]
