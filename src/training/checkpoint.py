"""
Checkpoint Manager with Rollback Support.

REQ-P2-003: Auto-rollback failsafe

Provides:
- Periodic checkpointing during training
- Rollback to previous state on PPL spike
- Checkpoint metadata tracking
- Cleanup of old checkpoints

The rollback is a FAILSAFE, not the primary strategy. Proactive
gradient clipping (Phase 4) is the primary defense. Rollback
catches cases where proactive measures fail.

Reference: PRD Section 7.4, P2-T6
"""

import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, List

import torch
from torch import nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for a saved checkpoint."""

    step: int
    loss: float
    perplexity: float
    path: Path
    timestamp: str
    size_mb: float


class CheckpointManager:
    """
    Manage training checkpoints with rollback support.

    Keeps the last N checkpoints and can rollback to any of them
    when training becomes unstable (PPL spike detected).

    Args:
        output_dir: Directory for checkpoints
        keep_last: Number of checkpoints to keep (default: 5)
        save_frequency: How often to save (in steps, default: 500)

    Example:
        >>> manager = CheckpointManager(Path("./checkpoints"))
        >>> for step in range(10000):
        ...     if manager.should_save(step):
        ...         manager.save(step, model, optimizer, loss, ppl)
        ...     if ppl_spike_detected:
        ...         manager.rollback(model, optimizer)
    """

    def __init__(
        self,
        output_dir: Path,
        keep_last: int = 5,
        save_frequency: int = 500,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.keep_last = keep_last
        self.save_frequency = save_frequency

        # Track checkpoints
        self.checkpoints: List[CheckpointMetadata] = []
        self.rollback_count = 0

        logger.info(
            f"CheckpointManager initialized: dir={self.output_dir}, "
            f"keep_last={keep_last}, save_freq={save_frequency}"
        )

    def should_save(self, step: int) -> bool:
        """Check if we should save at this step."""
        if step == 0:
            return True
        return step % self.save_frequency == 0

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optimizer,
        loss: float,
        perplexity: float,
        extra_state: Optional[dict] = None,
    ) -> CheckpointMetadata:
        """
        Save a checkpoint.

        Args:
            step: Current training step
            model: Model to save
            optimizer: Optimizer to save
            loss: Current loss value
            perplexity: Current perplexity
            extra_state: Additional state to save (e.g., scheduler)

        Returns:
            CheckpointMetadata for the saved checkpoint
        """
        # Create checkpoint path
        ckpt_path = self.output_dir / f"checkpoint_step{step:06d}.pt"

        # Build state dict
        state = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "perplexity": perplexity,
            "timestamp": datetime.now().isoformat(),
        }
        if extra_state:
            state["extra"] = extra_state

        # Save checkpoint
        torch.save(state, ckpt_path)

        # Get file size
        size_mb = ckpt_path.stat().st_size / (1024 * 1024)

        # Create metadata
        metadata = CheckpointMetadata(
            step=step,
            loss=loss,
            perplexity=perplexity,
            path=ckpt_path,
            timestamp=datetime.now().isoformat(),
            size_mb=size_mb,
        )

        self.checkpoints.append(metadata)

        # Cleanup old checkpoints
        self._cleanup()

        logger.info(
            f"Checkpoint saved: {ckpt_path.name} "
            f"(loss={loss:.4f}, ppl={perplexity:.2f}, size={size_mb:.1f}MB)"
        )

        return metadata

    def rollback(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        target_step: Optional[int] = None,
    ) -> Optional[int]:
        """
        Rollback to a previous checkpoint.

        Args:
            model: Model to restore
            optimizer: Optimizer to restore
            target_step: Specific step to rollback to (default: second-most-recent)

        Returns:
            Step that was rolled back to, or None if no checkpoint available
        """
        if not self.checkpoints:
            logger.warning("No checkpoints available for rollback")
            return None

        # Find target checkpoint
        if target_step is not None:
            # Find closest checkpoint at or before target
            candidates = [c for c in self.checkpoints if c.step <= target_step]
            if not candidates:
                logger.warning(f"No checkpoint at or before step {target_step}")
                return None
            checkpoint = max(candidates, key=lambda c: c.step)
        else:
            # Use second-most-recent (skip current which may be corrupt)
            if len(self.checkpoints) >= 2:
                checkpoint = self.checkpoints[-2]
            else:
                checkpoint = self.checkpoints[-1]

        # Verify checkpoint file exists
        if not checkpoint.path.exists():
            logger.error(f"Checkpoint file missing: {checkpoint.path}")
            return None

        # Load checkpoint
        state = torch.load(checkpoint.path, weights_only=False)

        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])

        self.rollback_count += 1

        logger.warning(
            f"ROLLBACK #{self.rollback_count}: Restored to step {checkpoint.step} "
            f"(loss={checkpoint.loss:.4f}, ppl={checkpoint.perplexity:.2f})"
        )

        return checkpoint.step

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        checkpoint_path: Optional[Path] = None,
    ) -> Optional[int]:
        """
        Load a specific checkpoint or the latest one.

        Args:
            model: Model to restore
            optimizer: Optimizer to restore
            checkpoint_path: Specific checkpoint to load (default: latest)

        Returns:
            Step of loaded checkpoint, or None if failed
        """
        if checkpoint_path is None:
            if not self.checkpoints:
                # Try to find checkpoints on disk
                self._discover_checkpoints()
            if not self.checkpoints:
                logger.warning("No checkpoints found")
                return None
            checkpoint_path = self.checkpoints[-1].path

        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return None

        state = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])

        step = int(state["step"])
        logger.info(f"Loaded checkpoint from step {step}")

        return step

    def _cleanup(self) -> None:
        """Remove old checkpoints beyond keep_last."""
        while len(self.checkpoints) > self.keep_last:
            old = self.checkpoints.pop(0)
            if old.path.exists():
                old.path.unlink()
                logger.debug(f"Removed old checkpoint: {old.path.name}")

    def _discover_checkpoints(self) -> None:
        """Discover existing checkpoints in output directory."""
        pattern = "checkpoint_step*.pt"
        found = sorted(self.output_dir.glob(pattern))

        for path in found:
            try:
                state = torch.load(path, weights_only=False)
                metadata = CheckpointMetadata(
                    step=state["step"],
                    loss=state.get("loss", 0.0),
                    perplexity=state.get("perplexity", 0.0),
                    path=path,
                    timestamp=state.get("timestamp", ""),
                    size_mb=path.stat().st_size / (1024 * 1024),
                )
                self.checkpoints.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {path}: {e}")

        self.checkpoints.sort(key=lambda c: c.step)
        logger.info(f"Discovered {len(self.checkpoints)} existing checkpoints")

    def get_latest(self) -> Optional[CheckpointMetadata]:
        """Get most recent checkpoint."""
        return self.checkpoints[-1] if self.checkpoints else None

    def get_statistics(self) -> dict:
        """Get checkpoint statistics."""
        total_size = sum(c.size_mb for c in self.checkpoints)
        return {
            "num_checkpoints": len(self.checkpoints),
            "rollback_count": self.rollback_count,
            "latest_step": self.checkpoints[-1].step if self.checkpoints else None,
            "total_size_mb": total_size,
            "output_dir": str(self.output_dir),
        }

    def clear(self) -> None:
        """Remove all checkpoints."""
        for ckpt in self.checkpoints:
            if ckpt.path.exists():
                ckpt.path.unlink()
        self.checkpoints.clear()
        logger.info("All checkpoints cleared")


def verify_rollback_trigger(
    model: nn.Module,
    optimizer: Optimizer,
    output_dir: Path,
) -> bool:
    """
    Test that rollback works correctly (for AC-P2-6).

    Creates a checkpoint, modifies the model, then rolls back
    and verifies the model was restored.

    Args:
        model: Model to test
        optimizer: Optimizer to test
        output_dir: Directory for test checkpoints

    Returns:
        True if rollback test passes
    """
    import tempfile

    # Use temp directory for test
    test_dir = Path(tempfile.mkdtemp())
    manager = CheckpointManager(test_dir, keep_last=3, save_frequency=100)

    try:
        # Get initial state
        initial_param = next(model.parameters()).clone()

        # Save checkpoint at step 100
        manager.save(
            step=100,
            model=model,
            optimizer=optimizer,
            loss=1.0,
            perplexity=2.7,
        )

        # Modify model (simulate training gone wrong)
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.5)

        # Verify model changed
        modified_param = next(model.parameters()).clone()
        if torch.allclose(initial_param, modified_param, atol=1e-6):
            logger.error("Model was not modified for rollback test")
            return False

        # Trigger rollback
        restored_step = manager.rollback(model, optimizer)

        if restored_step != 100:
            logger.error(f"Rollback failed: expected step 100, got {restored_step}")
            return False

        # Verify model restored
        restored_param = next(model.parameters())
        if not torch.allclose(initial_param, restored_param, atol=1e-6):
            logger.error("Model parameters not properly restored")
            return False

        logger.info("Rollback test PASSED")
        return True

    finally:
        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)
