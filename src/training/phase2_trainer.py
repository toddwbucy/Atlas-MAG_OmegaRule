"""
Phase 2 Training Loop.

Extends Phase 1 trainer with full Phase 2 infrastructure:
- NIAH probes for memory validation
- PPL delta telemetry
- Auto-rollback failsafe
- Gradient masking at shard boundaries

This is the complete training loop for Phase 2 validation.

Reference: PRD Phase 2 requirements P2-T1 through P2-T6
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, List, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

from src.model.qk_projection import QKProjection, create_qk_projection_for_model
from src.training.polarization import gate_polarization_loss, compute_gate_statistics
from src.training.fast_fail import GateMonitor, FastFailError
from src.training.niah_probe import NIAHProbe, NIAHResult
from src.training.telemetry import TelemetryLogger, StepMetrics
from src.training.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


@dataclass
class Phase2StepResult:
    """Result from a Phase 2 training step."""

    step: int
    lm_loss: float
    polar_loss: float
    total_loss: float
    perplexity: float
    ppl_delta: Optional[float]
    gate_mean: float
    gate_std: float
    polarized_ratio: float
    niah_accuracy: Optional[float]
    rollback_triggered: bool
    is_spike: bool


class Phase2Trainer:
    """
    Training loop for Phase 2 with full infrastructure.

    Integrates:
    - Language modeling loss (cross-entropy)
    - Gate polarization loss (with annealing lambda)
    - Fast-fail gate monitoring
    - NIAH retrieval probes (validates memory works)
    - PPL delta telemetry (detects training instability)
    - Auto-rollback on spike (failsafe)

    Args:
        model: AtlasMAGSkeleton or compatible model
        optimizer: PyTorch optimizer
        total_steps: Total training steps
        output_dir: Directory for checkpoints and logs
        niah_frequency: NIAH probe frequency (default: 1000)
        checkpoint_frequency: Checkpoint save frequency (default: 500)
        ppl_spike_threshold: PPL delta threshold for rollback (default: 0.05)
        log_interval: Console logging interval (default: 100)

    Example:
        >>> trainer = Phase2Trainer(model, optimizer, 10000, Path("./output"))
        >>> summary = trainer.run_training(train_loader, "cuda")
        >>> print(f"Final NIAH accuracy: {summary['niah_stats']['mean_accuracy']}")
    """

    def __init__(
        self,
        model: Any,
        optimizer: Optimizer,
        total_steps: int,
        output_dir: Path,
        niah_frequency: int = 1000,
        checkpoint_frequency: int = 500,
        ppl_spike_threshold: float = 0.05,
        log_interval: int = 100,
    ):
        self.model = model
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.output_dir = Path(output_dir)
        self.log_interval = log_interval

        # Get model dimension
        self.dim = model.dim if hasattr(model, "dim") else 768

        # Gate monitoring (from Phase 1)
        self.gate_monitor = GateMonitor()

        # NIAH probes
        self.niah_probe = NIAHProbe(
            dim=self.dim,
            probe_frequency=niah_frequency,
        )

        # Telemetry (includes PPL delta tracker)
        self.telemetry = TelemetryLogger(
            output_dir=self.output_dir / "telemetry",
            spike_threshold=ppl_spike_threshold,
        )

        # Checkpointing
        self.checkpoint_manager = CheckpointManager(
            output_dir=self.output_dir / "checkpoints",
            save_frequency=checkpoint_frequency,
        )

        # Q-K projection for NIAH
        self.qk_proj: Optional[QKProjection] = None

        # State
        self.current_step = 0
        self.results: List[Phase2StepResult] = []

        logger.info(
            f"Phase2Trainer initialized: total_steps={total_steps}, "
            f"niah_freq={niah_frequency}, ckpt_freq={checkpoint_frequency}"
        )

    def _ensure_qk_projection(self) -> Optional[QKProjection]:
        """Ensure QK projection exists for NIAH probes."""
        if self.qk_proj is None:
            self.qk_proj = create_qk_projection_for_model(self.model)
        return self.qk_proj

    def train_step(
        self,
        input_ids: Tensor,
        step: Optional[int] = None,
    ) -> Phase2StepResult:
        """
        Execute a single Phase 2 training step.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            step: Current step (uses internal counter if None)

        Returns:
            Phase2StepResult with all metrics

        Raises:
            FastFailError: If gate monitoring triggers abort
        """
        if step is None:
            step = self.current_step
        self.current_step = step + 1

        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        logits = self.model(input_ids)

        # Language modeling loss (next-token prediction)
        lm_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
        )

        # Gate values and polarization loss
        gate_list = self.model.get_gate_values()
        gate_values = torch.tensor(gate_list, device=logits.device)
        polar_loss = gate_polarization_loss(gate_values, step, self.total_steps)

        # Combined loss
        total_loss = lm_loss + polar_loss

        # Check for NaN (AC-P2-4)
        if torch.isnan(total_loss):
            raise FastFailError(f"NaN loss detected at step {step}", step=step)

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        # Gate monitoring (may raise FastFailError)
        self.gate_monitor.check(gate_values, step)

        # Compute statistics
        gate_stats = compute_gate_statistics(gate_values)

        # NIAH probe (if scheduled)
        niah_accuracy = None
        if self.niah_probe.should_probe(step):
            qk_proj = self._ensure_qk_projection()
            if qk_proj is not None:
                niah_result = self.niah_probe.run_probe_standalone(
                    qk_proj, step, str(input_ids.device)
                )
                niah_accuracy = niah_result.accuracy

        # Telemetry logging (ppl_tracker.update is called internally)
        step_metrics = self.telemetry.log_step(
            step=step,
            lm_loss=lm_loss.item(),
            polar_loss=polar_loss.item(),
            total_loss=total_loss.item(),
            gate_mean=gate_stats["mean"],
            gate_std=gate_stats["std"],
            polarized_ratio=gate_stats["polarized_ratio"],
            learning_rate=self.optimizer.param_groups[0]["lr"],
            niah_accuracy=niah_accuracy,
        )

        # Get PPL info from step_metrics (computed once in telemetry)
        perplexity = step_metrics.perplexity
        ppl_delta = step_metrics.ppl_delta
        is_spike = step_metrics.is_spike

        # Checkpointing
        if self.checkpoint_manager.should_save(step):
            self.checkpoint_manager.save(
                step=step,
                model=self.model,
                optimizer=self.optimizer,
                loss=total_loss.item(),
                perplexity=perplexity,
            )

        # Rollback check on spike
        rollback_triggered = False
        if is_spike:
            logger.warning(f"PPL spike at step {step}! Triggering rollback...")
            restored = self.checkpoint_manager.rollback(self.model, self.optimizer)
            if restored is not None:
                rollback_triggered = True

        # Console logging
        if step % self.log_interval == 0:
            niah_str = f", niah={niah_accuracy:.3f}" if niah_accuracy else ""
            logger.info(
                f"[Step {step}] loss={total_loss.item():.4f}, "
                f"ppl={perplexity:.2f}, gate_std={gate_stats['std']:.4f}"
                f"{niah_str}"
            )

        result = Phase2StepResult(
            step=step,
            lm_loss=lm_loss.item(),
            polar_loss=polar_loss.item(),
            total_loss=total_loss.item(),
            perplexity=perplexity,
            ppl_delta=ppl_delta,
            gate_mean=gate_stats["mean"],
            gate_std=gate_stats["std"],
            polarized_ratio=gate_stats["polarized_ratio"],
            niah_accuracy=niah_accuracy,
            rollback_triggered=rollback_triggered,
            is_spike=is_spike,
        )

        self.results.append(result)
        return result

    def run_training(
        self,
        data_iterator: Iterator[Tensor],
        device: str = "cpu",
    ) -> dict:
        """
        Run full Phase 2 training loop.

        Args:
            data_iterator: Iterator yielding batches of token IDs
            device: Device to train on

        Returns:
            Summary statistics dictionary
        """
        self.model.to(device)
        self.results.clear()

        logger.info(f"Starting Phase 2 training: {self.total_steps} steps on {device}")

        step = 0
        try:
            while step < self.total_steps:
                try:
                    batch = next(data_iterator)
                except StopIteration:
                    logger.warning("Data iterator exhausted; stopping training early")
                    break

                input_ids = batch.to(device) if hasattr(batch, "to") else batch

                try:
                    self.train_step(input_ids, step)
                    step += 1
                except FastFailError:
                    logger.exception("Fast-fail triggered")
                    break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        # Flush telemetry
        self.telemetry.flush()

        # Build summary
        summary = self._build_summary()
        logger.info(f"Phase 2 training complete: {summary['completed_steps']} steps")

        return summary

    def run_training_with_loader(
        self,
        train_loader: Any,
        device: str = "cpu",
    ) -> dict:
        """
        Run training with a DataLoader (convenience wrapper).

        Args:
            train_loader: PyTorch DataLoader
            device: Device to train on

        Returns:
            Summary statistics
        """
        def infinite_loader():
            while True:
                for batch in train_loader:
                    yield batch

        return self.run_training(infinite_loader(), device)

    def _build_summary(self) -> dict:
        """Build summary statistics from training results."""
        if not self.results:
            return {"completed_steps": 0}

        losses = [r.total_loss for r in self.results]
        rollbacks = [r for r in self.results if r.rollback_triggered]
        spikes = [r for r in self.results if r.is_spike]

        return {
            "completed_steps": len(self.results),
            "final_loss": losses[-1],
            "min_loss": min(losses),
            "mean_loss": sum(losses) / len(losses),
            "niah_stats": self.niah_probe.get_statistics(),
            "checkpoint_stats": self.checkpoint_manager.get_statistics(),
            "telemetry_summary": self.telemetry.get_summary(),
            "num_rollbacks": len(rollbacks),
            "num_spikes": len(spikes),
            "gate_monitor_summary": self.gate_monitor.get_summary(),
        }

    def check_all_acceptance_criteria(self) -> dict:
        """
        Check all Phase 2 acceptance criteria.

        Returns:
            Dictionary with pass/fail for each criterion
        """
        results = {}

        # AC-P2-3: Norm in projection
        qk = self._ensure_qk_projection()
        results["AC-P2-3_norm_in_projection"] = (
            qk is not None and qk.norm_persistent > 0
        )

        # AC-P2-4: 1000 steps stable (no NaN, checked during training)
        results["AC-P2-4_stable_training"] = len(self.results) >= 1000

        # AC-P2-5: NIAH passing (> 80%)
        niah_stats = self.niah_probe.get_statistics()
        results["AC-P2-5_niah_passing"] = (
            niah_stats.get("pass_rate", 0) >= 0.8
        )

        # AC-P2-6: Rollback tested
        # Only passes if rollback was actually triggered during training.
        # If no natural rollback occurred, run verify_rollback_trigger() separately.
        results["AC-P2-6_rollback_tested"] = (
            self.checkpoint_manager.rollback_count > 0
        )

        # AC-P2-7: PPL delta visible in telemetry
        results["AC-P2-7_ppl_delta_visible"] = (
            self.telemetry.check_ppl_delta_visible()
        )

        return results


def run_phase2_validation(
    model: Any,
    optimizer: Optimizer,
    output_dir: Path,
    num_steps: int = 1000,
    batch_size: int = 4,
    seq_len: int = 64,
    vocab_size: int = 1000,
    device: str = "cpu",
) -> dict:
    """
    Run Phase 2 validation with synthetic data.

    Useful for testing the training infrastructure without real data.

    Args:
        model: Model to train
        optimizer: Optimizer
        output_dir: Output directory
        num_steps: Number of steps to run
        batch_size: Batch size
        seq_len: Sequence length
        vocab_size: Vocabulary size
        device: Device to train on

    Returns:
        Summary statistics
    """
    trainer = Phase2Trainer(
        model=model,
        optimizer=optimizer,
        total_steps=num_steps,
        output_dir=output_dir,
        niah_frequency=100,  # More frequent for validation
        checkpoint_frequency=200,
        log_interval=50,
    )

    def synthetic_data():
        while True:
            yield torch.randint(0, vocab_size, (batch_size, seq_len))

    return trainer.run_training(synthetic_data(), device)
