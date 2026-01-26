"""
Phase 1 Training Loop for Architecture Validation.

Integrates:
- Standard language modeling loss (cross-entropy)
- Gate polarization loss (with annealing lambda)
- Fast-fail monitoring (early abort on broken training)

This trainer is designed to validate that the MAG architecture
can actually ROUTE between attention and memory, not just average.

Reference: PRD Phase 1 requirements P1-T1 through P1-T7
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

from src.training.polarization import gate_polarization_loss, compute_gate_statistics
from src.training.gate_monitor import GateMonitor, FastFailError

logger = logging.getLogger(__name__)


@dataclass
class TrainStepResult:
    """Result from a single training step."""

    lm_loss: float
    polar_loss: float
    total_loss: float
    gate_mean: float
    gate_std: float
    gate_min: float
    gate_max: float
    polarized_ratio: float
    lambda_polar: float
    step: int


class Phase1Trainer:
    """
    Training loop for Phase 1 Architecture Validation.

    This trainer validates that the MAG architecture can route
    between attention and memory by:
    1. Training with gate polarization loss
    2. Monitoring gate statistics
    3. Fast-failing on broken training runs

    Args:
        model: AtlasMAGSkeleton or compatible model with get_gate_values()
        optimizer: PyTorch optimizer
        total_steps: Total number of training steps
        log_interval: How often to log (default: every 100 steps)

    Example:
        >>> model = AtlasMAGSkeleton(...)
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> trainer = Phase1Trainer(model, optimizer, total_steps=10000)
        >>> for batch in dataloader:
        ...     result = trainer.train_step(batch, step)
        ...     if step % 100 == 0:
        ...         print(f"Step {step}: loss={result.total_loss:.4f}")
    """

    def __init__(
        self,
        model: Any,  # AtlasMAGSkeleton, but using Any for flexibility
        optimizer: Optimizer,
        total_steps: int,
        log_interval: int = 100,
    ):
        self.model = model
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.log_interval = log_interval

        # Gate monitor for fast-fail detection
        self.gate_monitor = GateMonitor()

        # Training state
        self.current_step = 0
        self.best_polar_ratio = 0.0  # Best polarization achieved

        logger.info(
            f"Phase1Trainer initialized: "
            f"total_steps={total_steps}, log_interval={log_interval}"
        )

    def train_step(
        self,
        input_ids: Tensor,
        step: Optional[int] = None,
    ) -> TrainStepResult:
        """
        Execute a single training step.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            step: Current step (uses internal counter if None)

        Returns:
            TrainStepResult with all metrics

        Raises:
            FastFailError: If fast-fail conditions are met
        """
        if step is None:
            step = self.current_step
        self.current_step = step + 1

        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        logits = self.model(input_ids)

        # Language modeling loss (next-token prediction)
        # logits: (batch, seq_len, vocab_size)
        # targets: input_ids shifted by 1
        lm_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
        )

        # Get gate values from all layers
        gate_list = self.model.get_gate_values()
        gate_values = torch.tensor(gate_list, device=logits.device)

        # Gate polarization loss
        polar_loss = gate_polarization_loss(gate_values, step, self.total_steps)

        # Combined loss
        total_loss = lm_loss + polar_loss

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        # Fast-fail check (may raise FastFailError)
        self.gate_monitor.check(gate_values, step)

        # Compute gate statistics
        gate_stats = compute_gate_statistics(gate_values)

        # Track best polarization
        if gate_stats["polarized_ratio"] > self.best_polar_ratio:
            self.best_polar_ratio = gate_stats["polarized_ratio"]

        # Logging
        if step % self.log_interval == 0:
            logger.info(
                f"[Step {step}] "
                f"lm_loss={lm_loss.item():.4f}, "
                f"polar_loss={polar_loss.item():.4f}, "
                f"gate_std={gate_stats['std']:.4f}, "
                f"polarized={gate_stats['polarized_ratio']:.1%}"
            )

        # Get current lambda for reporting
        from src.training.polarization import get_lambda_polar

        lambda_polar = get_lambda_polar(step, self.total_steps)

        return TrainStepResult(
            lm_loss=lm_loss.item(),
            polar_loss=polar_loss.item(),
            total_loss=total_loss.item(),
            gate_mean=gate_stats["mean"],
            gate_std=gate_stats["std"],
            gate_min=gate_stats["min"],
            gate_max=gate_stats["max"],
            polarized_ratio=gate_stats["polarized_ratio"],
            lambda_polar=lambda_polar,
            step=step,
        )

    def check_gates(self) -> dict:
        """
        Check current gate state.

        Returns:
            Dictionary with gate metrics
        """
        self.model.train(False)

        gate_list = self.model.get_gate_values()
        gate_values = torch.tensor(gate_list)

        stats = compute_gate_statistics(gate_values)

        return {
            "gate_values": gate_list,
            "statistics": stats,
            "best_polarized_ratio": self.best_polar_ratio,
            "monitor_summary": self.gate_monitor.get_summary(),
        }

    def reset(self) -> None:
        """Reset trainer state for a new run."""
        self.current_step = 0
        self.best_polar_ratio = 0.0
        self.gate_monitor.reset()


def verify_multiplicative_fusion(model: Any, device: str = "cpu") -> bool:
    """
    Verify that multiplicative fusion works correctly.

    When gate=0, output should not equal output when gate=1.
    This verifies the gate actually controls the output blend.

    This verifies P1-T6: Multiplicative fusion test.

    Args:
        model: AtlasMAGSkeleton or AtlasMAGBlock
        device: Device to run test on

    Returns:
        True if test passes, False otherwise
    """
    model = model.to(device)
    model.train(False)

    # Test with small input
    x = torch.randn(1, 32, model.dim if hasattr(model, "dim") else 768, device=device)

    # Get blocks to test (either model IS a block or has blocks)
    if hasattr(model, "blocks"):
        blocks = model.blocks
    else:
        blocks = [model]

    all_passed = True

    for i, block in enumerate(blocks):
        # Test gate=0 (pure attention)
        # Temporarily disable TTL for this test to avoid parameter modification
        original_ttl_enabled = getattr(block, 'ttl_enabled', False)
        block.ttl_enabled = False

        with torch.no_grad():
            # Store original gate
            original_gate = block.memory_gate.data.clone()

            # Force gate to 0
            block.memory_gate.data.fill_(-100)  # sigmoid(-100) is approximately 0
            output_gate0, _ = block(x.clone())

            # Force gate to 1
            block.memory_gate.data.fill_(100)  # sigmoid(100) is approximately 1
            output_gate1, _ = block(x.clone())

            # Restore original
            block.memory_gate.data.copy_(original_gate)

        # Restore TTL state
        block.ttl_enabled = original_ttl_enabled

        # Outputs should be different (attention vs memory)
        if torch.allclose(output_gate0, output_gate1, atol=1e-5):
            logger.warning(f"Block {i}: gate=0 and gate=1 outputs are identical!")
            all_passed = False
        else:
            diff = (output_gate0 - output_gate1).abs().mean().item()
            logger.info(f"Block {i}: gate difference = {diff:.6f}")

    return all_passed


def run_short_training(
    model: Any,
    optimizer: Optimizer,
    num_steps: int = 100,
    batch_size: int = 4,
    seq_len: int = 64,
    vocab_size: int = 1000,
    device: str = "cpu",
) -> list[TrainStepResult]:
    """
    Run a short training loop for testing.

    Useful for validating the training loop without full data.

    Args:
        model: Model to train
        optimizer: Optimizer
        num_steps: Number of steps to run
        batch_size: Batch size
        seq_len: Sequence length
        vocab_size: Vocabulary size for random data
        device: Device to train on

    Returns:
        List of TrainStepResult from each step
    """
    model = model.to(device)
    trainer = Phase1Trainer(model, optimizer, total_steps=num_steps, log_interval=10)

    results: list[TrainStepResult] = []

    for step in range(num_steps):
        # Generate random batch
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        try:
            result = trainer.train_step(input_ids, step)
            results.append(result)
        except FastFailError as e:
            logger.error(f"Fast-fail triggered: {e}")
            break

    return results
