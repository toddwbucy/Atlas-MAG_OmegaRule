"""
Fast-Fail Gate Monitoring for Phase 1 Architecture Validation.

Implements early abort conditions to detect broken training runs.
From PRD: "The system needs to scream 'I'm broken' immediately"

Fast-fail checks:
    1. Step 100: Record initial gate variance (baseline)
    2. Step 500: Verify variance increased ≥1.5× (gates are learning)
    3. Continuous: Abort if gate std < 0.01 (collapsed to single value)

Reference: PRD Section "Fast-Fail Checks"
"""

import logging
from typing import Optional

from torch import Tensor

from src.config import FAST_FAIL

logger = logging.getLogger(__name__)


class FastFailError(Exception):
    """
    Exception raised when fast-fail conditions are met.

    This indicates the training run is broken and should be aborted
    rather than wasting compute on a doomed experiment.

    Attributes:
        message: Description of the failure condition
        step: Training step where failure was detected
        details: Additional diagnostic information
    """

    def __init__(
        self,
        message: str,
        step: Optional[int] = None,
        details: Optional[dict] = None,
    ):
        self.message = message
        self.step = step
        self.details = details or {}

        # Build full message
        full_msg = f"FAST-FAIL: {message}"
        if step is not None:
            full_msg = f"[Step {step}] {full_msg}"
        if details:
            full_msg += f" | Details: {details}"

        super().__init__(full_msg)


class GateMonitor:
    """
    Monitor gate variance trajectory for fast-fail detection.

    The monitor tracks gate statistics and raises FastFailError if:
    1. Gate variance at step 500 < 1.5 × variance at step 100
    2. Gate std drops below 0.01 at any point after step 100

    These conditions indicate the gates aren't learning to route,
    meaning the architecture is stuck averaging.

    Example:
        >>> monitor = GateMonitor()
        >>> for step in range(1000):
        ...     gate_values = model.get_gate_values()
        ...     monitor.check(torch.tensor(gate_values), step)

    Raises:
        FastFailError: If any fast-fail condition is triggered
    """

    def __init__(self):
        self.initial_variance: Optional[float] = None
        self.baseline_step: int = 100
        self.check_step: int = FAST_FAIL.gate_variance_check_step  # 500
        self.variance_multiplier: float = FAST_FAIL.gate_variance_multiplier  # 1.5
        self.std_min: float = FAST_FAIL.gate_std_min  # 0.01

        # Statistics history (for debugging)
        self.history: list[dict] = []

    def check(self, gate_values: Tensor, step: int) -> dict:
        """
        Check gate statistics against fast-fail conditions.

        Args:
            gate_values: Tensor of gate values from model
            step: Current training step

        Returns:
            Dictionary with current gate statistics

        Raises:
            FastFailError: If any fast-fail condition is met
        """
        # Compute statistics
        gate_tensor = gate_values.detach().float()
        current_variance = gate_tensor.var().item()
        current_std = gate_tensor.std().item()
        current_mean = gate_tensor.mean().item()

        stats = {
            "step": step,
            "variance": current_variance,
            "std": current_std,
            "mean": current_mean,
            "min": gate_tensor.min().item(),
            "max": gate_tensor.max().item(),
        }

        # Record baseline at step 100
        if step == self.baseline_step:
            self.initial_variance = current_variance
            logger.info(
                f"[Step {step}] Gate baseline: "
                f"variance={current_variance:.6f}, "
                f"std={current_std:.4f}, "
                f"mean={current_mean:.4f}"
            )

        # Check 1: Variance increase at step 500
        if step == self.check_step and self.initial_variance is not None:
            expected_min = self.initial_variance * self.variance_multiplier
            if current_variance < expected_min:
                raise FastFailError(
                    f"Gate variance not increasing! "
                    f"Expected ≥{expected_min:.6f} (= {self.initial_variance:.6f} × {self.variance_multiplier}), "
                    f"got {current_variance:.6f}. "
                    f"ABORTING - architecture stuck in mushy middle.",
                    step=step,
                    details={
                        "initial_variance": self.initial_variance,
                        "current_variance": current_variance,
                        "expected_multiplier": self.variance_multiplier,
                    },
                )
            else:
                logger.info(
                    f"[Step {step}] Gate variance check PASSED: "
                    f"{current_variance:.6f} ≥ {expected_min:.6f}"
                )

        # Check 2: Continuous std check after step 100
        if step > self.baseline_step and current_std < self.std_min:
            raise FastFailError(
                f"Gate std collapsed to {current_std:.6f} < {self.std_min}. "
                f"All gates converged to same value - killing run.",
                step=step,
                details={
                    "std": current_std,
                    "threshold": self.std_min,
                    "mean": current_mean,
                },
            )

        # Store in history (keep last 100 entries)
        self.history.append(stats)
        if len(self.history) > 100:
            self.history = self.history[-100:]

        return stats

    def reset(self) -> None:
        """Reset monitor state for a new training run."""
        self.initial_variance = None
        self.history = []

    def get_summary(self) -> dict:
        """
        Get summary of gate monitoring.

        Returns:
            Dictionary with:
                - initial_variance: Baseline variance from step 100
                - latest_stats: Most recent statistics
                - num_checks: Number of checks performed
        """
        return {
            "initial_variance": self.initial_variance,
            "latest_stats": self.history[-1] if self.history else None,
            "num_checks": len(self.history),
        }


def check_gate_health(
    gate_values: Tensor,
    step: int,
    warn_threshold: float = 0.3,
) -> dict:
    """
    Quick health check for gate values (non-fatal).

    Unlike GateMonitor.check(), this doesn't raise exceptions.
    Use for logging and monitoring without aborting.

    Args:
        gate_values: Tensor of gate values
        step: Current step (for logging)
        warn_threshold: Threshold for warning about indecisive gates

    Returns:
        Dictionary with health status and diagnostics
    """
    values = gate_values.detach().float()

    # Count polarized vs indecisive
    polarized_low = (values < 0.1).sum().item()
    polarized_high = (values > 0.9).sum().item()
    indecisive = ((values > 0.4) & (values < 0.6)).sum().item()
    total = values.numel()

    indecisive_ratio = indecisive / total if total > 0 else 0.0
    polarized_ratio = (polarized_low + polarized_high) / total if total > 0 else 0.0

    healthy = indecisive_ratio < warn_threshold

    if not healthy:
        logger.warning(
            f"[Step {step}] High indecisive ratio: "
            f"{indecisive_ratio:.1%} of gates in (0.4, 0.6)"
        )

    return {
        "healthy": healthy,
        "polarized_ratio": polarized_ratio,
        "indecisive_ratio": indecisive_ratio,
        "polarized_low": polarized_low,
        "polarized_high": polarized_high,
        "indecisive": indecisive,
        "total": total,
    }
