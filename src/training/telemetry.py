"""
Training Telemetry for Phase 2.

Implements:
- PPL delta logging (every 100 steps in Stage 2)
- Spike detection for rollback triggering
- Metric aggregation and JSONL output
- Dashboard-ready format

Reference: PRD P2-T5, Section 10.1
"""

import json
import logging
import math
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Deque


logger = logging.getLogger(__name__)


@dataclass
class StepMetrics:
    """Metrics for a single training step."""

    step: int
    timestamp: str
    lm_loss: float
    polar_loss: float
    total_loss: float
    perplexity: float
    ppl_delta: Optional[float]
    gate_mean: float
    gate_std: float
    polarized_ratio: float
    niah_accuracy: Optional[float]
    learning_rate: float
    chunk_size: Optional[int] = None
    is_spike: bool = False


class PPLDeltaTracker:
    """
    Track perplexity delta for rollback detection.

    Computes delta from exponential moving average (EMA).
    Triggers spike warning when delta exceeds threshold.

    The PPL delta is defined as:
        delta = (current_ppl - ema_ppl) / ema_ppl

    A spike is detected when delta > spike_threshold (default 5%).

    Args:
        window_size: EMA window size (default: 100)
        spike_threshold: Percentage threshold for spike (default: 0.05 = 5%)
    """

    def __init__(
        self,
        window_size: int = 100,
        spike_threshold: float = 0.05,
    ):
        self.window_size = window_size
        self.spike_threshold = spike_threshold

        # EMA state
        self.ema: Optional[float] = None
        self.alpha = 2.0 / (window_size + 1)

        # History for analysis
        self.history: Deque[float] = deque(maxlen=1000)
        self.delta_history: Deque[float] = deque(maxlen=1000)
        self.spike_count = 0

    def update(self, perplexity: float) -> tuple[float, bool]:
        """
        Update tracker with new perplexity value.

        Args:
            perplexity: Current perplexity value

        Returns:
            Tuple of (ppl_delta, is_spike)
        """
        # Handle first value
        if self.ema is None:
            self.ema = perplexity
            self.history.append(perplexity)
            self.delta_history.append(0.0)
            return 0.0, False

        # Compute delta from EMA
        delta = (perplexity - self.ema) / max(self.ema, 1e-8)

        # Update EMA
        self.ema = self.alpha * perplexity + (1 - self.alpha) * self.ema

        # Record history
        self.history.append(perplexity)
        self.delta_history.append(delta)

        # Check for spike
        is_spike = delta > self.spike_threshold

        if is_spike:
            self.spike_count += 1
            logger.warning(
                f"PPL spike detected: delta={delta:.1%} > {self.spike_threshold:.1%} "
                f"(current={perplexity:.2f}, ema={self.ema:.2f})"
            )

        return delta, is_spike

    def get_moving_average(self) -> Optional[float]:
        """Get current EMA value."""
        return self.ema

    def get_statistics(self) -> dict:
        """Get tracker statistics."""
        if not self.history:
            return {"count": 0}

        values = list(self.history)
        deltas = list(self.delta_history)

        return {
            "count": len(values),
            "current_ema": self.ema,
            "latest_ppl": values[-1] if values else None,
            "min_ppl": min(values),
            "max_ppl": max(values),
            "mean_ppl": sum(values) / len(values),
            "latest_delta": deltas[-1] if deltas else None,
            "mean_delta": sum(deltas) / len(deltas) if deltas else 0.0,
            "spike_count": self.spike_count,
            "spike_threshold": self.spike_threshold,
        }

    def reset(self) -> None:
        """Reset tracker state."""
        self.ema = None
        self.history.clear()
        self.delta_history.clear()
        self.spike_count = 0


class TelemetryLogger:
    """
    Centralized telemetry logging.

    Writes metrics to JSONL file for dashboard consumption.
    Also logs to standard logger for console output.

    Output format (metrics.jsonl):
        {"step": 0, "lm_loss": 6.9, "perplexity": 1000.0, "ppl_delta": 0.0, ...}
        {"step": 100, "lm_loss": 5.2, "perplexity": 181.3, "ppl_delta": -0.82, ...}

    Args:
        output_dir: Directory for metrics files
        log_frequency: How often to write to file (steps)
    """

    def __init__(
        self,
        output_dir: Path,
        log_frequency: int = 100,
        spike_threshold: float = 0.05,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_frequency = log_frequency
        self.metrics_file = self.output_dir / "metrics.jsonl"

        # PPL delta tracker
        self.ppl_tracker = PPLDeltaTracker(spike_threshold=spike_threshold)

        # Accumulated metrics (in memory)
        self.all_metrics: List[StepMetrics] = []

        # Track how many metrics have been written to avoid duplicates in flush
        self._written_count = 0

        # Tracking
        self.start_time = datetime.now()

        logger.info(
            f"TelemetryLogger initialized: output={self.metrics_file}, "
            f"log_freq={log_frequency}"
        )

    def log_step(
        self,
        step: int,
        lm_loss: float,
        polar_loss: float,
        total_loss: float,
        gate_mean: float,
        gate_std: float,
        polarized_ratio: float,
        learning_rate: float,
        niah_accuracy: Optional[float] = None,
        chunk_size: Optional[int] = None,
    ) -> StepMetrics:
        """
        Log metrics for a training step.

        Args:
            step: Current training step
            lm_loss: Language modeling loss
            polar_loss: Gate polarization loss
            total_loss: Combined loss
            gate_mean: Mean gate value
            gate_std: Gate standard deviation
            polarized_ratio: Fraction of polarized gates
            learning_rate: Current learning rate
            niah_accuracy: NIAH probe accuracy (if run this step)
            chunk_size: Current chunk size (for TNT training)

        Returns:
            StepMetrics object with all metrics
        """
        # Compute perplexity from LM loss
        perplexity = math.exp(min(lm_loss, 20.0))  # Cap to avoid overflow

        # Track PPL delta
        ppl_delta, is_spike = self.ppl_tracker.update(perplexity)

        # Create metrics object
        metrics = StepMetrics(
            step=step,
            timestamp=datetime.now().isoformat(),
            lm_loss=lm_loss,
            polar_loss=polar_loss,
            total_loss=total_loss,
            perplexity=perplexity,
            ppl_delta=ppl_delta,
            gate_mean=gate_mean,
            gate_std=gate_std,
            polarized_ratio=polarized_ratio,
            niah_accuracy=niah_accuracy,
            learning_rate=learning_rate,
            chunk_size=chunk_size,
            is_spike=is_spike,
        )

        self.all_metrics.append(metrics)

        # Write to file periodically
        if step % self.log_frequency == 0:
            self._write_metrics(metrics)
            self._written_count = len(self.all_metrics)

        return metrics

    def _write_metrics(self, metrics: StepMetrics) -> None:
        """Append metrics to JSONL file."""
        with open(self.metrics_file, "a") as f:
            # Convert dataclass to dict, handling None values
            data = asdict(metrics)
            f.write(json.dumps(data) + "\n")

    def get_latest_metrics(self, n: int = 10) -> List[StepMetrics]:
        """Get the last N metrics."""
        return self.all_metrics[-n:]

    def get_summary(self) -> dict:
        """Get summary of all logged metrics."""
        if not self.all_metrics:
            return {"num_steps": 0}

        losses = [m.total_loss for m in self.all_metrics]
        ppl_values = [m.perplexity for m in self.all_metrics]
        spikes = [m for m in self.all_metrics if m.is_spike]

        return {
            "num_steps": len(self.all_metrics),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "final_loss": losses[-1],
            "min_loss": min(losses),
            "mean_loss": sum(losses) / len(losses),
            "final_perplexity": ppl_values[-1],
            "min_perplexity": min(ppl_values),
            "ppl_tracker": self.ppl_tracker.get_statistics(),
            "num_spikes": len(spikes),
            "metrics_file": str(self.metrics_file),
        }

    def flush(self) -> None:
        """Flush remaining metrics to file."""
        # Write unwritten metrics (those after _written_count)
        for metrics in self.all_metrics[self._written_count:]:
            self._write_metrics(metrics)
        self._written_count = len(self.all_metrics)

    def check_ppl_delta_visible(self) -> bool:
        """
        Verify that ppl_delta is being logged (for G2-7 checkpoint).

        Returns:
            True if metrics.jsonl exists and contains ppl_delta field
        """
        if not self.metrics_file.exists():
            return False

        try:
            with open(self.metrics_file, "r") as f:
                first_line = f.readline()
                if not first_line:
                    return False
                data = json.loads(first_line)
                return "ppl_delta" in data
        except (json.JSONDecodeError, IOError):
            return False
