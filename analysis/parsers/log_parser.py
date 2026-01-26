"""
Generic training log parser.

Parses training logs into structured data for analysis and visualization.
Designed to be extensible for different log formats.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class TrainingMetrics:
    """Container for parsed training metrics."""

    steps: list[int] = field(default_factory=list)
    tokens: list[float] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    ppl: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    tokens_per_sec: list[float] = field(default_factory=list)

    # Optional metrics (may not be present in all logs)
    polarization: list[float] = field(default_factory=list)
    gate_std: list[float] = field(default_factory=list)

    # Validation metrics (sparse - only at validation steps)
    val_steps: list[int] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_ppl: list[float] = field(default_factory=list)

    # Metadata
    config: dict = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert training metrics to pandas DataFrame."""
        data = {
            "step": self.steps,
            "tokens_M": self.tokens,
            "loss": self.loss,
            "ppl": self.ppl,
            "lr": self.lr,
            "grad_norm": self.grad_norm,
            "tokens_per_sec": self.tokens_per_sec,
        }

        # Add optional columns if they contain any non-None values
        # (None values will be converted to NaN by pandas)
        if any(v is not None for v in self.polarization):
            data["polarization"] = self.polarization
        if any(v is not None for v in self.gate_std):
            data["gate_std"] = self.gate_std

        return pd.DataFrame(data)

    def val_to_dataframe(self) -> pd.DataFrame:
        """Convert validation metrics to pandas DataFrame."""
        return pd.DataFrame(
            {
                "step": self.val_steps,
                "val_loss": self.val_loss,
                "val_ppl": self.val_ppl,
            }
        )


class LogParser:
    """
    Generic log parser with pluggable patterns.

    Supports:
        - Atlas-MAG training logs
        - Standard PyTorch training logs
        - Custom formats via pattern registration
    """

    # Atlas-MAG log pattern
    # Example: [Epoch 1/3] Step 100 | Tokens: 1.6M | LM Loss: 10.4500 | Polar: 2.7497 | PPL: 34546.05 | LR: 8.19e-08 | GradNorm: 5.96 | Gate std: 0.0838 | Tok/s: 20555
    ATLAS_PATTERN = re.compile(
        r"Step\s+(\d+)\s*\|"
        r"\s*Tokens:\s*([\d.]+)M\s*\|"
        r"\s*LM Loss:\s*([\d.]+)\s*\|"
        r"(?:\s*Polar:\s*([\d.]+)\s*\|)?"
        r"\s*PPL:\s*([\d.]+)\s*\|"
        r"\s*LR:\s*([\d.e+-]+)\s*\|"
        r"\s*GradNorm:\s*([\d.]+)\s*\|"
        r"(?:\s*Gate std:\s*([\d.NA/]+)\s*\|)?"
        r"\s*Tok/s:\s*(\d+)"
    )

    # Validation pattern
    # Example: [Validation] Loss: 3.3593, PPL: 28.77, Train/Val Gap: 0.83x
    VAL_PATTERN = re.compile(r"\[Validation\]\s*Loss:\s*([\d.]+),\s*PPL:\s*([\d.]+)")

    # Config pattern
    # Example: Config: dim=512, layers=6, heads=8
    CONFIG_PATTERN = re.compile(r"Config:\s*dim=(\d+),\s*layers=(\d+),\s*heads=(\d+)")

    # Mode pattern
    # Example: Mode: MEMORY+ATTENTION or Mode: ATTENTION-ONLY (ablation)
    MODE_PATTERN = re.compile(r"Mode:\s*([\w+-]+(?:\s*\([^)]+\))?)")

    def __init__(self):
        self.metrics = TrainingMetrics()
        self._last_step = -1

    def parse_file(self, log_path: str | Path) -> TrainingMetrics:
        """Parse a training log file."""
        log_path = Path(log_path)

        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")

        self.metrics = TrainingMetrics()
        self._last_step = -1

        with open(log_path, "r") as f:
            for line in f:
                self._parse_line(line)

        return self.metrics

    def _parse_line(self, line: str) -> None:
        """Parse a single log line."""
        # Try config pattern
        config_match = self.CONFIG_PATTERN.search(line)
        if config_match:
            self.metrics.config["dim"] = int(config_match.group(1))
            self.metrics.config["layers"] = int(config_match.group(2))
            self.metrics.config["heads"] = int(config_match.group(3))
            return

        # Try mode pattern
        mode_match = self.MODE_PATTERN.search(line)
        if mode_match:
            self.metrics.config["mode"] = mode_match.group(1).strip()
            return

        # Try training step pattern
        step_match = self.ATLAS_PATTERN.search(line)
        if step_match:
            step = int(step_match.group(1))

            # Avoid duplicates
            if step <= self._last_step:
                return
            self._last_step = step

            self.metrics.steps.append(step)
            self.metrics.tokens.append(float(step_match.group(2)))
            self.metrics.loss.append(float(step_match.group(3)))

            # Polarization (optional) - always append to keep aligned with steps
            polar = step_match.group(4)
            self.metrics.polarization.append(float(polar) if polar else None)

            self.metrics.ppl.append(float(step_match.group(5)))
            self.metrics.lr.append(float(step_match.group(6)))
            self.metrics.grad_norm.append(float(step_match.group(7)))

            # Gate std (optional, may be "N/A") - always append to keep aligned
            gate_std = step_match.group(8)
            if gate_std and gate_std != "N/A":
                self.metrics.gate_std.append(float(gate_std))
            else:
                self.metrics.gate_std.append(None)

            self.metrics.tokens_per_sec.append(int(step_match.group(9)))
            return

        # Try validation pattern (allow step 0)
        val_match = self.VAL_PATTERN.search(line)
        if val_match and self._last_step >= 0:
            self.metrics.val_steps.append(self._last_step)
            self.metrics.val_loss.append(float(val_match.group(1)))
            self.metrics.val_ppl.append(float(val_match.group(2)))
            return


def parse_training_log(log_path: str | Path) -> TrainingMetrics:
    """
    Convenience function to parse a training log.

    Args:
        log_path: Path to the training log file

    Returns:
        TrainingMetrics object with parsed data
    """
    parser = LogParser()
    return parser.parse_file(log_path)
