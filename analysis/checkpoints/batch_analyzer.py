"""
Batch checkpoint analyzer for processing multiple checkpoints.

Extracts metrics from each checkpoint to track model evolution over training.
"""

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class CheckpointMetrics:
    """Metrics extracted from a single checkpoint."""

    step: int
    path: str

    # Validation metrics (if available)
    val_loss: Optional[float] = None
    val_ppl: Optional[float] = None

    # Gate metrics
    gate_values: list[float] = field(default_factory=list)
    gate_mean: float = 0.0
    gate_std: float = 0.0
    gate_min: float = 0.0
    gate_max: float = 0.0

    # Memory metrics (for TTL models)
    has_memory: bool = False
    memory_param_norms: dict[str, float] = field(default_factory=dict)
    momentum_norms: dict[str, float] = field(default_factory=dict)

    # Model config
    dim: int = 0
    n_layers: int = 0
    n_heads: int = 0


class BatchCheckpointAnalyzer:
    """
    Analyze a series of checkpoints to track model evolution.

    Usage:
        analyzer = BatchCheckpointAnalyzer("runs/atlas_54m_ttl")
        results = analyzer.analyze_all()
        analyzer.save_results("output/checkpoint_analysis.json")
    """

    # Pattern to extract step number from checkpoint filename
    STEP_PATTERN = re.compile(r"checkpoint_step(\d+)\.pt")

    def __init__(self, run_dir: str | Path, device: str | None = None):
        self.run_dir = Path(run_dir)
        # Auto-detect CUDA availability if device not specified
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.results: list[CheckpointMetrics] = []

        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

    def find_checkpoints(self) -> list[tuple[int, Path]]:
        """Find all checkpoints and return sorted by step."""
        checkpoints = []

        for pt_file in self.run_dir.glob("*.pt"):
            # Skip best/final models for evolution analysis
            if pt_file.name in ("best_model.pt", "final_model.pt"):
                continue

            match = self.STEP_PATTERN.match(pt_file.name)
            if match:
                step = int(match.group(1))
                checkpoints.append((step, pt_file))

        return sorted(checkpoints, key=lambda x: x[0])

    def analyze_checkpoint(self, checkpoint_path: Path, step: int) -> CheckpointMetrics:
        """Extract metrics from a single checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        metrics = CheckpointMetrics(
            step=step,
            path=str(checkpoint_path),
        )

        # Extract validation metrics
        metrics.val_loss = checkpoint.get("val_loss")
        metrics.val_ppl = checkpoint.get("val_ppl")

        # Extract config
        config = checkpoint.get("config", {})
        metrics.dim = config.get("dim", 0)
        metrics.n_layers = config.get("n_layers", 0)
        metrics.n_heads = config.get("n_heads", 0)

        # Load model state dict for detailed analysis
        state_dict = checkpoint.get("model_state_dict", {})

        # Extract gate values
        gate_values = self._extract_gate_values(state_dict)
        if gate_values:
            metrics.gate_values = gate_values
            metrics.gate_mean = sum(gate_values) / len(gate_values)
            metrics.gate_std = self._std(gate_values)
            metrics.gate_min = min(gate_values)
            metrics.gate_max = max(gate_values)

        # Extract memory metrics
        memory_metrics = self._extract_memory_metrics(state_dict)
        metrics.has_memory = memory_metrics["has_memory"]
        metrics.memory_param_norms = memory_metrics["param_norms"]
        metrics.momentum_norms = memory_metrics["momentum_norms"]

        return metrics

    def _extract_gate_values(self, state_dict: dict) -> list[float]:
        """Extract gate values from state dict."""
        gate_values = []

        for key, value in state_dict.items():
            if "memory_gate" in key and value.numel() == 1:
                # Apply sigmoid to get actual gate value
                gate_val = torch.sigmoid(value).item()
                gate_values.append(gate_val)

        return gate_values

    def _extract_memory_metrics(self, state_dict: dict) -> dict:
        """Extract memory-related metrics from state dict."""
        result = {
            "has_memory": False,
            "param_norms": {},
            "momentum_norms": {},
        }

        for key, value in state_dict.items():
            # Check for Atlas memory parameters
            if "atlas_memory" in key or "poly_memory" in key:
                result["has_memory"] = True

                if "momentum_" in key:
                    # Momentum buffer
                    norm = value.norm().item()
                    result["momentum_norms"][key] = norm
                elif "weight" in key or "bias" in key:
                    # Memory parameters
                    norm = value.norm().item()
                    result["param_norms"][key] = norm

        return result

    def _std(self, values: list[float]) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance**0.5

    def analyze_all(self, verbose: bool = True) -> list[CheckpointMetrics]:
        """Analyze all checkpoints in the run directory."""
        checkpoints = self.find_checkpoints()

        if not checkpoints:
            print(f"No checkpoints found in {self.run_dir}")
            return []

        if verbose:
            print(f"Found {len(checkpoints)} checkpoints in {self.run_dir}")
            print("-" * 60)

        self.results = []

        for step, path in checkpoints:
            if verbose:
                print(f"  Analyzing step {step:>6,}...", end=" ")

            try:
                metrics = self.analyze_checkpoint(path, step)
                self.results.append(metrics)

                if verbose:
                    gate_info = (
                        f"gate_mean={metrics.gate_mean:.4f}" if metrics.gate_values else "no gates"
                    )
                    ppl_info = f"val_ppl={metrics.val_ppl:.2f}" if metrics.val_ppl else "no val"
                    print(f"{gate_info}, {ppl_info}")

            except Exception as e:
                if verbose:
                    print(f"ERROR: {e}")

        if verbose:
            print("-" * 60)
            print(f"Analyzed {len(self.results)} checkpoints successfully")

        return self.results

    def save_results(self, output_path: str | Path) -> None:
        """Save analysis results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        data = {
            "run_dir": str(self.run_dir),
            "num_checkpoints": len(self.results),
            "checkpoints": [asdict(m) for m in self.results],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved results to {output_path}")

    def get_metric_series(self, metric: str) -> tuple[list[int], list[float]]:
        """Extract a metric as time series (steps, values)."""
        steps = []
        values = []

        for m in self.results:
            val = getattr(m, metric, None)
            if val is not None:
                steps.append(m.step)
                values.append(val)

        return steps, values

    def get_gate_evolution(self) -> dict[str, tuple[list[int], list[float]]]:
        """Get gate values evolution per layer."""
        if not self.results:
            return {}

        n_layers = len(self.results[0].gate_values) if self.results[0].gate_values else 0
        if n_layers == 0:
            return {}

        evolution = {}
        for layer_idx in range(n_layers):
            steps = []
            values = []
            for m in self.results:
                if len(m.gate_values) > layer_idx:
                    steps.append(m.step)
                    values.append(m.gate_values[layer_idx])
            evolution[f"layer_{layer_idx}"] = (steps, values)

        return evolution


def analyze_checkpoint_series(
    run_dir: str | Path,
    output_dir: str | Path,
    device: str | None = None,
) -> BatchCheckpointAnalyzer:
    """
    Convenience function to analyze a checkpoint series and save results.

    Args:
        run_dir: Directory containing checkpoints
        output_dir: Directory to save analysis results
        device: Device for loading checkpoints

    Returns:
        BatchCheckpointAnalyzer with results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = BatchCheckpointAnalyzer(run_dir, device=device)
    analyzer.analyze_all()
    analyzer.save_results(output_dir / "checkpoint_metrics.json")

    return analyzer
