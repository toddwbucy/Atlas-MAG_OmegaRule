"""
Run comparison utilities.

Provides tools for comparing multiple training runs and generating reports.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..parsers.log_parser import TrainingMetrics, parse_training_log
from ..visualizations.training_curves import (
    plot_loss_comparison,
    plot_ppl_comparison,
    plot_throughput,
    plot_training_curves,
)


@dataclass
class RunSummary:
    """Summary statistics for a training run."""

    name: str
    config: dict
    total_steps: int
    total_tokens_m: float
    final_train_loss: float
    final_train_ppl: float
    best_val_loss: Optional[float]
    best_val_ppl: Optional[float]
    mean_throughput: float
    mean_grad_norm: float


class RunComparator:
    """
    Compare multiple training runs.

    Usage:
        comparator = RunComparator()
        comparator.add_run("TTL", "runs/atlas_54m_ttl/train.log")
        comparator.add_run("Ablation", "runs/atlas_54m_ablation_nomem/train.log")
        comparator.generate_comparison("output/comparison")
    """

    def __init__(self):
        self.runs: dict[str, TrainingMetrics] = {}

    def add_run(self, name: str, log_path: str | Path) -> None:
        """Add a run to the comparison."""
        metrics = parse_training_log(log_path)
        self.runs[name] = metrics
        print(
            f"Loaded '{name}': {len(metrics.steps)} steps, "
            f"{metrics.steps[-1] if metrics.steps else 0} max step"
        )

    def get_summary(self, name: str) -> RunSummary:
        """Get summary statistics for a run."""
        if name not in self.runs:
            raise KeyError(f"Run '{name}' not found")

        metrics = self.runs[name]

        # Filter None values from lists that may contain them for alignment
        val_losses = [v for v in metrics.val_loss if v is not None]
        val_ppls = [v for v in metrics.val_ppl if v is not None]
        throughput_vals = [v for v in metrics.tokens_per_sec if v is not None]
        grad_vals = [v for v in metrics.grad_norm if v is not None]

        return RunSummary(
            name=name,
            config=metrics.config,
            total_steps=metrics.steps[-1] if metrics.steps else 0,
            total_tokens_m=metrics.tokens[-1] if metrics.tokens else 0,
            final_train_loss=metrics.loss[-1] if metrics.loss else float("inf"),
            final_train_ppl=metrics.ppl[-1] if metrics.ppl else float("inf"),
            best_val_loss=min(val_losses) if val_losses else None,
            best_val_ppl=min(val_ppls) if val_ppls else None,
            mean_throughput=sum(throughput_vals) / len(throughput_vals) if throughput_vals else 0,
            mean_grad_norm=sum(grad_vals) / len(grad_vals) if grad_vals else 0,
        )

    def print_comparison_table(self) -> str:
        """Print a comparison table of all runs."""
        if not self.runs:
            return "No runs loaded."

        lines = []
        lines.append("=" * 80)
        lines.append("RUN COMPARISON SUMMARY")
        lines.append("=" * 80)

        # Header
        header = f"{'Metric':<25}"
        for name in self.runs.keys():
            header += f"{name:>20}"
        lines.append(header)
        lines.append("-" * 80)

        # Get summaries
        summaries = {name: self.get_summary(name) for name in self.runs}

        # Rows
        metrics_to_show = [
            (
                "Config (dim/layers/heads)",
                lambda s: f"{s.config.get('dim', '?')}/{s.config.get('layers', '?')}/{s.config.get('heads', '?')}",
            ),
            ("Mode", lambda s: s.config.get("mode", "N/A")),
            ("Total Steps", lambda s: f"{s.total_steps:,}"),
            ("Total Tokens (M)", lambda s: f"{s.total_tokens_m:.1f}"),
            ("Final Train Loss", lambda s: f"{s.final_train_loss:.4f}"),
            ("Final Train PPL", lambda s: f"{s.final_train_ppl:.2f}"),
            (
                "Best Val Loss",
                lambda s: f"{s.best_val_loss:.4f}" if s.best_val_loss is not None else "N/A",
            ),
            (
                "Best Val PPL",
                lambda s: f"{s.best_val_ppl:.2f}" if s.best_val_ppl is not None else "N/A",
            ),
            ("Mean Throughput (tok/s)", lambda s: f"{s.mean_throughput:,.0f}"),
            ("Mean Grad Norm", lambda s: f"{s.mean_grad_norm:.2f}"),
        ]

        for metric_name, getter in metrics_to_show:
            row = f"{metric_name:<25}"
            for name in self.runs.keys():
                value = getter(summaries[name])
                row += f"{value:>20}"
            lines.append(row)

        lines.append("=" * 80)

        # Highlight winner for key metrics
        if len(summaries) > 1:
            lines.append("\nKEY FINDINGS:")

            # Best validation PPL
            valid_ppl = [
                (name, s.best_val_ppl)
                for name, s in summaries.items()
                if s.best_val_ppl is not None
            ]
            if valid_ppl:
                best_name, best_ppl = min(valid_ppl, key=lambda x: x[1])
                lines.append(f"  - Best Val PPL: {best_name} ({best_ppl:.2f})")

            # Throughput comparison
            throughputs = [(name, s.mean_throughput) for name, s in summaries.items()]
            fastest_name, fastest_throughput = max(throughputs, key=lambda x: x[1])
            slowest_name, slowest_throughput = min(throughputs, key=lambda x: x[1])
            if fastest_throughput > 0 and slowest_throughput > 0:
                ratio = fastest_throughput / slowest_throughput
                lines.append(
                    f"  - Throughput: {fastest_name} is {ratio:.2f}x faster than {slowest_name}"
                )

        output = "\n".join(lines)
        print(output)
        return output

    def generate_comparison(
        self,
        output_dir: str | Path,
        smooth_window: int = 10,
    ) -> None:
        """
        Generate full comparison with plots and report.

        Args:
            output_dir: Directory to save outputs
            smooth_window: Window size for smoothing plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating comparison in {output_dir}/")
        print("-" * 50)

        # Generate plots
        print("Generating loss comparison plot...")
        plot_loss_comparison(
            self.runs,
            title="Training Loss Comparison",
            output_path=output_dir / "loss_comparison.png",
            smooth_window=smooth_window,
        )

        print("Generating PPL comparison plot...")
        plot_ppl_comparison(
            self.runs,
            title="Perplexity Comparison",
            output_path=output_dir / "ppl_comparison.png",
            smooth_window=smooth_window,
        )

        print("Generating throughput comparison plot...")
        plot_throughput(
            self.runs,
            title="Training Throughput Comparison",
            output_path=output_dir / "throughput_comparison.png",
            smooth_window=50,
        )

        # Generate individual run plots
        for name, metrics in self.runs.items():
            safe_name = name.lower().replace(" ", "_").replace("/", "_")
            print(f"Generating training curves for '{name}'...")
            plot_training_curves(
                metrics,
                title=f"Training Curves: {name}",
                output_path=output_dir / f"curves_{safe_name}.png",
                smooth_window=smooth_window,
            )

        # Generate text report
        print("Generating summary report...")
        report = self.print_comparison_table()
        with open(output_dir / "comparison_report.txt", "w") as f:
            f.write(report)

        # Generate JSON summary
        summaries = {name: self.get_summary(name) for name in self.runs}
        json_data = {}
        for name, summary in summaries.items():
            json_data[name] = {
                "config": summary.config,
                "total_steps": summary.total_steps,
                "total_tokens_m": summary.total_tokens_m,
                "final_train_loss": summary.final_train_loss,
                "final_train_ppl": summary.final_train_ppl,
                "best_val_loss": summary.best_val_loss,
                "best_val_ppl": summary.best_val_ppl,
                "mean_throughput": summary.mean_throughput,
                "mean_grad_norm": summary.mean_grad_norm,
            }

        with open(output_dir / "comparison_data.json", "w") as f:
            json.dump(json_data, f, indent=2)

        print("-" * 50)
        print(f"Comparison complete! Outputs saved to {output_dir}/")
        print("  - loss_comparison.png")
        print("  - ppl_comparison.png")
        print("  - throughput_comparison.png")
        print("  - curves_*.png (per run)")
        print("  - comparison_report.txt")
        print("  - comparison_data.json")


def compare_runs(
    runs: dict[str, str | Path],
    output_dir: str | Path,
    smooth_window: int = 10,
) -> RunComparator:
    """
    Convenience function to compare multiple runs.

    Args:
        runs: Dictionary mapping run names to log file paths
        output_dir: Directory to save comparison outputs
        smooth_window: Window size for smoothing

    Returns:
        RunComparator instance with loaded runs

    Example:
        compare_runs(
            {
                "TTL": "runs/atlas_54m_ttl/train.log",
                "Ablation": "runs/atlas_54m_ablation_nomem/train.log",
            },
            "output/comparison"
        )
    """
    comparator = RunComparator()

    for name, log_path in runs.items():
        comparator.add_run(name, log_path)

    comparator.generate_comparison(output_dir, smooth_window)

    return comparator


def generate_report(log_path: str | Path, output_dir: str | Path) -> None:
    """
    Generate a report for a single run.

    Args:
        log_path: Path to training log
        output_dir: Directory to save outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = parse_training_log(log_path)

    plot_training_curves(
        metrics,
        title="Training Curves",
        output_path=output_dir / "training_curves.png",
    )

    print(f"Report saved to {output_dir}/")
