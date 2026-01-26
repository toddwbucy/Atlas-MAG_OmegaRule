"""
Memory evolution tracking and visualization.

Tracks how memory parameters and momentum buffers evolve over training.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .batch_analyzer import BatchCheckpointAnalyzer

# Style configuration
STYLE_CONFIG = {
    "figure.figsize": (14, 10),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
}

# Colors for different metrics
MEMORY_COLORS = {
    "param_norm": "#2E86AB",  # Blue
    "momentum_norm": "#A23B72",  # Magenta
    "total": "#F18F01",  # Orange
}


class MemoryEvolutionTracker:
    """
    Track and visualize memory parameter evolution over training.

    Tracks:
    - Memory parameter norms (weights, biases)
    - Momentum buffer norms (for TTL models)
    - Memory state growth/stability
    """

    def __init__(self, analyzer: BatchCheckpointAnalyzer):
        self.analyzer = analyzer
        self.results = analyzer.results

    def get_summary_stats(self) -> dict:
        """Get summary statistics of memory evolution."""
        if not self.results:
            return {}

        # Find checkpoints with memory
        memory_checkpoints = [r for r in self.results if r.has_memory]
        if not memory_checkpoints:
            return {"has_memory": False}

        first = memory_checkpoints[0]
        last = memory_checkpoints[-1]

        # Calculate total parameter norms
        first_param_total = (
            sum(first.memory_param_norms.values()) if first.memory_param_norms else 0
        )
        last_param_total = sum(last.memory_param_norms.values()) if last.memory_param_norms else 0

        # Calculate total momentum norms
        first_momentum_total = sum(first.momentum_norms.values()) if first.momentum_norms else 0
        last_momentum_total = sum(last.momentum_norms.values()) if last.momentum_norms else 0

        return {
            "has_memory": True,
            "num_checkpoints": len(memory_checkpoints),
            "initial_step": first.step,
            "final_step": last.step,
            "initial_param_norm": first_param_total,
            "final_param_norm": last_param_total,
            "param_norm_change": last_param_total - first_param_total,
            "param_norm_growth_pct": (
                (last_param_total / first_param_total - 1) * 100 if first_param_total > 0 else 0
            ),
            "initial_momentum_norm": first_momentum_total,
            "final_momentum_norm": last_momentum_total,
            "momentum_norm_change": last_momentum_total - first_momentum_total,
            "num_param_keys": len(last.memory_param_norms),
            "num_momentum_keys": len(last.momentum_norms),
        }

    def print_summary(self) -> None:
        """Print memory evolution summary."""
        stats = self.get_summary_stats()

        print("=" * 60)
        print("MEMORY EVOLUTION SUMMARY")
        print("=" * 60)

        if not stats.get("has_memory", False):
            print("No memory data available (model may be attention-only)")
            return

        print(f"\nAnalyzed {stats['num_checkpoints']} checkpoints with memory")
        print(f"Steps: {stats['initial_step']:,} -> {stats['final_step']:,}")

        print("\nParameter Norms:")
        print(f"  Initial: {stats['initial_param_norm']:.4f}")
        print(f"  Final:   {stats['final_param_norm']:.4f}")
        print(
            f"  Change:  {stats['param_norm_change']:+.4f} ({stats['param_norm_growth_pct']:+.1f}%)"
        )

        if stats["num_momentum_keys"] > 0:
            print("\nMomentum Norms (TTL):")
            print(f"  Initial: {stats['initial_momentum_norm']:.4f}")
            print(f"  Final:   {stats['final_momentum_norm']:.4f}")
            print(f"  Change:  {stats['momentum_norm_change']:+.4f}")

        print("\nTracked Parameters:")
        print(f"  Memory params: {stats['num_param_keys']} tensors")
        print(f"  Momentum buffers: {stats['num_momentum_keys']} tensors")

        # Per-key breakdown for final checkpoint
        last = self.results[-1]
        if last.memory_param_norms:
            print("\nFinal Memory Parameter Norms:")
            for key, norm in sorted(last.memory_param_norms.items()):
                # Shorten key for display (handle short keys gracefully)
                parts = key.split(".")
                short_key = ".".join(parts[-2:]) if len(parts) >= 2 else key
                print(f"  {short_key}: {norm:.4f}")

    def get_norm_series(self) -> dict[str, tuple[list[int], list[float]]]:
        """
        Get time series of norm values.

        Returns dict with keys:
        - 'total_param_norm': (steps, values)
        - 'total_momentum_norm': (steps, values)
        - Individual parameter keys: (steps, values)
        """
        series = {
            "total_param_norm": ([], []),
            "total_momentum_norm": ([], []),
        }

        for m in self.results:
            if not m.has_memory:
                continue

            step = m.step

            # Total param norm
            total_param = sum(m.memory_param_norms.values()) if m.memory_param_norms else 0
            series["total_param_norm"][0].append(step)
            series["total_param_norm"][1].append(total_param)

            # Total momentum norm
            total_momentum = sum(m.momentum_norms.values()) if m.momentum_norms else 0
            series["total_momentum_norm"][0].append(step)
            series["total_momentum_norm"][1].append(total_momentum)

            # Individual param norms
            for key, norm in m.memory_param_norms.items():
                if key not in series:
                    series[key] = ([], [])
                series[key][0].append(step)
                series[key][1].append(norm)

            # Individual momentum norms
            for key, norm in m.momentum_norms.items():
                if key not in series:
                    series[key] = ([], [])
                series[key][0].append(step)
                series[key][1].append(norm)

        return series


def plot_memory_evolution(
    analyzer: BatchCheckpointAnalyzer,
    output_path: Optional[str | Path] = None,
    title: str = "Memory Evolution Over Training",
) -> plt.Figure:
    """
    Plot memory parameter evolution over training.

    Creates a multi-panel figure showing:
    1. Total parameter norm over time
    2. Total momentum norm over time (if TTL)
    3. Per-layer parameter norms
    4. Growth rate analysis

    Args:
        analyzer: BatchCheckpointAnalyzer with results
        output_path: If provided, save figure to this path
        title: Plot title

    Returns:
        matplotlib Figure (or None if no memory data)
    """
    plt.rcParams.update(STYLE_CONFIG)

    results = analyzer.results
    memory_checkpoints = [r for r in results if r.has_memory]

    if not memory_checkpoints:
        print("No memory data available for plotting (attention-only model?)")
        return None

    tracker = MemoryEvolutionTracker(analyzer)
    series = tracker.get_norm_series()

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # Panel 1: Total parameter norm
    ax1 = fig.add_subplot(gs[0, 0])

    steps, values = series["total_param_norm"]
    ax1.plot(
        steps, values, color=MEMORY_COLORS["param_norm"], linewidth=2, label="Total Param Norm"
    )

    ax1.set_xlabel("Step")
    ax1.set_ylabel("L2 Norm")
    ax1.set_title("Total Memory Parameter Norm")
    ax1.legend()

    # Panel 2: Total momentum norm (if present)
    ax2 = fig.add_subplot(gs[0, 1])

    steps, values = series["total_momentum_norm"]
    if any(v > 0 for v in values):
        ax2.plot(
            steps,
            values,
            color=MEMORY_COLORS["momentum_norm"],
            linewidth=2,
            label="Total Momentum Norm",
        )
        ax2.set_title("Total Momentum Buffer Norm (TTL)")
        ax2.legend()  # Only show legend when plot is drawn
    else:
        ax2.text(
            0.5,
            0.5,
            "No momentum buffers\n(non-TTL model)",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
            transform=ax2.transAxes,
        )
        ax2.set_title("Momentum Buffers")

    ax2.set_xlabel("Step")
    ax2.set_ylabel("L2 Norm")

    # Panel 3: Per-layer breakdown (stacked area or lines)
    ax3 = fig.add_subplot(gs[1, 0])

    # Get individual parameter keys (filter out totals and momentum)
    param_keys = [
        k
        for k in series.keys()
        if k not in ("total_param_norm", "total_momentum_norm") and "momentum" not in k.lower()
    ]

    if param_keys:
        colors = plt.cm.viridis(np.linspace(0, 1, len(param_keys)))

        for i, key in enumerate(sorted(param_keys)[:12]):  # Limit to 12 for readability
            steps, values = series[key]
            # Shorten key for legend
            short_key = ".".join(key.split(".")[-2:])
            ax3.plot(steps, values, color=colors[i], alpha=0.7, label=short_key)

        ax3.set_xlabel("Step")
        ax3.set_ylabel("L2 Norm")
        ax3.set_title("Per-Component Parameter Norms")
        ax3.legend(loc="upper left", ncol=2, fontsize=7)
    else:
        ax3.text(
            0.5,
            0.5,
            "No per-component data",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
            transform=ax3.transAxes,
        )

    # Panel 4: Growth rate / stability analysis
    ax4 = fig.add_subplot(gs[1, 1])

    steps, values = series["total_param_norm"]
    if len(values) > 1:
        # Calculate growth rate (derivative)
        growth_rates = []
        growth_steps = []
        for i in range(1, len(values)):
            step_diff = steps[i] - steps[i - 1]
            if step_diff > 0:
                rate = (values[i] - values[i - 1]) / step_diff * 1000  # per 1000 steps
                growth_rates.append(rate)
                growth_steps.append(steps[i])

        if growth_rates:
            ax4.plot(growth_steps, growth_rates, color=MEMORY_COLORS["total"], linewidth=1.5)
            ax4.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

            # Add trend line
            if len(growth_steps) > 2:
                z = np.polyfit(growth_steps, growth_rates, 1)
                p = np.poly1d(z)
                ax4.plot(growth_steps, p(growth_steps), "r--", alpha=0.5, label="Trend")

            ax4.set_xlabel("Step")
            ax4.set_ylabel("Norm Change Rate (per 1K steps)")
            ax4.set_title("Parameter Growth Rate")

            # Add annotation about stability
            final_rate = growth_rates[-1] if growth_rates else 0
            stability = (
                "Stable"
                if abs(final_rate) < 0.01
                else ("Growing" if final_rate > 0 else "Shrinking")
            )
            ax4.text(
                0.95,
                0.95,
                f"Final trend: {stability}",
                ha="right",
                va="top",
                fontsize=10,
                transform=ax4.transAxes,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved memory evolution plot to {output_path}")

    return fig
