"""
Gate evolution tracking and visualization.

Tracks how attention/memory routing gates evolve over training.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .batch_analyzer import BatchCheckpointAnalyzer, CheckpointMetrics


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

# Color map for layers
LAYER_COLORS = plt.cm.viridis(np.linspace(0, 1, 12))


class GateEvolutionTracker:
    """
    Track and visualize gate evolution over training.

    Gates control the routing between attention and memory paths.
    Values close to 0 = attention-dominant, close to 1 = memory-dominant.
    """

    def __init__(self, analyzer: BatchCheckpointAnalyzer):
        self.analyzer = analyzer
        self.results = analyzer.results

    def get_summary_stats(self) -> dict:
        """Get summary statistics of gate evolution."""
        if not self.results:
            return {}

        steps, means = self.analyzer.get_metric_series("gate_mean")
        _, stds = self.analyzer.get_metric_series("gate_std")

        return {
            "initial_mean": means[0] if means else None,
            "final_mean": means[-1] if means else None,
            "initial_std": stds[0] if stds else None,
            "final_std": stds[-1] if stds else None,
            "mean_change": (means[-1] - means[0]) if means else None,
            "std_change": (stds[-1] - stds[0]) if stds else None,
            "polarization_trend": "increasing" if stds and stds[-1] > stds[0] else "decreasing",
        }

    def print_summary(self) -> None:
        """Print gate evolution summary."""
        stats = self.get_summary_stats()

        print("=" * 60)
        print("GATE EVOLUTION SUMMARY")
        print("=" * 60)

        if not stats:
            print("No gate data available")
            return

        print(f"\nInitial State (step {self.results[0].step:,}):")
        print(f"  Mean: {stats['initial_mean']:.4f}")
        print(f"  Std:  {stats['initial_std']:.4f}")

        print(f"\nFinal State (step {self.results[-1].step:,}):")
        print(f"  Mean: {stats['final_mean']:.4f}")
        print(f"  Std:  {stats['final_std']:.4f}")

        print(f"\nEvolution:")
        print(f"  Mean change: {stats['mean_change']:+.4f}")
        print(f"  Std change:  {stats['std_change']:+.4f}")
        print(f"  Polarization: {stats['polarization_trend']}")

        # Per-layer final values
        if self.results[-1].gate_values:
            print(f"\nFinal Per-Layer Gate Values:")
            for i, val in enumerate(self.results[-1].gate_values):
                direction = "MEMORY" if val > 0.5 else "ATTENTION"
                bar_len = int(val * 30)
                bar = "#" * bar_len + "." * (30 - bar_len)
                print(f"  Layer {i}: {val:.4f} [{bar}] -> {direction}")


def plot_gate_evolution(
    analyzer: BatchCheckpointAnalyzer,
    output_path: Optional[str | Path] = None,
    title: str = "Gate Evolution Over Training",
) -> plt.Figure:
    """
    Plot gate evolution over training.

    Creates a multi-panel figure showing:
    1. Per-layer gate values over time
    2. Mean and std of gates over time
    3. Final gate distribution
    4. Gate heatmap over time

    Args:
        analyzer: BatchCheckpointAnalyzer with results
        output_path: If provided, save figure to this path
        title: Plot title

    Returns:
        matplotlib Figure
    """
    plt.rcParams.update(STYLE_CONFIG)

    results = analyzer.results
    if not results or not results[0].gate_values:
        print("No gate data available for plotting")
        return None

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # Panel 1: Per-layer gate evolution
    ax1 = fig.add_subplot(gs[0, 0])
    gate_evolution = analyzer.get_gate_evolution()

    for i, (layer_name, (steps, values)) in enumerate(gate_evolution.items()):
        color = LAYER_COLORS[i % len(LAYER_COLORS)]
        ax1.plot(steps, values, color=color, label=layer_name, alpha=0.8)

    ax1.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Balanced")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Gate Value")
    ax1.set_title("Per-Layer Gate Evolution")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper right", ncol=2, fontsize=8)

    # Panel 2: Mean and Std evolution
    ax2 = fig.add_subplot(gs[0, 1])

    steps, means = analyzer.get_metric_series("gate_mean")
    _, stds = analyzer.get_metric_series("gate_std")

    ax2.plot(steps, means, color="blue", label="Mean", linewidth=2)
    ax2.fill_between(
        steps,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.3,
        color="blue",
        label="±1 Std",
    )
    ax2.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Balanced")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Gate Value")
    ax2.set_title("Gate Mean ± Std Over Training")
    ax2.set_ylim(0, 1)
    ax2.legend()

    # Panel 3: Final gate distribution (bar chart)
    ax3 = fig.add_subplot(gs[1, 0])

    final_gates = results[-1].gate_values
    n_layers = len(final_gates)
    x = np.arange(n_layers)

    colors = ["#2E86AB" if g < 0.5 else "#A23B72" for g in final_gates]
    bars = ax3.bar(x, final_gates, color=colors, edgecolor="black", linewidth=0.5)

    ax3.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Balanced")
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("Gate Value")
    ax3.set_title(f"Final Gate Values (Step {results[-1].step:,})")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"L{i}" for i in range(n_layers)])
    ax3.set_ylim(0, 1)

    # Add value labels on bars
    for bar, val in zip(bars, final_gates):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2E86AB", label="Attention-dominant (<0.5)"),
        Patch(facecolor="#A23B72", label="Memory-dominant (>0.5)"),
    ]
    ax3.legend(handles=legend_elements, loc="upper right")

    # Panel 4: Heatmap of gate evolution
    ax4 = fig.add_subplot(gs[1, 1])

    # Build heatmap data
    steps_list = [r.step for r in results if r.gate_values]
    gate_matrix = np.array([r.gate_values for r in results if r.gate_values])

    if gate_matrix.size > 0:
        im = ax4.imshow(
            gate_matrix.T,
            aspect="auto",
            cmap="RdYlBu_r",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )

        # Set axis labels
        n_ticks = min(10, len(steps_list))
        tick_indices = np.linspace(0, len(steps_list) - 1, n_ticks, dtype=int)
        ax4.set_xticks(tick_indices)
        ax4.set_xticklabels([f"{steps_list[i]//1000}K" for i in tick_indices])
        ax4.set_yticks(range(gate_matrix.shape[1]))
        ax4.set_yticklabels([f"L{i}" for i in range(gate_matrix.shape[1])])

        ax4.set_xlabel("Step")
        ax4.set_ylabel("Layer")
        ax4.set_title("Gate Heatmap (Blue=Attention, Red=Memory)")

        # Colorbar
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label("Gate Value")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved gate evolution plot to {output_path}")

    return fig
