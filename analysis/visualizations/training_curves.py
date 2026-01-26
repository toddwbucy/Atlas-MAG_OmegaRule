"""
Training curve visualization utilities.

Provides publication-quality plots for training metrics comparison.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..parsers.log_parser import TrainingMetrics


# Style configuration for publication-quality plots
STYLE_CONFIG = {
    "figure.figsize": (12, 8),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "lines.linewidth": 2,
    "axes.grid": True,
    "grid.alpha": 0.3,
}

# Color palette for different runs
COLORS = {
    "primary": "#2E86AB",    # Blue
    "secondary": "#A23B72",  # Magenta
    "tertiary": "#F18F01",   # Orange
    "quaternary": "#C73E1D", # Red
    "success": "#3A7D44",    # Green
}


def _apply_style():
    """Apply consistent plot styling."""
    plt.rcParams.update(STYLE_CONFIG)


def _smooth(values: list, window: int = 10) -> np.ndarray:
    """Apply moving average smoothing."""
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_training_curves(
    metrics: TrainingMetrics,
    title: str = "Training Curves",
    output_path: Optional[str | Path] = None,
    smooth_window: int = 10,
) -> plt.Figure:
    """
    Plot comprehensive training curves for a single run.

    Args:
        metrics: Parsed training metrics
        title: Plot title
        output_path: If provided, save figure to this path
        smooth_window: Window size for smoothing

    Returns:
        matplotlib Figure object
    """
    _apply_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Loss curve
    ax = axes[0, 0]
    steps = metrics.steps[smooth_window - 1:] if smooth_window > 1 else metrics.steps
    loss_smooth = _smooth(metrics.loss, smooth_window)
    ax.plot(steps, loss_smooth, color=COLORS["primary"], label="Train Loss")
    if metrics.val_steps:
        ax.scatter(metrics.val_steps, metrics.val_loss, color=COLORS["secondary"],
                   s=50, zorder=5, label="Val Loss", marker="o")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()

    # PPL curve (log scale)
    ax = axes[0, 1]
    ppl_smooth = _smooth(metrics.ppl, smooth_window)
    ax.semilogy(steps, ppl_smooth, color=COLORS["primary"], label="Train PPL")
    if metrics.val_steps:
        ax.scatter(metrics.val_steps, metrics.val_ppl, color=COLORS["secondary"],
                   s=50, zorder=5, label="Val PPL", marker="o")
    ax.set_xlabel("Step")
    ax.set_ylabel("Perplexity (log scale)")
    ax.set_title("Perplexity")
    ax.legend()

    # Learning rate
    ax = axes[1, 0]
    ax.plot(metrics.steps, metrics.lr, color=COLORS["tertiary"])
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    # Gradient norm
    ax = axes[1, 1]
    grad_smooth = _smooth(metrics.grad_norm, smooth_window)
    ax.plot(steps, grad_smooth, color=COLORS["quaternary"])
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    return fig


def plot_loss_comparison(
    runs: dict[str, TrainingMetrics],
    title: str = "Loss Comparison",
    output_path: Optional[str | Path] = None,
    smooth_window: int = 10,
    include_validation: bool = True,
) -> plt.Figure:
    """
    Plot loss comparison across multiple runs.

    Args:
        runs: Dictionary mapping run names to their metrics
        title: Plot title
        output_path: If provided, save figure to this path
        smooth_window: Window size for smoothing
        include_validation: Whether to include validation points

    Returns:
        matplotlib Figure object
    """
    _apply_style()

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = list(COLORS.values())

    for i, (name, metrics) in enumerate(runs.items()):
        color = colors[i % len(colors)]
        steps = metrics.steps[smooth_window - 1:] if smooth_window > 1 else metrics.steps
        loss_smooth = _smooth(metrics.loss, smooth_window)

        ax.plot(steps, loss_smooth, color=color, label=f"{name} (train)", alpha=0.8)

        if include_validation and metrics.val_steps:
            ax.scatter(metrics.val_steps, metrics.val_loss, color=color,
                       s=60, zorder=5, marker="o", edgecolors="white", linewidths=1)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    return fig


def plot_ppl_comparison(
    runs: dict[str, TrainingMetrics],
    title: str = "Perplexity Comparison",
    output_path: Optional[str | Path] = None,
    smooth_window: int = 10,
    log_scale: bool = True,
    include_validation: bool = True,
) -> plt.Figure:
    """
    Plot perplexity comparison across multiple runs.

    Args:
        runs: Dictionary mapping run names to their metrics
        title: Plot title
        output_path: If provided, save figure to this path
        smooth_window: Window size for smoothing
        log_scale: Use log scale for y-axis
        include_validation: Whether to include validation points

    Returns:
        matplotlib Figure object
    """
    _apply_style()

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = list(COLORS.values())

    for i, (name, metrics) in enumerate(runs.items()):
        color = colors[i % len(colors)]
        steps = metrics.steps[smooth_window - 1:] if smooth_window > 1 else metrics.steps
        ppl_smooth = _smooth(metrics.ppl, smooth_window)

        if log_scale:
            ax.semilogy(steps, ppl_smooth, color=color, label=f"{name} (train)", alpha=0.8)
        else:
            ax.plot(steps, ppl_smooth, color=color, label=f"{name} (train)", alpha=0.8)

        if include_validation and metrics.val_steps:
            ax.scatter(metrics.val_steps, metrics.val_ppl, color=color,
                       s=60, zorder=5, marker="o", edgecolors="white", linewidths=1)

    ax.set_xlabel("Step")
    ax.set_ylabel("Perplexity" + (" (log scale)" if log_scale else ""))
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    return fig


def plot_throughput(
    runs: dict[str, TrainingMetrics],
    title: str = "Training Throughput",
    output_path: Optional[str | Path] = None,
    smooth_window: int = 50,
) -> plt.Figure:
    """
    Plot throughput comparison (tokens/sec) across runs.

    Args:
        runs: Dictionary mapping run names to their metrics
        title: Plot title
        output_path: If provided, save figure to this path
        smooth_window: Window size for smoothing

    Returns:
        matplotlib Figure object
    """
    _apply_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = list(COLORS.values())

    for i, (name, metrics) in enumerate(runs.items()):
        color = colors[i % len(colors)]
        steps = metrics.steps[smooth_window - 1:] if smooth_window > 1 else metrics.steps
        throughput_smooth = _smooth(metrics.tokens_per_sec, smooth_window)

        ax.plot(steps, throughput_smooth, color=color, label=name, alpha=0.8)

        # Add mean line
        mean_throughput = np.mean(metrics.tokens_per_sec)
        ax.axhline(y=mean_throughput, color=color, linestyle="--", alpha=0.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Tokens/sec")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    return fig
