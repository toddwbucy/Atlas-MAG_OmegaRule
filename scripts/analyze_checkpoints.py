#!/usr/bin/env python3
"""
Analyze checkpoint evolution for Atlas-MAG TTL training run.

Extracts gate dynamics, memory norms, and weight statistics across all checkpoints.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.checkpoints import (
    BatchCheckpointAnalyzer,
    plot_gate_evolution,
    plot_memory_evolution,
)


def main():
    checkpoint_dir = Path("runs/atlas_54m_ttl")
    output_dir = Path("reports/atlas_ttl_analysis/figures")
    data_dir = Path("reports/atlas_ttl_analysis/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CHECKPOINT EVOLUTION ANALYSIS")
    print("=" * 60)
    print(f"\nCheckpoint directory: {checkpoint_dir}")
    print(f"Output directory: {output_dir}\n")

    # Initialize analyzer with run directory
    analyzer = BatchCheckpointAnalyzer(checkpoint_dir)

    # Analyze all checkpoints
    results = analyzer.analyze_all(verbose=True)

    # Save results to JSON
    analyzer.save_results(data_dir / "checkpoint_metrics.json")

    # Print gate evolution summary
    print("\n" + "=" * 60)
    print("GATE EVOLUTION SUMMARY")
    print("=" * 60)

    if results and results[0].gate_values:
        first = results[0]
        last = results[-1]

        print(f"\nInitial state (step {first.step:,}):")
        print(f"  Gate mean: {first.gate_mean:.4f}")
        print(f"  Gate std:  {first.gate_std:.4f}")
        print(f"  Gate range: [{first.gate_min:.4f}, {first.gate_max:.4f}]")

        print(f"\nFinal state (step {last.step:,}):")
        print(f"  Gate mean: {last.gate_mean:.4f}")
        print(f"  Gate std:  {last.gate_std:.4f}")
        print(f"  Gate range: [{last.gate_min:.4f}, {last.gate_max:.4f}]")

        print("\nEvolution:")
        print(f"  Mean change: {last.gate_mean - first.gate_mean:+.4f}")
        print(f"  Std change:  {last.gate_std - first.gate_std:+.4f}")

        # Per-layer final gate values
        print("\nFinal per-layer gate values:")
        for i, val in enumerate(last.gate_values):
            direction = "MEMORY" if val > 0.5 else "ATTENTION"
            bar_len = int(val * 30)
            bar = "#" * bar_len + "." * (30 - bar_len)
            print(f"  Layer {i}: {val:.4f} [{bar}] -> {direction}")

    # Generate plots
    print("\n" + "-" * 60)
    print("Generating visualization plots...")

    try:
        fig = plot_gate_evolution(analyzer, output_path=output_dir / "gate_evolution.png")
        if fig:
            print("  Saved gate_evolution.png")
        else:
            print("  Skipped gate_evolution.png (no gate data)")
    except Exception as e:
        print(f"  Error generating gate_evolution.png: {e}")

    try:
        fig = plot_memory_evolution(analyzer, output_path=output_dir / "memory_evolution.png")
        if fig:
            print("  Saved memory_evolution.png")
        else:
            print("  Skipped memory_evolution.png (no memory data)")
    except Exception as e:
        print(f"  Error generating memory_evolution.png: {e}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
