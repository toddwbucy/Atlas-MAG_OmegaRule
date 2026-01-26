#!/usr/bin/env python3
"""
Analysis CLI - Command-line interface for training analysis tools.

Usage:
    python -m analysis.cli compare runs/run1/train.log runs/run2/train.log -o output/
    python -m analysis.cli report runs/run1/train.log -o output/
    python -m analysis.cli summary runs/run1/train.log
"""

import argparse
import sys
from pathlib import Path


def cmd_compare(args):
    """Compare multiple training runs."""
    from .comparisons import compare_runs

    # Build runs dictionary from positional args
    runs = {}
    for i, log_path in enumerate(args.logs):
        # Use directory name as run name if available
        path = Path(log_path)
        if path.parent.name != ".":
            name = path.parent.name
        else:
            name = f"Run {i + 1}"

        # Handle duplicates
        base_name = name
        counter = 1
        while name in runs:
            name = f"{base_name}_{counter}"
            counter += 1

        runs[name] = log_path

    compare_runs(runs, args.output, smooth_window=args.smooth)


def cmd_report(args):
    """Generate report for a single run."""
    from .comparisons import generate_report

    generate_report(args.log, args.output)


def cmd_summary(args):
    """Print summary statistics for a run."""
    from .parsers import parse_training_log

    metrics = parse_training_log(args.log)

    print("=" * 60)
    print(f"TRAINING SUMMARY: {args.log}")
    print("=" * 60)

    if metrics.config:
        print(f"\nConfig: {metrics.config}")

    print(f"\nTraining Progress:")
    print(f"  Total steps: {metrics.steps[-1] if metrics.steps else 0:,}")
    print(f"  Total tokens: {metrics.tokens[-1] if metrics.tokens else 0:.1f}M")

    print(f"\nFinal Metrics:")
    print(f"  Train Loss: {metrics.loss[-1] if metrics.loss else 'N/A':.4f}")
    print(f"  Train PPL:  {metrics.ppl[-1] if metrics.ppl else 'N/A':.2f}")

    if metrics.val_loss:
        print(f"\nValidation:")
        print(f"  Best Val Loss: {min(metrics.val_loss):.4f}")
        print(f"  Best Val PPL:  {min(metrics.val_ppl):.2f}")
        print(f"  Validation points: {len(metrics.val_loss)}")

    if metrics.tokens_per_sec:
        mean_throughput = sum(metrics.tokens_per_sec) / len(metrics.tokens_per_sec)
        print(f"\nThroughput:")
        print(f"  Mean: {mean_throughput:,.0f} tok/s")
        print(f"  Min:  {min(metrics.tokens_per_sec):,} tok/s")
        print(f"  Max:  {max(metrics.tokens_per_sec):,} tok/s")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Training Analysis Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two runs
  python -m analysis.cli compare runs/ttl/train.log runs/ablation/train.log -o comparison/

  # Generate report for single run
  python -m analysis.cli report runs/ttl/train.log -o report/

  # Print quick summary
  python -m analysis.cli summary runs/ttl/train.log
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare multiple training runs",
    )
    compare_parser.add_argument(
        "logs",
        nargs="+",
        help="Log files to compare",
    )
    compare_parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory for comparison",
    )
    compare_parser.add_argument(
        "--smooth",
        type=int,
        default=10,
        help="Smoothing window size (default: 10)",
    )
    compare_parser.set_defaults(func=cmd_compare)

    # Report command
    report_parser = subparsers.add_parser(
        "report",
        help="Generate report for a single run",
    )
    report_parser.add_argument(
        "log",
        help="Log file to analyze",
    )
    report_parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory for report",
    )
    report_parser.set_defaults(func=cmd_report)

    # Summary command
    summary_parser = subparsers.add_parser(
        "summary",
        help="Print summary statistics",
    )
    summary_parser.add_argument(
        "log",
        help="Log file to summarize",
    )
    summary_parser.set_defaults(func=cmd_summary)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
