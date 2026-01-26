# Analysis Tools

Loosely-coupled training analysis utilities designed for reuse across ML projects.

## Design Philosophy

This package is intentionally **decoupled** from the training code. It:
- Parses generic log formats (not tied to specific model architectures)
- Uses only standard data science libraries (pandas, matplotlib)
- Can be copied to other projects with minimal modification
- Provides both CLI and Python API interfaces

## Installation

The analysis tools use these dependencies (add to your project's pyproject.toml):

```toml
[tool.poetry.dependencies]
pandas = "^2.0"
matplotlib = "^3.7"
```

## Quick Start

### Command Line

```bash
# Compare two training runs
python -m analysis.cli compare \
    runs/atlas_54m_ttl/train.log \
    runs/atlas_54m_ablation_nomem/train.log \
    -o output/comparison

# Generate report for single run
python -m analysis.cli report runs/atlas_54m_ttl/train.log -o output/report

# Print quick summary
python -m analysis.cli summary runs/atlas_54m_ttl/train.log
```

### Python API

```python
from analysis.parsers import parse_training_log
from analysis.comparisons import compare_runs, RunComparator
from analysis.visualizations import plot_loss_comparison

# Parse a single log
metrics = parse_training_log("runs/my_run/train.log")
print(f"Final PPL: {metrics.ppl[-1]:.2f}")

# Compare multiple runs
compare_runs(
    {
        "TTL": "runs/atlas_54m_ttl/train.log",
        "Ablation": "runs/atlas_54m_ablation_nomem/train.log",
    },
    output_dir="output/comparison"
)

# Or use the class directly for more control
comparator = RunComparator()
comparator.add_run("TTL", "runs/atlas_54m_ttl/train.log")
comparator.add_run("Ablation", "runs/atlas_54m_ablation_nomem/train.log")
comparator.print_comparison_table()
comparator.generate_comparison("output/comparison")
```

## Package Structure

```
analysis/
├── __init__.py              # Package metadata
├── cli.py                   # Command-line interface
├── README.md                # This file
├── parsers/
│   ├── __init__.py
│   └── log_parser.py        # Generic log parsing
├── visualizations/
│   ├── __init__.py
│   └── training_curves.py   # Plotting utilities
└── comparisons/
    ├── __init__.py
    └── run_comparator.py    # Multi-run comparison
```

## Supported Log Formats

### Atlas-MAG Format (default)

```
[Epoch 1/3] Step 100 | Tokens: 1.6M | LM Loss: 10.4500 | Polar: 2.7497 | PPL: 34546.05 | LR: 8.19e-08 | GradNorm: 5.96 | Gate std: 0.0838 | Tok/s: 20555
```

### Validation Lines

```
[Validation] Loss: 3.3593, PPL: 28.77, Train/Val Gap: 0.83x
```

### Adding Custom Formats

Extend `LogParser` with custom patterns:

```python
from analysis.parsers.log_parser import LogParser
import re

class MyLogParser(LogParser):
    MY_PATTERN = re.compile(r"step=(\d+) loss=([\d.]+)")

    def _parse_line(self, line):
        match = self.MY_PATTERN.search(line)
        if match:
            self.metrics.steps.append(int(match.group(1)))
            self.metrics.loss.append(float(match.group(2)))
            return
        super()._parse_line(line)
```

## Output Files

When running `compare`, the following files are generated:

| File | Description |
|------|-------------|
| `loss_comparison.png` | Training loss curves overlaid |
| `ppl_comparison.png` | Perplexity curves (log scale) |
| `throughput_comparison.png` | Tokens/sec comparison |
| `curves_<run>.png` | Full training curves per run |
| `comparison_report.txt` | Text summary table |
| `comparison_data.json` | Machine-readable metrics |

## Reusing in Other Projects

To use these tools in another project:

1. Copy the `analysis/` directory to your project
2. Add dependencies: `pandas`, `matplotlib`
3. Update import paths if needed
4. (Optional) Add custom log patterns for your format

The tools are designed to work with any training log that includes step numbers and metrics.
