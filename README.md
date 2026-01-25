# Atlas-MAG with Omega Rule

A hybrid sequence model combining **Sliding Window Attention (SWA)** with **Deep Neural Long-Term Memory**, implementing the **Omega Rule** from the Atlas paper for context-aware memorization.

## Overview

Atlas-MAG is a proof-of-concept implementation exploring a novel approach to sequence modeling:

```
Input → [Persistent Memory Tokens] → ┬→ [Sliding Window Attention] ─┐
                                     │                              │
                                     └→ [Neural Memory (Omega)] ────┤
                                              ↓                      │
                                         [Gate σ()]                  │
                                              ↓                      │
                                     output = SWA × σ(Memory) ←─────┘
```

### Key Features

- **Omega Rule Memory**: Sliding context window with exponential decay for efficient long-range dependencies
- **Input-Dependent Gates**: Per-position γ gates that modulate memory decay based on content relevance
- **Gate Polarization**: Training objective that pushes gates toward 0 or 1 for interpretable memory/attention routing
- **Persistent Memory**: Precomputed M_persistent provides stable long-term memory accessible from first token

### Architecture Highlights

| Component | Description |
|-----------|-------------|
| Q-K Memory Projection | Projects queries through accumulated key space with proper normalization |
| Causal Context Window | Only attends to past positions within configurable window (default: 256) |
| Exponential Decay | Recent tokens weighted more heavily (decay_base=0.95) |
| RMSNorm + SwiGLU | Modern transformer building blocks |
| Rotary Embeddings | Position encoding for attention |

## Installation

```bash
# Clone the repository
git clone https://github.com/toddwbucy/Atlas-MAG_OmegaRule.git
cd Atlas-MAG_OmegaRule

# Install dependencies with Poetry
poetry install

# Download tokenizer (uses HuggingFace SmolLM tokenizer)
# The tokenizer is created automatically on first run if not present
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (tested on RTX A6000 48GB)
- ~50GB disk space for SmolLM-Corpus subset

## Quick Start

### Training a Model

```bash
# Train a 54M parameter model (Chinchilla-optimal: ~1B tokens)
poetry run python scripts/train_smollm.py \
    --dim 512 \
    --layers 6 \
    --heads 8 \
    --output-dir runs/atlas_54m \
    --batch-size 8 \
    --gradient-accumulation-steps 4 \
    --max-steps 61000

# Ablation: Train without memory (attention-only baseline)
poetry run python scripts/train_smollm.py \
    --dim 512 \
    --layers 6 \
    --heads 8 \
    --output-dir runs/atlas_54m_no_memory \
    --batch-size 8 \
    --gradient-accumulation-steps 4 \
    --max-steps 61000 \
    --disable-memory
```

### Model Configurations

| Config | Params | dim | layers | heads | Chinchilla Tokens |
|--------|--------|-----|--------|-------|-------------------|
| Tiny | 33.7M | 384 | 6 | 6 | ~675M |
| Small | 54.4M | 512 | 6 | 8 | ~1.1B |
| Medium | 67.0M | 512 | 8 | 8 | ~1.3B |
| Base | 195.1M | 768 | 12 | 12 | ~3.9B |

### Inference

```bash
# Test a trained checkpoint
poetry run python scripts/quick_inference.py \
    --model runs/atlas_54m/best_model.pt \
    --prompt "The quick brown fox" \
    --max-tokens 100
```

## Training Data

This implementation uses the [SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) with weighted sampling:

| Subset | Weight | Description |
|--------|--------|-------------|
| cosmopedia-v2 | 40% | Synthetic textbooks and educational content |
| fineweb-edu-dedup | 50% | High-quality web text |
| python-edu-cleaned | 10% | Python code with educational comments |

## Project Structure

```
Atlas-MAG_OmegaRule/
├── src/
│   ├── model/
│   │   ├── skeleton.py          # Main model architecture
│   │   ├── qk_projection.py     # Omega Rule Q-K memory projection
│   │   ├── persistent_memory.py # M_persistent computation
│   │   └── projections.py       # QKV, rotary embeddings
│   ├── data/
│   │   ├── smollm_dataset.py    # SmolLM-Corpus data loading
│   │   └── tokenizer.py         # BPE tokenizer wrapper
│   ├── training/
│   │   ├── polarization.py      # Gate polarization loss
│   │   ├── niah_probe.py        # Needle-in-haystack retrieval tests
│   │   └── telemetry.py         # Training metrics logging
│   └── nn/
│       ├── rmsnorm.py           # RMS normalization
│       └── swiglu.py            # SwiGLU activation
├── scripts/
│   ├── train_smollm.py          # Main training script
│   ├── quick_inference.py       # Checkpoint testing
│   └── analyze_checkpoint.py    # Checkpoint analysis tools
├── tests/                       # Comprehensive test suite (200+ tests)
└── docs/                        # Design documents and summaries
```

## Key Equations

### Omega Rule (Memory Update)

```
M_t = M_persistent + Σ(i=t-c+1 to t) γ_i^(t) × (k_i ⊗ k_i)
q'_t = M_t @ q_t / norm_sum_t
```

Where:
- `c` = context window size
- `γ_i^(t)` = decay_base^(t-i) × gate_i (input-dependent decay)
- `norm_sum_t` = norm_persistent + Σ weighted ||k_i||²

### Gate Polarization Loss

```
L_polar = λ(t) × mean(1 - |2g - 1|)
```

Maximum penalty at g=0.5, zero at g∈{0,1}. Encourages gates to commit to memory or attention.

## Testing

```bash
# Run full test suite
poetry run pytest tests/ -v

# Run specific test categories
poetry run pytest tests/test_phase2.py -v  # Q-K projection tests
poetry run pytest tests/test_smollm_dataset.py -v  # Data loading tests
```

## References

- **Atlas Paper**: [arXiv:2505.23735](https://arxiv.org/abs/2505.23735) - Deep Neural Long-Term Memory
- **Titans Paper**: [arXiv:2501.00663](https://arxiv.org/abs/2501.00663) - Titans: Learning to Memorize at Test Time
- **SmolLM**: [HuggingFace SmolLM](https://huggingface.co/HuggingFaceTB/SmolLM-135M)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Status

🔬 **Research/Proof-of-Concept** - This is an experimental implementation for architecture validation. Not intended for production use.

### Current Training Runs

- 54M model with memory (Chinchilla-optimal training)
- 54M model without memory (ablation baseline)

Results will be published after training completes.
