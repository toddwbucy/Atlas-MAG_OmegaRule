# Atlas-MAG with Omega Rule

A hybrid sequence model combining **Sliding Window Attention (SWA)** with **Deep Neural Long-Term Memory**, implementing the **Omega Rule** from the Atlas paper ([arXiv:2505.23735](https://arxiv.org/abs/2505.23735)).

## Paper Reference

> **Atlas: Learning to Optimally Memorize the Context at Test Time**
> arXiv:2505.23735

This implementation aims to be a faithful reproduction of the Atlas paper, serving as a "boundary object" between the mathematical formalism and operational code. Each module includes detailed paper references with equation numbers and section citations.

## Overview

Atlas-MAG combines the efficiency of local attention with the long-range associative memory of deep neural networks:

```
Input → Embedding → [MAGBlock × N] → RMSNorm → LM Head → Output

MAGBlock:
    ┌─────────────────────────────────────────────────────────┐
    │  x ──┬──→ [Sliding Window Attention] ──→ attn_out       │
    │      │                                       │          │
    │      └──→ [Deep Polynomial Memory] ──→ mem_out          │
    │                                              │          │
    │      output = x + attn_out × sigmoid(mem_out) ←─────────│
    └─────────────────────────────────────────────────────────┘
```

### Key Features from the Atlas Paper

| Feature | Paper Section | Description |
|---------|---------------|-------------|
| **Omega Rule** | Section 3.2, Eq. 9 | Context-aware memory update over sliding window |
| **Polynomial Features** | Section 3.1, Props 1-2 | Increases memory capacity from O(d_k) to O(d_k²) |
| **MAG Architecture** | Section 4, 5.1 | Memory-as-Gate: memory output gates attention |
| **TTL (Test-Time Learning)** | Section 3.2 | Inner-loop optimization of memory at inference |
| **Newton-Schulz (Muon)** | Table 1 | Orthogonalization for stable momentum updates |
| **Input-Dependent γ Gates** | Section 3.2 | Per-position decay for context pruning |

### Memory Capacity (Propositions 1 & 2)

| Configuration | Capacity | Associations per Layer |
|---------------|----------|------------------------|
| Matrix memory (no φ) | O(d_k) | ~64 |
| With polynomial φ_2 | O(d_k²) | ~4,096 |

## Installation

```bash
# Clone the repository
git clone https://github.com/toddwbucy/Atlas-MAG_OmegaRule.git
cd Atlas-MAG_OmegaRule

# Install dependencies with Poetry
poetry install
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (tested on RTX A6000 48GB)
- ~50GB disk space for SmolLM-Corpus subset

## Quick Start

### Training

```bash
# Train the 42.6M parameter model (Small config)
poetry run python scripts/train.py \
    --dim 512 \
    --layers 6 \
    --heads 8 \
    --output-dir runs/atlas_42m \
    --batch-size 24 \
    --gradient-accumulation-steps 4 \
    --max-steps 11000

# Ablation: Train without memory (attention-only baseline)
poetry run python scripts/train.py \
    --disable-memory \
    --output-dir runs/atlas_42m_ablation
```

### Concurrent Evaluation

```bash
# Run eval worker on separate GPU (watches for checkpoints)
poetry run python scripts/eval_worker.py \
    --checkpoint-dir runs/atlas_42m \
    --device cuda:1
```

### Model Configurations

| Config | Params | dim | layers | heads | Purpose |
|--------|--------|-----|--------|-------|---------|
| Small | 42.6M | 512 | 6 | 8 | Architecture validation |
| Base | 124.7M | 768 | 12 | 12 | First real evaluation |

## Project Structure

```
Atlas-MAG_OmegaRule/
├── src/
│   ├── model/
│   │   ├── skeleton.py          # Full model assembly (Section 4)
│   │   ├── blocks.py            # MAGBlock, AttentionOnlyBlock (Section 5.1)
│   │   ├── atlas_memory.py      # Deep polynomial memory (Section 3.1)
│   │   ├── qk_projection.py     # Omega Rule Q-K projection (Eq. 9)
│   │   ├── persistent_memory.py # M_persistent computation
│   │   └── projections.py       # QKV, rotary embeddings
│   ├── training/
│   │   ├── ttl_update.py        # Test-Time Learning (Eq. 32-33)
│   │   ├── omega_loss.py        # Omega Rule loss (Eq. 9)
│   │   ├── niah_probe.py        # Needle-in-haystack memory probe
│   │   ├── validation.py        # Validation utilities
│   │   └── checkpoint.py        # Checkpoint management
│   ├── data/
│   │   ├── smollm_dataset.py    # SmolLM-Corpus streaming
│   │   └── tokenizer.py         # BPE tokenizer wrapper
│   ├── nn/
│   │   ├── newton_schulz.py     # NS-5 orthogonalization (Table 1)
│   │   ├── rmsnorm.py           # RMS normalization
│   │   └── swiglu.py            # SwiGLU activation
│   └── utils/
│       └── logging.py           # Logging utilities
├── scripts/
│   ├── train.py                 # Main training script
│   ├── eval_worker.py           # Async evaluation worker
│   └── quick_inference.py       # Checkpoint testing
└── tests/                       # Test suite (109 tests)
```

## Key Equations

### Omega Rule (Section 3.2, Equation 9)

The Omega Rule optimizes memory over a sliding context window:

```
ℓ_Omega(M; t) = Σ(i=t-c+1 to t) γ_i^(t) × ||M(φ(k_i)) - v_i||²
```

For our outer-product memory implementation:
```
M_t = M_persistent + Σ(i=t-c+1 to t) γ^(t-i) × (k_i ⊗ k_i)
q'_t = M_t @ q_t / norm_sum_t
```

Where:
- `c` = context window size (default: 256)
- `γ` = decay_base^(t-i) × gate_i (exponential decay with learned gates)
- `φ` = polynomial feature map for increased capacity

### TTL Update (Section 3.2, Equations 32-33)

Test-Time Learning uses gradient descent with momentum:

```
S_t = θ × S_{t-1} + ∇ℓ(M_{t-1}; k_t, v_t)    # Momentum accumulation
M_t = α × M_{t-1} - η × NS-5(S_t)             # Memory update with Muon
```

Where NS-5 is Newton-Schulz iteration with 5 steps for orthogonalization.

## Training Data

Uses [SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) with weighted sampling:

| Subset | Weight | Description |
|--------|--------|-------------|
| cosmopedia-v2 | 40% | Synthetic textbooks |
| fineweb-edu-dedup | 50% | High-quality web text |
| python-edu-cleaned | 10% | Educational Python code |

## Testing

```bash
# Run full test suite (109 tests)
poetry run pytest tests/ -v

# Run specific test files
poetry run pytest tests/test_phase0.py -v  # Core components
poetry run pytest tests/test_ttl.py -v     # TTL/Omega tests
```

## References

- **Atlas Paper**: [arXiv:2505.23735](https://arxiv.org/abs/2505.23735) - Atlas: Learning to Optimally Memorize the Context at Test Time
- **Titans Paper**: [arXiv:2501.00663](https://arxiv.org/abs/2501.00663) - Titans: Learning to Memorize at Test Time
- **SwiGLU Paper**: [arXiv:2002.05202](https://arxiv.org/abs/2002.05202) - GLU Variants Improve Transformer (Shazeer, 2020)
- **SmolLM-Corpus**: [HuggingFace](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Status

**Active Development** - Paper-faithful implementation with comprehensive test coverage.

Training validation in progress. Results will be published upon completion.
