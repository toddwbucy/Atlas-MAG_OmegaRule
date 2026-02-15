# Atlas-MAG with Omega Rule

An implementation of the **Atlas** paper's Memory-As-Gate (MAG) architecture with polynomial memory, test-time learning (TTL), and the Omega Rule.

> **Atlas: Learning to Optimally Memorize the Context at Test Time**
> Behrouz, Li, Kacham, Daliri, Deng, Zhong, Razaviyayn, Mirrokni (Google Research)
> [arXiv:2505.23735](https://arxiv.org/abs/2505.23735)

**Checkpoint**: [r3d91ll/Atlas-MAG_OmegaRule on HuggingFace](https://huggingface.co/r3d91ll/Atlas-MAG_OmegaRule)

## The Infrastructure Problem

This model demonstrates a fundamental gap between how Nested Learning models are *designed* to run and how existing infrastructure *allows* them to run.

Atlas-MAG uses **test-time learning (TTL)**: during the forward pass, the model's memory updates itself via gradient descent. This is not training — it is how the model processes context. But PyTorch gates the TTL inner loop behind `if self.training`, and every serving framework (vLLM, TGI, TensorRT-LLM) sets models to inference mode before serving.

The result: the model's memory architecture is silenced at serve time.

Two scripts let you see this for yourself:

### Demo: The Train-Flag Problem

```bash
# Auto-downloads the 473MB checkpoint from HuggingFace
pip install huggingface_hub
python scripts/demo_ttl_inference.py
```

Runs the same model with the same weights on the same input twice — once with TTL silenced (inference mode), once with TTL active (training mode). You'll see different outputs from identical weights.

### Benchmark: NIAH Memory Probe

```bash
python scripts/benchmark_niah.py
```

Measures memory contribution at positions beyond the sliding window attention range under three conditions:

| Condition | What It Is | What It Represents |
|-----------|-----------|-------------------|
| **Attention Only** | Memory disabled | Baseline |
| **TTL OFF** | Memory exists, is static | What serving frameworks give you |
| **TTL ON** | Memory adapts during forward pass | How the model was designed |

The gap between TTL OFF and TTL ON is what inference mode costs you.

## Model

| | |
|---|---|
| **Parameters** | 43M (dim=512, 6 layers, 8 heads) |
| **Memory** | Polynomial degree-2, rank-512 |
| **TTL** | Muon optimizer (Newton-Schulz 5-iter), momentum=0.9 |
| **Training Data** | SmolLM-Corpus (cosmopedia 40%, fineweb-edu 50%, python-edu 10%) |
| **Training** | 8,800 steps on dual A6000 48GB |
| **NIAH Accuracy** | 85.9% (memory contribution at beyond-window positions) |

## Architecture

```
Input -> Embedding -> [MAGBlock x 6] -> RMSNorm -> LM Head -> Output

MAGBlock:
    x --+--> [Sliding Window Attention] --> attn_out
        |                                      |
        +--> [Deep Polynomial Memory]  --> mem_out
                                               |
        output = x + attn_out * sigmoid(mem_out)
```

Each MAGBlock combines local attention (window=512) with a polynomial memory module. The memory output *gates* the attention output, controlling how much attention contributes at each position.

The polynomial feature map (Section 3.1, Props 1-2) increases memory capacity from O(d_k) to O(d_k^2) — roughly 64x more associations per layer.

## Installation

```bash
git clone https://github.com/toddwbucy/Atlas-MAG_OmegaRule.git
cd Atlas-MAG_OmegaRule
pip install torch huggingface_hub tokenizers

# Run the demo (auto-downloads checkpoint)
python scripts/demo_ttl_inference.py
```

For full development (training, tests):

```bash
poetry install
poetry run pytest tests/ -v  # 109 tests
```

## Project Structure

```
Atlas-MAG_OmegaRule/
├── src/
│   ├── model/
│   │   ├── skeleton.py          # AtlasMAGSkeleton (Section 4)
│   │   ├── blocks.py            # MAGBlock with gamma gates (Section 5.1)
│   │   ├── atlas_memory.py      # Polynomial memory (Section 3.1)
│   │   ├── qk_projection.py     # Omega Rule Q-K projection (Eq. 9)
│   │   ├── persistent_memory.py # M_persistent computation
│   │   └── projections.py       # QKV, rotary embeddings
│   ├── runtime/
│   │   ├── ttl_update.py        # Test-Time Learning (Eq. 32-33)
│   │   ├── omega_loss.py        # Omega Rule loss (Eq. 9)
│   │   ├── niah_probe.py        # Needle-in-haystack memory probe
│   │   ├── validation.py        # Validation utilities
│   │   └── checkpoint.py        # Checkpoint management
│   ├── data/                    # SmolLM-Corpus streaming + tokenizer
│   └── nn/                      # Newton-Schulz, RMSNorm
├── scripts/
│   ├── demo_ttl_inference.py    # Train-flag problem demo
│   ├── benchmark_niah.py        # NIAH memory probe (TTL ON vs OFF)
│   ├── train.py                 # Training script
│   ├── eval_worker.py           # Async evaluation worker
│   └── quick_inference.py       # Quick text generation
├── tests/                       # 109 tests
├── ISSUES.md                    # NL graph compliance tracking
└── LESSONS_LEARNED.md           # What we learned building this
```

## Key Equations

**Omega Rule** (Section 3.2, Eq. 9) — memory update over sliding context window:
```
l_Omega(M; t) = sum(i=t-c+1 to t) gamma_i^(t) * ||M(phi(k_i)) - v_i||^2
```

**TTL Update** (Section 3.2, Eq. 32-33) — gradient descent with Muon momentum:
```
S_t = theta * S_{t-1} + grad_l(M_{t-1}; k_t, v_t)   # Momentum
M_t = alpha * M_{t-1} - eta * NS-5(S_t)              # Memory update
```

## References

- [Atlas (arXiv:2505.23735)](https://arxiv.org/abs/2505.23735) — Learning to Optimally Memorize the Context at Test Time
- [Titans (arXiv:2501.00663)](https://arxiv.org/abs/2501.00663) — Learning to Memorize at Test Time
- [Nested Learning (arXiv:2512.24695)](https://arxiv.org/abs/2512.24695) — The capstone paper unifying the research program

## License

MIT
