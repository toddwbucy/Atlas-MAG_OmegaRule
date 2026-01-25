# Titans MAG Architecture Summary

**Source**: "Titans: Learning to Memorize at Test Time" (arXiv:2501.00663)
**Authors**: Ali Behrouz, Peilin Zhong, Vahab Mirrokni (Google Research)

---

## Executive Summary

Titans introduces a family of neural architectures that combine **short-term memory** (attention) with **long-term memory** (a neural network that learns to memorize at test time). The **MAG (Memory as a Gate)** variant is one of three architectural patterns for incorporating memory, using a gating mechanism to combine sliding window attention with a neural memory module.

---

## Core Concepts

### The Memory Perspective

The paper frames sequence modeling through a memory lens:

| Component | Role | Implementation |
|-----------|------|----------------|
| **Short-term Memory** | Precise, limited context | Sliding Window Attention (SWA) |
| **Long-term Memory** | Fading, persistent storage | Neural Memory Module (LMM) |
| **Persistent Memory** | Task knowledge (data-independent) | Learnable parameters |

### The Neural Long-term Memory Module (LMM)

The heart of Titans is the **Long-term Memory Module (LMM)** - a meta-model that learns to memorize at test time.

#### Surprise-Based Learning

The key insight: **events that violate expectations (surprises) are more memorable**.

Memory update formula:
```
M_t = (1 - α_t) M_{t-1} + S_t                    # Memory update with forgetting
S_t = η_t S_{t-1} - θ_t ∇ℓ(M_{t-1}; x_t)         # Surprise = past + momentary
```

Where:
- `M_t` = Memory state at time t (neural network weights)
- `α_t` = Adaptive forgetting gate (data-dependent)
- `S_t` = Total surprise metric
- `η_t` = Surprise decay (controls how past surprise propagates)
- `θ_t` = Learning rate for momentary surprise
- `∇ℓ` = Gradient of associative memory loss

#### Associative Memory Loss

The memory learns key-value associations:
```
k_t = x_t W_K          # Key projection
v_t = x_t W_V          # Value projection

ℓ(M; x_t) = ||M(k_t) - v_t||²₂   # Reconstruction loss
```

#### Memory Retrieval

To retrieve from memory, use forward pass without weight update:
```
q_t = x_t W_Q          # Query projection
y_t = M*(q_t)          # Forward pass (no gradient)
```

---

## MAG Architecture (Memory as a Gate)

### Overview

MAG runs two parallel branches and combines them via gating:

```
┌─────────────────────────────────────────────────────────────┐
│                    MAG Architecture                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: x ∈ R^(N×d_in)                                      │
│                                                             │
│  1. Prepend Persistent Memory:                              │
│     x̃ = [p₁, p₂, ..., p_Np] || x                           │
│                                                             │
│  2. Two Parallel Branches:                                  │
│     ┌──────────────────┐    ┌──────────────────┐           │
│     │ Sliding Window   │    │ Neural Memory    │           │
│     │ Attention (SWA)  │    │ Module (LMM)     │           │
│     │                  │    │                  │           │
│     │  y = SW-Attn*(x̃) │    │  m = M(x̃)       │           │
│     └────────┬─────────┘    └────────┬─────────┘           │
│              │                       │                      │
│              └───────────┬───────────┘                      │
│                          │                                  │
│  3. Gated Combination:   ▼                                  │
│     o = y ⊗ M(x̃)                                           │
│                                                             │
│  (⊗ = non-linear gating with learned normalization)        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Mathematical Formulation

```python
# Input sequence
x ∈ R^(N × d_in)

# Step 1: Prepend persistent memory
x̃ = [p₁, p₂, ..., p_Np] || x

# Step 2: Parallel processing
y = SW-Attn*(x̃)    # Sliding window attention with prefix
m = M(x̃)           # Neural memory forward pass

# Step 3: Gated combination
o = y ⊗ m          # Non-linear gating
```

### Attention Mask Pattern

```
MAG Attention Mask (Figure 3b from paper):

      ← Sliding Window →  ← Long-term Memory →  ← Persistent →
    ┌────────────────────┬────────────────────┬───────────────┐
    │ ▓▓▓▓▓▓▓░░░░░░░░░░░ │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │ ▓▓▓▓▓▓▓▓▓▓▓▓▓ │
    │ ░▓▓▓▓▓▓▓░░░░░░░░░░ │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │ ▓▓▓▓▓▓▓▓▓▓▓▓▓ │
    │ ░░▓▓▓▓▓▓▓░░░░░░░░░ │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │ ▓▓▓▓▓▓▓▓▓▓▓▓▓ │
    │ ░░░▓▓▓▓▓▓▓░░░░░░░░ │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │ ▓▓▓▓▓▓▓▓▓▓▓▓▓ │
    │ ░░░░▓▓▓▓▓▓▓░░░░░░░ │ ...                │ ...           │
    └────────────────────┴────────────────────┴───────────────┘

    ▓ = Attention active    ░ = Masked (no attention)

    - Sliding window: Local causal attention (window size W)
    - Long-term memory: Processed separately by neural memory
    - Persistent memory: Always attended to (prefix)
```

### Key Characteristics

| Feature | MAG Behavior |
|---------|--------------|
| **Segmentation** | No segmentation (processes full sequence) |
| **Short-term** | Sliding window attention (precise, limited) |
| **Long-term** | Neural memory module (fading, persistent) |
| **Combination** | Gating mechanism (learned weighting) |
| **Multi-head view** | Different head structures in parallel |

---

## Comparison: MAG vs MAC vs MAL

| Aspect | MAC (Context) | MAG (Gate) | MAL (Layer) |
|--------|---------------|------------|-------------|
| **Memory Integration** | Concatenate to context | Gated parallel branch | Sequential layers |
| **Segmentation** | Yes (chunks) | No | No |
| **Attention Type** | Full causal (in segment) | Sliding window | Sliding window |
| **Memory Flow** | Attention decides storage | Direct memory update | Pre-attention compression |
| **Strength** | Best long-context | Balanced efficiency | Fastest training |

### Performance Comparison (from Table 5)

| Variant | Language Modeling (ppl↓) | Reasoning (acc↑) | Long Context (acc↑) |
|---------|--------------------------|------------------|---------------------|
| LMM (no attention) | 27.01 | 47.83 | 92.68 |
| +Attn (MAC) | 26.67 | 48.65 | **97.95** |
| +Attn (MAG) | **25.70** | 48.60 | 96.70 |
| +Attn (MAL) | 25.91 | 47.87 | 96.91 |

**MAG Strengths**:
- Best perplexity in language modeling
- Competitive reasoning accuracy
- Good long-context performance (slightly below MAC)

---

## Deep Memory Architecture

The paper argues for **deep memory** (L_M ≥ 2 layers):

```
Why Deep Memory?

Linear memory (L_M = 1):
  - Equivalent to online linear regression
  - Assumes linear dependency in historical data
  - Limited expressiveness

Deep memory (L_M ≥ 2):
  - MLPs are strictly more expressive than linear
  - Can capture non-linear patterns
  - Better performance on long sequences
```

### Memory Architecture Details

```python
class NeuralMemory:
    """Long-term Memory Module (LMM)"""

    def __init__(self, d_in, L_M=2):
        # L_M layers of MLP
        self.layers = [MLP(d_in, d_in) for _ in range(L_M)]

        # Projections for Q, K, V
        self.W_Q = Linear(d_in, d_in)
        self.W_K = Linear(d_in, d_in)
        self.W_V = Linear(d_in, d_in)

        # 1D depthwise-separable convolution after projections
        self.conv_q = Conv1D(d_in, kernel_size=4)
        self.conv_k = Conv1D(d_in, kernel_size=4)
        self.conv_v = Conv1D(d_in, kernel_size=4)

        # Learnable gates
        self.alpha = ...  # Forgetting gate
        self.eta = ...    # Surprise decay
        self.theta = ...  # Learning rate

    def forward(self, x):
        # Project to Q, K, V with convolution
        q = self.conv_q(self.W_Q(x))
        k = self.conv_k(self.W_K(x))
        v = self.conv_v(self.W_V(x))

        # Apply SiLU activation
        q, k, v = silu(q), silu(k), silu(v)

        # Normalize Q, K with L2 norm
        q = l2_normalize(q)
        k = l2_normalize(k)

        # Update memory weights using surprise
        # ... (gradient-based update)

        # Retrieve from memory
        return self.memory_forward(q)
```

---

## Persistent Memory

Persistent memory serves three purposes:

1. **Memory Perspective**: Stores task-related knowledge (data-independent)
2. **FFN Perspective**: Acts like input-independent attention weights
3. **Technical Fix**: Mitigates attention sink problem (bias toward initial tokens)

```python
# Persistent memory tokens prepended to sequence
P = [p_1, p_2, ..., p_Np]  # Learnable parameters
x_new = P || x             # Concatenate with input
```

---

## Parallelizable Training

The paper shows how to parallelize the gradient-based memory update:

### Within-Chunk (Linear)
```
# Mini-batch gradient descent can be tensorized
∇ℓ(W_0; x_t) = (W_0 x_t - x_t) x_t^T

Σ θ_i (β_b/β_i) ∇ℓ(W_0; x_i) = Θ_b B_b (W_0 X - X) X^T
```

### Cross-Chunk (Non-Linear)
```
# Momentum is a linear recurrence
S_t = η_t S_{t-1} - θ_t u_t

# Can use parallel associative scan
```

---

## Key Experimental Results

### Needle-in-Haystack (S-NIAH)

| Model | 2K | 4K | 8K | 16K |
|-------|-----|-----|-----|------|
| TTT | 78.8 | 28.0 | 4.4 | 0.0 |
| Mamba2 | 42.2 | 4.2 | 0.0 | 0.0 |
| DeltaNet | 46.2 | 20.0 | 1.6 | 0.0 |
| **Titans (MAG)** | **98.0** | **98.0** | **90.2** | **88.2** |

### BABILong Benchmark

Titans (MAC) outperforms GPT-4, Llama3.1-70B, and other large models on reasoning across distributed facts - with ~70x fewer parameters than Llama3.1-8B+RAG.

### Training Throughput

MAG shows competitive training throughput, benefiting from FlashAttention optimization for sliding window attention.

---

## Ablation Study

| Component Removed | Language ppl↓ | Reasoning acc↑ | Long Context acc↑ |
|-------------------|---------------|----------------|-------------------|
| Full LMM | 27.01 | 47.83 | 92.68 |
| → Linear Memory | 28.49 | 46.97 | 85.34 |
| → w/o Convolution | 28.73 | 45.82 | 90.28 |
| → w/o Momentum | 28.98 | 45.49 | 87.12 |
| → w/o Weight Decay | 29.04 | 45.11 | 85.60 |
| → w/o Persistent Memory | 27.63 | 46.35 | 92.49 |

**Most important components** (by impact):
1. Weight Decay (forgetting mechanism)
2. Momentum (past surprise)
3. Convolution
4. Persistent Memory

---

## Theoretical Properties

> **Theorem 4.1**: Contrary to Transformers, diagonal linear recurrent models, and DeltaNet, all of which are limited to TC⁰, Titans are capable of solving problems beyond TC⁰, meaning that Titans are theoretically more expressive than Transformers and most modern linear recurrent models in state tracking tasks.

---

## Implementation Notes

### Architectural Details

1. **Activation**: SiLU for Q, K, V projections
2. **Normalization**: L2-norm for queries and keys
3. **Convolution**: 1D depthwise-separable after Q, K, V projections
4. **Gating**: Normalization + linear layer before output projection
5. **Residual**: Used in all blocks

### MAG-Specific Implementation

```python
class TitansMAG(nn.Module):
    def __init__(self, d_model, n_heads, window_size, n_persistent):
        self.persistent_memory = nn.Parameter(torch.randn(n_persistent, d_model))
        self.sliding_window_attn = SlidingWindowAttention(d_model, n_heads, window_size)
        self.neural_memory = NeuralMemory(d_model)
        self.gate_norm_attn = nn.LayerNorm(d_model)
        self.gate_norm_mem = nn.LayerNorm(d_model)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        # Prepend persistent memory
        batch_size = x.size(0)
        persistent = self.persistent_memory.unsqueeze(0).expand(batch_size, -1, -1)
        x_aug = torch.cat([persistent, x], dim=1)

        # Parallel branches
        y_attn = self.sliding_window_attn(x_aug)
        y_mem = self.neural_memory(x_aug)

        # Gated combination
        y_attn_norm = self.gate_norm_attn(y_attn)
        y_mem_norm = self.gate_norm_mem(y_mem)

        # Non-linear gating (paper uses σ(.) for this)
        output = y_attn_norm * self.gate_activation(y_mem_norm)

        return output
```

---

## Summary: Why MAG?

**Choose MAG when you need**:
- Balanced efficiency and long-context capability
- No sequence segmentation (simpler implementation)
- Parallel processing of attention and memory
- Good perplexity on language modeling tasks

**Choose MAC instead when**:
- Maximum long-context performance is critical
- Attention should decide what to memorize
- Working with very long sequences (>100K tokens)

**Choose MAL instead when**:
- Training speed is the priority
- Following existing hybrid architecture patterns
- Memory should pre-process before attention

---

## References

- Paper: https://arxiv.org/abs/2501.00663
- Related: TTT (Test-Time Training), DeltaNet, Gated DeltaNet, Mamba2
