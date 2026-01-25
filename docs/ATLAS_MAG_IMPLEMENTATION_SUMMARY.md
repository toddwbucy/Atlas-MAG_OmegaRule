# Atlas-MAG Implementation Summary

## Overview

**Paper**: "Atlas: Learning to Optimally Memorize the Context at Test Time" (arXiv:2505.23735)
**Authors**: Ali Behrouz et al. (Google Research)
**Focus**: Complete implementation guide for Atlas-MAG with the Omega Rule

Atlas is a long-term memory module that learns to memorize **context** (not individual tokens) by optimizing memory based on current AND past tokens within a sliding window. Atlas-MAG is the hybrid variant combining Atlas with Sliding Window Attention (SWA).

---

## Core Problem Addressed

Modern recurrent models suffer from three limitations:
1. **Online nature**: Memory optimized only w.r.t. the last input token
2. **Limited capacity**: Bounded by architecture and feature mapping
3. **Weak memory management**: Simple gradient descent leads to suboptimal local minima

Atlas addresses all three through:
- **Omega Rule**: Sliding window context memorization
- **Polynomial Feature Mapping**: Increased memory capacity
- **Muon Optimizer**: Second-order information for better memory management

---

## The Omega Rule

### Definition

The Omega Rule optimizes memory w.r.t. a **window of past tokens** rather than just the current token:

```
min_M  Σ(i=t-c+1 to t) γ_i^(t) · ||M(k_i) - v_i||²₂
```

Where:
- `c` = local context window length (c=1 reduces to online Delta rule)
- `γ_i^(t)` = input-dependent decay/gating parameters for token i
- `M(·)` = memory module (neural network)
- `k_i, v_i` = key-value pairs at position i

### Key Insight

Instead of measuring the "surprise" of a single token, Omega Rule measures the **surprise of a local context** based on context-aware combination of tokens within the window.

### Gradient Descent Update (OmegaNet baseline)

```python
# OmegaNet update rule (gradient descent)
M_t = α_t * M_{t-1} - η_t * Σ(i=t-c+1 to t) γ_i^(t) * ∇||M_{t-1}(φ(k_i)) - v_i||²₂
```

For linear memory this simplifies to:
```python
M_t = (diag(α_t) - Σ γ_i^(t) * φ(k_i) * φ(k_i)ᵀ) * M_{t-1} - Σ γ_i^(t) * v_i * φ(k_i)ᵀ
```

---

## Atlas: Full Update Rules

Atlas enhances OmegaNet by using the **Muon optimizer** (Newton-Schulz iteration) instead of simple gradient descent:

### Primary Update Equations

```python
# Atlas memory update with Muon optimizer
M_t = α_t * M_{t-1} - η_t * NewtonSchulz_k(S_t)

# Momentum accumulation
S_t = θ_t * S_{t-1} + ∇[Σ(i=t-c+1 to t) γ_i^(t) * ||M(φ(k_i)) - v_i||²₂]
```

Where:
- `α_t` = weight decay parameter (input-dependent)
- `η_t` = learning rate (input-dependent)
- `θ_t` = momentum coefficient
- `k` = Newton-Schulz iterations (more iterations → better 2nd-order approximation)
- `φ(·)` = polynomial feature mapping

### Newton-Schulz Algorithm (NS-5)

The Newton-Schulz iteration approximates the nearest semi-orthogonal matrix:

```python
def newton_schulz_5(G, num_iters=5):
    """
    Compute orthogonalized version of gradient matrix.
    Approximates: (G @ G.T)^{-1/2} @ G
    """
    a, b, c = (3.4445, -4.7750, 2.0315)  # Coefficients for 5th order
    X = G
    for _ in range(num_iters):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X
```

---

## Polynomial Feature Mapping

### Purpose
Increases memory capacity from O(d_k) to O(d_k^p) where p is polynomial degree.

### Definition

```python
def polynomial_mapping(x, degree=2):
    """
    φ_p(x) = [x^β]_{|β|≤p}

    For degree 2 with input x ∈ R^d:
    Output includes: 1, x_1, x_2, ..., x_d, x_1², x_1*x_2, ..., x_d²
    """
    features = [torch.ones_like(x[..., :1])]  # degree 0
    features.append(x)  # degree 1

    if degree >= 2:
        # Add degree 2 terms (outer product)
        x_outer = x.unsqueeze(-1) * x.unsqueeze(-2)  # [B, d, d]
        # Extract upper triangular (including diagonal) to avoid duplicates
        features.append(x_outer.flatten(-2))

    return torch.cat(features, dim=-1)
```

### Capacity Analysis

| Memory Type | Capacity |
|-------------|----------|
| Linear (matrix) | O(d_k) |
| Deep MLP (L layers) | O(d_k · d_v · Σ min{d_h^(j)} · d_h^(j+1)) |
| With polynomial degree p | O(d_k^p) |

---

## Memory Architecture

### Standard Atlas Memory (2-layer MLP with residual)

```python
class AtlasMemory(nn.Module):
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.W1 = nn.Linear(dim * expansion, dim)
        self.W2 = nn.Linear(dim, dim * expansion)
        self.activation = nn.GELU()

    def forward(self, x):
        # M(x) = x + W1 * σ(W2 * x)
        return x + self.W1(self.activation(self.W2(x)))
```

### Atlas++ Memory (Gated MLP)

```python
class AtlasPlusPlusMemory(nn.Module):
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.W1 = nn.Linear(dim * expansion, dim)
        self.W2 = nn.Linear(dim, dim * expansion)
        self.W3 = nn.Linear(dim, dim * expansion)  # Gate
        self.activation = nn.GELU()

    def forward(self, x):
        # M(x) = x + W1 * (σ(W2 * x) ⊙ W3 * x)
        gate = self.activation(self.W2(x))
        value = self.W3(x)
        return x + self.W1(gate * value)
```

---

## Atlas Layer Implementation

### Complete Atlas Layer

```python
class AtlasLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        context_window: int = 8,  # Omega rule window size
        poly_degree: int = 2,
        memory_expansion: int = 4,
        conv_size: int = 4,
        ns_iters: int = 5,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.context_window = context_window
        self.ns_iters = ns_iters

        # Projections for K, Q, V
        self.W_k = nn.Linear(dim, dim)
        self.W_q = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)

        # Short convolutions (following recent linear RNN practices)
        self.conv_k = nn.Conv1d(dim, dim, conv_size, padding=conv_size-1, groups=dim)
        self.conv_q = nn.Conv1d(dim, dim, conv_size, padding=conv_size-1, groups=dim)

        # Normalization for stability
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.q_norm = nn.RMSNorm(self.head_dim)

        # Learnable gates (input-dependent)
        self.W_alpha = nn.Linear(dim, num_heads)  # Weight decay
        self.W_eta = nn.Linear(dim, num_heads)    # Learning rate
        self.W_theta = nn.Linear(dim, num_heads)  # Momentum
        self.W_gamma = nn.Linear(dim, num_heads * context_window)  # Context gates

        # Memory module
        self.memory = AtlasMemory(self.head_dim, memory_expansion)

        # Polynomial feature mapping
        self.poly_degree = poly_degree

        # Output projection
        self.W_o = nn.Linear(dim, dim)

    def polynomial_features(self, x):
        """Apply polynomial feature mapping to keys."""
        if self.poly_degree == 1:
            return x
        elif self.poly_degree == 2:
            # Quadratic features: [x, x⊗x]
            x_sq = x.unsqueeze(-1) * x.unsqueeze(-2)
            triu_idx = torch.triu_indices(x.size(-1), x.size(-1))
            x_sq_flat = x_sq[..., triu_idx[0], triu_idx[1]]
            return torch.cat([x, x_sq_flat], dim=-1)
        else:
            raise NotImplementedError(f"Degree {self.poly_degree} not implemented")

    def newton_schulz(self, G):
        """Newton-Schulz iteration for orthogonalization."""
        a, b, c = 3.4445, -4.7750, 2.0315
        X = G
        for _ in range(self.ns_iters):
            A = X @ X.transpose(-2, -1)
            B = b * A + c * A @ A
            X = a * X + B @ X
        return X

    def forward(self, x, memory_state=None, momentum_state=None):
        """
        Args:
            x: Input tensor [B, L, D]
            memory_state: Previous memory state (for recurrent inference)
            momentum_state: Previous momentum state

        Returns:
            output: [B, L, D]
            new_memory_state: Updated memory
            new_momentum_state: Updated momentum
        """
        B, L, D = x.shape
        H = self.num_heads
        d = self.head_dim
        c = self.context_window

        # Project and apply convolutions
        k = self.W_k(x).transpose(1, 2)
        k = self.conv_k(k)[..., :L].transpose(1, 2)

        q = self.W_q(x).transpose(1, 2)
        q = self.conv_q(q)[..., :L].transpose(1, 2)

        v = self.W_v(x)

        # Reshape for multi-head
        k = k.view(B, L, H, d)
        q = q.view(B, L, H, d)
        v = v.view(B, L, H, d)

        # Normalize K and Q
        k = self.k_norm(k)
        q = self.q_norm(q)

        # Compute gates
        alpha = torch.sigmoid(self.W_alpha(x)).view(B, L, H, 1, 1)  # [B, L, H, 1, 1]
        eta = torch.sigmoid(self.W_eta(x)).view(B, L, H, 1, 1)
        theta = torch.sigmoid(self.W_theta(x)).view(B, L, H, 1, 1)
        gamma = torch.sigmoid(self.W_gamma(x)).view(B, L, H, c)  # [B, L, H, c]

        # Apply polynomial features to keys
        k_poly = self.polynomial_features(k)  # [B, L, H, d_poly]
        q_poly = self.polynomial_features(q)

        # Initialize states if None
        if memory_state is None:
            memory_state = self._init_memory_state(B, H, d)
        if momentum_state is None:
            momentum_state = torch.zeros_like(memory_state)

        outputs = []

        # Process sequence (chunk-wise for efficiency in practice)
        for t in range(L):
            # Get current tokens
            k_t = k_poly[:, t]  # [B, H, d_poly]
            v_t = v[:, t]       # [B, H, d]
            q_t = q_poly[:, t]  # [B, H, d_poly]

            # Compute context window indices
            start_idx = max(0, t - c + 1)

            # Compute Omega gradient (sum over context window)
            grad = torch.zeros_like(memory_state)
            for i in range(start_idx, t + 1):
                k_i = k_poly[:, i]
                v_i = v[:, i]
                gamma_i = gamma[:, t, :, i - start_idx:i - start_idx + 1]

                # Gradient of ||M(k_i) - v_i||²
                pred = self.memory(k_i)
                error = pred - v_i
                grad_i = self._compute_memory_gradient(k_i, error)
                grad = grad + gamma_i.unsqueeze(-1) * grad_i

            # Update momentum
            momentum_state = theta[:, t] * momentum_state + grad

            # Newton-Schulz orthogonalization
            ns_momentum = self.newton_schulz(momentum_state)

            # Update memory
            memory_state = alpha[:, t] * memory_state - eta[:, t] * ns_momentum

            # Read from memory
            output_t = self.memory(q_t)
            outputs.append(output_t)

        # Stack outputs
        output = torch.stack(outputs, dim=1)  # [B, L, H, d]
        output = output.view(B, L, D)
        output = self.W_o(output)

        return output, memory_state, momentum_state

    def _init_memory_state(self, B, H, d):
        """Initialize memory parameters."""
        # Initialize memory MLP weights
        return {
            'W1': torch.zeros(B, H, d * 4, d),
            'W2': torch.zeros(B, H, d, d * 4),
        }

    def _compute_memory_gradient(self, k, error):
        """Compute gradient w.r.t. memory parameters."""
        # This is a simplified version; actual implementation
        # computes gradients through the MLP
        return error.unsqueeze(-1) * k.unsqueeze(-2)
```

---

## Atlas-MAG Architecture

Atlas-MAG (Memory as a Gate) is the hybrid architecture combining Atlas with Sliding Window Attention.

### Architecture Pattern

```
Input
  │
  ├──────────────────┐
  │                  │
  ▼                  │
[Atlas Layer] ───────┤
  │                  │
  ▼                  │
[RMSNorm]            │
  │                  │
  ├──────────────────┘ (residual)
  │
  ▼
[SWA Layer] ◄── Sliding Window Attention
  │
  ▼
[Norm]
  │
  ├──────────────────┐
  │                  │
  ▼                  │
[MLP (SwiGLU)]       │
  │                  │
  ├──────────────────┘ (residual)
  │
  ▼
Output
```

### Complete Atlas-MAG Block

```python
class AtlasMAGBlock(nn.Module):
    """
    Atlas Memory-as-Gate Block
    Combines Atlas layer with Sliding Window Attention
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        atlas_context_window: int = 8,
        swa_window_size: int = 512,
        mlp_expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Atlas layer
        self.atlas = AtlasLayer(
            dim=dim,
            num_heads=num_heads,
            context_window=atlas_context_window,
        )
        self.atlas_norm = nn.RMSNorm(dim)

        # Sliding Window Attention
        self.swa = SlidingWindowAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=swa_window_size,
        )
        self.swa_norm = nn.RMSNorm(dim)

        # MLP (SwiGLU)
        self.mlp = SwiGLU(dim, expansion=mlp_expansion)
        self.mlp_norm = nn.RMSNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory_state=None, momentum_state=None):
        # Atlas branch
        atlas_out, memory_state, momentum_state = self.atlas(
            self.atlas_norm(x), memory_state, momentum_state
        )
        x = x + self.dropout(atlas_out)

        # SWA branch
        swa_out = self.swa(self.swa_norm(x))
        x = x + self.dropout(swa_out)

        # MLP
        mlp_out = self.mlp(self.mlp_norm(x))
        x = x + self.dropout(mlp_out)

        return x, memory_state, momentum_state


class SlidingWindowAttention(nn.Module):
    """Standard Sliding Window Attention."""
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        self.W_qkv = nn.Linear(dim, 3 * dim)
        self.W_o = nn.Linear(dim, dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, L, D = x.shape

        qkv = self.W_qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        # Compute attention with sliding window mask
        attn = torch.einsum('blhd,bmhd->bhlm', q, k) * self.scale

        # Create sliding window mask
        mask = self._create_sliding_window_mask(L)
        attn = attn.masked_fill(~mask, float('-inf'))

        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum('bhlm,bmhd->blhd', attn, v)
        out = out.reshape(B, L, D)

        return self.W_o(out)

    def _create_sliding_window_mask(self, L):
        """Create causal sliding window attention mask."""
        mask = torch.ones(L, L, dtype=torch.bool)
        for i in range(L):
            start = max(0, i - self.window_size + 1)
            mask[i, :start] = False
            mask[i, i+1:] = False  # Causal
        return mask


class SwiGLU(nn.Module):
    """SwiGLU activation MLP."""
    def __init__(self, dim, expansion=4):
        super().__init__()
        hidden = int(dim * expansion * 2 / 3)
        self.W1 = nn.Linear(dim, hidden)
        self.W2 = nn.Linear(dim, hidden)
        self.W3 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.W3(F.silu(self.W1(x)) * self.W2(x))
```

---

## Complete Atlas-MAG Model

```python
class AtlasMAG(nn.Module):
    """
    Full Atlas-MAG Language Model
    """
    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 1536,
        num_layers: int = 24,
        num_heads: int = 16,
        atlas_context_window: int = 8,
        swa_window_size: int = 512,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_encoding = nn.Embedding(max_seq_len, dim)

        self.layers = nn.ModuleList([
            AtlasMAGBlock(
                dim=dim,
                num_heads=num_heads,
                atlas_context_window=atlas_context_window,
                swa_window_size=swa_window_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, memory_states=None, momentum_states=None):
        B, L = input_ids.shape

        # Embeddings
        x = self.embedding(input_ids)
        positions = torch.arange(L, device=input_ids.device)
        x = x + self.pos_encoding(positions)

        # Initialize states if needed
        if memory_states is None:
            memory_states = [None] * len(self.layers)
        if momentum_states is None:
            momentum_states = [None] * len(self.layers)

        new_memory_states = []
        new_momentum_states = []

        # Process through layers
        for i, layer in enumerate(self.layers):
            x, mem, mom = layer(x, memory_states[i], momentum_states[i])
            new_memory_states.append(mem)
            new_momentum_states.append(mom)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, new_memory_states, new_momentum_states
```

---

## Parallel Training Algorithm

For efficient training, Atlas uses chunk-wise parallelization:

```python
def parallel_atlas_forward(
    x: torch.Tensor,           # [B, L, D]
    chunk_size: int = 64,
    context_window: int = 8,
):
    """
    Efficient parallel training with chunk-wise recurrence.

    Within each chunk: parallel gradient computation
    Across chunks: sequential state updates
    """
    B, L, D = x.shape
    num_chunks = L // chunk_size

    memory_state = None
    momentum_state = None
    all_outputs = []

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = chunk_start + chunk_size
        x_chunk = x[:, chunk_start:chunk_end]

        # Compute all keys, values in parallel
        k_chunk = compute_keys(x_chunk)      # [B, chunk_size, H, d]
        v_chunk = compute_values(x_chunk)    # [B, chunk_size, H, d]
        q_chunk = compute_queries(x_chunk)   # [B, chunk_size, H, d]

        # Compute gates in parallel
        alpha_chunk = compute_alpha(x_chunk)
        eta_chunk = compute_eta(x_chunk)
        theta_chunk = compute_theta(x_chunk)
        gamma_chunk = compute_gamma(x_chunk)

        # Compute all gradients w.r.t. chunk start state
        # Apply sliding window mask for Omega rule
        sliding_mask = create_sliding_window_mask(chunk_size, context_window)

        # Parallel gradient computation with einsum
        gradients = compute_omega_gradients_parallel(
            k_chunk, v_chunk, memory_state, gamma_chunk, sliding_mask
        )  # [B, chunk_size, H, d, d]

        # Sequential momentum/memory updates within chunk
        outputs_chunk = []
        for t in range(chunk_size):
            # Update momentum
            momentum_state = theta_chunk[:, t] * momentum_state + gradients[:, t]

            # Newton-Schulz
            ns_momentum = newton_schulz_5(momentum_state)

            # Update memory
            memory_state = alpha_chunk[:, t] * memory_state - eta_chunk[:, t] * ns_momentum

            # Read output
            output_t = memory_read(q_chunk[:, t], memory_state)
            outputs_chunk.append(output_t)

        all_outputs.extend(outputs_chunk)

    return torch.stack(all_outputs, dim=1)


def create_sliding_window_mask(chunk_size, context_window):
    """
    Create mask for Omega rule within chunk.

    M_s is identity + (c-1) positions before each diagonal
    """
    mask = torch.eye(chunk_size, dtype=torch.bool)
    for offset in range(1, context_window):
        mask = mask | torch.eye(chunk_size, dtype=torch.bool).roll(offset, dims=1)
    # Make causal
    mask = torch.tril(mask)
    return mask
```

---

## Hyperparameters

### Recommended Settings from Paper

| Model Size | Blocks | Dim  | Heads | Peak LR | Tokens |
|------------|--------|------|-------|---------|--------|
| 340M       | 24     | 1024 | 16    | 1.5e-3  | 15B    |
| 760M       | 24     | 1536 | 16    | 1.25e-3 | 30B    |
| 1.3B       | 18     | 2048 | 8     | 7e-4    | 100B   |

### Atlas-Specific Settings

| Parameter | Recommended Value |
|-----------|-------------------|
| Context window (c) | 4-16 |
| Polynomial degree | 2 |
| Memory expansion | 4x |
| Conv kernel size | 4 |
| NS iterations | 5 |
| SWA window (hybrid) | 512-2048 |

---

## Key Performance Results

### Language Modeling (760M, 30B tokens)

| Model | Wiki PPL | LMB PPL | Avg Acc |
|-------|----------|---------|---------|
| Transformer++ | 25.21 | 27.64 | 48.69% |
| Titans (LMM) | 20.04 | 21.96 | 51.56% |
| Atlas | **18.92** | **21.01** | **52.77%** |
| Atlas (MAG) | **18.62** | 21.18 | **53.08%** |

### Long Context (BABILong)

- Atlas achieves **80%+ accuracy at 10M context length**
- Titans performance drops significantly beyond 1M context
- Attributed to better memory capacity via polynomial features + Muon optimizer

---

## Implementation Checklist

- [ ] Polynomial feature mapping for keys/queries
- [ ] Deep MLP memory with residual connections
- [ ] Newton-Schulz iteration (5 steps recommended)
- [ ] Input-dependent gates (α, η, θ, γ)
- [ ] Key/Query normalization (RMSNorm)
- [ ] Short convolutions on K, Q projections
- [ ] Sliding window mask for Omega gradient
- [ ] Chunk-wise parallel training
- [ ] Hybrid integration with SWA (for MAG variant)

---

## References

- Paper: arXiv:2505.23735
- Muon Optimizer: Jordan et al. 2024
- Titans: Behrouz, Zhong, et al. 2024
- DeltaNet: Schlag et al. 2021
