# Memory Architecture Fix: TNT Q-K Projection

## Implementation Status: ✅ COMPLETE

**Date**: 2026-01-25

### Changes Made
1. **`src/model/qk_projection.py`**: Added `CausalQKMemoryProjection` class
   - Causally accumulates key outer products: `M_t = M_persistent + Σ(k_i ⊗ k_i)`
   - Projects queries through accumulated memory: `q'_t = (M_t @ q_t) / norm_sum_t`

2. **`src/model/skeleton.py`**:
   - `AtlasMAGBlock` now accepts `persistent_memory` parameter
   - Added `qk_memory = CausalQKMemoryProjection(...)` in each block
   - Gate initialization changed from 0.0 to -2.0 (sigmoid ≈ 0.12)
   - Forward pass modified to: `q = (1 - gate) * q + gate * q_projected`

3. **`AtlasMAGSkeleton`**:
   - Now passes `self.persistent_memory` to each block

### Verification
- ✅ 209/209 tests passing
- ✅ mypy type check clean
- ✅ Forward pass produces finite outputs
- ✅ Gates start at 0.12 (favor attention initially)
- ✅ Each block's `qk_memory` references shared `m_persistent`

### Next Steps
- Start new training run with fixed memory on GPU0
- Compare with attention-only baseline on GPU1
- If Option A fails: Fall back to Option B (fixed AtlasMemory)

---

## Problem Statement

The current implementation has two disconnected memory systems:
- `PersistentMemory` (TNT-style) - created but never used
- `AtlasMemory` (gated MLP) - used but provides noise

This causes gates to stay at ~0.4 (model prefers attention over noise).

## Solution: Implement TNT Q-K Projection

### The Formula (from TNT paper)

```
M_t = M_persistent + Σ(k_i ⊗ k_i)   # Accumulate key outer products
norm_sum_t = norm_persistent + Σ||k_i||²  # Accumulate norms

q'_t = (M_t @ q_t) / norm_sum_t     # Project query through memory
```

### How It Works

1. **M_persistent** = precomputed from 64 persistent keys (provides "background knowledge")
2. **During forward pass**: accumulate each key's outer product into M_t
3. **For each query**: project through accumulated memory to retrieve relevant info
4. **Gate decides**: use projected query (memory) vs original query (attention)

### Architecture Changes

```
BEFORE (broken):
┌─────────────────────────────────────────────────────────────┐
│ AtlasMAGBlock                                               │
│                                                             │
│   x ──► norm1 ──┬──► Attention(Q,K,V) ────────┬──► + ──► x │
│                 │                              │            │
│                 └──► AtlasMemory(MLP) ──► gate─┘            │
│                      (random noise!)                        │
└─────────────────────────────────────────────────────────────┘

AFTER (fixed):
┌─────────────────────────────────────────────────────────────┐
│ AtlasMAGBlock                                               │
│                                                             │
│   x ──► norm1 ──► QKV projection ──► Q, K, V               │
│                                        │                    │
│                   ┌────────────────────┘                    │
│                   ▼                                         │
│              Q ──┬──► Attention(Q, K, V) ────────┬──► + ──► │
│                  │                                │         │
│                  └──► QKProjection(Q, M_t) ──► gate─┘       │
│                       (M_t from PersistentMemory)           │
└─────────────────────────────────────────────────────────────┘
```

### Key Insight: Gate Controls Q vs Q'

```python
# The gate now has a meaningful choice:
Q_original = Q                           # Pure attention (no memory)
Q_projected = (M_t @ Q) / norm_sum       # Memory-enhanced

# Gate blends them:
Q_final = (1 - gate) * Q_original + gate * Q_projected

# Then attention uses Q_final:
attn_weights = softmax(Q_final @ K^T / sqrt(d))
```

## Implementation Plan

### File Changes

1. **`src/model/qk_projection.py`** (NEW)
   - `QKMemoryProjection` class
   - Handles M_t accumulation and Q projection
   - Integrates with PersistentMemory

2. **`src/model/skeleton.py`** (MODIFY)
   - `AtlasMAGBlock`: Replace AtlasMemory with QKMemoryProjection
   - Connect PersistentMemory to blocks
   - Gate now controls Q vs Q' blending

3. **`src/model/atlas_memory.py`** (KEEP as Option B backup)
   - Don't delete, just don't use for now

4. **`scripts/calibrate_persistent_keys.py`** (NEW)
   - Initialize persistent_keys from real data
   - Compute M_persistent and norm_persistent
   - Save calibrated values

### QKMemoryProjection Class

```python
class QKMemoryProjection(nn.Module):
    """
    TNT-style Q-K projection through accumulated memory.

    M_t = M_persistent + Σ(k_i ⊗ k_i)
    q'_t = (M_t @ q_t) / norm_sum_t
    """

    def __init__(self, dim: int, persistent_memory: PersistentMemory):
        super().__init__()
        self.dim = dim
        self.persistent_memory = persistent_memory

    def forward(self, q: Tensor, k: Tensor) -> Tensor:
        """
        Project queries through accumulated key memory.

        Args:
            q: Query tensor (batch, n_heads, seq_len, head_dim)
            k: Key tensor (batch, n_heads, seq_len, head_dim)

        Returns:
            Projected queries (batch, n_heads, seq_len, head_dim)
        """
        batch, n_heads, seq_len, head_dim = q.shape

        # Start with persistent memory
        M_t = self.persistent_memory.m_persistent.clone()  # (dim, dim)
        norm_sum = self.persistent_memory.norm_persistent

        # Reshape for projection (combine heads for now)
        q_flat = q.transpose(1, 2).reshape(batch, seq_len, -1)  # (B, L, dim)
        k_flat = k.transpose(1, 2).reshape(batch, seq_len, -1)  # (B, L, dim)

        # Accumulate and project causally
        q_projected = []
        for t in range(seq_len):
            # Project current query through accumulated memory
            q_t = q_flat[:, t, :]  # (batch, dim)
            q_prime = (M_t @ q_t.T).T / norm_sum  # (batch, dim)
            q_projected.append(q_prime)

            # Accumulate current key into memory (for future positions)
            k_t = k_flat[:, t, :]  # (batch, dim)
            # M_t += k_t ⊗ k_t (outer product, averaged over batch)
            M_t = M_t + torch.einsum('bd,be->de', k_t, k_t) / batch
            norm_sum = norm_sum + (k_t.norm(dim=-1) ** 2).mean().item()

        # Stack and reshape back to head format
        q_proj = torch.stack(q_projected, dim=1)  # (batch, seq_len, dim)
        q_proj = q_proj.reshape(batch, seq_len, n_heads, head_dim).transpose(1, 2)

        return q_proj
```

### Modified AtlasMAGBlock

```python
class AtlasMAGBlock(nn.Module):
    def __init__(self, dim, n_heads, persistent_memory):
        # ... existing init ...

        # Replace AtlasMemory with QKMemoryProjection
        self.qk_memory = QKMemoryProjection(dim, persistent_memory)

        # Gate now controls Q vs Q' blending
        self.memory_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, attention_mask=None):
        h = self.norm1(x)
        q, k, v = self.qkv(h, reshape_heads=True)
        q, k = self.rope(q, k)

        # Project Q through memory
        q_projected = self.qk_memory(q, k)

        # Gate blends original Q with projected Q
        gate = torch.sigmoid(self.memory_gate)
        q_final = (1 - gate) * q + gate * q_projected

        # Attention with blended Q
        attn = scaled_dot_product_attention(q_final, k, v, attention_mask)
        # ... rest of forward ...
```

## Persistent Key Initialization

The persistent keys should be initialized from real data, not random:

```python
def calibrate_persistent_keys(dataloader, tokenizer, n_samples=10000):
    """
    Initialize persistent keys from real data embeddings.

    Strategy: Use k-means clustering on key vectors from a sample of data.
    The 64 cluster centers become the persistent keys.
    """
    all_keys = []

    # Collect key vectors from real data
    for batch in dataloader:
        with torch.no_grad():
            # Get key projections from a trained/initialized model
            keys = model.get_key_vectors(batch['input_ids'])
            all_keys.append(keys)

        if len(all_keys) * batch_size >= n_samples:
            break

    # Cluster to find 64 representative keys
    all_keys = torch.cat(all_keys, dim=0)
    kmeans = KMeans(n_clusters=64)
    kmeans.fit(all_keys.cpu().numpy())

    persistent_keys = torch.tensor(kmeans.cluster_centers_)
    return persistent_keys
```

## Testing Plan

1. **Unit tests**: QKMemoryProjection produces correct shapes
2. **Integration test**: Full forward pass works
3. **Gradient test**: Gradients flow through memory gate
4. **Sanity check**: Gate learns to move (not stuck at 0.5)

## Fallback: Option B

If Option A doesn't work (gates still stuck, no improvement):

1. Revert to AtlasMemory (gated MLP)
2. Initialize gate at -3.0 (sigmoid ≈ 0.05, favor attention initially)
3. Initialize MLP weights to near-identity
4. Increase polarization λ to 50-100

## Success Criteria

- [ ] Gate std > 0.1 after 1000 steps (gates are learning)
- [ ] Some gates < 0.2 or > 0.8 (polarization happening)
- [ ] Memory+Attention performs ≥ Attention-only (memory helps, not hurts)
- [ ] Validation PPL improves vs baseline
