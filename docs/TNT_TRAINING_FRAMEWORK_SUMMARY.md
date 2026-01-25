# TNT Training Framework for Atlas-MAG

**Paper**: "TNT: Improving Chunkwise Training for Test-Time Memorization" (arXiv:2511.07343)
**Authors**: Zeman Li, Ali Behrouz, Yuan Deng, Peilin Zhong, et al. (Google Research)
**Date**: November 2025

---

## Executive Summary

TNT is a **two-stage training paradigm** that solves the fundamental efficiency-accuracy tradeoff in deep memory modules like Titans and Atlas. It achieves up to **17× training speedup** while **improving model accuracy**, making it the recommended training framework for Atlas-MAG implementations.

The core insight: **different components should process information at different granularities during distinct training stages**.

---

## The Problem TNT Solves

### Why Atlas-MAG Training Is Hard

From the Atlas-MAG architecture (see `ATLAS_MAG_IMPLEMENTATION_SUMMARY.md`), the neural memory module requires:
- Gradient-based weight updates at each token
- Non-linear recurrences (LayerNorm, MLP forward passes)
- Sequential state dependencies

This creates a fundamental conflict:

| Chunk Size | Training Speed | Model Quality |
|------------|----------------|---------------|
| Large (128-512) | Fast (hardware saturated) | Poor (coarse learning signal) |
| Small (8-16) | Slow (5-10% utilization) | Good (fine-grained updates) |

**Current practice**: Fixed small chunk size (16-64 tokens) as a compromise → **prohibitively slow training**.

### Three Fundamental Challenges

#### Challenge 1: Inefficient Training
- Deep memory modules achieve only **5-10% of peak hardware performance**
- Small chunks required for quality don't saturate accelerators
- Cannot use parallel scan tricks (those require linear state transitions)

#### Challenge 2: Compression-Retrieval Mismatch
```
Memory Compression: W_t ← W_{t-1} - η_t ∇_W L(f(W_{t-1}, k_t), v_t)  # Uses KEYS
Memory Retrieval:   o_t = f(W_t, q_t)                                 # Uses QUERIES
```
The memory is trained to map **keys → values**, but queried with **queries** which may lie outside the learned key domain.

#### Challenge 3: Train-Test Chunk Size Sensitivity
Models become **over-specialized** to their training chunk size:
- Model trained with C=64 performs optimally only at C=64 inference
- Smaller chunks at inference (which should be better) actually hurt performance
- This contradicts intuition and limits flexibility

---

## TNT Architecture

### Overview: Two-Stage Training

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TNT Training Paradigm                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ STAGE 1: Efficiency-Focused Pre-training (~95% of compute)        │  │
│  │                                                                    │  │
│  │  • Hierarchical Memory (Global + N Local modules)                 │  │
│  │  • Large chunk sizes for hardware saturation                      │  │
│  │  • Periodic local memory resets → context parallelism             │  │
│  │  • Q-K Projection for retrieval alignment                         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ STAGE 2: Performance-Focused Fine-tuning (~5% of compute)         │  │
│  │                                                                    │  │
│  │  • Smaller local chunk sizes (ideally C_L' = 1)                   │  │
│  │  • Adapts model for high-resolution inference                     │  │
│  │  • Brief training phase, minimal overhead                         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Hierarchical Memory Architecture

```
Sequence L × D ─────────────────────────────────────────────────────────▶ t
    ╱   ╱   ╱   ╱   ╱   ╱   ╱   ╱   ╱   ╱   ╱   ╱   ╱   ╱   ╱   ╱
┌──────────────────────────────────────────────────────────────────────┐
│ GLOBAL MEMORY (W_G)                                                   │
│                                                                       │
│  ├─────────── C_G = 2048 ───────────┤                                │
│  t=0                                 t=1                      t=2 ···│
│  Sequential state carry-over between large chunks                     │
└──────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────────┐
│ LOCAL MEMORY 1 (W_L1)           ◄── Local Memory Window S_L ──►      │
│                                                                       │
│  ├── C_L ──┤                    ├── C_L ──┤                          │
│  t=0  t=1  t=2  ···  RESET      t=0  t=1  t=2  ···  RESET    ···    │
│                         │                             │               │
│                    W_init                        W_init              │
└──────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────────┐
│ LOCAL MEMORY N (W_LN)           ◄── Different resolution ──►         │
│                                                                       │
│  All local memories process in PARALLEL                              │
│  Each can have different chunk sizes: C_L = {8, 16, 32, ...}         │
└──────────────────────────────────────────────────────────────────────┘
```

**Key Innovation**: **Periodic state reset** for local memories breaks sequential dependencies, enabling massive context parallelization.

---

## TNT Update Rules

### Global Memory Update

The global memory evolves sequentially with large chunk size C_G (e.g., 2048):

```python
# Global memory update (Eq. 5 from paper)
V_{(n+1)C_G} = V_{nC_G} - Σ_{t=nC_G}^{(n+1)C_G} η_t ∇_V L(f(V_{nC_G}, k_t), v_t)

# For n = {0, ..., L // C_G}
```

- State carried over sequentially between large chunks
- Gradients computed in parallel within each chunk (standard chunkwise training)
- Captures **long-range dependencies** efficiently

### Local Memory Update

Local memories operate in parallel with periodic resets:

```python
# Local memory update (Eq. 6 from paper)
W_t = {
    W_init                                                      if 0 ≡ t (mod S_L)
    W_{t-1} - Σ_{τ=ξ(t,C_L)}^t η_τ ∇_W L(f(W_{ξ(t,C_L)}, k_τ), v_τ)  otherwise
}

# Where:
# - S_L = local memory window size (shard length)
# - C_L = local memory chunk size
# - ξ(t, C_L) = t - (t mod C_L)  # Start of current chunk
# - W_init = learnable initial state (shared across all shards)
```

**Critical**: At the start of each shard (length S_L), local memory resets to `W_init`. This:
1. Breaks long-range sequential dependencies
2. Enables processing shards as **independent parallel units**
3. Allows massive context parallelism (distribute shards across devices)

### Q-K Projection for Retrieval

Fixes the key-query domain mismatch:

```python
# TNT Memory Retrieval (Eq. 7 from paper)
o_t = f(V_{ξ(t,C_G)}, q_t) + f(W_t, Σ_{τ=ξ(t,C_L)}^t (k_τ k_τ^T / ||k_τ||²) q_t)
     └────────────────────┘   └──────────────────────────────────────────────────┘
        Global: raw query              Local: projected query onto key space
```

**Implementation Efficiency**: The projection matrix `Σ k_τ k_τ^T / ||k_τ||²` can be maintained as a **running sum** (constant-size d×d matrix), updated efficiently via parallel prefix scan.

---

## Stage 2: Fine-Tuning for Inference

After efficient pre-training, a brief fine-tuning phase adapts the model:

```python
# Stage 2 configuration
C_L' < C_L           # Smaller chunk size than Stage 1
# Ideally C_L' = 1   # Per-token updates for maximum resolution
```

**Why this works**:
- Train-test chunk mismatch can be **rectified with minimal overhead**
- Only ~5% additional compute
- Often **surpasses** original performance

**Alignment with Inference**:
- Global memory handles context **prefill** (large chunks)
- Optimized local memory handles iterative **decoding** (small chunks)

---

## Integration with Atlas-MAG

### How TNT Enhances Atlas-MAG Training

From `ATLAS_MAG_IMPLEMENTATION_SUMMARY.md`, Atlas-MAG has:
- Omega Rule (sliding window context memorization)
- Newton-Schulz orthogonalization (Muon optimizer)
- Polynomial feature mapping
- Sliding Window Attention + Neural Memory hybrid

TNT wraps this architecture with:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Atlas-MAG + TNT                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input ──┬──► [TNT Global Memory]                               │
│          │         │                                             │
│          │         │ ◄── Large chunks (C_G=2048)                │
│          │         │ ◄── Atlas Omega Rule                        │
│          │         │ ◄── Muon Optimizer (Newton-Schulz)         │
│          │         ▼                                             │
│          │    Global Output                                      │
│          │         │                                             │
│          └──► [TNT Local Memory 1..N] ◄── Parallel shards       │
│                    │                                             │
│                    │ ◄── Smaller chunks (C_L={8,16,...})        │
│                    │ ◄── Periodic resets to W_init              │
│                    │ ◄── Q-K Projection                         │
│                    ▼                                             │
│               Local Output                                       │
│                    │                                             │
│          ┌────────┴────────┐                                    │
│          ▼                 ▼                                     │
│    [Sliding Window    [Gated                                    │
│     Attention]         Combination]                             │
│          │                 │                                     │
│          └────────┬────────┘                                    │
│                   ▼                                              │
│              [SwiGLU MLP]                                        │
│                   │                                              │
│                   ▼                                              │
│               Output                                             │
└─────────────────────────────────────────────────────────────────┘
```

### Configuration Mapping

| Atlas-MAG Component | TNT Enhancement |
|---------------------|-----------------|
| Neural Memory Module | Split into Global + Local hierarchy |
| Omega Rule (context window c) | Applied within each memory module |
| Muon Optimizer | Can be used as inner-loop optimizer |
| Polynomial Features | Applied to keys in both global/local |
| SWA | Remains unchanged (parallel branch) |

---

## Training Configuration

### Recommended Settings (from TNT paper)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Global chunk size (C_G) | 2048 | Hardware-saturating |
| Local chunk sizes (C_L) | {8, 16} or {4, 8, 16, 32} | Multi-resolution |
| Local window size (S_L) | 2048-4096 | Shard length for parallelism |
| Stage 2 chunk sizes | {1} or {2, 4} or {2, 4, 8, 16} | Fine-grained |
| Stage 2 compute | ~5% of Stage 1 | Minimal overhead |

### Training Time Comparison (150M params, 10B tokens)

| Model | Chunk Config | Training Time | Speedup |
|-------|--------------|---------------|---------|
| Titans (baseline) | C=8 | 19.48 hrs | 1.00× |
| Titans | C=64 | 4.18 hrs | 4.67× |
| TNT Stage 1 | C_L={8} | 2.54 hrs | **7.68×** |
| TNT Stage 1 | C_L={64} | 1.12 hrs | **17.37×** |
| TNT Stage 2 | C_L'={1} | +0.15 hrs | - |

---

## Performance Results

### Language Modeling (150M params, 10B tokens)

| Model | Config | Avg PPL ↓ | Avg Acc ↑ |
|-------|--------|-----------|-----------|
| Transformer (w gating) | - | 22.39 | 39.7% |
| Titans | C=8 | 25.07 | 39.0% |
| **TNT Stage 1** | {4,8,16,32} | **23.13** | 40.6% |
| **TNT Stage 2** | {2,4,8,16} | **23.09** | **40.9%** |

### Key Findings

1. **TNT Stage 1 alone** beats Titans and vanilla Transformers
2. **Stage 2 fine-tuning** provides additional gains for ~5% compute overhead
3. **Multi-resolution locals** (multiple C_L values) improve over single resolution
4. **Q-K Projection is essential**: Removing it increases PPL from 21.04 → 22.01
5. **Global memory is critical**: Removing it increases PPL from 21.04 → 25.60

---

## Implementation Pseudocode

### TNT Forward Pass (Stage 1)

```python
class TNTAtlasMAG(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        global_chunk_size: int = 2048,
        local_chunk_sizes: list = [8, 16],  # Multiple resolutions
        local_window_size: int = 2048,
    ):
        super().__init__()
        self.C_G = global_chunk_size
        self.C_L = local_chunk_sizes
        self.S_L = local_window_size
        self.N_local = len(local_chunk_sizes)

        # Global memory (Atlas-style)
        self.global_memory = AtlasMemory(dim)

        # N Local memories (one per chunk size)
        self.local_memories = nn.ModuleList([
            AtlasMemory(dim) for _ in range(self.N_local)
        ])

        # Learnable initial states for each local memory
        self.W_init = nn.ParameterList([
            nn.Parameter(torch.zeros(dim, dim))
            for _ in range(self.N_local)
        ])

        # Projections
        self.W_k = nn.Linear(dim, dim)
        self.W_q = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)

        # SWA for MAG architecture
        self.swa = SlidingWindowAttention(dim, num_heads)

    def forward(self, x, global_state=None, local_states=None):
        B, L, D = x.shape

        # Project to K, Q, V
        k = self.W_k(x)
        q = self.W_q(x)
        v = self.W_v(x)

        # ========== Global Memory (sequential over large chunks) ==========
        global_outputs = []
        if global_state is None:
            global_state = self.global_memory.init_state(B)

        for chunk_start in range(0, L, self.C_G):
            chunk_end = min(chunk_start + self.C_G, L)
            k_chunk = k[:, chunk_start:chunk_end]
            v_chunk = v[:, chunk_start:chunk_end]
            q_chunk = q[:, chunk_start:chunk_end]

            # Parallel gradient computation within chunk
            # Memory update (chunkwise)
            global_state = self.update_memory_chunkwise(
                self.global_memory, global_state, k_chunk, v_chunk
            )

            # Retrieval with raw query
            global_out = self.global_memory.retrieve(global_state, q_chunk)
            global_outputs.append(global_out)

        global_output = torch.cat(global_outputs, dim=1)

        # ========== Local Memories (parallel over shards) ==========
        local_outputs = []

        for i, (local_mem, C_L_i, W_init_i) in enumerate(
            zip(self.local_memories, self.C_L, self.W_init)
        ):
            # Split sequence into parallel shards
            shards = self.split_into_shards(k, q, v, self.S_L)

            shard_outputs = []
            for shard_k, shard_q, shard_v in shards:
                # Reset to learnable initial state
                local_state = W_init_i.unsqueeze(0).expand(B, -1, -1)

                # Q-K projection matrix (running sum)
                M_proj = torch.zeros(B, D, D, device=x.device)

                for chunk_start in range(0, self.S_L, C_L_i):
                    chunk_end = min(chunk_start + C_L_i, self.S_L)
                    k_chunk = shard_k[:, chunk_start:chunk_end]
                    v_chunk = shard_v[:, chunk_start:chunk_end]
                    q_chunk = shard_q[:, chunk_start:chunk_end]

                    # Update memory (chunkwise parallel)
                    local_state = self.update_memory_chunkwise(
                        local_mem, local_state, k_chunk, v_chunk
                    )

                    # Update Q-K projection matrix
                    M_proj = self.update_qk_projection(M_proj, k_chunk)

                    # Project query onto key space
                    q_projected = torch.bmm(M_proj, q_chunk.transpose(-1, -2)).transpose(-1, -2)

                    # Retrieval with projected query
                    local_out = local_mem.retrieve(local_state, q_projected)
                    shard_outputs.append(local_out)

            local_outputs.append(torch.cat(shard_outputs, dim=1))

        # Sum all local outputs
        local_output = sum(local_outputs)

        # ========== Combine Global + Local ==========
        memory_output = global_output + local_output

        # ========== MAG: Gate with SWA ==========
        swa_output = self.swa(x)
        output = swa_output * torch.sigmoid(memory_output)  # Gated combination

        return output, global_state, local_states

    def update_qk_projection(self, M, k_chunk):
        """Update running Q-K projection matrix."""
        # M_t = M_{t-1} + Σ k_τ k_τ^T / ||k_τ||²
        k_norm = k_chunk / (k_chunk.norm(dim=-1, keepdim=True) + 1e-6)
        M_update = torch.einsum('bld,ble->bde', k_norm, k_norm)
        return M + M_update
```

### Stage 2 Fine-Tuning

```python
def stage2_finetune(model, dataloader, num_steps=1000):
    """
    Fine-tune with smaller local chunk sizes.
    Only ~5% of Stage 1 compute.
    """
    # Reduce local chunk sizes
    original_C_L = model.C_L.copy()
    model.C_L = [max(1, c // 4) for c in original_C_L]  # e.g., {8,16} → {2,4}

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break

        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return model
```

---

## Summary: TNT Benefits for Atlas-MAG

| Aspect | Without TNT | With TNT |
|--------|-------------|----------|
| Training Speed | 19.48 hrs (C=8) | 1.12 hrs (C_L={64}) |
| Speedup | 1× | **17.37×** |
| PPL (150M/10B) | 25.07 | **23.09** |
| Accuracy | 39.0% | **40.9%** |
| Hardware Utilization | 5-10% | Near-optimal |
| Inference Flexibility | Fixed chunk size | Adaptable (Stage 2) |

**Key Takeaways**:
1. TNT is **essential** for practical Atlas-MAG training
2. Two-stage approach **decouples** efficiency from accuracy
3. Hierarchical memory captures **both** long-range and fine-grained patterns
4. Q-K projection fixes domain mismatch (measurable accuracy gain)
5. Stage 2 fine-tuning is cheap (~5%) but impactful

---

## References

- TNT Paper: arXiv:2511.07343
- Titans Paper: arXiv:2501.00663 (see `TITANS_MAG_SUMMARY.md`)
- Atlas Paper: arXiv:2505.23735 (see `ATLAS_MAG_IMPLEMENTATION_SUMMARY.md`)
- TTT (Test-Time Training): arXiv:2407.04620
