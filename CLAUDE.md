# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AtlasMAG-TNT** is a proof-of-concept implementation of a 60M parameter hybrid sequence model combining:
- **Atlas-MAG**: Memory-as-Gate architecture with neural long-term memory + sliding window attention
- **TNT**: Titans-in-Titans two-stage training framework for 17× speedup

**Current State**: Pre-implementation (design documents only, no code yet)

**Target Hardware**:
- 2× RTX A6000 48GB with NVLink (96GB total VRAM)
- RTX 2000 16GB for validation
- Threadripper 7960x, 256GB ECC RAM

## Architecture Summary

```
Input → [Persistent Memory Tokens] → ┬→ [Sliding Window Attention] ─┐
                                     │                              │
                                     └→ [Atlas Neural Memory] ──────┤
                                              ↓                      │
                                         [Gate σ()]                  │
                                              ↓                      │
                                     output = SWA × σ(Atlas) ←──────┘
```

### TNT Hierarchical Memory

```
Global Memory (C_G=2048): Long-range dependencies, sequential state
     │
     ├── Local Memory 1 (C_L={8,16,...}): Parallel shards, periodic resets
     ├── Local Memory 2
     └── Local Memory N
```

## Key Implementation Files (Planned)

Based on PRD phases:
- `src/model/atlas_memory.py` - Gated MLP memory with polynomial features
- `src/model/mag_block.py` - Memory-as-Gate block with SWA
- `src/model/tnt_wrapper.py` - TNT hierarchical memory (global + local)
- `src/training/stage1.py` - Efficient pre-training (95% compute)
- `src/training/stage2.py` - Fine-tuning with smaller chunks (5% compute)
- `src/utils/newton_schulz.py` - Muon optimizer (K=5)
- `src/utils/probes.py` - NIAH retrieval probes

## Critical Constants (From PRD)

```python
# Architecture (decisions required in Phase 1)
D = 768                    # Model dimension
L_M = 2 or 3               # Memory depth (Phase 1 decision)
POLY_DEGREE = 2            # Polynomial feature degree (likely)
N_PERSISTENT = 64          # Persistent memory tokens

# TNT Training
C_G = 2048                 # Global chunk size
C_L = [8, 16, 32, 64]      # Local chunk sizes (multi-resolution)
S_L = 2048                 # Local shard length
K = 5                      # Newton-Schulz iterations (unified)

# Stage 2 Fine-tuning
GAMMA = 0.9993             # Geometric decay for chunk size
STAGE2_RATIO = 0.05        # ~5% of Stage 1 compute

# Gate Polarization
LAMBDA_INITIAL = 10.0      # First 10% of training
LAMBDA_FINAL = 0.1         # After annealing
```

## Implementation Phases

| Phase | Goal | Key Deliverable |
|-------|------|-----------------|
| 0 | Foundation | W_init, M_persistent, norm_persistent |
| 1 | Architecture | Gate polarization, polynomial features |
| 2 | Training | NIAH probes, Q-K projection with normalization |
| 3 | Optimization | K=5 Newton-Schulz, ≥10× throughput |
| 4 | Stage 2 | Dynamic gradient clipping, resolution robustness |
| 5 | Inference | Prefill/decode modes, no resets |

**Philosophy**: Phases are **diagnostic checkpoints**, not blockers. Log gaps, keep building.

## Key Equations

### Atlas Memory Update (Muon Optimizer)
```
M_t = α_t * M_{t-1} - η_t * NewtonSchulz_5(S_t)
S_t = θ_t * S_{t-1} + ∇[Ω-rule loss]
```

### Gate Polarization Loss
```
L_polar = λ(t) × (1 - |2g - 1|)
```
Maximum penalty at g=0.5, zero at g∈{0,1}

### Q-K Projection (TNT)
```
q'_t = M_t @ q_t / norm_sum_t
M_t = M_persistent + Σ(k_i @ k_i^T)
norm_sum_t = norm_persistent + Σ||k_i||²
```

### Dynamic Gradient Clip (Stage 2)
```
κ(t) = κ_base × √(32 / C_L(t))
```
Where κ_base = 0.5, scales with chunk size

## Validation Metrics

| Metric | Target | Phase |
|--------|--------|-------|
| NIAH retrieval accuracy | >80% | Phase 2+ |
| Gate polarization | ≥20% tokens at <0.1 or >0.9 | Phase 4 |
| Throughput | ≥10× baseline | Phase 3 |
| PPL resolution sensitivity | Flat curve {1-64} | Phase 4 |
| Reset shock | <5% loss spike | Phase 0 |

## Fast-Fail Checks

```python
# Step 500: Abort if gates stuck in mushy middle
if step == 500 and gate_variance < initial_variance * 1.5:
    raise FastFailError("Gates not learning - abort run")

# Continuous: Abort if gates collapse
if step > 100 and gate_values.std() < 0.01:
    raise FastFailError("Gate std < 0.01 - architecture broken")
```

## Reference Documentation

- `PRD_Atlas_MAG_TNT.md` - Complete implementation specification (v5.4)
- `docs/TITANS_MAG_SUMMARY.md` - Titans paper summary (arXiv:2501.00663)
- `docs/ATLAS_MAG_IMPLEMENTATION_SUMMARY.md` - Atlas implementation guide
- `docs/TNT_TRAINING_FRAMEWORK_SUMMARY.md` - TNT training details

## Common Patterns

### Newton-Schulz Orthogonalization
```python
def newton_schulz(G, num_iters=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G
    for _ in range(num_iters):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X
```

### Polynomial Feature Mapping (Degree 2)
```python
def polynomial_features(x):
    # x: [B, L, d] → [B, L, d + d*(d+1)/2]
    x_outer = x.unsqueeze(-1) * x.unsqueeze(-2)
    triu_idx = torch.triu_indices(x.size(-1), x.size(-1))
    return torch.cat([x, x_outer[..., triu_idx[0], triu_idx[1]]], dim=-1)
```

### M_persistent Initialization
```python
# Compute at runtime on rank 0, broadcast via NCCL
if rank == 0:
    norm_persistent = sum(torch.linalg.norm(k_p)**2 for k_p in persistent_keys)
dist.broadcast(norm_persistent, src=0)
```

## What NOT To Do

- ❌ Static gradient clip throughout Stage 2 (use dynamic κ(t))
- ❌ Option B (high-variance init) for gates (use annealing λ)
- ❌ Decoupled K values (K=5 everywhere for this POC)
- ❌ Parallel A/B tests splitting GPU capacity
- ❌ Defense-grade blocking gates (research-grade: log and continue)
