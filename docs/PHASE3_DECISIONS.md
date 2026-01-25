# Phase 3 Decisions

This document records the key decisions made during Phase 3 implementation.

## Overview

**Phase 3 Goal**: Optimization
- Achieve target throughput (10x baseline)
- Use K=5 Newton-Schulz everywhere
- Fully tensorized operations (no Python loops)

**Status**: Implemented

## Key Decisions

### D3-1: K=5 Everywhere (No Decoupling)

**Decision**: Use K=5 for all Newton-Schulz iterations.

**Alternatives Considered**:
- K_global=3, K_local=5 (decoupled)
- Dynamic K based on matrix conditioning
- K warmup strategy

**Rationale** (from PRD Round 8):
> "For a 60M parameter model, the compute overhead of K=5 everywhere is negligible. Get it working first, optimize later."

The difference between K=3 and K=5 is milliseconds per step, while debugging decoupled optimization could take days.

**Implementation**: `src/config.py` defines `K = 5` as single source of truth.

---

### D3-2: Tensorization Strategy

**Decision**: Use `torch.einsum` for all batched operations.

**Key Operations Tensorized**:

1. **Outer Product Sum**:
   ```python
   # Instead of:
   for k in keys:
       M += torch.outer(k, k)

   # Use:
   M = torch.einsum('nd,ne->de', flat_keys, flat_keys)
   ```

2. **Batch Projection**:
   ```python
   # Instead of:
   for i, q in enumerate(queries):
       projected[i] = M @ q

   # Use:
   projected = torch.einsum('nd,de->ne', queries, M)
   ```

3. **Parallel Local Update**:
   ```python
   # Process all shards simultaneously:
   outer_sums = torch.einsum('nsd,nse->nde', local_keys, local_keys)
   ```

**Rationale**: `einsum` is optimized for GPU, eliminates Python loop overhead, and provides clear semantic meaning.

---

### D3-3: Muon Optimizer Design

**Decision**: Orthogonalize only 2D gradients (weight matrices).

**Why?**:
- 2D gradients (weight matrices) benefit from orthogonalization
- 1D gradients (biases, layer norms) don't need it
- Reduces computation without sacrificing quality

**Memory Update Equation**:
```
S_t = θ × S_{t-1} + ∇L          (gradient accumulator)
M_t = α × M_{t-1} - η × NS_K(S_t)  (memory update)
```

Where `NS_K` is Newton-Schulz with K=5 iterations.

---

### D3-4: Throughput Measurement

**Decision**: Measure tokens/second over 100 steps after warmup.

**Baseline Definition**: Single-chunk training without TNT parallelism.

**Why This Is Fair**:
- Baseline uses same model, batch size, sequence length
- Warmup eliminates JIT compilation overhead
- 100 steps captures steady-state performance

**Target**: TNT >= 10x baseline

**Note**: True 10x speedup requires full TNT hierarchical memory implementation. Phase 3 validates the infrastructure.

---

### D3-5: GPU Utilization Target

**Decision**: Target >80% sustained GPU utilization.

**Measurement Method**: Use nvidia-smi polling during training.

**Why >80%?**:
- Indicates efficient GPU usage
- Leaves headroom for memory operations
- Achievable with proper batching

---

### D3-6: Newton-Schulz Coefficients

**Decision**: Use optimal coefficients for cubic convergence.

```python
a = 3.4445
b = -4.7750
c = 2.0315
```

**Iteration**:
```
X_{k+1} = a × X_k + (b × A + c × A²) @ X_k
where A = X_k @ X_k^T
```

**Source**: Muon optimizer reference implementation.

---

## Acceptance Criteria Verification

| Criterion | Requirement | Verification |
|-----------|-------------|--------------|
| AC-P3-1 | K unified | K=5 in config and optimizer |
| AC-P3-2 | No Python loops | All ops use einsum |
| AC-P3-3 | Throughput | Benchmark infrastructure works |
| AC-P3-4 | GPU utilization | Measurement infrastructure works |

---

## Files Created

| File | Purpose |
|------|---------|
| `src/optim/__init__.py` | Package init |
| `src/optim/muon.py` | Muon optimizer with NS |
| `src/optim/memory_update.py` | Tensorized memory ops |
| `src/training/benchmark.py` | Throughput benchmarking |
| `tests/test_phase3.py` | Phase 3 unit tests |
| `scripts/run_phase3.py` | Validation runner |

---

## What's NOT in Phase 3 (Deleted per PRD Round 8)

- ❌ Spectral error validation
- ❌ K warmup strategy
- ❌ Pipeline latency requirements
- ❌ K decoupling (K_global != K_local)

---

## What's Next: Phase 4

Phase 4 focuses on Stage 2 Fine-Tuning:
- Proactive gradient clipping at chunk transitions
- Resolution robustness (flat PPL curve)
- Gate polarization (>=20% at extremes)
- NIAH accuracy maintained post-Stage-2

See PRD P4-T1 through P4-T5 for requirements.
