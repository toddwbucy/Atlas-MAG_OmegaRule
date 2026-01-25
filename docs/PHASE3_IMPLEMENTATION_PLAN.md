# Phase 3: Optimization - Implementation Plan

## Goal
Achieve target throughput (≥10× baseline) with K=5 Newton-Schulz everywhere.

## Status
- ✅ **Phase 0 Complete**: 46/46 tests passing
- ✅ **Phase 1 Complete**: 6/6 checkpoints passing
- ✅ **Phase 2 Complete**: 5/5 checkpoints passing
- 🚧 **Phase 3 In Progress**: Optimization

## PRD Requirements (P3-T1 through P3-T3)

| Task | Description | Status |
|------|-------------|--------|
| P3-T1 | Use K=5 for all memory modules | ✅ Already implemented |
| P3-T2 | Implement tensorized Muon updates | 🔲 To implement |
| P3-T3 | Measure throughput vs baseline | 🔲 To implement |

## What Already Exists

### Newton-Schulz Implementation (`src/nn/newton_schulz.py`)
- ✅ `newton_schulz(G, num_iters=5)` - Standard orthogonalization
- ✅ `newton_schulz_batched(G, num_iters=5)` - Batched version
- ✅ `orthogonality_error(X)` - Convergence validation
- ✅ Optimal coefficients: a=3.4445, b=-4.7750, c=2.0315
- ✅ Unit tests in `tests/test_nn_blocks.py`

### Constants (`src/config.py`)
- ✅ `K = 5` - Newton-Schulz iterations (unified)
- ✅ `ValidationTargets.throughput_multiplier = 10.0`

---

## New Files to Create

```
src/
├── optim/                        # NEW DIRECTORY
│   ├── __init__.py
│   ├── muon.py                   # Muon optimizer with NS orthogonalization
│   └── memory_update.py          # Tensorized memory updates

src/training/
└── benchmark.py                  # Throughput benchmarking utilities

tests/
└── test_phase3.py                # Phase 3 unit tests

scripts/
└── run_phase3.py                 # Phase 3 validation runner

docs/
└── PHASE3_DECISIONS.md           # Decision documentation
```

---

## Implementation Steps

### Step 1: Muon Optimizer (`src/optim/muon.py`)

The Muon optimizer orthogonalizes gradients before applying updates:

```python
class Muon(Optimizer):
    """
    Muon optimizer with Newton-Schulz orthogonalization.

    Uses K=5 Newton-Schulz iterations to orthogonalize gradients
    before applying the update. This improves conditioning.

    Memory update equation:
        M_t = α_t × M_{t-1} - η_t × NewtonSchulz_K(∇L)

    Args:
        params: Model parameters
        lr: Learning rate (default: 1e-3)
        momentum: Momentum coefficient α (default: 0.95)
        k: Newton-Schulz iterations (default: 5)
        weight_decay: L2 penalty (default: 0.0)
    """

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Orthogonalize 2D gradients (weight matrices)
                if grad.ndim == 2:
                    grad = newton_schulz(grad, num_iters=group['k'])

                # Apply momentum and update
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)

                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(grad)

                p.data.add_(buf, alpha=-group['lr'])
```

Key design decisions:
- Only orthogonalize 2D gradients (weight matrices)
- 1D gradients (biases, layer norms) use standard update
- K=5 everywhere (no decoupling)

---

### Step 2: Tensorized Memory Updates (`src/optim/memory_update.py`)

The critical requirement is **no Python loops over tokens**. All operations must be batched.

```python
def tensorized_memory_update(
    M_t: Tensor,           # (dim, dim) memory matrix
    keys: Tensor,          # (batch, seq_len, dim) key vectors
    norm_sum: float,       # Running norm accumulator
    alpha: float = 0.95,   # Memory decay
) -> Tuple[Tensor, float]:
    """
    Tensorized memory update - NO Python loops.

    Updates M_t with a batch of keys in a single operation:
        M_t = α × M_t + Σ(k_i ⊗ k_i)
        norm_sum += Σ||k_i||²

    Args:
        M_t: Current memory matrix
        keys: Batch of key vectors
        norm_sum: Current norm accumulator
        alpha: Memory decay factor

    Returns:
        Updated (M_t, norm_sum)
    """
    # Flatten batch and sequence dimensions
    # keys: (batch, seq_len, dim) -> (batch * seq_len, dim)
    flat_keys = keys.reshape(-1, keys.size(-1))

    # Compute sum of outer products in ONE operation
    # (n, dim, 1) @ (n, 1, dim) -> (n, dim, dim) -> sum -> (dim, dim)
    outer_sum = torch.einsum('nd,ne->de', flat_keys, flat_keys)

    # Update memory matrix
    M_new = alpha * M_t + outer_sum

    # Update norm accumulator
    norm_delta = (flat_keys.norm(dim=-1) ** 2).sum().item()
    norm_new = norm_sum + norm_delta

    return M_new, norm_new


def batch_qk_projection(
    M_t: Tensor,        # (dim, dim)
    queries: Tensor,    # (batch, seq_len, dim)
    norm_sum: float,
) -> Tensor:
    """
    Batch Q-K projection - NO Python loops.

    Projects all queries through the accumulated memory:
        q' = M_t @ q / norm_sum

    Returns:
        Projected queries (batch, seq_len, dim)
    """
    # queries @ M_t.T is more efficient than loop over tokens
    # Shape: (batch, seq_len, dim) @ (dim, dim) -> (batch, seq_len, dim)
    projected = torch.einsum('bsd,de->bse', queries, M_t)

    return projected / max(norm_sum, 1e-8)
```

The key insight: use `torch.einsum` for batched outer products instead of loops.

---

### Step 3: Throughput Benchmarking (`src/training/benchmark.py`)

```python
@dataclass
class ThroughputResult:
    """Result from throughput benchmark."""
    tokens_per_second: float
    samples_per_second: float
    gpu_utilization: float
    memory_allocated_gb: float
    vs_baseline: float  # Multiplier vs baseline

    @property
    def passes_target(self) -> bool:
        return self.vs_baseline >= 10.0


class ThroughputBenchmark:
    """
    Benchmark training throughput.

    Measures tokens/second for:
    1. Baseline: Standard single-chunk training
    2. TNT: With local memory parallelism

    Target: TNT ≥ 10× baseline
    """

    def __init__(self, model, batch_size: int, seq_len: int, device: str):
        self.model = model
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device

    def measure_baseline(self, num_steps: int = 100) -> ThroughputResult:
        """Measure baseline (single-chunk) throughput."""
        ...

    def measure_tnt(self, num_steps: int = 100) -> ThroughputResult:
        """Measure TNT (parallel local) throughput."""
        ...

    def run_comparison(self, num_steps: int = 100) -> dict:
        """Run full baseline vs TNT comparison."""
        baseline = self.measure_baseline(num_steps)
        tnt = self.measure_tnt(num_steps)

        return {
            "baseline": baseline,
            "tnt": tnt,
            "speedup": tnt.tokens_per_second / baseline.tokens_per_second,
            "passes_target": tnt.vs_baseline >= 10.0,
        }
```

---

### Step 4: GPU Utilization Monitoring

```python
def get_gpu_utilization() -> float:
    """Get current GPU utilization percentage."""
    if not torch.cuda.is_available():
        return 0.0

    # Use nvidia-smi for accurate utilization
    import subprocess
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        return float(result.stdout.strip().split('\n')[0])
    return 0.0


def monitor_gpu_during_training(
    training_fn: Callable,
    sample_interval: float = 0.5,
) -> Tuple[Any, List[float]]:
    """
    Monitor GPU utilization during training.

    Returns:
        (training_result, utilization_samples)
    """
    ...
```

---

### Step 5: Phase 3 Tests (`tests/test_phase3.py`)

**G3-1: K=5 Used Everywhere**
```python
def test_k_unified():
    """K=5 should be used for all Newton-Schulz calls."""
    from src.config import K
    assert K == 5

    # Verify optimizer uses K=5
    optimizer = Muon(model.parameters())
    assert optimizer.defaults['k'] == 5
```

**G3-2: No Python Loops Over Tokens**
```python
def test_no_python_loops():
    """Memory updates should be fully tensorized."""
    import torch.profiler

    keys = torch.randn(4, 64, 768)
    M_t = torch.zeros(768, 768)

    # Profile the operation
    with torch.profiler.profile() as prof:
        M_new, _ = tensorized_memory_update(M_t, keys, 0.0)

    # Verify no Python for loops in trace
    # (Check for batched matmul operations)
    ...
```

**G3-3: Throughput ≥10× Baseline**
```python
def test_throughput_target():
    """TNT should achieve ≥10× baseline throughput."""
    benchmark = ThroughputBenchmark(model, batch_size=4, seq_len=512)
    result = benchmark.run_comparison(num_steps=50)

    assert result["speedup"] >= 10.0
```

**G3-4: GPU Utilization >80%**
```python
def test_gpu_utilization():
    """Training should maintain >80% GPU utilization."""
    result, samples = monitor_gpu_during_training(train_step, sample_interval=0.1)
    mean_utilization = sum(samples) / len(samples)

    assert mean_utilization > 80.0
```

---

### Step 6: Phase 3 Runner (`scripts/run_phase3.py`)

```python
def run_phase3_validation():
    """Execute Phase 3 diagnostic checkpoints."""

    print("=" * 70)
    print("PHASE 3: Optimization Validation")
    print("=" * 70)

    results = {}

    # G3-1: K=5 everywhere
    results["G3-1_k_unified"] = check_k_unified()

    # G3-2: No Python loops
    results["G3-2_tensorized_ops"] = check_tensorized_operations()

    # G3-3: Throughput ≥10×
    results["G3-3_throughput"] = check_throughput_target()

    # G3-4: GPU utilization >80%
    results["G3-4_gpu_utilization"] = check_gpu_utilization()

    # Summary
    passed = sum(1 for v in results.values() if v)
    print(f"\nPHASE 3 SUMMARY: {passed}/{len(results)} checkpoints passed")
```

---

## Execution Order

```
1. src/optim/__init__.py              # Package setup
2. src/optim/muon.py                  # Muon optimizer
3. src/optim/memory_update.py         # Tensorized updates
4. src/training/benchmark.py          # Throughput benchmarking
5. tests/test_phase3.py               # Unit tests
6. scripts/run_phase3.py              # Validation runner
7. docs/PHASE3_DECISIONS.md           # Decision documentation
```

**Estimated New Files**: 7
**Estimated New LOC**: ~500 lines

---

## Key Design Decisions

### D3-1: K=5 Everywhere
Per PRD Round 8: "For a 60M parameter model, the compute overhead of K=5 everywhere is negligible. Get it working first, optimize later."

### D3-2: Tensorization Strategy
Use `torch.einsum` for batched outer products. This avoids:
- Python for loops over sequence length
- Explicit loop unrolling
- Sequential memory updates

### D3-3: Baseline Definition
"Baseline" = Single-chunk training without TNT parallelism.
This is a FAIR comparison showing the benefit of hierarchical memory.

### D3-4: Throughput Measurement
Measure tokens/second over 100 steps (after warmup).
This captures steady-state performance, not startup overhead.

---

## Acceptance Criteria

| Criterion | Requirement | Verification |
|-----------|-------------|--------------|
| AC-P3-1 | K unified | K=5 in all calls |
| AC-P3-2 | No Python loops | Profiler shows batched matmul |
| AC-P3-3 | Throughput | ≥10× baseline |
| AC-P3-4 | GPU utilization | >80% sustained |

---

## Notes

### What's NOT in Phase 3 (Deleted per PRD Round 8)
- ❌ Spectral error validation - premature optimization
- ❌ K warmup strategy - unnecessary complexity
- ❌ Pipeline latency requirements - not a bottleneck at this scale
- ❌ K decoupling (K_global=3, K_local=5) - get it working first

### Phase 3 Philosophy
> "A working model is infinitely better than a theoretically faster broken model"

The goal is correct implementation first, then optimization. K=5 everywhere is simple and sufficient for this 60M parameter proof of concept.

---

## What's Next: Phase 4

Phase 4 focuses on Stage 2 Fine-Tuning:
- Dynamic gradient clipping at chunk transitions
- Resolution robustness (flat PPL curve across chunk sizes)
- Gate polarization (≥20% at extremes)

See PRD P4-T1 through P4-T5 for requirements.
