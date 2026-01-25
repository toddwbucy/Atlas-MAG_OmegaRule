# Phase 2: Training Infrastructure - Implementation Plan

## Goal
Build training loop that won't silently fail, with NIAH probes to validate memory actually works.

## Status
- ✅ **Phase 0 Complete**: 46/46 tests passing
- ✅ **Phase 1 Complete**: 31/31 tests passing, 6/6 checkpoints passing
- 🚧 **Phase 2 In Progress**: Training infrastructure

## PRD Requirements (P2-T1 through P2-T6)

| Task | Description | Status |
|------|-------------|--------|
| P2-T1 | Lock multi-resolution chunk sizes | 🔲 Decision protocol |
| P2-T2 | Determine NIAH probe frequency | 🔲 Depends on P2-T4 |
| P2-T3 | Q-K projection with norm_persistent | 🔲 To implement |
| P2-T4 | Implement NIAH probes | 🔲 To implement |
| P2-T5 | Implement PPL delta telemetry | 🔲 To implement |
| P2-T6 | Implement auto-rollback (failsafe) | 🔲 To implement |

## Phase 2 Acceptance Criteria (from PRD 11.3)

| ID | Criterion | Test |
|----|-----------|------|
| AC-P2-1 | Chunk sizes locked | Documented decision |
| AC-P2-2 | NIAH overhead | < 1% throughput loss |
| AC-P2-3 | Norm in projection | norm_persistent in init |
| AC-P2-4 | 1000 steps stable | No crash, no NaN |
| AC-P2-5 | NIAH passing | > 80% accuracy |
| AC-P2-6 | Rollback tested | Triggers on forced spike |
| AC-P2-7 | Gradient mask at boundaries | grad M_t = 0 at shard boundaries |

---

## What Already Exists

### From Phase 0/1
- `PersistentMemory` with `m_persistent` and `norm_persistent` ✅
- `AtlasMemory` gated MLP ✅
- `AtlasMAGBlock` with gate infrastructure ✅
- `QKVProjection` for attention (NOT for memory retrieval) ✅
- Gate polarization loss with annealing ✅
- Fast-fail gate monitoring ✅
- `Phase1Trainer` training loop ✅
- Data loading infrastructure (`calibration.py`) ✅

### Config Constants (already defined)
```python
# src/config.py
C_G: int = 2048                   # Global chunk size
C_L_OPTIONS: List[int] = [8, 16, 32, 64]  # Local chunk candidates
S_L: int = 2048                   # Local shard length
K: int = 5                        # Newton-Schulz iterations
```

---

## New Files to Create

```
src/
├── model/
│   └── qk_projection.py         # Q-K projection with M_t accumulator
├── training/
│   ├── niah_probe.py            # NIAH retrieval validation
│   ├── telemetry.py             # PPL delta logging + metrics
│   ├── checkpoint.py            # Checkpoint manager + rollback
│   └── phase2_trainer.py        # Extended training loop

tests/
└── test_phase2.py               # Phase 2 unit tests

scripts/
└── run_phase2.py                # Phase 2 validation runner

docs/
└── PHASE2_DECISIONS.md          # Chunk size + frequency decisions
```

---

## Implementation Steps

### Step 1: Q-K Projection with Normalization (`src/model/qk_projection.py`)

This is the **Silent Killer #2 Fix** from the PRD - M_persistent needs a denominator.

```python
class QKProjection(nn.Module):
    """
    Q-K Projection with running M_t accumulator.

    At shard boundaries, M_t is reset to M_persistent and
    norm_sum is reset to norm_persistent. This ensures
    persistent memory is accessible from the first token.

    Key equations:
        M_t = M_persistent + sum(k_i outer k_i)
        norm_sum = norm_persistent + sum(||k_i||^2)
        q'_t = M_t @ q_t / norm_sum
    """

    def __init__(self, persistent_memory, dim):
        self.M_t = torch.zeros(dim, dim)
        self.norm_sum = 0.0

    def reset_at_shard_boundary(self):
        """CRITICAL: Inject BOTH numerator AND denominator."""
        self.M_t = self.persistent_memory.m_persistent.clone()
        self.norm_sum = self.persistent_memory.norm_persistent

    def update(self, k_t):
        self.M_t = self.M_t + torch.outer(k_t, k_t)
        self.norm_sum += (k_t.norm() ** 2).item()

    def project(self, q_t):
        return (self.M_t @ q_t) / max(self.norm_sum, 1e-8)
```

### Step 2: NIAH Retrieval Probes (`src/training/niah_probe.py`)

```python
class NIAHProbe:
    """
    Needle-in-a-Haystack probe for memory validation.

    1. Inject a known key-value pair (needle)
    2. Run haystack (random forward passes)
    3. Query with the key
    4. Measure retrieval accuracy via cosine similarity

    Target: > 80% accuracy
    """

    def run_probe(self, model, step, device):
        # Generate random needle
        needle_key = F.normalize(torch.randn(dim), dim=0)
        needle_value = F.normalize(torch.randn(dim), dim=0)

        # Inject needle
        qk_proj.inject_memory(needle_key, needle_value)

        # Bury under haystack
        for _ in range(100):
            qk_proj.update(F.normalize(torch.randn(dim), dim=0))

        # Query
        retrieved = qk_proj.query_memory(needle_key)
        accuracy = F.cosine_similarity(retrieved, needle_value)

        return NIAHResult(accuracy=accuracy, passed=accuracy >= 0.8)
```

### Step 3: PPL Delta Telemetry (`src/training/telemetry.py`)

```python
class PPLDeltaTracker:
    """Track perplexity delta for rollback detection."""

    def update(self, perplexity):
        # Compute delta from EMA
        delta = (perplexity - self.ema) / self.ema

        # Update EMA
        self.ema = alpha * perplexity + (1 - alpha) * self.ema

        # Check for spike (> 5% from moving avg)
        is_spike = delta > 0.05

        return delta, is_spike


class TelemetryLogger:
    """Write metrics to JSONL for dashboard."""

    def log_step(self, step, loss, perplexity, ppl_delta, ...):
        # Write to metrics.jsonl
        # Log to console every 100 steps
```

### Step 4: Checkpoint Manager (`src/training/checkpoint.py`)

```python
class CheckpointManager:
    """
    Manage checkpoints with rollback support.

    REQ-P2-003: Auto-rollback failsafe
    """

    def save(self, step, model, optimizer, loss, perplexity):
        # Save state dict to checkpoint_step{N}.pt
        # Keep last 5 checkpoints

    def rollback(self, model, optimizer, target_step=None):
        # Restore from checkpoint
        # Log warning about rollback
        self.rollback_count += 1
```

### Step 5: Phase 2 Trainer (`src/training/phase2_trainer.py`)

```python
class Phase2Trainer:
    """
    Extended training loop with full Phase 2 infrastructure.

    Integrates:
    - Language modeling loss
    - Gate polarization loss
    - Fast-fail monitoring
    - NIAH probes (every 1000 steps)
    - PPL delta telemetry
    - Auto-rollback on spike
    """

    def train_step(self, input_ids, step):
        # Forward + backward
        # Gate monitoring
        # PPL delta tracking
        # NIAH probe (if scheduled)
        # Checkpointing
        # Rollback check
```

### Step 6: Phase 2 Tests (`tests/test_phase2.py`)

Key tests:
- `test_qk_projection_norm_persistent` - AC-P2-3
- `test_niah_accuracy_above_80` - AC-P2-5
- `test_niah_overhead_below_1_percent` - AC-P2-2
- `test_ppl_delta_spike_detection`
- `test_rollback_trigger` - AC-P2-6
- `test_1000_steps_no_nan` - AC-P2-4
- `test_gradient_mask_at_boundary` - AC-P2-7

### Step 7: Phase 2 Runner (`scripts/run_phase2.py`)

```python
def main():
    # G2-1: Chunk sizes
    results["chunk_sizes"] = test_chunk_sizes()

    # G2-2: NIAH frequency
    results["niah_frequency"] = test_niah_overhead()

    # G2-3: norm_persistent in projection
    results["norm_in_projection"] = test_qk_norm()

    # G2-4: 1000 steps stable
    results["stability"] = run_1000_steps()

    # G2-5: NIAH accuracy
    results["niah_accuracy"] = test_niah_accuracy()

    # G2-6: Rollback test
    results["rollback"] = test_rollback()

    # G2-7: PPL delta visible
    results["ppl_delta"] = test_telemetry()

    # Summary: 7/7 checkpoints
```

---

## Phase 2 Exit Gate Checkpoints

| Checkpoint | Description | Test |
|------------|-------------|------|
| G2-1 | Chunk sizes locked | Document decision with throughput data |
| G2-2 | NIAH frequency locked | Measure overhead < 1% |
| G2-3 | norm_persistent in projection | `assert qk.norm_sum == norm_persistent` |
| G2-4 | 1000 steps stable | No OOM, no NaN |
| G2-5 | NIAH accuracy > 80% | `niah_probe.run_probe()` passes |
| G2-6 | Rollback tested | Force spike, verify rollback |
| G2-7 | PPL delta visible | Check `metrics.jsonl` has `ppl_delta` |

---

## Verification

### Unit Tests
```bash
pytest tests/test_phase2.py -v
```

### Integration Test
```bash
python scripts/run_phase2.py --device cuda --steps 1000
```

Expected output:
```
PHASE 2: Training Infrastructure
============================================================
[G2-1] Chunk sizes... DECIDED (Set A: {8,16,32,64})
[G2-2] NIAH frequency... LOCKED (1000 steps, 0.3% overhead)
[G2-3] Q-K projection norm... PASSED
[G2-4] 1000 steps stable... PASSED (no NaN, no OOM)
[G2-5] NIAH accuracy... PASSED (87.3% > 80%)
[G2-6] Rollback test... PASSED
[G2-7] PPL delta telemetry... PASSED

PHASE 2 SUMMARY: 7/7 checkpoints passed
*** ALL PHASE 2 CHECKS PASSED ***
```

### Type Check
```bash
python -m mypy src/training src/model/qk_projection.py --ignore-missing-imports
```

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/model/qk_projection.py` | CREATE | Q-K projection with M_t |
| `src/training/niah_probe.py` | CREATE | NIAH retrieval validation |
| `src/training/telemetry.py` | CREATE | PPL delta logging |
| `src/training/checkpoint.py` | CREATE | Checkpoint + rollback |
| `src/training/phase2_trainer.py` | CREATE | Phase 2 training loop |
| `src/training/__init__.py` | MODIFY | Add new exports |
| `tests/test_phase2.py` | CREATE | Phase 2 unit tests |
| `scripts/run_phase2.py` | CREATE | Phase 2 runner |
| `docs/PHASE2_DECISIONS.md` | CREATE | Chunk size + frequency docs |

---

## Execution Order

```
1. src/model/qk_projection.py           # Q-K projection with normalization
2. src/training/niah_probe.py           # NIAH probes
3. src/training/telemetry.py            # PPL delta tracking
4. src/training/checkpoint.py           # Checkpointing + rollback
5. src/training/phase2_trainer.py       # Phase 2 training loop
6. src/training/__init__.py             # Update exports
7. tests/test_phase2.py                 # Unit tests
8. scripts/run_phase2.py                # Integration runner
9. docs/PHASE2_DECISIONS.md             # Decision documentation
```

**Estimated New Files**: 9
**Estimated New LOC**: ~1,200 lines

---

## Key Design Decisions

### NIAH Probe Design
- **Frequency**: Start with 1000 steps (adjust if overhead > 1%)
- **Haystack size**: 100 tokens (enough noise without excessive cost)
- **Threshold**: 80% cosine similarity (from PRD)
- **On failure**: Log warning, don't abort (research-grade tolerance)

### Rollback Strategy
- **Trigger**: PPL delta > 5% from moving average
- **Action**: Restore model + optimizer from last checkpoint
- **Limit**: Track rollback count (target: <= 3 in Stage 2)

### Telemetry Format
- **File**: `metrics.jsonl` (append-only, dashboard-friendly)
- **Fields**: step, loss, perplexity, ppl_delta, gate stats, NIAH accuracy
- **Frequency**: Every step internally, write to file every 100 steps

---

## Dependencies

**Required for Phase 2**:
- Phase 0/1 complete ✅
- `PersistentMemory` with `m_persistent` and `norm_persistent` ✅
- Gate infrastructure ✅
- Training data (use `calibration.py`)

**Not required yet** (Phase 3+):
- TNT hierarchical memory (global + local)
- Newton-Schulz optimizer integration
- Multi-GPU training
