# Phase 2 Decisions

This document records the key decisions made during Phase 2 implementation.

## Overview

**Phase 2 Goal**: Training Infrastructure Validation
- Prove the training loop can detect and recover from failures
- Validate memory actually works via NIAH probes
- Establish telemetry for monitoring training health

**Status**: Implemented

## Key Decisions

### D2-1: Silent Killer #2 Fix (norm_persistent)

**Problem**: M_persistent alone causes volume mismatch in Q-K projection.

**Solution**: Include norm_persistent as denominator:
```python
q'_t = M_t @ q_t / norm_sum_t
```

Where:
- `M_t = M_persistent + Σ(k_i ⊗ k_i)`
- `norm_sum_t = norm_persistent + Σ||k_i||²`

**Rationale**: Without the denominator, persistent memory is either too quiet (drowned by new keys) or too loud (overwhelming). The normalization ensures proper volume scaling.

**Implementation**: `src/model/qk_projection.py`

---

### D2-2: NIAH Probe Frequency

**Decision**: Default probe frequency = 1000 steps

**Alternatives Considered**:
- 100 steps: Too frequent, >1% overhead
- 500 steps: Acceptable but more compute
- 2000 steps: Risk missing memory degradation

**Rationale**:
- Target overhead: <1% of training time
- Need enough probes to catch degradation early
- 1000 steps is ~10 probes per 10K training run

**Validation**: `measure_niah_overhead()` function confirms <1% overhead

---

### D2-3: PPL Delta Spike Threshold

**Decision**: 5% threshold for PPL spike detection

**Formula**:
```python
delta = (current_ppl - ema_ppl) / ema_ppl
is_spike = delta > 0.05  # 5%
```

**Alternatives Considered**:
- 2%: Too sensitive, false positives
- 10%: Too lenient, miss real spikes
- Absolute threshold: Doesn't scale with training stage

**Rationale**:
- EMA-relative catches context-appropriate spikes
- 5% balances sensitivity vs. false positives
- PRD recommends this as starting point

**Tuning**: Threshold configurable in TelemetryLogger

---

### D2-4: Checkpoint Rollback Strategy

**Decision**: Rollback to second-most-recent checkpoint

**Why not most recent?**
The most recent checkpoint might have been saved just before the spike, when the model was already corrupted. Going back one more step provides safety margin.

**Rollback Trigger**: PPL spike detected (delta > threshold)

**Implementation**:
```python
if is_spike:
    checkpoint_manager.rollback(model, optimizer)
    # Continues from step N-2, not N-1
```

**PRD Reference**: This is a FAILSAFE, not primary defense. Proactive gradient clipping (Phase 4) is the primary defense.

---

### D2-5: NIAH Accuracy Threshold

**Decision**: 80% accuracy for NIAH pass

**What it measures**: Cosine similarity between retrieved value and injected needle.

**Why 80%?**
- Random baseline: ~0% (orthogonal vectors)
- Perfect retrieval: 100%
- 80% indicates memory is working but not perfect
- Lower threshold for POC, can increase for production

**What failing means**: Memory system is broken, abort training.

---

### D2-6: Haystack Size

**Decision**: 100 random keys for NIAH haystack

**Tradeoff**:
- Smaller haystack: Easier retrieval, less diagnostic
- Larger haystack: Harder retrieval, slower probe

**Rationale**: 100 provides meaningful challenge without excessive probe time.

---

### D2-7: Telemetry Output Format

**Decision**: JSONL (JSON Lines) for metrics

**Format**:
```json
{"step": 0, "lm_loss": 6.9, "ppl_delta": 0.0, "niah_accuracy": null, ...}
{"step": 100, "lm_loss": 5.2, "ppl_delta": -0.25, "niah_accuracy": 0.85, ...}
```

**Alternatives Considered**:
- CSV: Harder to add new fields
- TensorBoard: Overkill for POC
- WandB: External dependency

**Rationale**:
- Easy to parse with any tool
- Append-only (no file locking)
- Human readable
- Supports optional fields (niah_accuracy = null when not probed)

---

### D2-8: Checkpoint File Format

**Decision**: PyTorch `.pt` format

**Contents**:
```python
{
    "step": 100,
    "model_state_dict": {...},
    "optimizer_state_dict": {...},
    "loss": 5.0,
    "perplexity": 150.0,
    "timestamp": "2025-01-24T..."
}
```

**Keep Last**: 5 checkpoints by default (configurable)

**Rationale**: Standard format, easy to load/save, includes all necessary state.

---

## Acceptance Criteria Verification

| Criterion | Requirement | Verification |
|-----------|-------------|--------------|
| AC-P2-3 | norm_persistent in projection | `qk.norm_sum == qk.norm_persistent` after reset |
| AC-P2-4 | 1000 steps stable | Trainer completes without NaN |
| AC-P2-5 | NIAH > 80% | `result.accuracy >= 0.8` |
| AC-P2-6 | Rollback tested | `test_rollback_trigger()` passes |
| AC-P2-7 | PPL delta visible | `telemetry.check_ppl_delta_visible()` |

---

## Files Created

| File | Purpose |
|------|---------|
| `src/model/qk_projection.py` | Q-K projection with normalization |
| `src/training/niah_probe.py` | NIAH retrieval validation |
| `src/training/telemetry.py` | PPL delta tracking |
| `src/training/checkpoint.py` | Checkpoint with rollback |
| `src/training/phase2_trainer.py` | Full Phase 2 training loop |
| `tests/test_phase2.py` | Phase 2 unit tests |
| `scripts/run_phase2.py` | Validation runner |

---

## What's Next: Phase 3

Phase 3 focuses on Optimization:
- K=5 Newton-Schulz iterations
- ≥10× throughput improvement
- Muon optimizer integration

See PRD P3-T1 through P3-T4 for requirements.
