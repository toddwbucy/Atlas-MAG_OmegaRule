# Phase 4: Stage 2 Fine-Tuning - Design Decisions

## Overview

This document captures the key design decisions made during Phase 4 implementation.
Each decision references the relevant PRD section and explains the rationale.

---

## D4-1: Inherit from Phase2Trainer

**Decision**: `Stage2Trainer` extends `Phase2Trainer` rather than being a standalone class.

**Rationale**:
Phase2Trainer already implements critical infrastructure:
- Checkpoint management with save/load
- NIAH probe integration for memory validation
- PPL delta telemetry for training stability
- Auto-rollback failsafe on PPL spikes
- Gate monitoring with fast-fail checks

Stage 2 only adds chunk scheduling and dynamic gradient clipping. By inheriting, we:
- Reuse tested, validated code
- Maintain consistent interfaces
- Avoid code duplication
- Ensure rollback works seamlessly with new features

**Trade-offs**:
- Less flexibility (constrained by parent class design)
- Must maintain compatibility with parent interfaces

**Alternatives Considered**:
- Composition: Have Stage2Trainer wrap Phase2Trainer
- Standalone: Implement everything from scratch
- Both rejected as more complex with no clear benefit

---

## D4-2: Proactive Clipping Before Reactive Rollback

**Decision**: Apply aggressive gradient clipping at chunk transitions BEFORE problems occur.

**PRD Reference**: Section 7.4 - "Proactive clipping FIRST, rollback as last resort"

**Rationale**:
Rollback is expensive:
- Requires checkpoint loading (I/O)
- Loses training progress (wasted compute)
- May introduce discontinuities

Proactive clipping is cheap:
- Simple norm scaling (in-place)
- No state changes
- Prevents problems instead of fixing them

**Implementation**:
```python
# At chunk transition, apply extra-aggressive clipping
transition_threshold = normal_threshold * 0.5  # 50% of normal

if is_chunk_transition:
    clip_gradients(model, transition_threshold)
```

**Metrics**: Track `transition_count` and `clip_count` to verify proactive measures working.

---

## D4-3: Soft Boundaries Optional (Default: Off)

**Decision**: Start with hard chunk transitions. Soft interpolation only if needed.

**PRD Reference**: P4-T2 - "Optionally implement soft boundaries"

**Rationale**:
Soft boundaries add complexity:
- Requires interpolation logic
- More state to track
- Harder to debug

Hard transitions are simpler:
- Clean chunk size changes
- Proactive clipping handles discontinuities
- Easier to reason about

**Implementation**:
```python
class Stage2Trainer:
    def __init__(self, ..., enable_soft_boundaries: bool = False):
        self.enable_soft_boundaries = enable_soft_boundaries
```

Flag preserved for future use if hard transitions prove unstable.

---

## D4-4: kappa_base = 0.5 as Default

**Decision**: Use `kappa_base = 0.5` at reference chunk size C_L = 32.

**PRD Reference**: Section 7.3 - "Verify kappa_base = 0.5 at C_L = 32"

**Rationale**:
The PRD specifies this value based on calibration:
- At C_L = 32: kappa = 0.5 (base threshold)
- At C_L = 8: kappa = 1.0 (2x relaxed)
- At C_L = 1: kappa = 2.83 (5.66x relaxed)

This scaling compensates for increased gradient variance at smaller chunk sizes.

**Verification**: G4-1 exit gate checks formula at all chunk sizes.

---

## D4-5: Maximum 3 Rollbacks

**Decision**: Abort training if more than 3 auto-rollbacks occur during Stage 2.

**PRD Reference**: G4-2 - "Rollback count <= 3 auto-rollbacks"

**Rationale**:
Frequent rollbacks indicate:
- Proactive measures are failing
- Training is fundamentally unstable
- Need to investigate root cause

3 rollbacks is generous:
- Allows for occasional instability
- Catches systematic problems
- Prevents infinite rollback loops

**Implementation**:
```python
if is_spike:
    if self.rollback_count >= self.max_rollbacks:
        logger.error("Max rollbacks exceeded. Training unstable.")
    else:
        self.checkpoint_manager.rollback(...)
        self.rollback_count += 1
```

---

## D4-6: Geometric Decay with gamma = 0.9993

**Decision**: Use gamma = 0.9993 for chunk size decay.

**PRD Reference**: Section A.7 - "Geometric decay for chunk size"

**Formula**: `C_L(t) = max(1, floor(C_start * gamma^t))`

**Decay Schedule**:
| Step | C_L(t) | % of Start |
|------|--------|------------|
| 0 | 32 | 100% |
| 1000 | 16 | 50% |
| 2000 | 8 | 25% |
| 5000 | ~1-2 | ~5% |
| 10000 | 1 | minimum |

**Rationale**:
This provides gradual transition from Stage 1 (large chunks) to fine-grained learning (small chunks).
The model learns:
1. Global patterns first (large context)
2. Local details later (small context)

---

## D4-7: Resolution Sensitivity Threshold = 10% CV

**Decision**: PPL coefficient of variation < 10% across chunk sizes for "resolution flat".

**PRD Reference**: G4-4 - "Resolution flat across {1-64}"

**Rationale**:
A truly resolution-robust model should:
- Maintain similar performance regardless of chunk size
- Not be sensitive to inference-time chunk decisions

10% CV threshold means:
- If mean PPL = 100, std must be < 10
- Small variations acceptable (noise, measurement)
- Large variations indicate resolution dependence

**Measurement**:
```python
ppl_curve = {1: ppl_1, 4: ppl_4, 8: ppl_8, ...}
cv = std(ppl_curve.values()) / mean(ppl_curve.values())
passed = cv < 0.10
```

---

## Exit Gate Summary

| Gate | Criterion | Threshold | Rationale |
|------|-----------|-----------|-----------|
| G4-1 | Dynamic clip formula | Exact match | Mathematical correctness |
| G4-2 | Rollback count | <= 3 | Indicates stable training |
| G4-3 | PPL spike count | < 10 | Threshold properly calibrated |
| G4-4 | Resolution CV | < 10% | Resolution robust |
| G4-5 | Gate polarization | >= 20% | Learning to specialize |
| G4-6 | NIAH accuracy | > 80% | Memory still works |

---

## Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `src/training/gradient_clipping.py` | Dynamic kappa(t), ProactiveStabilizer | ~200 |
| `src/training/stage2_trainer.py` | Stage 2 trainer with chunk scheduling | ~400 |
| `src/training/phase4_validation.py` | Exit gate checks G4-1 through G4-6 | ~250 |
| `tests/test_phase4.py` | Unit tests for Phase 4 | ~400 |
| `scripts/run_phase4.py` | Validation runner | ~200 |
| `docs/PHASE4_DECISIONS.md` | This document | ~200 |

---

## Future Considerations

1. **Soft Boundaries**: If hard transitions cause instability, implement interpolation
2. **Adaptive kappa_base**: Could learn optimal kappa_base during warmup
3. **Per-Layer Clipping**: Different thresholds for different model layers
4. **Chunk Size Scheduling**: Non-geometric schedules (cosine, step)
