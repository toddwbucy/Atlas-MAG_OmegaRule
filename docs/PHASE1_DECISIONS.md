# Phase 1 Architecture Decisions

**Status**: Implementation Complete
**Date**: 2025-01-24
**Goal**: Validate MAG architecture can ROUTE (not average) between attention and memory

---

## Pre-Resolved Decisions

### P1-T3: Polynomial Degree = 2

**Decision**: Use degree-2 polynomial features

**Rationale**: Memory constraint
- **Degree 2**: `768 + 295,296 = 296,064 features` (~1.1 MB/token)
- **Degree 3**: `75,583,584 features` (~289 MB/token) - Exceeds VRAM

**Implementation**: `AtlasMemoryPoly` in `src/model/atlas_memory.py`
- Only degree 2 is implemented
- Attempting degree 3 raises `NotImplementedError`

**Trade-off**: Degree 2 provides sufficient expressiveness for 60M POC while staying within hardware limits.

---

### P1-T4: Memory Depth L_M = 2

**Decision**: Use 2-layer memory depth

**Rationale**: Parameter budget
- Current model: ~110M params with L_M implicit in `AtlasMemory`
- `AtlasMemory` gated MLP structure provides sufficient expressiveness
- Deeper memory would exceed toy model budget

**Implementation**: `L_M = 2` in `src/config.py`

**Trade-off**: Shallower memory may limit long-range dependencies but keeps model tractable for validation.

---

## Phase 1 Requirements

| Task | Description | Implementation |
|------|-------------|----------------|
| P1-T1 | Gate polarization with annealing λ | `src/training/polarization.py` |
| P1-T2 | Gate noise suppression test | `tests/test_phase1.py` |
| P1-T3 | Polynomial degree = 2 | Pre-resolved (see above) |
| P1-T4 | Memory depth L_M = 2 | Pre-resolved (see above) |
| P1-T5 | Fast-fail at step 500 | `src/training/fast_fail.py` |
| P1-T6 | Multiplicative fusion | `src/training/trainer.py` |
| P1-T7 | Gated MLP (Atlas++) | `src/model/atlas_memory.py` |

---

## Gate Polarization Loss

**Formula**:
```
L_polar = λ(t) × (1 - |2g - 1|)
```

**Properties**:
- Maximum penalty (λ) at g=0.5 (indecisive)
- Zero penalty at g∈{0,1} (decisive routing)

**Annealing Schedule**:
- First 10% of training: λ = 10.0 (high pressure to polarize)
- Remaining 90%: Exponential decay λ → 0.1 (allow subtle mixing)

**Rationale**: High initial λ forces architecture to prove it can route. Decay allows nuanced mixing once routing is established.

---

## Fast-Fail Conditions

| Condition | Step | Action |
|-----------|------|--------|
| Record baseline variance | 100 | Store initial gate variance |
| Variance not increasing | 500 | ABORT if < 1.5× baseline |
| Std collapse | Any > 100 | ABORT if std < 0.01 |

**Rationale**: From PRD - "The system needs to scream 'I'm broken' immediately"

These conditions detect when gates are stuck averaging (not learning to route), allowing early termination rather than wasting compute.

---

## Validation Checkpoints

Run with: `python scripts/run_phase1.py`

| Checkpoint | Test |
|------------|------|
| G1-1 | Gate noise suppression (gates move from 0.5) |
| G1-2 | λ annealing calibration (10.0 → 0.1) |
| G1-3 | Polynomial degree = 2 |
| G1-4 | Memory depth L_M = 2 |
| G1-5 | Non-trivial gate variance (std > 0.05) |
| G1-6 | Multiplicative fusion (gate=0 ≠ gate=1) |

**Success Criteria**: All 6 checkpoints must pass.

---

## Files Created

| File | Purpose |
|------|---------|
| `src/training/__init__.py` | Package init |
| `src/training/polarization.py` | Gate polarization loss + annealing |
| `src/training/fast_fail.py` | Fast-fail monitoring |
| `src/training/trainer.py` | Phase 1 training loop |
| `tests/test_phase1.py` | Unit tests |
| `scripts/run_phase1.py` | Integration runner |
| `docs/PHASE1_DECISIONS.md` | This document |

---

## Next Steps (Phase 2)

After Phase 1 validation passes:
1. Add NIAH (Needle-in-a-Haystack) retrieval probes
2. Implement Q-K projection with normalization
3. Test cross-attention retrieval accuracy (target: >80%)

---

## References

- PRD: `PRD_Atlas_MAG_TNT.md` (Phase 1 requirements)
- Atlas paper: arXiv:2505.23735 (Gated MLP memory)
- TNT paper: arXiv:2511.07343 (Two-stage training)
