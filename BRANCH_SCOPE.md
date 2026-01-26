# Branch: refactor/output-level-combination

## Purpose
Fix the architectural blending location to match the Atlas paper (Figure 3).

## Problem
Current implementation blends at **Q-level** (before attention):
```python
q = (1 - gate) * q + gate * q_projected  # WRONG
attn_out = attention(q, k, v)
```

Paper specifies **output-level** combination (parallel branches):
```
Input → [Atlas Layer] ──┐
                        ├──► Add → Output
      → [SWA] ─────────┘
```

## Changes Required

### 1. Modify `AtlasMAGBlock.forward()` in `src/model/skeleton.py`

**Before:**
```python
# Q-level blending
q_projected = self.memory_projection(q, k, gamma_gates)
q = (1 - gate) * q + gate * q_projected
attn_out = self.attention(q, k, v)
```

**After:**
```python
# Parallel branches with output-level combination
attn_out = self.attention(q, k, v)
mem_out = self.memory_module(k)  # or appropriate input
combined = attn_out + mem_out    # Or gated: gate * mem_out + (1-gate) * attn_out
```

### 2. Files Affected
- `src/model/skeleton.py` - Main architectural change

## Dependencies
- **None** - This is the first branch in the sequence

## Blocked By
- Nothing

## Blocks
- `feature/wire-atlas-memory`

## Acceptance Criteria
- [ ] Atlas layer and SWA operate as parallel branches
- [ ] Combination happens at output level (after both computations)
- [ ] Tests pass
- [ ] No regression in training convergence

## References
- Issue #12
- Atlas Paper Figure 3 (MAG Architecture)
- Atlas Paper Section 5 (Architectural Backbone)
