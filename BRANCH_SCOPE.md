# Branch: feature/wire-atlas-memory

## Purpose
Wire the existing `AtlasMemory` and `AtlasMemoryPoly` modules into the forward pass,
replacing the current outer-product approximation with the paper's deep MLP memory.

## Problem
Current implementation uses closed-form outer products:
```python
M_t = Σ γ^(t-i) * (k_i ⊗ k_i)  # Linear approximation
```

Paper specifies deep MLP memory with polynomial features:
```python
M(φ(k)) = φ(k) + W1 * (σ(W2(φ(k))) ⊗ W3(φ(k)))  # Gated MLP with residual
```

## Why Polynomial Features Are ESSENTIAL

From Atlas Paper (Section 3.1, Propositions 1 & 2):

**Without polynomial features:**
```
Capacity = O(d_k)
```

**With polynomial features φ_p:**
```
Capacity = O(d_k^p)
```

For d_k = 64 and p = 2:
- Without: 64 associations
- With: 4,096 associations (64× improvement)

**Paper Ablation (Table 6):**
| Model | Perplexity | Accuracy |
|-------|-----------|----------|
| Atlas | 19.97 | 52.77% |
| w/o Polynomial | 22.14 | 50.57% |

**Critical for independent researchers**: With smaller d_k due to hardware constraints,
polynomial features are even MORE important to maintain adequate memory capacity.

## Changes Required

### 1. Wire `AtlasMemory` into forward pass

**File:** `src/model/skeleton.py`

```python
from src.model.atlas_memory import AtlasMemoryPoly

class AtlasMAGBlock(nn.Module):
    def __init__(self, ...):
        # Replace CausalQKMemoryProjection with AtlasMemoryPoly
        self.memory = AtlasMemoryPoly(
            dim=dim,
            hidden_dim=dim * 4,
            poly_degree=2,  # φ_2 for quadratic capacity
        )

    def forward(self, x):
        # Memory branch (parallel to attention)
        mem_out = self.memory(k)  # M(φ(k))
        ...
```

### 2. Enable polynomial feature mapping

**File:** `src/model/atlas_memory.py`

The `AtlasMemoryPoly` class already implements:
```python
def _polynomial_features(self, x):
    # φ_2(x) = [x, upper_tri(x ⊗ x)]
    outer = x.unsqueeze(-1) * x.unsqueeze(-2)
    triu = outer[..., triu_idx[0], triu_idx[1]]
    return torch.cat([x, triu], dim=-1)
```

Ensure this is called in the forward pass.

### 3. Update configuration

**File:** `src/config.py`

```python
# Ensure these are used
POLY_DEGREE = 2  # Already exists, needs wiring
ATLAS_MEMORY_HIDDEN_MULT = 4
```

## Files Affected
- `src/model/skeleton.py` - Wire AtlasMemoryPoly
- `src/model/atlas_memory.py` - Verify polynomial feature implementation
- `src/config.py` - Update hyperparameters if needed
- `src/model/qk_projection.py` - May be deprecated

## Dependencies
- **Requires:** `refactor/output-level-combination` (Branch 1)

## Blocked By
- `refactor/output-level-combination`

## Blocks
- `feature/ttl-update-loop`

## Acceptance Criteria
- [ ] `AtlasMemoryPoly` used as memory module M(·)
- [ ] Polynomial features φ(k) applied to keys
- [ ] Memory output dimension matches expected
- [ ] Outer-loop training (standard backprop) works
- [ ] Tests pass
- [ ] Memory capacity improved (can verify via associative recall tests)

## Mathematical Reference

**Polynomial Feature Mapping (Eq. 22):**
```
φ_p(x) = x^⊗p    (self-tensoring via Kronecker product)

Lifted dimension:
D = C(d_k + p, p) = (d_k + p)! / (p! * d_k!)
```

**Gated MLP Memory (Eq. 43):**
```
M(x) = x + W1 * (σ(W2(x)) ⊗ W3(x))

Where:
- W1 ∈ R^(d_out × d_hidden)
- W2, W3 ∈ R^(d_hidden × d_in)
- σ = SiLU activation
- ⊗ = element-wise product (gating)
```

## References
- Issue #12
- Atlas Paper Section 3.1 (Memory Capacity)
- Atlas Paper Propositions 1 & 2
- Atlas Paper Equation 43 (Gated MLP)
- Atlas Paper Table 6 (Ablation Study)
