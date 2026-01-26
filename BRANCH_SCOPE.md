# Branch: feature/ttl-update-loop

## Purpose

Implement the **core Atlas innovation**: gradient-based Test-Time Learning (TTL) memory updates.
This replaces standard backprop-only training with actual test-time memory optimization.

## The Core Problem

**Current state (after Branch 2):**
- Memory module `AtlasMemoryPoly` exists and is wired in
- Memory weights are updated via **outer-loop backprop only** (standard training)
- No test-time adaptation - memory is frozen at inference

**What Atlas paper specifies:**
- Memory weights are updated **during the forward pass** via gradient descent
- This enables the model to memorize context at test time (true TTL)

---

## Key Equations from Atlas Paper

### Omega Rule Loss (Eq. 9)
```
L_omega = sum(i=t-c+1 to t) gamma_i^(t) * ||M(phi(k_i)) - v_i||^2
```

The memory M should map keys to values over a context window of size `c`.

### Momentum Accumulation (Eq. 33)
```
S_t = theta_t * S_{t-1} + grad(L_omega)
```

Where:
- `S_t` = momentum buffer (same shape as memory parameters)
- `theta_t` = momentum decay (input-dependent or fixed)
- `grad(L_omega)` = gradient of Omega loss w.r.t. memory parameters

### Memory Update Rule (Eq. 32)
```
M_t = alpha_t * M_{t-1} - eta_t * NewtonSchulz_k(S_t)
```

Where:
- `alpha_t` = weight decay (typically close to 1.0)
- `eta_t` = learning rate for memory updates
- `NewtonSchulz_k` = k iterations of Newton-Schulz orthogonalization (Muon optimizer)
- `k = 5` per paper recommendation

### Why Newton-Schulz (Muon)?

Standard SGD accumulates gradient magnitude over time, causing instability.
Newton-Schulz orthogonalizes the update, keeping it on the Stiefel manifold.

```
NS(G) iteratively computes: X_{i+1} = X_i * (aI + bX_i^T X_i + c(X_i^T X_i)^2)

With optimal coefficients: a=3.4445, b=-4.7750, c=2.0315
```

---

## Current State Analysis

### What EXISTS and is READY

| Component | File | Status |
|-----------|------|--------|
| `AtlasMemoryPoly` | `atlas_memory.py` | Wired into forward pass |
| `newton_schulz()` | `newton_schulz.py` | Implemented, NOT wired |
| `GammaGate` | `skeleton.py` | Working (input-dependent decay) |
| Context window | `config.py` | `OMEGA_CONTEXT_WINDOW = 256` |
| Decay base | `config.py` | `OMEGA_DECAY_BASE = 0.95` |

### What NEEDS to be IMPLEMENTED

| Component | Description | Complexity |
|-----------|-------------|------------|
| Omega loss computation | `||M(phi(k)) - v||^2` over context window | Medium |
| Gradient computation | `grad(L_omega)` w.r.t. memory params | Medium |
| Momentum buffer `S_t` | Persistent state across forward passes | Medium |
| Memory update loop | Wire NS into parameter update | Medium |
| State management | Handle batch boundaries, reset conditions | High |

---

## Design Decisions (DECIDED)

### Decision 1: Where does TTL happen? ✅ DECIDED: Option A

**Inside forward pass (paper approach)**
```python
def forward(self, x):
    # ... compute k, v ...

    # TTL update (modifies self.memory weights)
    with torch.enable_grad():
        loss = compute_omega_loss(self.memory, k, v, gamma)
        self.ttl_step(loss)

    # Use updated memory
    mem_out = self.memory(h, return_contribution=True)
```

### Decision 2: Gradient computation scope ✅ DECIDED: Option A

**Full autograd through memory**
```python
loss = omega_loss(self.memory, k, v)
grads = torch.autograd.grad(loss, self.memory.parameters())
```

### Decision 3: Momentum state storage ✅ DECIDED: Option A

**Store in module as registered buffers**
```python
class AtlasMemoryPoly(nn.Module):
    def __init__(self):
        self.register_buffer('momentum_w1', torch.zeros_like(self.w1.weight))
        self.register_buffer('momentum_w2', ...)
```

This keeps state with module and survives serialization.

### Decision 4: When to reset momentum? ✅ DECIDED: Configurable

**Configurable via `TTL_RESET_MODE`**
- `"sequence"`: Reset at start of each sequence
- `"batch"`: Reset at start of each batch
- `"never"`: Never reset (most aggressive, accumulates across training)

Default: `"sequence"` (conservative). Configurable allows testing which works best.

### Decision 5: Training vs Inference TTL ✅ DECIDED: Both (Configurable)

**TTL happens at BOTH training and inference** - this is the core innovation.
The whole point of TTL is learning at inference time, not just training.

Configurable via `TTL_ENABLED` flag for ablation studies:
- `TTL_ENABLED = True`: Full TTL at train and inference (default)
- `TTL_ENABLED = False`: Disabled for ablation comparison

---

## Proposed Architecture

```
AtlasMAGBlock.forward(x):
    |
    +-- h = norm(x)
    |
    +-- q, k, v = QKV(h)
    |
    +-- if ttl_enabled:
    |       |
    |       +-- L_omega = sum(gamma_i * ||M(phi(k_i)) - v_i||^2)
    |       |
    |       +-- grad = autograd.grad(L_omega, M.parameters())
    |       |
    |       +-- S_t = theta * S_{t-1} + grad
    |       |
    |       +-- M.params -= eta * NewtonSchulz(S_t)
    |
    +-- mem_out = M(h, return_contribution=True)
    |
    +-- attn_out = SWA(q, k, v)
    |
    +-- return x + attn_out + gate * mem_out
```

---

## Implementation Plan

### Phase 1: Omega Loss Computation
**File:** `src/training/omega_loss.py` (new)

```python
def compute_omega_loss(
    memory: AtlasMemoryPoly,
    keys: Tensor,      # (batch, seq, dim)
    values: Tensor,    # (batch, seq, dim)
    gamma: Tensor,     # (batch, seq, 1) decay weights
    context_window: int = 256,
) -> Tensor:
    """
    Compute Omega Rule loss over context window.

    L = sum(gamma_i * ||M(phi(k_i)) - v_i||^2)
    """
    # Get last `context_window` positions
    k_ctx = keys[:, -context_window:]
    v_ctx = values[:, -context_window:]
    g_ctx = gamma[:, -context_window:]

    # Forward through memory
    predicted = memory(k_ctx)  # M(phi(k))

    # Weighted MSE
    diff = predicted - v_ctx
    loss = (g_ctx * (diff ** 2).sum(dim=-1, keepdim=True)).mean()

    return loss
```

### Phase 2: Momentum Buffer
**File:** `src/model/atlas_memory.py` (modify)

```python
class AtlasMemoryPoly(nn.Module):
    def __init__(self, ...):
        ...
        # Momentum buffers (registered as buffers for serialization)
        self._init_momentum_buffers()

    def _init_momentum_buffers(self):
        """Initialize momentum buffers matching parameter shapes."""
        for name, param in self.named_parameters():
            buffer_name = f'momentum_{name.replace(".", "_")}'
            self.register_buffer(buffer_name, torch.zeros_like(param))

    def get_momentum(self, name: str) -> Tensor:
        buffer_name = f'momentum_{name.replace(".", "_")}'
        return getattr(self, buffer_name)

    def reset_momentum(self):
        """Reset all momentum buffers to zero."""
        for name, param in self.named_parameters():
            buffer_name = f'momentum_{name.replace(".", "_")}'
            getattr(self, buffer_name).zero_()
```

### Phase 3: TTL Update Step
**File:** `src/training/ttl_update.py` (new)

```python
def ttl_step(
    memory: AtlasMemoryPoly,
    loss: Tensor,
    theta: float = 0.9,      # Momentum decay
    alpha: float = 0.999,    # Weight decay
    eta: float = 0.01,       # Learning rate
    ns_iterations: int = 5,  # Newton-Schulz iterations
) -> dict:
    """
    Perform one TTL update step on memory parameters.

    S_t = theta * S_{t-1} + grad
    M_t = alpha * M_{t-1} - eta * NS(S_t)
    """
    # Compute gradients
    grads = torch.autograd.grad(
        loss,
        memory.parameters(),
        create_graph=False,  # Don't need second-order
        retain_graph=False,
    )

    stats = {}

    # Update each parameter
    for (name, param), grad in zip(memory.named_parameters(), grads):
        # Get momentum buffer
        momentum = memory.get_momentum(name)

        # Update momentum: S_t = theta * S_{t-1} + grad
        momentum.mul_(theta).add_(grad)

        # Orthogonalize via Newton-Schulz
        if momentum.dim() >= 2:
            update = newton_schulz(momentum, num_iters=ns_iterations)
        else:
            update = momentum  # Skip NS for 1D (bias terms)

        # Update parameter: M_t = alpha * M_{t-1} - eta * NS(S_t)
        param.data.mul_(alpha).sub_(update, alpha=eta)

        stats[f'{name}_grad_norm'] = grad.norm().item()
        stats[f'{name}_update_norm'] = update.norm().item()

    return stats
```

### Phase 4: Integration into Forward Pass
**File:** `src/model/skeleton.py` (modify)

```python
class AtlasMAGBlock(nn.Module):
    def __init__(self, ..., ttl_enabled: bool = True):
        ...
        self.ttl_enabled = ttl_enabled

    def forward(self, x, v_targets=None):
        """
        Args:
            x: Input tensor
            v_targets: Optional value targets for TTL (if None, uses v from QKV)
        """
        h = self.norm1(x)
        q, k, v = self.qkv(h, reshape_heads=True)

        # TTL update (if enabled and training)
        if self.ttl_enabled and self.training:
            # Use v as targets (self-supervised)
            v_flat = v.transpose(1, 2).reshape(batch, seq, -1)
            gamma = self.gamma_gate(h) if self.gamma_gate else None

            omega_loss = compute_omega_loss(
                self.memory, h, v_flat, gamma
            )
            ttl_step(self.memory, omega_loss)

        # Rest of forward...
```

---

## Configuration Additions

**File:** `src/config.py`

```python
# TTL (Test-Time Learning) Configuration
TTL_ENABLED: bool = True              # Master switch (both train & inference)
TTL_THETA: float = 0.9                # Momentum decay
TTL_ALPHA: float = 0.999              # Weight decay
TTL_ETA: float = 0.01                 # Memory learning rate
TTL_NS_ITERATIONS: int = 5            # Newton-Schulz iterations
TTL_RESET_MODE: str = "sequence"      # When to reset: "sequence", "batch", "never"
```

---

## Files Affected

| File | Changes |
|------|---------|
| `src/training/omega_loss.py` | **NEW** - Omega loss computation |
| `src/training/ttl_update.py` | **NEW** - TTL update step |
| `src/model/atlas_memory.py` | Add momentum buffers |
| `src/model/skeleton.py` | Integrate TTL into forward |
| `src/model/newton_schulz.py` | Minor: ensure batch support |
| `src/config.py` | Add TTL hyperparameters |
| `tests/test_ttl.py` | **NEW** - TTL-specific tests |

---

## Testing Strategy

### Unit Tests

1. **test_omega_loss_shape**: Loss is scalar, gradient flows
2. **test_omega_loss_zero_when_perfect**: Loss=0 when M(k)=v exactly
3. **test_momentum_accumulation**: S_t updates correctly
4. **test_newton_schulz_on_momentum**: NS produces orthogonal update
5. **test_memory_params_change**: Parameters actually update
6. **test_momentum_reset**: Reset works correctly

### Integration Tests

1. **test_ttl_in_forward_pass**: Full forward with TTL enabled
2. **test_ttl_disabled_unchanged**: TTL disabled = no param changes
3. **test_ttl_training_vs_eval**: Different behavior in train/eval modes
4. **test_gradient_flow_outer_loop**: Outer-loop backprop still works

### Validation Tests

1. **test_associative_recall**: Can memorize k->v pairs (capacity test)
2. **test_ttl_improves_recall**: TTL should improve recall vs no-TTL

---

## Acceptance Criteria

- [x] Omega loss computed over context window
- [x] Gradients computed w.r.t. memory parameters
- [x] Momentum buffers accumulate correctly
- [x] Newton-Schulz applied to momentum before update
- [x] Memory updated via: `M_t = alpha*M_{t-1} - eta*NS(S_t)`
- [x] TTL can be enabled/disabled via config
- [x] Momentum resets at sequence boundaries (configurable)
- [x] All existing tests still pass (208/209 - 1 pre-existing failure)
- [x] New TTL tests pass (21/21)
- [x] Training converges (no NaN/Inf) - verified in numerical stability test

### Decision 6: Target values for Omega loss ✅ DECIDED: Option A

**Use `v` from QKV projection** (Atlas paper Eq. 9, Appendix D)

From the paper:
- Equation 9: `L = sum(γ_i * ||M(φ(k_i)) - v_i||^2)`
- Appendix D: `K = xW_K`, `V = xW_V` (standard projections)

The memory learns to map keys to values, matching the attention pattern.

### Decision 7: Which layers get TTL? ✅ DECIDED: All layers

**All layers have TTL enabled** (Titans paper architecture)

The Titans paper shows memory applied across all layers in MAC/MAG/MAL variants.
No restriction to "only later layers" - memory at every level provides maximum flexibility.

### Decision 8: Batch handling ✅ DECIDED: Shared momentum

**Shared momentum across batch** (Atlas formulations)

The update equations `S_t = θ*S_{t-1} + grad(L_omega)` don't use per-sample subscripts.
Omega loss is computed as mean over batch, so gradients naturally aggregate.
This gives "batch consensus" momentum - simpler and faster.

---

## Open Questions (Non-blocking)

1. **Interaction with gate polarization**: Does TTL affect gates?
   - Gates are separate from memory parameters, so TTL shouldn't directly affect them
   - May observe indirect effects through changed memory behavior
   - Monitor during training, address if issues arise

---

## References

- Issue #12 (Implementation Gap Analysis)
- Atlas Paper arXiv:2505.23735
  - Eq. 9: Omega Rule
  - Eq. 32-33: Update Rule
  - Section 4.2: Muon Optimizer
- PR #13 (Branch 1: Output-level combination)
- PR #14 (Branch 2: Wire AtlasMemoryPoly)
