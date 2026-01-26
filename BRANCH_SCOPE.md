# Branch: feature/training-integration

## Purpose
Integrate the TTL system into the training harness, handling memory state
management across batches and updating monitoring/logging systems.

## Context

After Branches 1-3 are complete, we have:
- ✅ Output-level combination (Branch 1)
- ✅ AtlasMemoryPoly with polynomial features (Branch 2)
- ✅ TTL update loop with Muon optimizer (Branch 3)

This branch connects everything to the actual training pipeline.

## Key Challenges

### 1. Memory State Persistence Across Batches

The TTL system maintains state:
- `M` - Memory module parameters (updated each step)
- `S` - Momentum buffers (accumulated across steps)

**Questions to resolve:**
- Should memory reset at sequence boundaries?
- Should memory persist across batches in same epoch?
- How to handle gradient accumulation with memory state?

**Paper guidance (Section 1):**
> "Through the paper, we use the terminology 'Test Time Memorization' because
> the process involves storing and retrieving information strictly within the
> global context, without updating the model's core learned parameters (i.e.,
> outer-loop) or initial states from pre-training."

This suggests memory should reset at context boundaries (new sequences).

### 2. Two-Level Optimization

Atlas has two optimization levels:
1. **Inner Loop (TTL):** Updates M's parameters based on Omega loss
2. **Outer Loop (Standard):** Updates all other parameters via backprop

```python
# Training step pseudocode
for batch in dataloader:
    # Forward pass (includes inner-loop TTL updates)
    output = model(batch)

    # Outer-loop loss (language modeling)
    loss = cross_entropy(output, targets)

    # Outer-loop update (AdamW on non-memory params)
    # Memory params already updated by TTL in forward pass
    loss.backward()
    optimizer.step()
```

**Key question:** Should gradients from outer loss also flow through M?

### 3. Detaching Memory State

To prevent outer-loop gradients from interfering with TTL:

```python
# In forward pass, after TTL update
mem_out = memory(k)
mem_out = mem_out.detach()  # Stop outer gradients here?
```

Or let both loops update M (potential conflict).

## Changes Required

### 1. Modify Training Loop

**File:** `scripts/train_smollm.py`

```python
def train_step(model, batch, optimizer):
    # TTL updates happen inside model.forward()

    # Forward pass
    logits = model(batch.input_ids)

    # Outer-loop loss
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        batch.labels.view(-1)
    )

    # Outer-loop backward (memory params excluded or detached)
    loss.backward()
    optimizer.step()

    return loss.item()
```

### 2. Memory State Reset Logic

```python
class AtlasModel(nn.Module):
    def reset_memory_state(self):
        """Reset memory and momentum to initial state."""
        for block in self.blocks:
            if hasattr(block, 'ttl_module'):
                block.ttl_module.reset()

    def forward(self, x, reset_memory=False):
        if reset_memory:
            self.reset_memory_state()
        ...
```

### 3. Update Gate Monitoring

**File:** `src/training/gate_monitor.py`

Add monitoring for:
- Memory update magnitudes
- Momentum norm
- Newton-Schulz convergence
- Omega loss values

```python
class GateMonitor:
    def log_ttl_metrics(self, ttl_module):
        self.log("memory/omega_loss", ttl_module.last_omega_loss)
        self.log("memory/momentum_norm", ttl_module.momentum_norm)
        self.log("memory/update_magnitude", ttl_module.update_magnitude)
```

### 4. Checkpoint Handling

Memory state should be saved/loaded with checkpoints:

```python
def save_checkpoint(model, optimizer, path):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'memory_state': model.get_memory_state(),  # New
    }
    torch.save(state, path)

def load_checkpoint(model, optimizer, path):
    state = torch.load(path)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    model.set_memory_state(state['memory_state'])  # New
```

## Files Affected

| File | Changes |
|------|---------|
| `scripts/train_smollm.py` | Training loop integration |
| `src/training/gate_monitor.py` | Add TTL metrics |
| `src/training/polarization.py` | Verify compatibility |
| `src/model/skeleton.py` | Memory reset API |
| `src/config.py` | Training hyperparameters |

## New Configuration Parameters

```python
# Memory state management
RESET_MEMORY_PER_SEQUENCE = True   # Reset at sequence boundaries
DETACH_MEMORY_OUTPUT = True        # Prevent outer gradients to M

# TTL scheduling
TTL_WARMUP_STEPS = 1000           # Steps before enabling TTL
TTL_UPDATE_FREQUENCY = 1           # TTL update every N tokens
```

## Dependencies
- **Requires:** `feature/ttl-update-loop` (Branch 3)
- **Requires:** `feature/wire-atlas-memory` (Branch 2)
- **Requires:** `refactor/output-level-combination` (Branch 1)

## Blocked By
- `feature/ttl-update-loop`

## Blocks
- Nothing (final branch)

## Acceptance Criteria
- [ ] Training loop handles TTL updates correctly
- [ ] Memory state resets at appropriate boundaries
- [ ] Two-level optimization works (inner TTL + outer backprop)
- [ ] Checkpoints save/restore memory state
- [ ] Gate monitor tracks TTL metrics
- [ ] Training converges (loss decreases)
- [ ] No memory leaks from state accumulation
- [ ] Gradient accumulation works correctly
- [ ] Multi-GPU training compatible (if applicable)

## Testing Plan

### 1. Sanity Check
```bash
poetry run python scripts/train_smollm.py --max-steps 100
# Verify loss decreases, no crashes
```

### 2. Memory State Test
```python
def test_memory_reset():
    model = AtlasModel()
    x1 = torch.randn(1, 100, 512)
    model(x1)  # Updates memory
    state1 = model.get_memory_state()

    model.reset_memory_state()
    state2 = model.get_memory_state()

    # Memory should be reset
    assert not torch.equal(state1, state2)
```

### 3. Convergence Test
Compare training curves with:
- Full Atlas (TTL enabled)
- Ablation (TTL disabled, `--disable-memory`)

Atlas should show better or equal convergence.

## References
- Issue #12
- Atlas Paper Section 1 (Test Time Memorization definition)
- Atlas Paper Section 5.1 (Parallel Training)
- Atlas Paper Section 6 (Experiments)
