# Atlas Code Smell Violations & Issues

Tracked issues against the NL graph's code smell rules (nl_code_smells collection).
Each issue references the specific code smell, affected files, and fix strategy.

**GitHub Issues**: [toddwbucy/Atlas-MAG_OmegaRule](https://github.com/toddwbucy/Atlas-MAG_OmegaRule/issues?q=label%3Anl-graph-compliance) (label: `nl-graph-compliance`)

## Status Key

| Status | Meaning |
|--------|---------|
| OPEN | Not yet addressed |
| IN_PROGRESS | Fix started |
| DONE | Fixed and tests passing |
| WONTFIX | Deliberately kept (with rationale) |

---

## Critical (Must Fix for Graph Compliance)

### ATL-01: Remove model.train()/model.eval() distinction [CS-10] -- [Issue #60](https://github.com/toddwbucy/Atlas-MAG_OmegaRule/issues/60) / [PR #63](https://github.com/toddwbucy/Atlas-MAG_OmegaRule/pull/63)
- **Status**: PR OPEN (2026-02-09)
- **Code Smell**: CS-10 -- No `model.train()` / `model.eval()`
- **Location**: `src/atlas/model/blocks.py:245` -- `if self.ttl_enabled and self.training:`
- **Impact**: TTL inner loop only runs when `self.training` is True. This means memory is NOT updated during inference, violating the NL principle that context processing is always the same.
- **Fix**: Remove the `self.training` gate. TTL should always run when `ttl_enabled=True`. The `training` flag is a PyTorch concept foreign to NL ontology.
- **Risk**: Low -- TTL during inference is actually what the paper specifies ("test-time learning").

### ATL-02: Remove DataLoader class [CS-11] -- [GitHub #52](https://github.com/toddwbucy/Atlas-MAG_OmegaRule/issues/52)
- **Status**: OPEN
- **Code Smell**: CS-11 -- No `TrainingLoop` or `DataLoader` class
- **Location**: `src/atlas/data/smollm_dataset.py:453-524` -- `create_smollm_dataloader()`, `create_smollm_val_dataloader()`
- **Also**: `scripts/train.py:199`, `scripts/eval_worker.py:100`, `tests/test_smollm_dataset.py:284-316`
- **Impact**: Uses `torch.utils.data.DataLoader` directly. Should use a context processor pattern.
- **Fix**: Replace DataLoader with a streaming context processor that yields token sequences. The underlying `IterableDataset` is fine -- it's the DataLoader wrapper that introduces the foreign concept.
- **Risk**: Medium -- affects training script and tests. Must maintain functional equivalence.

### ATL-03: Rename "training" concepts [CS-13] -- [Issue #59](https://github.com/toddwbucy/Atlas-MAG_OmegaRule/issues/59) / [PR #62](https://github.com/toddwbucy/Atlas-MAG_OmegaRule/pull/62)
- **Status**: PR OPEN (2026-02-09) -- training/ renamed to runtime/
- **Code Smell**: CS-13 -- The word "training" is a code smell
- **Location**: Multiple files -- `src/atlas/training/` directory name, function names, docstrings
- **Impact**: Pervasive use of "training" vocabulary throughout `training/` subpackage.
- **Fix**: Rename `training/` to `runtime/` or `processing/`. Rename `train.py` to `run.py`. Update all references. The concept is "context processing", not "training".
- **Risk**: Low (renaming only) but many files touched.

### ATL-04: Replace AdamW with M3 optimizer [CS-28] -- [GitHub #53](https://github.com/toddwbucy/Atlas-MAG_OmegaRule/issues/53)
- **Status**: OPEN
- **Code Smell**: CS-28 -- Adam/SGD/AdamW are forbidden
- **Location**: `scripts/train.py:39,226-255` -- imports and uses `torch.optim.AdamW`
- **Impact**: Outer loop uses AdamW instead of M3 (the NL-native optimizer). This is the most visible violation.
- **Fix**: Replace with M3 optimizer (already defined in HOPE's graph as `hope_algorithms/alg-m3`). Reuse the M3 implementation from HOPE or implement Atlas-specific variant.
- **Risk**: Medium -- different optimizer may change convergence behavior. Need to validate.

### ATL-05: Remove independent optimizer selection [CS-27] -- [GitHub #54](https://github.com/toddwbucy/Atlas-MAG_OmegaRule/issues/54)
- **Status**: OPEN
- **Code Smell**: CS-27 -- No independent optimizer selection
- **Location**: `scripts/train.py:226-255` -- manual parameter group construction
- **Impact**: Training script manually excludes memory parameters from AdamW. This is independent optimizer selection logic.
- **Fix**: Follows from ATL-04. Once M3 is used, the optimizer is determined by the graph, not by script logic.
- **Risk**: Low (follows ATL-04).

---

## High (Architectural Alignment)

### ATL-06: Memory is a separate module [CS-01] -- [GitHub #55](https://github.com/toddwbucy/Atlas-MAG_OmegaRule/issues/55)
- **Status**: OPEN
- **Code Smell**: CS-01 -- No `MemoryModule` class (memory IS the parameters)
- **Location**: `src/atlas/model/atlas_memory.py` -- `AtlasMemoryPoly` class (313 lines)
- **Also**: `src/atlas/model/blocks.py:193` -- `self.memory = AtlasMemoryPoly(...)`
- **Impact**: Memory is implemented as an independent `nn.Module` that is composed into the block. In NL ontology, memory IS the parameters -- there shouldn't be a separate "memory module".
- **Fix**: Large refactor -- integrate memory weights directly into the block as parameters. The polynomial feature expansion and Q-K projection would become part of the block's forward pass, not a separate module.
- **Risk**: High -- largest refactor, affects tests. Defer to later pass.

### ATL-07: Multiple external APIs beyond forward [CS-18] -- [Issue #61](https://github.com/toddwbucy/Atlas-MAG_OmegaRule/issues/61) / [PR #64](https://github.com/toddwbucy/Atlas-MAG_OmegaRule/pull/64)
- **Status**: PR OPEN (2026-02-09) -- removed forward_memory_only, get_gate_values, set_ttl_enabled; renamed reset_ttl_momentum to _reset_ttl_momentum
- **Code Smell**: CS-18 -- Forward pass IS the only external API
- **Location**: `src/atlas/model/skeleton.py:238-293`
  - `forward_memory_only()` -- returns memory state alongside logits
  - `get_gate_values()` -- extracts gate monitoring values
  - `reset_ttl_momentum()` -- resets momentum buffers
  - `set_ttl_enabled()` -- toggles TTL on/off
  - `count_parameters()` -- parameter counting utility
- **Impact**: Multiple external methods beyond `forward()`. Only `forward()` should be the external API.
- **Fix**: Move monitoring data into `forward()` return (optional dict). Move reset logic to internal mechanisms. Remove `set_ttl_enabled` (TTL always on per ATL-01). `count_parameters` is a diagnostic and may stay.
- **Risk**: Medium -- affects training script and evaluation scripts.

### ATL-08: Verify observe-then-advance in TTL [CS-32]
- **Status**: DONE (2026-02-09) -- verified compliant
- **Code Smell**: CS-32 -- Observe then advance: stateful counters mutate AFTER all observers
- **Location**: `src/atlas/runtime/ttl_update.py:260-271` -- `TTLUpdater.step()`
- **Impact**: Step counter increments at line 270 AFTER the TTL update at line 260. This appears correct (observe the update, then advance counter). Need to verify no other stateful counters violate this.
- **Fix**: Audit all stateful counters in TTL and memory. Verify observe-then-advance ordering.
- **Risk**: Low -- likely already correct, but audit needed.

---

## Enhancement (Graph-Informed Improvements)

### ATL-09: Outer loop should use Muon/M3, not AdamW -- [GitHub #56](https://github.com/toddwbucy/Atlas-MAG_OmegaRule/issues/56)
- **Status**: OPEN
- **Source**: Atlas IS #3 (each optimization level has its own optimizer)
- **Description**: The outer loop should use the NL-native M3 optimizer, which includes Newton-Schulz orthogonalization. The inner loop already uses Muon (NS-5) -- outer loop should match.
- **Fix**: Superseded by ATL-04. When ATL-04 is implemented, this is automatically resolved.

### ATL-10: Add MAL variant alongside MAG -- [GitHub #57](https://github.com/toddwbucy/Atlas-MAG_OmegaRule/issues/57)
- **Status**: OPEN
- **Source**: Atlas IS #6 (composable building blocks)
- **Description**: Current implementation only has MAG (Memory-as-Gate). Atlas also defines MAL (Memory-as-Layer) where memory output is additive, not multiplicative. Adding MAL would validate the composability axiom.
- **Fix**: Create `MALBlock` alongside `MAGBlock` in `blocks.py`. Separate session -- stretch goal.

### ATL-11: Fix pseudocode-identified mismatches -- [GitHub #58](https://github.com/toddwbucy/Atlas-MAG_OmegaRule/issues/58)
- **Status**: OPEN
- **Source**: Phase 2 pseudocode extraction
- **Description**: Two concrete mismatches found:
  1. **Learnable coefficients `a_i`** from Eq 5 not implemented -- polynomial expansion hardcodes degree-2 without learnable weighting
  2. **Residual at key-dim vs full-dim** -- intentional deviation (WONTFIX)
- **Fix**: Add `nn.Parameter` for `a_1, a_2` in `AtlasMemoryPoly` and scale polynomial terms accordingly.

---

## Cross-Reference: GitHub Issues

| Local ID | GitHub Issue | PR | Code Smell | Priority | Status |
|----------|-------------|-----|-----------|----------|--------|
| ATL-01 | [#60](https://github.com/toddwbucy/Atlas-MAG_OmegaRule/issues/60) | [#63](https://github.com/toddwbucy/Atlas-MAG_OmegaRule/pull/63) | CS-10 | Critical | PR OPEN |
| ATL-02 | [#52](https://github.com/toddwbucy/Atlas-MAG_OmegaRule/issues/52) | -- | CS-11 | Critical | OPEN |
| ATL-03 | [#59](https://github.com/toddwbucy/Atlas-MAG_OmegaRule/issues/59) | [#62](https://github.com/toddwbucy/Atlas-MAG_OmegaRule/pull/62) | CS-13 | Critical | PR OPEN |
| ATL-04 | [#53](https://github.com/toddwbucy/Atlas-MAG_OmegaRule/issues/53) | -- | CS-28 | Critical | OPEN |
| ATL-05 | [#54](https://github.com/toddwbucy/Atlas-MAG_OmegaRule/issues/54) | -- | CS-27 | Critical | OPEN |
| ATL-06 | [#55](https://github.com/toddwbucy/Atlas-MAG_OmegaRule/issues/55) | -- | CS-01 | High | OPEN |
| ATL-07 | [#61](https://github.com/toddwbucy/Atlas-MAG_OmegaRule/issues/61) | [#64](https://github.com/toddwbucy/Atlas-MAG_OmegaRule/pull/64) | CS-18 | High | PR OPEN |
| ATL-08 | -- | -- | CS-32 | High | DONE (audit) |
| ATL-09 | [#56](https://github.com/toddwbucy/Atlas-MAG_OmegaRule/issues/56) | -- | -- | Enhancement | OPEN |
| ATL-10 | [#57](https://github.com/toddwbucy/Atlas-MAG_OmegaRule/issues/57) | -- | -- | Enhancement | OPEN |
| ATL-11 | [#58](https://github.com/toddwbucy/Atlas-MAG_OmegaRule/issues/58) | -- | -- | Bug | OPEN |

## Fix Order (Phase 4)

1. ~~**ATL-08** (CS-32): Verify observe-then-advance~~ -- DONE
2. ~~**ATL-01** (CS-10): Remove train/eval distinction~~ -- DONE
3. ~~**ATL-03** (CS-13): Rename training concepts~~ -- DONE
4. ~~**ATL-07** (CS-18): Consolidate to forward-only API~~ -- DONE
5. **ATL-02** (CS-11): Replace DataLoader with streaming context processor
6. **ATL-04 + ATL-05** (CS-28/27): Replace AdamW with M3 optimizer
7. **ATL-06** (CS-01): Integrate memory into model (largest refactor)
8. **ATL-09**: Wire up M3 as outer-loop optimizer (follows ATL-04)
9. **ATL-11**: Fix pseudocode-identified mismatches
10. **ATL-10**: Add MAL variant (stretch goal -- separate session)
