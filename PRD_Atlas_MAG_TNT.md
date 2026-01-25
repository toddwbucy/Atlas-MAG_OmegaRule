# Product Requirements Document: Atlas-MAG with TNT Training

**Document Version**: 5.4
**Status**: ✅ **ALL PHASES GO** | Diagnostic checkpoints (not blockers)
**Last Updated**: 2026-01-24
**Authors**: [Engineering Team]

---

## ⚠️ PROOF OF CONCEPT SCOPE

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  This PRD defines a 60M PARAMETER TOY MODEL to validate the    │
│  Atlas-MAG + TNT architecture BEFORE committing to scale-up.   │
│                                                                 │
│  CURRENT PHASE: Architecture Validation                        │
│  CONTEXT: Single engineer, rapid iteration, empirical focus    │
│                                                                 │
│  HARDWARE (FULL CAPACITY - NOT SPLIT):                         │
│    • GPU 0 + GPU 1: RTX A6000 48GB × 2 with NVLink             │
│      → Both GPUs run ONE model at FULL 2048 token context      │
│    • GPU 2: RTX 2000 16GB — Validation when needed             │
│                                                                 │
│  PHILOSOPHY (Round 8):                                          │
│    ✓ Diagnostic checkpoints, NOT blocking gates                │
│    ✓ Keep momentum — code while calibrating                    │
│    ✓ Research-grade tolerance (note issues, keep going)        │
│    ✓ Full-capacity runs (no A/B splitting of compute)          │
│                                                                 │
│  WHAT THIS IS:                                                  │
│    ✓ Prove the architecture works                              │
│    ✓ Get to training FAST                                      │
│    ✓ First epoch loss curves → committee review                │
│                                                                 │
│  WHAT THIS IS NOT:                                              │
│    ✗ Production system                                         │
│    ✗ Defense-grade safety (that's for scale-up)                │
│    ✗ Premature optimization                                    │
│                                                                 │
│  SUCCESS CRITERIA: Loss curves from first epoch                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

NOTE ON MOMENTUM: "The loudest killer of any side project is losing
momentum." This PRD is a MAP, not a LAW. Phases are diagnostic
checkpoints, not blockers. Start coding immediately.
```

---

> **Round 8 Verdict**: "OVER-ENGINEERED FOR TEAM OF ONE"
>
> "You are clearly ready to build. This PRD is a map, not a law. Ignore the blockers. Run the code."
>
> "The loudest killer of any side project is losing momentum."

---

## Executive Summary

This PRD defines the requirements for implementing **Atlas-MAG**, a hybrid sequence model combining Sliding Window Attention (SWA) with a Deep Neural Long-Term Memory (Atlas), trained using the **TNT (Titans-in-Titans)** two-stage training framework.

**Objective**: Build a scalable sequence model that achieves:
- **Higher capacity** via deep memory and polynomial feature maps
- **Context-aware memorization** via the Omega Rule (sliding window updates)
- **17× training speedup** via TNT hierarchical memory with context parallelism
- **Inference optimization** via two-stage training resolving train-test chunk mismatch

**Target Hardware**:
- 2× RTX A6000 48GB with NVLink (96GB total VRAM)
- Threadripper 7960x with 256GB ECC RAM
- Toy model: 50-60M parameters (d=768)

---

## Critical Issues Summary (All Rounds)

### Round 8 (Current) - SIMPLIFY FOR SOLO DEV

> "The loudest killer of any side project is losing momentum."

| Issue | Category | The Problem | Fix |
|-------|----------|-------------|-----|
| Sequential Phase Blocking | Process | Can't code while waiting | **Diagnostic checkpoints** |
| Static norm_persistent | Infrastructure | Manual recalc kills experimentation | **Runtime compute + NCCL** |
| K Decoupling | Optimization | Premature optimization | **K=5 everywhere** |
| Parallel A/B Tests | Testing | Halves context window | **CANCELLED** |
| Slow Gate Validation | Architecture | Feedback too slow | **Fast-fail at step 100-500** |

### Round 7 - 90% OF THE WAY THERE (Superseded by R8)

| Issue | Category | Fix | R8 Update |
|-------|----------|-----|-----------|
| Subjective K Validation | Phase 3 | Automated 1e-4 trigger | **DELETED** - K=5 everywhere |
| Slow Annealing Feedback | Phase 1 | Polarization velocity check | **Fast-fail** at step 100-500 |
| Configuration Drift | Phase 0 | Deferred | **Runtime NCCL** |
| Unbounded Gradient Clip | Phase 4 | A/B Test | **CANCELLED** - trust theory |

### Round 6 - GATES ARE CAUTION TAPE (Incorporated, Simplified by R8)

> "The restructuring was a huge step forward... but the devils are in the floating point details."

| Issue | Category | The Weakness | Fix |
|-------|----------|--------------|-----|
| Option B Too Passive | Initialization | Model drifts back to mushy middle | **Remove Option B**, mandate annealing λ |
| norm_persistent Runtime | Infrastructure | FP divergence across GPUs | **Runtime compute + NCCL broadcast** (R8) |
| K Decoupling | Optimization | Premature optimization | **K=5 everywhere** (R8 simplified) |
| Static Clip Scale | Training | Semi-truck brakes on go-kart | **Dynamic κ(t)** scales with chunk size |

### Round 5 - SILENT KILLERS (Incorporated)

> "These four issues are silent killers for phase two."

| Issue | Category | The Trap | Fix | R6 Update |
|-------|----------|----------|-----|-----------|
| Gate Mushy Middle | Initialization | Model averages instead of routes | Polarization loss | **Annealing λ mandatory** |
| M_persistent Normalization | Infrastructure | Volume mismatch corrupts retrieval | Add norm_persistent scalar | **Static constant** |
| Reactive Rollback | Training | Treats symptom, not disease | Proactive gradient clipping | **Dynamic threshold** |
| Muon Pipeline Bubble | Optimization | Latency kills 17× speedup | Decouple K values | **K warmup if needed** |

### Round 4 - ✅ RESOLVED

| Issue | Resolution |
|-------|------------|
| OQ-2: Muon Scope | Full Muon (K=5 both) → **REVISED R5**: Decouple K values |
| OQ-8: Gamma Baseline | γ = 0.9993 (TNT paper Table 4) |
| OQ-10: Persistent Tokens | N_p = 64 tokens (~2.6 MB for d=768) |
| Auto-Rollback | Implemented → **EXTENDED R5**: Add proactive gradient clipping |

### Round 3 - ✅ INCORPORATED

| Issue | Resolution |
|-------|------------|
| Newton-Schulz K | **K=5 everywhere** (R8: "get it working first") |
| M_persistent | Initialize M_t with M_persistent → **EXTENDED R5**: Add norm_persistent |
| Stage 2 Decay | Continuous geometric decay (γ=0.9993) |

### Round 2 - ✅ INCORPORATED

| Issue | Resolution |
|-------|------------|
| Vanity Metrics | NIAH probes replace gradient metrics |
| W_init Reset Shock | Steady-state initialization |
| Stage 2 Cliff | Geometric annealing |

---

## Table of Contents

1. [Goals and Non-Goals](#1-goals-and-non-goals)
2. [Implementation Phases (Diagnostic Checkpoints)](#2-implementation-phases-diagnostic-checkpoints)
3. [Phase 0: Foundation & Calibration](#3-phase-0-foundation--calibration)
4. [Phase 1: Architecture Validation](#4-phase-1-architecture-validation)
5. [Phase 2: Training Infrastructure](#5-phase-2-training-infrastructure)
6. [Phase 3: Optimization](#6-phase-3-optimization-simplified---round-8)
7. [Phase 4: Stage 2 Fine-Tuning](#7-phase-4-stage-2-fine-tuning)
8. [Phase 5: Inference & Release](#8-phase-5-inference--release)
9. [Technical Requirements Reference](#9-technical-requirements-reference)
10. [Hyperparameter Specifications](#10-hyperparameter-specifications)
11. [Acceptance Criteria](#11-acceptance-criteria)
12. [Appendix: Mathematical Definitions](#appendix-mathematical-definitions)

---

## 1. Goals and Non-Goals

### 1.1 Goals

| ID | Goal | Success Metric | Validated At |
|----|------|----------------|--------------|
| G1 | MAG dual-branch with verified memory utility | NIAH retrieval accuracy > 80% | Phase 2 Gate |
| G2 | Super-linear memory capacity | O(d^p) where p ≥ 2 | Phase 1 Gate |
| G3 | TNT training efficiency | Throughput ≥10× baseline | Phase 3 Gate |
| G4 | Dynamic inference modes | Seamless prefill→decode | Phase 5 Gate |
| G5 | Constant memory complexity | O(d²) regardless of seq_len | Phase 2 Gate |
| G6 | Resolution-robust inference | Flat PPL curve {1-64} | Phase 4 Gate |
| G7 | Persistent memory retrievable | Accessible from first token | Phase 0 Gate |
| G8 | Gate makes decisive routing | ≥20% tokens polarized (<0.1 or >0.9) | Phase 4 Gate |

### 1.2 Non-Goals

- Custom CUDA kernel development (use existing FlashAttention, cuBLAS)
- Distributed training across multiple nodes (single-node multi-GPU only for v1)
- Quantization or model compression (full precision only for v1)
- Integration with specific inference serving frameworks

---

## 2. Implementation Phases (Diagnostic Checkpoints)

### 2.1 Phase Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   PHASE FLOW (ALL PHASES GO)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐      │
│   │ PHASE 0  │────▶│ PHASE 1  │────▶│ PHASE 2  │────▶│ PHASE 3  │      │
│   │Foundation│     │  Arch    │     │ Training │     │  Optim   │      │
│   └──────────┘     └──────────┘     └──────────┘     └──────────┘      │
│        │                │                │                │             │
│        ▼                ▼                ▼                ▼             │
│   [CHECKPOINT]    [CHECKPOINT]    [CHECKPOINT]    [CHECKPOINT]         │
│   (log gaps)      (log gaps)      (log gaps)      (log gaps)           │
│                                                          │              │
│                                          ┌───────────────┘              │
│                                          ▼                              │
│                                    ┌──────────┐     ┌──────────┐       │
│                                    │ PHASE 4  │────▶│ PHASE 5  │       │
│                                    │ Stage 2  │     │ Inference│       │
│                                    └──────────┘     └──────────┘       │
│                                          │                │             │
│                                          ▼                ▼             │
│                                    [CHECKPOINT]    [CHECKPOINT]        │
│                                                          │              │
│                                                          ▼              │
│                                                   [LOSS CURVES]         │
│                                                   (to committee)        │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Phase Status (All GO)

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0: Foundation | ✅ **GO** | Start immediately |
| Phase 1: Architecture | ✅ **GO** | Code while Phase 0 runs |
| Phase 2: Training | ✅ **GO** | Code while calibrating |
| Phase 3: Optimization | ✅ **GO** | K=5 everywhere (simplified) |
| Phase 4: Stage 2 | ✅ **GO** | Dynamic clip ready |
| Phase 5: Inference | ✅ **GO** | Test when training converges |

### 2.3 Checkpoint Philosophy (Round 8)

> **Diagnostic checkpoints, NOT blocking gates.**
>
> - Log gaps, keep building
> - "You can keep building the car even if the engine isn't perfectly tuned"
> - Momentum > perfection for solo dev
> - Research-grade tolerance: note issues, continue

```python
# OLD - Blocking gate
if not phase_0_complete:
    raise BlockingError("Cannot proceed to Phase 1")

# NEW - Diagnostic checkpoint
if not phase_0_complete:
    log_warning("⚠️ Phase 0 incomplete - proceeding with known gaps")
    tech_debt.register("norm_persistent not finalized")
```

---

## 3. Phase 0: Foundation & Calibration

**Goal**: Establish initialization primitives that all subsequent phases depend on

**Duration**: Week 1-2

### 3.1 Engineering Tasks

| Task ID | Description | Former OQ | Acceptance Criteria |
|---------|-------------|-----------|---------------------|
| P0-T1 | Compute W_init via steady-state calibration | OQ-9 | Reset shock < 5% loss spike |
| P0-T2 | Compute M_persistent from 64 persistent keys | - | All keys included in outer product sum |
| P0-T3 | **Compute norm_persistent scalar** | NEW (R5) | Denominator matches TNT formula |
| P0-T4 | Implement hash verification infrastructure | - | All GPUs report identical hash |
| P0-T5 | Determine calibration batch size | OQ-9 | Start at 10K tokens, increase if reset shock > 5% |

### 3.2 Deliverables

```
□ W_init parameter (learnable, shared across shards)
□ M_persistent matrix (d × d, precomputed)
□ norm_persistent scalar (sum of ||k_p||² for all persistent keys)
□ Hash verification function
□ Calibration data pipeline
□ Skeleton model code that compiles and runs forward pass
```

### 3.3 Requirements

**REQ-P0-001**: Steady-State W_init Initialization

```python
def compute_steady_state_init(model, calibration_data, num_samples=10000):
    """
    Phase 0: Compute steady-state W_init before training.

    ACCEPTANCE TEST:
      After initialization, measure loss spike at first reset boundary.
      If spike > 5%, INCREASE num_samples and retry.
    """
    memory_states = []

    with torch.no_grad():
        for batch in calibration_data:
            if sum(len(s) for s in memory_states) >= num_samples:
                break
            _, memory_state = model.forward_memory_only(batch)
            memory_states.append(memory_state)

    W_init = torch.stack(memory_states).mean(dim=0)
    return W_init
```

**REQ-P0-002**: M_persistent with Normalization (REVISED - Round 8)

> **ROUND 8 SIMPLIFICATION**: Compute at runtime, broadcast via NCCL
>
> **WHY**: "If you want to change persistent tokens, you'd have to manually recalculate. That friction kills experimentation."

```python
# ============================================================
# ROUND 8 SIMPLIFIED: Runtime compute + NCCL broadcast
# ============================================================
def compute_and_broadcast_norm_persistent(persistent_keys, rank, world_size):
    """
    Phase 0: Compute at runtime on rank 0, broadcast to all.

    NCCL guarantees bitwise identical values across GPUs.
    "Trust NCCL to do its job."

    Round 8: "Don't manually calculate norm_persistent.
              Compute at runtime. Trust NCCL."
    """
    if rank == 0:
        # Compute on primary GPU
        norm_persistent = sum(
            torch.linalg.norm(k_p) ** 2
            for k_p in persistent_keys
        )
        if isinstance(norm_persistent, (int, float)):
            norm_persistent = torch.tensor([norm_persistent], device='cuda')
    else:
        norm_persistent = torch.zeros(1, device='cuda')

    # NCCL broadcast guarantees identical bits
    dist.broadcast(norm_persistent, src=0)

    log_info(f"✓ norm_persistent computed and broadcast: {norm_persistent.item():.6f}")
    return norm_persistent.item()
```

**Optional Drift Check (Warning, Not Fatal)**:
```python
def check_norm_drift(norm_persistent, world_size, rank):
    """
    Research-grade: Log warning if drift detected, but continue.

    Round 8: "Defense grade stops if a butterfly flaps its wings.
              Research grade notes the butterfly and keeps going."
    """
    all_values = [torch.zeros(1, device='cuda') for _ in range(world_size)]
    dist.all_gather(all_values, torch.tensor([norm_persistent], device='cuda'))

    values = [v.item() for v in all_values]
    drift = max(values) - min(values)

    if drift > 1e-8:
        log_warning(f"Minor norm drift: {drift:.2e} (acceptable for POC)")
    else:
        log_info("✓ norm_persistent consistent across cluster")
```

**REQ-P0-003**: Hash Verification

```python
def verify_m_persistent_consistency():
    """
    MUST PASS before training starts.
    """
    import hashlib

    local_hash = hashlib.sha256(
        M_persistent.cpu().numpy().tobytes()
    ).hexdigest()

    all_hashes = [None] * dist.get_world_size()
    dist.all_gather_object(all_hashes, local_hash)

    if len(set(all_hashes)) != 1:
        raise RuntimeError(f"Persistent memory divergence detected!")

    log_info("✓ M_persistent consistency verified")
```

### 3.4 DIAGNOSTIC CHECKPOINT

```
┌─────────────────────────────────────────────────────────────────┐
│                 PHASE 0 DIAGNOSTIC CHECKPOINT                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  □ G0-1: W_init computed                                        │
│          Test: Reset shock at shard boundary < 5% loss spike    │
│          If fail: Log warning, continue anyway                  │
│                                                                 │
│  □ G0-2: M_persistent computed with all 64 persistent keys      │
│          Test: M_persistent.shape == (d, d)                     │
│                                                                 │
│  □ G0-3: norm_persistent computed at runtime (R8)               │
│          Method: Compute on rank 0, NCCL broadcast              │
│          "Trust NCCL to do its job"                             │
│                                                                 │
│  □ G0-4: Hash verification passes across all GPUs               │
│          Test: All hashes match                                 │
│          If fail: WARNING (not fatal for POC)                   │
│                                                                 │
│  □ G0-5: Skeleton code runs forward pass without crash          │
│          Test: forward(random_input) returns valid tensor       │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  ALL BOXES CHECKED → Great! Proceed confidently                 │
│  SOME BOXES UNCHECKED → Log gaps, proceed anyway (R8)           │
│  "Keep momentum. You can fix issues while training runs."       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Phase 1: Architecture Validation

**Goal**: Prove the MAG architecture can actually route (not average)

**Duration**: Week 3-5

**Prerequisites**: Phase 0 checkpoint logged

### 4.1 Engineering Tasks

| Task ID | Description | Decision Protocol |
|---------|-------------|-------------------|
| P1-T1 | Gate polarization mechanism | See §4.2 Decision Protocol |
| P1-T2 | Gate noise suppression test | Must pass: gate < 0.01 on noise |
| P1-T3 | Polynomial feature degree | See §4.3 Decision Protocol |
| P1-T4 | Memory depth | See §4.4 Decision Protocol |
| P1-T5 | **Fast-fail gate check (R8)** | Abort if gate.std < 0.01 at step 500 |
| P1-T6 | Multiplicative fusion | Unit test: gate=0 → output=0 |
| P1-T7 | Gated MLP structure | Matches Atlas++ formula |

---

### 4.2 Gate Polarization: Annealing λ (REVISED - Round 6)

> **ROUND 6 FIX**: Option B (high-variance init) REMOVED - "too passive, hope and pray"
>
> **MANDATE**: Polarization loss with annealing schedule
>
> **WHY**: "Neural networks are lazy. They will find the path of least resistance. The gate becomes a screen door."

**REQ-P1-001**: Polarization Loss with Annealing Schedule

```python
def get_lambda_polar(step, total_steps):
    """
    Annealing schedule for polarization penalty.

    First 10% of training: HIGH λ
      - "Basically forbids outputting anything other than 0 or 1"
      - "Force it to learn the routing primitive first"

    Remaining 90%: Decay λ
      - "Allow for subtle mixing later on"
    """
    warmup_steps = int(0.1 * total_steps)

    if step < warmup_steps:
        return 10.0  # HIGH - winner-take-all regime
    else:
        # Decay from 10.0 → 0.1 over remaining steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 10.0 * (0.01 ** progress)  # Exponential decay


def gate_polarization_loss(gate_values, step, total_steps):
    """
    Add penalty for gates near 0.5.

    L_polar = λ(t) × (1 - |2g - 1|)
    """
    lambda_polar = get_lambda_polar(step, total_steps)
    polarization = 1.0 - torch.abs(2 * gate_values - 1)
    return lambda_polar * polarization.mean()
```

**FORBIDDEN OPTIONS**:
- ❌ Option B (high-variance init) - "Too passive, gates drift back"
- ❌ Static zero bias - Causes mushy middle from step 1
- ❌ Static λ without annealing - Doesn't "teach routing first"

**CALIBRATION PROTOCOL**:

```
┌─────────────────────────────────────────────────────────────────┐
│            ANNEALING λ CALIBRATION PROTOCOL                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STEP 1: Run 1000 steps with annealing λ schedule               │
│          - λ_initial = 10.0 (first 10% = 100 steps)             │
│          - λ_final = 0.1 (by step 1000)                         │
│                                                                 │
│  STEP 2: Measure at checkpoints                                 │
│          ┌────────────────────────────────────────────────┐     │
│          │ Step │ λ(t)  │ Gate Polar% │ Loss │ NIAH %    │     │
│          ├────────────────────────────────────────────────┤     │
│          │ 100  │ 10.0  │ _______     │ ____ │ _______   │     │
│          │ 500  │ ~1.0  │ _______     │ ____ │ _______   │     │
│          │ 1000 │ 0.1   │ _______     │ ____ │ _______   │     │
│          └────────────────────────────────────────────────┘     │
│                                                                 │
│  STEP 3: Validation Rule                                        │
│          ┌────────────────────────────────────────────────┐     │
│          │ AT STEP 100 (peak λ):                         │     │
│          │   Polarization MUST be ≥ 80%                  │     │
│          │   (gates forced to extremes)                  │     │
│          │                                                │     │
│          │ AT STEP 1000 (low λ):                         │     │
│          │   Polarization MUST remain ≥ 20%              │     │
│          │   (routing behavior "stuck")                  │     │
│          │                                                │     │
│          │ IF polarization drops < 20% after warmup:     │     │
│          │   → Increase λ_final (e.g., 0.5 instead of 0.1)│     │
│          │   → Restart calibration                       │     │
│          └────────────────────────────────────────────────┘     │
│                                                                 │
│  STEP 4: Document final λ schedule with polarization curve      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**FAST-FAIL GATE CHECK (Round 8)**:

> "The system needs to scream 'I'm broken' immediately, not whisper it three hours later."

```python
def fast_fail_gate_check(gate_values, step, initial_variance=None):
    """
    Don't wait for λ decay. Check trajectory immediately.

    Round 8: "You could be 10 or 20 percent into your training run
              before you even know if the gates are working."
    """
    current_variance = gate_values.var().item()

    if step == 100:
        # Store initial variance for trajectory check
        return current_variance

    if step == 500:
        if current_variance < initial_variance * 1.5:
            raise FastFailError(
                f"Gate variance not increasing! "
                f"Step 100: {initial_variance:.4f}, Step 500: {current_variance:.4f}. "
                f"ABORTING - architecture is stuck in mushy middle."
            )

    # Continuous check: if gates collapse, abort immediately
    if step > 100 and current_variance < 0.01:
        raise FastFailError(
            f"Gate std < 0.01 after step {step}. "
            f"Killing run - gates not learning."
        )

    return current_variance
```

---

### 4.3 Decision Protocol: Polynomial Degree (P1-T3)

> **GOAL**: Choose between degree 2 and degree 3 polynomial feature expansion

**Trade-off Analysis**:
```
Degree 2: d=768 → 768 + 295,296 = 296,064 features
          Memory: ~1.1 MB per token (fp32)
          Compute: O(d²)

Degree 3: d=768 → 768 + 295,296 + 75,287,520 = 75,583,584 features
          Memory: ~289 MB per token (fp32) ← LIKELY TOO LARGE
          Compute: O(d³)
```

**DECISION PROTOCOL**:

```
┌─────────────────────────────────────────────────────────────────┐
│            POLYNOMIAL DEGREE DECISION PROTOCOL                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STEP 1: Calculate memory requirements for both                 │
│          d=768, batch_size=32, seq_len=2048                     │
│                                                                 │
│          Degree 2 memory: _______ GB                            │
│          Degree 3 memory: _______ GB                            │
│          Available VRAM:  96 GB                                 │
│                                                                 │
│  STEP 2: Decision Rule                                          │
│          ┌────────────────────────────────────────────────┐     │
│          │ IF degree 3 memory > 50% of VRAM:             │     │
│          │   → Use degree 2 (memory constraint)          │     │
│          │                                                │     │
│          │ IF degree 3 memory ≤ 50% of VRAM:             │     │
│          │   → Run 100-step comparison                   │     │
│          │   → Measure: loss convergence rate            │     │
│          │   → Choose: faster convergence wins           │     │
│          │   → Tie-breaker: degree 2 (simpler)           │     │
│          └────────────────────────────────────────────────┘     │
│                                                                 │
│  EXPECTED OUTCOME: Degree 2 (degree 3 likely exceeds memory)    │
│                                                                 │
│  STEP 3: Document decision with memory calculation              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### 4.4 Decision Protocol: Memory Depth (P1-T4)

> **GOAL**: Choose between L_M = 2 and L_M = 3 gated MLP layers

**Trade-off Analysis**:
```
L_M = 2: Fewer parameters, faster forward pass
         May have limited expressiveness

L_M = 3: More parameters, slower forward pass
         Higher expressiveness, risk of overfitting on toy model
```

**DECISION PROTOCOL**:

```
┌─────────────────────────────────────────────────────────────────┐
│            MEMORY DEPTH DECISION PROTOCOL                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STEP 1: Calculate parameter counts                             │
│          (using chosen polynomial degree from P1-T3)            │
│                                                                 │
│          L_M=2 parameters: _______ M                            │
│          L_M=3 parameters: _______ M                            │
│          Target model size: 50-60M                              │
│                                                                 │
│  STEP 2: Decision Rule                                          │
│          ┌────────────────────────────────────────────────┐     │
│          │ IF L_M=3 pushes total params > 70M:           │     │
│          │   → Use L_M=2 (stay within toy model budget)  │     │
│          │                                                │     │
│          │ IF both fit within 70M:                       │     │
│          │   → Run gate noise suppression test (P1-T2)   │     │
│          │     with both depths                          │     │
│          │   → Measure: steps to reach gate < 0.1        │     │
│          │   → Choose: fewer steps wins                  │     │
│          │   → Tie-breaker: L_M=2 (simpler)              │     │
│          └────────────────────────────────────────────────┘     │
│                                                                 │
│  STEP 3: Document decision with parameter count                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### 4.5 Architectural Requirements

**REQ-P1-002**: Gated MLP Structure (Atlas++)
```
M(x) = x + W₁ · σ(W₂ · x) ⊙ W₃ · x
```

**REQ-P1-003**: Multiplicative Fusion
```python
# REQUIRED implementation
gate = sigmoid(linear_projection(atlas_output))
output = attention_output * gate

# FORBIDDEN implementation
output = attention_output + atlas_output  # Cannot suppress attention
```

---

### 4.6 DIAGNOSTIC CHECKPOINT

```
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 1 EXIT GATE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  □ G1-1: Gate noise suppression test PASSES (TIGHTENED - R6)    │
│          Test: Gate suppresses noise branch to mean < 0.01      │
│          (NOT 0.1 - "10% leakage is huge when it's pure noise") │
│                                                                 │
│  □ G1-2: Annealing λ calibration COMPLETED (§5.2 Protocol)      │
│          Evidence: Polarization curve at steps 100, 500, 1000   │
│          REQUIRE: ≥80% at step 100, ≥20% at step 1000           │
│          Document: "λ schedule: [values], polarization: [curve]"│
│                                                                 │
│  □ G1-3: Polynomial degree DECIDED (§5.3 Protocol)              │
│          Evidence: Memory calculation for d=768                 │
│          Decision: Degree [2|3] selected per decision rule      │
│          Document: "Degree [N] because [memory/perf reason]"    │
│                                                                 │
│  □ G1-4: Memory depth DECIDED (§5.4 Protocol)                   │
│          Evidence: Parameter count for both options             │
│          Decision: L_M=[2|3] selected per decision rule         │
│          Document: "L_M=[N] because [param count/perf reason]"  │
│                                                                 │
│  □ G1-5: Forward pass produces non-trivial gates                │
│          Test: gate.std() > 0.1 (not all clustered at 0.5)      │
│                                                                 │
│  □ G1-6: Multiplicative fusion verified                         │
│          Test: gate=0 → output=0 regardless of attention        │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  GATE CRITERIA:                                                 │
│    - ALL boxes checked with documented evidence                 │
│    - Decision protocols followed (not just "we picked one")     │
│    - Measurements recorded in experiment log                    │
│                                                                 │
│  ALL CRITERIA MET → PROCEED TO PHASE 2                          │
│  ANY CRITERIA UNMET → ITERATE ON PHASE 1                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Phase 2: Training Infrastructure

**Goal**: Build training loop that won't silently fail

**Duration**: Week 6-9

**Prerequisites**: Phase 1 EXIT GATE passed

### 5.1 Engineering Tasks

| Task ID | Description | Decision Protocol |
|---------|-------------|-------------------|
| P2-T1 | Lock multi-resolution chunk sizes | See §5.2 Decision Protocol |
| P2-T2 | Determine NIAH probe frequency | See §5.3 Decision Protocol |
| P2-T3 | Implement Q-K projection with norm | M_t reset includes norm_persistent |
| P2-T4 | Implement NIAH probes | Accuracy > 80% |
| P2-T5 | Implement PPL delta telemetry | Logs every 100 steps in Stage 2 |
| P2-T6 | Implement auto-rollback (failsafe) | Triggers on 5% spike |

---

### 5.2 Decision Protocol: Chunk Sizes (P2-T1)

> **GOAL**: Select the set of local chunk sizes C_L for multi-resolution training

**Candidate Sets**:
```
Set A: {8, 16, 32, 64}      - Fine-grained, more memory checkpoints
Set B: {16, 32, 64, 128}    - Coarser, higher throughput
Set C: {8, 32, 64}          - Sparse, fewer transitions
```

**DECISION PROTOCOL**:

```
┌─────────────────────────────────────────────────────────────────┐
│            CHUNK SIZE DECISION PROTOCOL                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STEP 1: Benchmark throughput for each candidate set            │
│          Run 100 steps, measure tokens/sec                      │
│                                                                 │
│          ┌────────────────────────────────────────────────┐     │
│          │ Set                │ tokens/sec │ vs baseline │     │
│          ├────────────────────────────────────────────────┤     │
│          │ Set A {8,16,32,64} │ _______    │ ____×       │     │
│          │ Set B {16,32,64,128}│ _______   │ ____×       │     │
│          │ Set C {8,32,64}    │ _______    │ ____×       │     │
│          │ Baseline (C=8 only)│ _______    │ 1.0×        │     │
│          └────────────────────────────────────────────────┘     │
│                                                                 │
│  STEP 2: Check memory at smallest chunk                         │
│          - C_L=8 requires more memory checkpoints               │
│          - Verify no OOM at smallest chunk size                 │
│                                                                 │
│  STEP 3: Decision Rule                                          │
│          ┌────────────────────────────────────────────────┐     │
│          │ REQUIRE: All chunks in set must fit in VRAM   │     │
│          │                                                │     │
│          │ IF multiple sets fit:                         │     │
│          │   → Choose set with highest throughput        │     │
│          │   → Must achieve ≥ 5× baseline                │     │
│          │                                                │     │
│          │ IF no set achieves ≥ 5× baseline:             │     │
│          │   → STOP. Investigate parallelization bug.    │     │
│          └────────────────────────────────────────────────┘     │
│                                                                 │
│  STEP 4: Document choice with throughput measurements           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### 5.3 Decision Protocol: NIAH Frequency (P2-T2)

> **GOAL**: Set probe frequency that catches memory degradation without killing throughput

**DECISION PROTOCOL**:

```
┌─────────────────────────────────────────────────────────────────┐
│            NIAH FREQUENCY DECISION PROTOCOL                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STEP 1: Measure baseline throughput (no probes)                │
│          tokens/sec baseline: _______                           │
│                                                                 │
│  STEP 2: Test candidate frequencies                             │
│          ┌────────────────────────────────────────────────┐     │
│          │ Frequency    │ tokens/sec │ overhead │         │     │
│          ├────────────────────────────────────────────────┤     │
│          │ Every 500    │ _______    │ ____%    │         │     │
│          │ Every 1000   │ _______    │ ____%    │         │     │
│          │ Every 2000   │ _______    │ ____%    │         │     │
│          └────────────────────────────────────────────────┘     │
│                                                                 │
│  STEP 3: Decision Rule                                          │
│          ┌────────────────────────────────────────────────┐     │
│          │ REQUIRE: Overhead < 1%                        │     │
│          │                                                │     │
│          │ Among frequencies with < 1% overhead:         │     │
│          │   → Choose LOWEST frequency (most frequent)   │     │
│          │   → More probes = earlier detection           │     │
│          │                                                │     │
│          │ IF all frequencies have > 1% overhead:        │     │
│          │   → Use every 2000 steps                      │     │
│          │   → Accept higher overhead for safety         │     │
│          └────────────────────────────────────────────────┘     │
│                                                                 │
│  STEP 4: Document with overhead measurement                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### 5.4 Silent Killer #2 Fix: Normalization

> **PROBLEM**: M_persistent lacks denominator → volume mismatch
>
> **SYMPTOM**: Code runs, no errors, attention scores are garbage
>
> **RESULT**: Persistent memory either too quiet or deafening

**REQ-P2-001**: Q-K Projection with Normalization

```python
class QKProjection:
    def __init__(self, M_persistent, norm_persistent):
        self.M_persistent = M_persistent
        self.norm_persistent = norm_persistent

    def reset_at_shard_boundary(self):
        """
        At shard boundary, inject BOTH numerator AND denominator.
        """
        self.M_t = self.M_persistent.clone()
        self.norm_sum = self.norm_persistent  # CRITICAL: Include this!

    def update(self, k_t):
        """
        Update running sum with new key.
        """
        self.M_t = self.M_t + torch.outer(k_t, k_t)
        self.norm_sum = self.norm_sum + (k_t.norm() ** 2).item()

    def project(self, q_t):
        """
        Project query onto key subspace with proper normalization.
        """
        return (self.M_t @ q_t) / self.norm_sum
```

**REQ-P2-002**: NIAH Retrieval Probe

```python
def niah_probe(model, step, probe_frequency=1000):
    """
    Validates that Atlas memory actually WORKS.

    DECISION REQUIRED (OQ-7): Is 1000 steps sufficient?
    Test: If overhead > 1% throughput, increase frequency.
    """
    if step % probe_frequency != 0:
        return

    key = random_vector(dim)
    value = random_vector(dim)

    model.inject_memory(key, value)
    model.forward(generate_random_tokens(100))  # haystack
    retrieved = model.query_memory(key)

    accuracy = cosine_similarity(retrieved, value)
    log_metric("niah_retrieval_accuracy", accuracy)

    if accuracy < 0.8:
        log_warning("Atlas memory retrieval degraded")
```

### 5.5 DIAGNOSTIC CHECKPOINT

```
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 2 EXIT GATE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  □ G2-1: Multi-resolution chunk sizes LOCKED (§6.2 Protocol)    │
│          Decision: C_L ∈ {___}, documented with throughput data │
│          Evidence: Benchmark measured per decision rule         │
│                                                                 │
│  □ G2-2: NIAH probe frequency LOCKED (§6.3 Protocol)            │
│          Decision: frequency=___ with measured overhead         │
│          Test: Overhead < 1% throughput at chosen frequency     │
│                                                                 │
│  □ G2-3: Q-K projection includes norm_persistent                │
│          Test: self.norm_sum initialized with norm_persistent   │
│                                                                 │
│  □ G2-4: Training loop runs 1000 steps without crash            │
│          Test: No OOM, no NaN, forward/backward complete        │
│                                                                 │
│  □ G2-5: NIAH accuracy > 80% after 1000 steps                   │
│          Test: niah_retrieval_accuracy logged and passing       │
│                                                                 │
│  □ G2-6: Auto-rollback tested                                   │
│          Test: Force spike, verify rollback triggers            │
│                                                                 │
│  □ G2-7: PPL delta telemetry visible                            │
│          Test: Dashboard shows ppl_delta metric                 │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  ALL BOXES CHECKED → PROCEED TO PHASE 3                         │
│  ANY BOX UNCHECKED → ITERATE ON PHASE 2                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Phase 3: Optimization (SIMPLIFIED - Round 8)

**Goal**: Achieve target throughput with simple K=5 everywhere

**Duration**: Week 10-12

**Prerequisites**: Phase 2 checkpoint logged

> **ROUND 8 SIMPLIFICATION**: "For a 60M parameter model, the compute overhead of K=5 everywhere is negligible. Get it working first, optimize later."

### 6.1 Engineering Tasks

| Task ID | Description | Acceptance Criteria |
|---------|-------------|---------------------|
| P3-T1 | Use K=5 for all memory modules | Simple, uniform |
| P3-T2 | Implement tensorized Muon updates | No Python loops over tokens |
| P3-T3 | Measure throughput vs baseline | ≥10× single-chunk baseline |

### 6.2 Newton-Schulz Implementation (K=5 Everywhere)

```python
# ROUND 8 SIMPLIFIED: K=5 everywhere
# "A working model is infinitely better than a theoretically faster broken model"

K = 5  # Same for global AND local memory

def newton_schulz(G, num_iters=K):
    """
    Standard Newton-Schulz orthogonalization.

    K=5 for everything. Simple.

    Round 8: "For a 60M model on two A6000s, the K=3 vs K=5
              difference is milliseconds per step. Debugging
              decoupled optimization could take DAYS."
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G
    for _ in range(num_iters):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X

# Usage - same K for both:
global_update = newton_schulz(global_grad)  # K=5
local_update = newton_schulz(local_grad)    # K=5
```

**DELETED (Per Round 8)**:
- ❌ Spectral error validation - premature optimization
- ❌ K warmup strategy - unnecessary complexity
- ❌ Pipeline latency requirements - not a bottleneck at this scale
- ❌ K decoupling (K_global=3, K_local=5) - get it working first

### 6.3 DIAGNOSTIC CHECKPOINT

```
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 3 DIAGNOSTIC CHECKPOINT                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  □ G3-1: K=5 used everywhere (no decoupling)                    │
│          Test: Single K constant in code                        │
│                                                                 │
│  □ G3-2: No Python loops over token dimension                   │
│          Test: Profiler shows batched matmul only               │
│                                                                 │
│  □ G3-3: Training throughput ≥ 10× baseline                     │
│          Test: tokens/sec with TNT vs single-chunk baseline     │
│                                                                 │
│  □ G3-4: GPU utilization > 80%                                  │
│          Test: nvidia-smi shows sustained high utilization      │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  ALL BOXES CHECKED → Great! Proceed confidently                 │
│  SOME BOXES UNCHECKED → Log gaps, proceed anyway (R8)           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Phase 4: Stage 2 Fine-Tuning

**Goal**: Achieve resolution robustness without blowing up

**Duration**: Week 13-15

**Prerequisites**: Phase 3 EXIT GATE passed

### 7.1 Engineering Tasks

| Task ID | Description | Former OQ | Acceptance Criteria |
|---------|-------------|-----------|---------------------|
| P4-T1 | **Implement proactive gradient clipping** | NEW (R5) | Clips at chunk transitions |
| P4-T2 | Optionally implement soft boundaries | NEW (R5) | Interpolated chunk size |
| P4-T3 | Verify gamma decay schedule | - | 32 × γ^T ≈ 1 |
| P4-T4 | Tune PPL delta threshold | OQ-11 | Based on observed variance |
| P4-T5 | Validate gate polarization | AC-27 | ≥20% tokens polarized |

### 7.2 Decision Protocol: PPL Delta Threshold (P4-T4)

> **GOAL**: Determine the rollback trigger threshold based on observed variance
>
> **PROBLEM**: Current spec says "5% from moving average" but optimal threshold depends on actual training dynamics

**DECISION PROTOCOL**:

```
┌─────────────────────────────────────────────────────────────────┐
│            PPL DELTA THRESHOLD CALIBRATION                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STEP 1: Run Stage 2 for 1000 steps WITHOUT rollback enabled    │
│          Record all PPL delta values                            │
│          (This is OBSERVATION only - do not checkpoint)         │
│                                                                 │
│  STEP 2: Analyze PPL delta distribution                         │
│          ┌────────────────────────────────────────────────┐     │
│          │ Statistic                    │ Value           │     │
│          ├────────────────────────────────────────────────┤     │
│          │ Mean PPL delta               │ _______         │     │
│          │ Std deviation                │ _______         │     │
│          │ 95th percentile              │ _______         │     │
│          │ 99th percentile              │ _______         │     │
│          │ Max observed spike           │ _______         │     │
│          └────────────────────────────────────────────────┘     │
│                                                                 │
│  STEP 3: Decision Rule                                          │
│          ┌────────────────────────────────────────────────┐     │
│          │ Threshold = 99th percentile + 1 std deviation  │     │
│          │                                                │     │
│          │ Rationale: Trigger only on TRUE anomalies      │     │
│          │   - Not on normal training variance            │     │
│          │   - But catch real instability events          │     │
│          │                                                │     │
│          │ IF 99th percentile < 5% of moving average:     │     │
│          │   → Use 5% as default (training is stable)     │     │
│          │                                                │     │
│          │ IF 99th percentile > 10% of moving average:    │     │
│          │   → STOP. Training is fundamentally unstable.  │     │
│          │   → Investigate proactive clipping parameters. │     │
│          └────────────────────────────────────────────────┘     │
│                                                                 │
│  STEP 4: Document threshold with statistical justification      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### 7.3 Dynamic Gradient Clip Threshold (REVISED - Round 6)

> **ROUND 6 FIX**: Static clip_scale = 0.5 is "semi-truck brakes on a go-kart"
>
> **WHY**: As chunk size shrinks → gradient variance INCREASES
>         Static threshold strangles learning at small chunk sizes
>
> **FIX**: κ(t) scales with √(32 / C_L(t))

**REQ-STAGE2-006**: Dynamic Gradient Clip Scaling

```python
def get_dynamic_clip_threshold(chunk_size, kappa_base=0.5):
    """
    Dynamic threshold that scales with chunk size.

    Chunk size 32: "Like driving a semi-truck"
      - Has momentum, stable
      - κ = 0.5 (base threshold)

    Chunk size 1: "Like driving a go-kart"
      - Twitchy, high variance
      - κ = 2.83 (relaxed by 5.66×)

    "Aligns safety rails with the physics of training"
    """
    return kappa_base * math.sqrt(32.0 / chunk_size)


# Example progression:
# C_L = 32: κ = 0.5 × √(32/32) = 0.5 × 1.0  = 0.5
# C_L = 16: κ = 0.5 × √(32/16) = 0.5 × 1.41 = 0.71
# C_L = 8:  κ = 0.5 × √(32/8)  = 0.5 × 2.0  = 1.0
# C_L = 4:  κ = 0.5 × √(32/4)  = 0.5 × 2.83 = 1.41
# C_L = 1:  κ = 0.5 × √(32/1)  = 0.5 × 5.66 = 2.83
```

**FORBIDDEN**: Static clip_scale throughout Stage 2

**CALIBRATION PROTOCOL**:

```
┌─────────────────────────────────────────────────────────────────┐
│            DYNAMIC CLIP CALIBRATION (REVISED R6)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STEP 1: Verify κ_base = 0.5 at C_L = 32                        │
│          Run 500 steps at C_L = 32 with κ = 0.5                 │
│          - If stable: κ_base = 0.5 confirmed                    │
│          - If unstable: lower κ_base to 0.3                     │
│                                                                 │
│  STEP 2: Verify dynamic scaling at small chunks                 │
│          Run 500 steps at C_L = 1 with κ = κ_base × √32         │
│                                                                 │
│          ┌────────────────────────────────────────────────┐     │
│          │ C_L │ κ(t)  │ Clip Events │ Learning OK?      │     │
│          ├────────────────────────────────────────────────┤     │
│          │ 32  │ 0.5   │ _______     │ [YES|NO]          │     │
│          │ 8   │ 1.0   │ _______     │ [YES|NO]          │     │
│          │ 1   │ 2.83  │ _______     │ [YES|NO]          │     │
│          └────────────────────────────────────────────────┘     │
│                                                                 │
│  STEP 3: Validation Rule                                        │
│          ┌────────────────────────────────────────────────┐     │
│          │ Learning is OK if:                            │     │
│          │   - Loss continues to decrease                │     │
│          │   - Clip events are RARE (not every step)     │     │
│          │   - NIAH accuracy maintained > 80%            │     │
│          │                                                │     │
│          │ IF clipping every step at C_L=1:              │     │
│          │   → Increase κ_base (e.g., 0.7)               │     │
│          │   → Recalculate dynamic scaling               │     │
│          └────────────────────────────────────────────────┘     │
│                                                                 │
│  STEP 4: Document κ_base and verify dynamic formula works       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### 7.4 Silent Killer #3 Fix: Reactive Rollback

> **PROBLEM**: Auto-rollback treats symptom, not disease
>
> **SYMPTOM**: Training eventually completes after many resets
>
> **RESULT**: "Reversing car after hitting wall, then driving into same wall"

**REQ-P4-001**: Proactive Boundary Stabilization (NEW - Round 5)

```python
class ProactiveStabilizer:
    """
    Stop the crash before it happens.

    Rollback is FAILSAFE, not primary strategy.
    """

    def __init__(self, clip_scale=0.5):
        self.clip_scale = clip_scale
        self.previous_chunk_size = None

    def on_chunk_transition(self, current_chunk_size, gradients):
        """
        At chunk size changes, gradients WILL spike.
        Clip them preemptively.
        """
        if self.previous_chunk_size is not None:
            if current_chunk_size != self.previous_chunk_size:
                # Transition detected - apply aggressive clipping
                max_norm = self.compute_normal_grad_norm() * self.clip_scale
                torch.nn.utils.clip_grad_norm_(gradients, max_norm)
                log_info(f"Proactive clip at {self.previous_chunk_size}→{current_chunk_size}")

        self.previous_chunk_size = current_chunk_size
```

**REQ-P4-002**: Soft Boundaries (Optional, Preferred)

```python
def soft_chunk_transition(step, transition_start, transition_end,
                          chunk_before, chunk_after):
    """
    Smoothly interpolate effective chunk size.

    Instead of hard cut: 32 → 16
    Use gradual: 32 → 28 → 24 → 20 → 16

    Source: TNT paper's idea of soft boundaries
    """
    if step < transition_start:
        return chunk_before
    elif step > transition_end:
        return chunk_after
    else:
        # Linear interpolation
        progress = (step - transition_start) / (transition_end - transition_start)
        return chunk_before + progress * (chunk_after - chunk_before)
```

**REQ-P4-003**: Auto-Rollback (Failsafe Only)

```python
def stage2_training_step(model, batch, step, checkpoint_manager, stabilizer):
    """
    Proactive clipping FIRST, rollback as last resort.
    """
    # 1. Proactive stabilization
    current_chunk = get_current_chunk_size(step)

    # 2. Forward/backward
    loss = model(batch)
    loss.backward()

    # 3. Proactive gradient clipping at transitions
    stabilizer.on_chunk_transition(current_chunk, model.parameters())

    # 4. Compute PPL delta
    delta = perplexity_delta_probe(model, validation_batch, step)

    # 5. Rollback ONLY if proactive measures fail
    if delta > moving_average_delta * 1.05:
        log_warning(f"Rollback triggered despite proactive clipping")
        checkpoint_manager.rollback(step - 500)
        set_gamma(get_gamma() * 0.5)
```

### 7.5 DIAGNOSTIC CHECKPOINT

```
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 4 EXIT GATE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  □ G4-1: DYNAMIC gradient clipping calibrated (§7.3 R6)         │
│          Decision: κ_base=___ with κ(t) = κ_base × √(32/C_L)    │
│          Test: Learning OK at all chunk sizes (rare clip events)│
│          FORBIDDEN: Static clip_scale throughout Stage 2        │
│                                                                 │
│  □ G4-2: Stage 2 completes without > 3 auto-rollbacks           │
│          Test: Rollback count in training logs                  │
│                                                                 │
│  □ G4-3: PPL delta threshold calibrated (§7.2 Protocol)         │
│          Decision: threshold=___% with statistical justification│
│          Test: ppl_delta metric doesn't trend upward            │
│                                                                 │
│  □ G4-4: Resolution sensitivity curve is FLAT                   │
│          Test: PPL variance < threshold across {1-64}           │
│                                                                 │
│  □ G4-5: Gate polarization ≥ 20%                                │
│          Test: ≥20% tokens have gate < 0.1 OR > 0.9             │
│                                                                 │
│  □ G4-6: NIAH accuracy still > 80%                              │
│          Test: Post-Stage-2 NIAH probe passes                   │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  ALL BOXES CHECKED → PROCEED TO PHASE 5                         │
│  ANY BOX UNCHECKED → ITERATE ON PHASE 4                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Phase 5: Inference & Release

**Goal**: Validate inference modes work correctly

**Duration**: Week 16-18

**Prerequisites**: Phase 4 EXIT GATE passed

### 8.1 Engineering Tasks

| Task ID | Description | Acceptance Criteria |
|---------|-------------|---------------------|
| P5-T1 | Implement prefill mode | No resets, M_t init with M_persistent |
| P5-T2 | Implement decode mode | C_L=1, no resets |
| P5-T3 | Dynamic chunk switching | No recompilation needed |
| P5-T4 | Persistent memory read-only | ∇P = 0 during inference |
| P5-T5 | End-to-end generation test | Coherent output |

### 8.2 Requirements

**REQ-P5-001**: Prefill Mode
- Process prompt using Global Memory
- Use larger chunk sizes for efficiency
- DISABLE periodic resets
- Initialize M_t with M_persistent + norm_persistent

**REQ-P5-002**: Decode Mode
- Use Local Memory for token generation
- Set chunk size C_L = 1
- Maintain memory state across generated tokens
- DISABLE periodic resets

**REQ-P5-003**: Explicit Reset Removal
```
EXPLICIT REQUIREMENT: Periodic resets are REMOVED during inference
```

### 8.3 DIAGNOSTIC CHECKPOINT

```
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 5 EXIT GATE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  □ G5-1: Prefill → Decode transition seamless                   │
│          Test: End-to-end prompt + generation works             │
│                                                                 │
│  □ G5-2: Generated text is coherent                             │
│          Test: Human evaluation of sample outputs               │
│                                                                 │
│  □ G5-3: Persistent memory accessible from first token          │
│          Test: Query persistent key at t=0 returns correct val  │
│                                                                 │
│  □ G5-4: No resets during inference                             │
│          Test: Memory state persists across S_L boundary        │
│                                                                 │
│  □ G5-5: Dynamic chunk switch without reload                    │
│          Test: Change C_L at runtime, no recompilation          │
│                                                                 │
│  □ G5-6: All acceptance criteria (AC-1 through AC-34) pass      │
│          Test: Full validation suite green                      │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  ALL BOXES CHECKED → RELEASE CANDIDATE                          │
│  ANY BOX UNCHECKED → ITERATE ON PHASE 5                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Technical Requirements Reference

This section consolidates all requirements from phases into a reference list.

### 9.1 Initialization Requirements
- REQ-P0-001: Steady-state W_init
- REQ-P0-002: M_persistent with norm_persistent
- REQ-P0-003: Hash verification

### 9.2 Architecture Requirements
- REQ-P1-001: Gate polarization (loss OR init)
- REQ-P1-002: Gated MLP structure (Atlas++)
- REQ-P1-003: Polynomial feature mapping
- REQ-P1-004: Multiplicative fusion

### 9.3 Training Requirements
- REQ-P2-001: Q-K projection with normalization
- REQ-P2-002: NIAH retrieval probe

### 9.4 Optimization Requirements
- REQ-P3-001: Unified Newton-Schulz K=5 (SIMPLIFIED - Round 8)
- REQ-P3-002: Tensorized Muon updates (no Python loops)

### 9.5 Stage 2 Requirements
- REQ-P4-001: Proactive boundary stabilization
- REQ-P4-002: Soft boundaries (optional)
- REQ-P4-003: Auto-rollback (failsafe)

### 9.6 Inference Requirements
- REQ-P5-001: Prefill mode
- REQ-P5-002: Decode mode
- REQ-P5-003: Reset removal

---

## 10. Hyperparameter Specifications

### 10.1 Architecture Parameters

| Parameter | Value | Decision Point |
|-----------|-------|----------------|
| Architecture | MAG | Fixed |
| Memory depth (L_M) | **[2 or 3]** | Phase 1 Gate |
| Memory structure | Atlas++ (gated MLP) | Fixed |
| Feature degree (p) | **[2 or 3]** | Phase 1 Gate |
| Fusion | Multiplicative | Fixed |

### 10.2 Initialization Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| W_init method | Steady-state mean | NOT zero |
| Calibration batch | **[10K+ tokens]** | Phase 0, increase if reset shock > 5% |
| N_persistent | 64 tokens | ~2.6 MB for d=768 |
| norm_persistent | **Runtime compute + NCCL** | REVISED (R8) - Trust NCCL broadcast |
| Gate init | **Annealing λ polarization loss** | REVISED (R6) - Option B REMOVED |
| λ_initial | 10.0 | First 10% of training (winner-take-all) |
| λ_final | 0.1 | After annealing (allow subtle mixing) |
| Noise test threshold | **< 0.01** | TIGHTENED (R6) - was 0.1 |

### 10.3 Newton-Schulz Parameters (SIMPLIFIED - Round 8)

| Parameter | Value | Notes |
|-----------|-------|-------|
| K | **5 everywhere** | SIMPLIFIED (R8) - no decoupling, get it working first |
| Muon scope | Both global and local | Same K=5 for both |

### 10.4 TNT Training Parameters

| Parameter | Stage 1 | Stage 2 |
|-----------|---------|---------|
| Global chunk (C_G) | 2048 | 2048 |
| Local chunk (C_L) | **[{8,16,32,64}]** | Geometric decay 32→1 |
| Shard length (S_L) | 2048-4096 | 2048-4096 |
| Compute fraction | 95% | 5% |
| M_t reset | M_persistent + norm_persistent | Same |
| γ (gamma) | - | 0.9993 |

### 10.5 Metrics Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| NIAH frequency | **[1000 steps]** | Phase 2, adjust if overhead > 1% |
| NIAH threshold | 80% accuracy | Below = warning |
| PPL delta frequency | 100 steps | Stage 2 only |
| PPL spike threshold | 5% from moving avg | Triggers rollback |
| Gradient clip | **κ(t) = κ_base × √(32/C_L)** | REVISED (R6) - DYNAMIC not static |
| κ_base | 0.5 | Base threshold at C_L=32 |
| Gate polarization | ≥20% tokens | At <0.01 OR >0.99 (TIGHTENED R6) |

---

## 11. Acceptance Criteria

### 11.1 Phase 0 Acceptance

| ID | Criterion | Test |
|----|-----------|------|
| AC-P0-1 | W_init via steady-state | Reset shock < 5% |
| AC-P0-2 | M_persistent computed | Contains all 64 key outer products |
| AC-P0-3 | norm_persistent computed | Equals Σ||k_p||² |
| AC-P0-4 | Hash verification | All GPUs match |

### 11.2 Phase 1 Acceptance

| ID | Criterion | Test |
|----|-----------|------|
| AC-P1-1 | Gate noise suppression | Suppresses to < 0.01 (TIGHTENED R6) |
| AC-P1-2 | Polarization implemented | Annealing λ (Option B REMOVED R6) |
| AC-P1-3 | Polynomial verified | Correct dimension expansion |
| AC-P1-4 | Multiplicative fusion | gate=0 → output=0 |
| AC-P1-5 | Non-trivial gates | gate.std() > 0.1 |

### 11.3 Phase 2 Acceptance

| ID | Criterion | Test |
|----|-----------|------|
| AC-P2-1 | Chunk sizes locked | Documented decision |
| AC-P2-2 | NIAH overhead | < 1% throughput loss |
| AC-P2-3 | Norm in projection | norm_persistent in init |
| AC-P2-4 | 1000 steps stable | No crash, no NaN |
| AC-P2-5 | NIAH passing | > 80% accuracy |
| AC-P2-6 | Rollback tested | Triggers on forced spike |
| AC-P2-7 | Gradient mask at boundaries | ∇M_t = 0 at shard boundaries (R2) |

### 11.4 Phase 3 Acceptance

| ID | Criterion | Test |
|----|-----------|------|
| AC-P3-1 | K unified | K=5 everywhere (no decoupling) |
| AC-P3-2 | No Python loops | Batched matmul only |
| AC-P3-3 | Throughput | ≥ 10× baseline |
| AC-P3-4 | GPU utilization | > 80% |

### 11.5 Phase 4 Acceptance

| ID | Criterion | Test |
|----|-----------|------|
| AC-P4-1 | Proactive clipping | Clips at transitions |
| AC-P4-2 | Rollback count | ≤ 3 during Stage 2 |
| AC-P4-3 | PPL delta stable | No upward trend |
| AC-P4-4 | Resolution flat | Low variance {1-64} |
| AC-P4-5 | Gate polarized | ≥ 20% |
| AC-P4-6 | NIAH still passing | > 80% post-Stage-2 |

### 11.6 Phase 5 Acceptance

| ID | Criterion | Test |
|----|-----------|------|
| AC-P5-1 | Mode transition | Seamless prefill→decode |
| AC-P5-2 | Coherent output | Human evaluation |
| AC-P5-3 | Persistent accessible | Query at t=0 works |
| AC-P5-4 | No inference resets | Memory persists |
| AC-P5-5 | Dynamic C_L | No recompilation |

---

## Appendix: Mathematical Definitions

### A.1 Omega Rule Loss Function

$$L_t = \sum_{i=t-c+1}^{t} \gamma_i^{(t)} \|\mathcal{M}(\phi(k_i)) - v_i\|^2_2$$

### A.2 Muon Memory Update (K=5 Everywhere)

$$\mathcal{M}_t = \alpha_t \mathcal{M}_{t-1} - \eta_t \text{NewtonSchulz}_K(S_t)$$

Where K = 5 for both global and local memory (SIMPLIFIED - Round 8).

### A.3 Newton-Schulz Iteration

$$X_{n+1} = aX_n + bX_nX_n^TX_n + cX_nX_n^TX_nX_n^TX_n$$

Where $a = 3.4445$, $b = -4.7750$, $c = 2.0315$

### A.4 Persistent Memory with Normalization (REVISED - Round 5)

$$M_{persistent} = \sum_{p=1}^{N_p} k_p k_p^\top$$

$$norm_{persistent} = \sum_{p=1}^{N_p} \|k_p\|^2$$

### A.5 Q-K Projection with Normalization (REVISED - Round 5)

$$q'_t = \frac{M_t \cdot q_t}{norm\_sum_t}$$

Where:
- $M_t = M_{persistent} + \sum_{i=start}^{t} k_i k_i^\top$
- $norm\_sum_t = norm_{persistent} + \sum_{i=start}^{t} \|k_i\|^2$

### A.6 Gate Polarization Loss (NEW - Round 5)

$$L_{polar} = \lambda (1 - |2g - 1|)$$

Where:
- $g$ = gate output ∈ [0, 1]
- $\lambda$ = penalty weight
- Maximum penalty at g = 0.5, zero penalty at g ∈ {0, 1}

### A.7 Stage 2 Geometric Decay

$$C_L(t) = \max\left(1, \lfloor C_{start} \cdot \gamma^t \rfloor\right)$$

Where:
- $C_{start}$ = 32
- $\gamma$ = 0.9993
- $T_2$ = Total Stage 2 steps

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-23 | Engineering | Initial PRD |
| 2.0 | 2026-01-24 | Engineering | Round 2: NIAH, W_init, M_t reset |
| 3.0 | 2026-01-24 | Engineering | Round 3: Lock K=5, M_persistent, geometric decay |
| 4.0 | 2026-01-24 | Engineering | Round 4: Blockers identified |
| 4.1 | 2026-01-24 | Engineering | Round 4.1: Blockers resolved |
| 5.0 | 2026-01-24 | Engineering | **MAJOR RESTRUCTURE**: Gated phases, Round 5 silent killers |
| 5.1 | 2026-01-24 | Engineering | **GATES TIGHTENED**: Round 6 engineering fixes |
| 5.2 | 2026-01-24 | Engineering | **POC SCOPE + A/B TESTS**: Round 7 empirical validation focus |
| 5.3 | 2026-01-24 | Engineering | **SIMPLIFIED FOR SOLO DEV**: Round 8 - momentum over perfection |
| 5.4 | 2026-01-24 | Engineering | **COMPLIANCE FIX**: Removed all K decoupling contradictions |

---

## Change Summary (v5.3 → v5.4)

| Change | Description |
|--------|-------------|
| **AC-P3-1 fixed** | Changed from "K decoupled" to "K unified | K=5 everywhere" |
| **AC-P3-2 deleted** | Removed "Pipeline bubble" (obsolete with K=5 everywhere) |
| **Appendix A.2 fixed** | Changed "K_global(3), K_local(5)" to "K=5 everywhere" |
| **Section 9.4 fixed** | Changed "Decoupled Newton-Schulz K" to "Unified K=5" |
| **Section 10.3 fixed** | Removed "With decoupled K" note |
| **Section numbering** | Fixed all inconsistent subsection numbers |

**The Fix**: v5.3 change summary correctly said "K=5 everywhere" but the body still had old K decoupling references. v5.4 removes all contradictions.

---

## Change Summary (v5.2 → v5.3)

| Change | Description |
|--------|-------------|
| **Philosophy shift** | From defense-grade to research-grade |
| **EXIT GATES → CHECKPOINTS** | Diagnostic only, not blocking |
| **norm_persistent** | Runtime compute + NCCL (not static config) |
| **K values** | K=5 everywhere (no decoupling) |
| **A/B Tests CANCELLED** | Sequential full-capacity runs instead |
| **Fast-fail gate check** | Abort at step 500 if gates stuck |
| **Section renumbering** | Removed A/B Test section, renumbered phases |

**DELETED (Premature for POC)**:
- ❌ Spectral error validation
- ❌ K warmup strategy
- ❌ Pipeline latency requirements
- ❌ AB-01, AB-02, AB-03
- ❌ Bitwise check (downgraded to warning)

**The Core Philosophy (Round 8)**:
> "The loudest killer of any side project is losing momentum."
>
> "You are clearly ready to build. This PRD is a map, not a law. Ignore the blockers. Run the code."

---

## Change Summary (v5.1 → v5.2)

| Change | Description |
|--------|-------------|
| **POC SCOPE box** | Prominent 60M toy model framing at document top |
| **A/B Test Protocols** | New Section 3 with parallel testing workflow |
| **Hardware config** | GPU 0/1 for A/B, GPU 2 (RTX 2000) for validation |
| **AB-01** | Gradient clip ceiling test (clamp vs no-clamp) |
| **AB-02** | Polarization velocity threshold test (0.05 vs 0.03) |
| **AB-03** | K warmup necessity test (K=3 vs K=5→3) |
| **CI/CD deferred** | Explicitly acknowledged as "single engineer, not team problem yet" |
| **Section renumbering** | All phases renumbered after A/B Test section insertion |
| **AC-P1-1 FIXED** | Noise threshold now correctly states < 0.01 |
| **AC-P2-7 ADDED** | Gradient mask unit test at shard boundaries |
| **AC-P1-2 CLARIFIED** | Explicitly states Option B REMOVED |

**The Core Philosophy (Round 7)**:
> "60M toy model, single engineer, not a team problem yet."
>
> "We have the hardware to quickly A/B test many of these questions away."

---

## Change Summary (v5.0 → v5.1)

| Change | Description |
|--------|-------------|
| **Option B REMOVED** | High-variance init eliminated - "too passive, gates drift back" |
| **Annealing λ MANDATED** | λ = 10.0 → 0.1 schedule, "teach routing before choosing" |
| **Noise test TIGHTENED** | Gate < 0.01 (not 0.1) - "10% leakage is huge" |
| **norm_persistent STATIC** | Config constant, not runtime computation |
| **Bitwise broadcast check** | All GPUs verify identical bytes, not just values |
| **Spectral error validation** | K=3 validated before use, warmup if error > 1e-4 |
| **K warmup strategy** | K=5 for 1000 steps → K=3 if spectral error high |
| **Dynamic gradient clip** | κ(t) = κ_base × √(32/C_L(t)), not static 0.5 |

**The Core Philosophy (Round 6)**:
> "The restructuring was a huge step forward... but the engineering inside those gates needs to be much, much more rigid."
>
> "Tighten the bolts, lock the gates. And you'll have something special."

---

## Change Summary (v4.1 → v5.0)

| Change | Description |
|--------|-------------|
| **Structure** | Complete restructure around gated phases |
| **OQs eliminated** | Converted to engineering tasks with acceptance criteria |
| **Silent Killer #1** | Gate init: added polarization loss / high-var options |
| **Silent Killer #2** | norm_persistent: added denominator to M_persistent |
| **Silent Killer #3** | Proactive gradient clipping before rollback |
| **Silent Killer #4** | Decoupled K: K_global=3, K_local=5 |
| **EXIT GATES** | Added concrete pass/fail tests for each phase |
| **Phase authorization** | Clear GO/CONDITIONAL/BLOCKED status |

**The Core Change**:
> Before: "OQ-5: Resolve by week 6"
> After: "P2-T1: Lock chunk sizes. EXIT GATE G2-1: Decision documented with rationale."

---

*End of Document*
