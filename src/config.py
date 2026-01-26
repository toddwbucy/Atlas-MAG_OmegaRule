"""
Configuration constants for Atlas-MAG with Omega Rule.

All hyperparameters are defined here as the single source of truth.
Phase decisions and architectural constants are documented with their origins.

Note: This implements Atlas-MAG (Memory-as-Gate) with the Omega Rule context window.
TNT (Titans-in-Titans) hierarchical memory and two-stage training are NOT implemented.
"""

from dataclasses import dataclass
from typing import Tuple


# =============================================================================
# Architecture Constants (PRD Section: Critical Constants)
# =============================================================================

# Model dimensions
D: int = 768                      # Model dimension (hidden size)
N_HEADS: int = 12                 # Number of attention heads

# Fast-fail guard: ensure D is divisible by N_HEADS to avoid silent truncation
if D % N_HEADS != 0:
    raise ValueError(f"D ({D}) must be divisible by N_HEADS ({N_HEADS})")

HEAD_DIM: int = D // N_HEADS      # Per-head dimension (64)

# Memory configuration
N_PERSISTENT: int = 64            # Number of persistent memory tokens
L_M: int = 2                      # Memory depth (Phase 1 decision: start with 2)
POLY_DEGREE: int = 2              # Polynomial feature degree (φ_2 for quadratic capacity)
MEMORY_EXPANSION: int = 4         # SwiGLU expansion factor
USE_POLY_MEMORY: bool = True      # Use polynomial features for memory (ESSENTIAL per paper)

# Sliding window attention
WINDOW_SIZE: int = 512            # Local attention window

# =============================================================================
# Omega Rule Configuration (Atlas paper Eq. 9)
# =============================================================================

# Context window for memory (c in the paper)
# Only the last OMEGA_CONTEXT_WINDOW tokens contribute to memory
OMEGA_CONTEXT_WINDOW: int = 256   # Context window size for Omega rule

# Base decay rate for exponential weighting within context window
# γ_base^(t-i) gives higher weight to recent tokens
OMEGA_DECAY_BASE: float = 0.95    # Base decay rate for γ

# Input-dependent gate configuration
GAMMA_GATE_HIDDEN_DIM: int = 64   # Hidden dim for γ gate MLP


# =============================================================================
# Algorithm Constants
# =============================================================================

# Newton-Schulz iterations (unified across all uses)
K: int = 5                        # Newton-Schulz iterations for Muon optimizer


# =============================================================================
# Gate Polarization Constants (PRD Section: Gate Polarization)
# =============================================================================

LAMBDA_INITIAL: float = 10.0      # First 10% of training
LAMBDA_FINAL: float = 0.1         # After annealing
POLARIZATION_ANNEAL_RATIO: float = 0.1  # When to switch from initial to final


# =============================================================================
# Phase 0 Calibration Constants
# =============================================================================

CALIBRATION_TOKENS: int = 10_000  # Initial calibration size
CALIBRATION_BATCH_SIZE: int = 32  # Batch size for calibration
CALIBRATION_SEQ_LEN: int = 512    # Sequence length for calibration
RESET_SHOCK_THRESHOLD: float = 0.05  # Maximum acceptable loss spike at reset


# =============================================================================
# Tokenizer Constants
# =============================================================================

VOCAB_SIZE: int = 32_000          # BPE vocabulary size
SPECIAL_TOKENS: Tuple[str, ...] = ("<pad>", "<unk>", "<bos>", "<eos>")


# =============================================================================
# Validation Targets (PRD Section: Validation Metrics)
# =============================================================================

@dataclass
class ValidationTargets:
    """Target metrics for validation at each phase."""
    niah_accuracy: float = 0.80           # >80% NIAH retrieval (Phase 2+)
    gate_polarization_ratio: float = 0.20  # ≥20% tokens at <0.1 or >0.9 (Phase 4)
    throughput_multiplier: float = 10.0    # ≥10× baseline (Phase 3)
    reset_shock_max: float = 0.05          # <5% loss spike (Phase 0)


# =============================================================================
# Fast-Fail Thresholds (PRD Section: Fast-Fail Checks)
# =============================================================================

@dataclass
class FastFailThresholds:
    """Thresholds for early abort conditions."""
    gate_variance_check_step: int = 500
    gate_variance_multiplier: float = 1.5  # Abort if < initial * 1.5
    gate_std_min: float = 0.01             # Abort if std < 0.01 after step 100


# =============================================================================
# Device Configuration
# =============================================================================

@dataclass
class DeviceConfig:
    """Hardware configuration."""
    # Primary training GPUs
    primary_gpus: Tuple[int, ...] = (0, 1)  # RTX A6000 pair
    validation_gpu: int = 2                  # RTX 2000 for validation

    # Memory limits (GB)
    primary_gpu_memory: int = 48
    validation_gpu_memory: int = 16

    # For Phase 0: single GPU mode
    single_gpu_mode: bool = True
    default_gpu: int = 0


# Create default instances
VALIDATION_TARGETS = ValidationTargets()
FAST_FAIL = FastFailThresholds()
DEVICE_CONFIG = DeviceConfig()
