"""
Configuration constants for Atlas-MAG with Omega Rule.

All hyperparameters are defined here as the single source of truth.
This implements Atlas-MAG (Memory-as-Gate) with the Omega Rule context window.

Reference: Atlas paper (arXiv:2505.23735)
"""

from dataclasses import dataclass
from typing import Tuple


# =============================================================================
# Architecture Constants
# =============================================================================

# Model dimensions (defaults, can be overridden by training script)
D: int = 768                      # Model dimension (hidden size)
N_HEADS: int = 12                 # Number of attention heads

# Fast-fail guard: ensure D is divisible by N_HEADS
if D % N_HEADS != 0:
    raise ValueError(f"D ({D}) must be divisible by N_HEADS ({N_HEADS})")

HEAD_DIM: int = D // N_HEADS      # Per-head dimension (64)

# Memory configuration
N_PERSISTENT: int = 64            # Number of persistent memory tokens
L_M: int = 2                      # Memory depth
POLY_DEGREE: int = 2              # Polynomial feature degree (2 = quadratic capacity)
MEMORY_EXPANSION: int = 4         # SwiGLU expansion factor
POLY_RANK: int = 512              # Low-rank compression dim for poly features

# Sliding window attention
WINDOW_SIZE: int = 512            # Local attention window


# =============================================================================
# Omega Rule Configuration (Atlas paper Eq. 9)
# =============================================================================

OMEGA_CONTEXT_WINDOW: int = 256   # Context window size for Omega rule
OMEGA_DECAY_BASE: float = 0.95    # Base decay rate for γ
GAMMA_GATE_HIDDEN_DIM: int = 64   # Hidden dim for γ gate MLP


# =============================================================================
# Algorithm Constants
# =============================================================================

K: int = 10                       # Newton-Schulz iterations


# =============================================================================
# TTL (Test-Time Learning) Configuration (Atlas paper Eq. 32-33)
# =============================================================================

TTL_ENABLED: bool = True
TTL_THETA: float = 0.9            # Momentum decay
TTL_ALPHA: float = 0.999          # Weight decay (close to 1 = minimal)
TTL_ETA: float = 0.01             # Memory learning rate
TTL_NS_ITERATIONS: int = 10       # Newton-Schulz iterations
TTL_ADAPTIVE_ETA: bool = True     # Scale LR by inverse gradient norm
TTL_RESET_MODE: str = "sequence"  # When to reset momentum buffers


# =============================================================================
# Tokenizer Constants
# =============================================================================

VOCAB_SIZE: int = 32_000
SPECIAL_TOKENS: Tuple[str, ...] = ("<pad>", "<unk>", "<bos>", "<eos>")


# =============================================================================
# Fast-Fail Thresholds (for tests)
# =============================================================================

@dataclass
class FastFailThresholds:
    """Thresholds for early abort conditions."""
    gate_variance_check_step: int = 500
    gate_variance_multiplier: float = 1.5
    gate_std_min: float = 0.01


FAST_FAIL = FastFailThresholds()
