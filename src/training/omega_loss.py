"""
Omega Rule Loss for Test-Time Learning (TTL).

Implements Equation 9 from the Atlas paper (arXiv:2505.23735):

    L_omega = sum(i=t-c+1 to t) gamma_i^(t) * ||M(phi(k_i)) - v_i||^2

The memory M learns to map keys to values over a sliding context window.
The exponential decay gamma weights recent tokens more heavily.

Reference:
    - Atlas paper Eq. 9: Omega Rule loss formulation
    - Atlas paper Appendix D: Key-Value projection definitions
"""

import torch
from torch import Tensor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.model.atlas_memory import AtlasMemoryPoly


def compute_omega_loss(
    memory: "AtlasMemoryPoly",
    keys: Tensor,
    values: Tensor,
    gamma: Tensor | None = None,
    context_window: int = 256,
    decay_base: float = 0.95,
) -> Tensor:
    """
    Compute Omega Rule loss over a context window.

    The loss measures how well the memory can map keys to values:
        L = mean(gamma_i * ||M(k_i) - v_i||^2)

    Args:
        memory: AtlasMemoryPoly module (applies polynomial features internally)
        keys: Key tensor of shape (batch, seq, dim) - typically h (hidden state)
        values: Target values of shape (batch, seq, dim) - v from QKV projection
        gamma: Optional per-position decay weights (batch, seq, 1).
               If None, uses exponential decay from decay_base.
        context_window: Number of recent positions to include (default: 256)
        decay_base: Base for exponential decay if gamma is None (default: 0.95)

    Returns:
        Scalar loss tensor (with gradients for memory parameters)

    Note:
        - Memory input goes through polynomial features φ(k) internally
        - Uses return_contribution=True to get raw memory output (no residual)
        - Gradients flow through memory parameters for TTL update
    """
    batch, seq, dim = keys.shape

    # Determine context window bounds
    ctx_start = max(0, seq - context_window)
    ctx_len = seq - ctx_start

    # Extract context window
    k_ctx = keys[:, ctx_start:]  # (batch, ctx_len, dim)
    v_ctx = values[:, ctx_start:]  # (batch, ctx_len, dim)

    # Compute decay weights if not provided
    if gamma is None:
        # Exponential decay: gamma_base^(t-i) for i in [t-c+1, t]
        # Position 0 (oldest) gets lowest weight, position ctx_len-1 gets highest
        positions = torch.arange(ctx_len, device=keys.device, dtype=keys.dtype)
        gamma_weights = decay_base ** (ctx_len - 1 - positions)  # (ctx_len,)
        gamma_weights = gamma_weights.view(1, -1, 1)  # (1, ctx_len, 1)
    else:
        # Use provided per-position gates
        gamma_weights = gamma[:, ctx_start:]  # (batch, ctx_len, 1)

    # Forward through memory (gets raw contribution without residual)
    # The memory applies polynomial features internally via proj_down + _polynomial_features
    predicted = memory(k_ctx, return_contribution=True)  # (batch, ctx_len, dim)

    # Compute weighted MSE loss
    # ||M(phi(k_i)) - v_i||^2 per position, weighted by gamma
    diff = predicted - v_ctx  # (batch, ctx_len, dim)
    squared_error = (diff ** 2).sum(dim=-1, keepdim=True)  # (batch, ctx_len, 1)
    weighted_error = gamma_weights * squared_error  # (batch, ctx_len, 1)

    # Mean over batch and context positions
    loss: Tensor = weighted_error.mean()

    return loss


def compute_omega_loss_with_stats(
    memory: "AtlasMemoryPoly",
    keys: Tensor,
    values: Tensor,
    gamma: Tensor | None = None,
    context_window: int = 256,
    decay_base: float = 0.95,
) -> tuple[Tensor, dict[str, float]]:
    """
    Compute Omega loss with diagnostic statistics.

    Same as compute_omega_loss but also returns useful diagnostics
    for monitoring TTL behavior during training.

    Returns:
        Tuple of (loss, stats_dict) where stats_dict contains:
        - 'omega_loss': The loss value
        - 'prediction_norm': Mean norm of memory predictions
        - 'target_norm': Mean norm of target values
        - 'error_norm': Mean norm of prediction error
        - 'context_len': Effective context window length used
    """
    batch, seq, dim = keys.shape
    ctx_start = max(0, seq - context_window)
    ctx_len = seq - ctx_start

    k_ctx = keys[:, ctx_start:]
    v_ctx = values[:, ctx_start:]

    if gamma is None:
        positions = torch.arange(ctx_len, device=keys.device, dtype=keys.dtype)
        gamma_weights = decay_base ** (ctx_len - 1 - positions)
        gamma_weights = gamma_weights.view(1, -1, 1)
    else:
        gamma_weights = gamma[:, ctx_start:]

    predicted = memory(k_ctx, return_contribution=True)
    diff = predicted - v_ctx
    squared_error = (diff ** 2).sum(dim=-1, keepdim=True)
    weighted_error = gamma_weights * squared_error
    loss: Tensor = weighted_error.mean()

    # Compute diagnostics (detached to avoid affecting gradients)
    with torch.no_grad():
        stats = {
            "omega_loss": loss.item(),
            "prediction_norm": predicted.norm(dim=-1).mean().item(),
            "target_norm": v_ctx.norm(dim=-1).mean().item(),
            "error_norm": diff.norm(dim=-1).mean().item(),
            "context_len": ctx_len,
        }

    return loss, stats
