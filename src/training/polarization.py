"""
Gate Polarization Loss for Phase 1 Architecture Validation.

The polarization loss encourages gates to commit to either attention (gate=0)
or memory (gate=1), rather than averaging (gate=0.5). This is essential for
proving the MAG architecture can actually ROUTE, not just average.

Loss function:
    L_polar = λ(t) × (1 - |2g - 1|)

Properties:
    - Maximum penalty (λ) at g=0.5 (indecisive)
    - Zero penalty at g∈{0,1} (decisive)
    - λ starts high (10.0) to force polarization, then decays to 0.1

Reference: PRD Section "Gate Polarization"
"""

import torch
from torch import Tensor

from src.config import LAMBDA_INITIAL, LAMBDA_FINAL, POLARIZATION_ANNEAL_RATIO


def get_lambda_polar(step: int, total_steps: int) -> float:
    """
    Compute annealing schedule for polarization penalty.

    Schedule:
        - First 10% of training: HIGH λ = 10.0 (winner-take-all pressure)
        - Remaining 90%: Exponential decay λ → 0.1 (allow subtle mixing)

    The high initial λ forces gates to pick a side early. The decay allows
    nuanced mixing once the architecture has proven it can route.

    Args:
        step: Current training step (0-indexed)
        total_steps: Total number of training steps

    Returns:
        Lambda value for polarization loss at this step

    Example:
        >>> get_lambda_polar(0, 10000)     # Start of training
        10.0
        >>> get_lambda_polar(500, 10000)   # Still in warmup (5% < 10%)
        10.0
        >>> get_lambda_polar(10000, 10000) # End of training
        0.1
    """
    if total_steps <= 0:
        return LAMBDA_INITIAL

    warmup_steps = int(POLARIZATION_ANNEAL_RATIO * total_steps)

    if step < warmup_steps:
        # During warmup: constant high λ
        return LAMBDA_INITIAL
    else:
        # After warmup: exponential decay from LAMBDA_INITIAL to LAMBDA_FINAL
        remaining_steps = total_steps - warmup_steps
        if remaining_steps <= 0:
            return LAMBDA_FINAL

        progress = (step - warmup_steps) / remaining_steps
        # Clamp progress to [0, 1]
        progress = max(0.0, min(1.0, progress))

        # Exponential interpolation: λ_init * (λ_final/λ_init)^progress
        # This gives smooth exponential decay
        ratio = LAMBDA_FINAL / LAMBDA_INITIAL
        return float(LAMBDA_INITIAL * (ratio ** progress))


def gate_polarization_loss(
    gate_values: Tensor,
    step: int,
    total_steps: int,
) -> Tensor:
    """
    Compute gate polarization loss.

    The loss function:
        L_polar = λ(t) × mean(1 - |2g - 1|)

    This has the following properties:
        - At g=0.5: |2(0.5)-1| = 0, so penalty = λ × 1 (maximum)
        - At g=0.0: |2(0.0)-1| = 1, so penalty = λ × 0 (zero)
        - At g=1.0: |2(1.0)-1| = 1, so penalty = λ × 0 (zero)

    The loss encourages gates to commit to either extreme, proving the
    architecture can route rather than just average.

    Args:
        gate_values: Tensor of gate values in [0, 1], any shape
        step: Current training step
        total_steps: Total number of training steps

    Returns:
        Scalar tensor with polarization loss

    Example:
        >>> gates = torch.tensor([0.5, 0.5, 0.5])  # All indecisive
        >>> loss = gate_polarization_loss(gates, step=0, total_steps=1000)
        >>> loss.item()  # High penalty (λ=10.0)
        10.0

        >>> gates = torch.tensor([0.0, 1.0, 0.0])  # All decisive
        >>> loss = gate_polarization_loss(gates, step=0, total_steps=1000)
        >>> loss.item()  # Zero penalty
        0.0
    """
    lambda_polar = get_lambda_polar(step, total_steps)

    # Compute polarization: how far from decisive (0 or 1)?
    # |2g - 1| = 1 when g∈{0,1}, = 0 when g=0.5
    decisiveness = torch.abs(2 * gate_values - 1)

    # Penalty is inverse of decisiveness
    # indecisive (g=0.5) → penalty=1, decisive (g∈{0,1}) → penalty=0
    polarization_penalty = 1.0 - decisiveness

    # Mean across all gates, scaled by λ
    loss: Tensor = lambda_polar * polarization_penalty.mean()

    return loss


def compute_gate_statistics(gate_values: Tensor) -> dict:
    """
    Compute diagnostic statistics for gate values.

    Useful for monitoring gate behavior during training.

    Args:
        gate_values: Tensor of gate values in [0, 1]

    Returns:
        Dictionary with:
            - mean: Mean gate value
            - std: Standard deviation
            - min: Minimum value
            - max: Maximum value
            - polarized_ratio: Fraction at <0.1 or >0.9
            - indecisive_ratio: Fraction in (0.4, 0.6)
    """
    values = gate_values.detach().float()

    # Polarized: gates that have committed to a side
    polarized = ((values < 0.1) | (values > 0.9)).float().mean().item()

    # Indecisive: gates stuck in the middle
    indecisive = ((values > 0.4) & (values < 0.6)).float().mean().item()

    return {
        "mean": values.mean().item(),
        "std": values.std().item(),
        "min": values.min().item(),
        "max": values.max().item(),
        "polarized_ratio": polarized,
        "indecisive_ratio": indecisive,
    }
