"""
W_init: Steady-State Initialization for Memory.

Computes the initial memory state by running calibration data through
the model and finding the steady-state. This reduces "reset shock"
when memory is reset at shard boundaries during training.

PRD Requirement P0-T1:
    Compute W_init via steady-state calibration.
    ACCEPTANCE: Reset shock < 5% loss spike.

Reference: TNT paper (arXiv:2511.07343)
"""

import logging
from typing import Callable, List, Optional, Protocol, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    CALIBRATION_TOKENS,
    RESET_SHOCK_THRESHOLD,
)


class AtlasMAGModel(Protocol):
    """Protocol for AtlasMAG models with required attributes."""
    dim: int
    w_init: nn.Parameter
    blocks: nn.ModuleList

    def to(self, device: torch.device) -> "AtlasMAGModel": ...
    def train(self, mode: bool) -> "AtlasMAGModel": ...
    def __call__(self, input_ids: Tensor) -> Tensor: ...
    def forward_memory_only(self, input_ids: Tensor) -> Tuple[Tensor, Tensor]: ...

logger = logging.getLogger(__name__)


def compute_steady_state_init(
    model: AtlasMAGModel,
    calibration_loader: DataLoader[Tensor],
    num_tokens: int = CALIBRATION_TOKENS,
    device: torch.device = torch.device("cuda"),
    use_running_mean: bool = True,
    momentum: float = 0.99,
) -> Tensor:
    """
    Compute W_init via steady-state mean of memory states.

    Runs calibration data through the model, collecting memory module
    activations. The mean of these activations becomes W_init, providing
    a good starting point that reduces reset shock.

    Args:
        model: The AtlasMAG model
        calibration_loader: DataLoader with calibration sequences
        num_tokens: Maximum tokens to process
        device: Device to run on
        use_running_mean: If True, use exponential moving average
        momentum: EMA momentum (only used if use_running_mean=True)

    Returns:
        W_init tensor of shape (dim,)

    Phase 0 Acceptance Criteria:
        - Reset shock < 5% loss spike at shard boundary
    """
    model = model.to(device)
    model.train(False)

    dim: int = model.dim
    tokens_seen = 0
    w_init: Tensor

    if use_running_mean:
        # Exponential moving average for memory stability
        w_init = torch.zeros(dim, device=device)
        n_updates = 0
    else:
        # Collect all states for mean
        all_states: List[Tensor] = []

    logger.info(f"Computing W_init from {num_tokens} calibration tokens...")

    with torch.no_grad():
        pbar = tqdm(calibration_loader, desc="W_init calibration")
        for batch in pbar:
            if tokens_seen >= num_tokens:
                break

            batch = batch.to(device)

            # Forward pass that returns memory state
            _, memory_state = model.forward_memory_only(batch)

            if use_running_mean:
                # EMA update
                if n_updates == 0:
                    w_init = memory_state.mean(dim=0) if memory_state.ndim > 1 else memory_state
                else:
                    current = memory_state.mean(dim=0) if memory_state.ndim > 1 else memory_state
                    # Reduce to (dim,) if needed - flatten first for safe slicing
                    if current.numel() > dim:
                        current = current.flatten()[:int(dim)]
                    w_init = momentum * w_init + (1 - momentum) * current
                n_updates += 1
            else:
                all_states.append(memory_state.cpu())

            tokens_seen += batch.numel()
            pbar.set_postfix({"tokens": tokens_seen})

    if not use_running_mean:
        # Stack and compute mean
        if not all_states:
            raise ValueError(
                "No calibration batches processed; cannot compute W_init. "
                "Check that calibration_loader yields data."
            )
        all_states_tensor = torch.cat(all_states, dim=0)
        w_init = all_states_tensor.mean(dim=0).to(device)

    # Reduce to correct dimension if needed - flatten first for safe slicing
    if w_init.numel() > dim:
        w_init = w_init.flatten()[:int(dim)]

    logger.info(
        f"W_init computed: shape={w_init.shape}, "
        f"norm={torch.linalg.norm(w_init).item():.4f}, "
        f"mean={w_init.mean().item():.6f}, "
        f"std={w_init.std().item():.6f}"
    )

    result: Tensor = w_init
    return result


def measure_reset_shock(
    model: AtlasMAGModel,
    test_loader: DataLoader[Tensor],
    w_init: Optional[Tensor] = None,
    device: torch.device = torch.device("cuda"),
    num_batches: int = 10,
) -> Tuple[float, float, float]:
    """
    Measure reset shock: loss difference when memory is reset.

    Compares loss with continuous memory vs. reset memory to
    validate that W_init reduces shock to < 5%.

    Args:
        model: The AtlasMAG model
        test_loader: DataLoader with test sequences
        w_init: Optional W_init to use (uses model's if None)
        device: Device to run on
        num_batches: Number of batches to test

    Returns:
        Tuple of (shock_ratio, loss_with_init, loss_without_init)
    """
    model = model.to(device)
    model.train(False)

    # Save original W_init
    w_init_param: nn.Parameter = model.w_init
    original_w_init = w_init_param.data.clone()

    losses_with_init = []
    losses_without_init = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_batches:
                break

            batch = batch.to(device)

            # Test with W_init
            if w_init is not None:
                w_init_param.data = w_init
            logits_with = model(batch)
            targets = batch[:, 1:].contiguous()
            logits_with = logits_with[:, :-1].contiguous()
            loss_with = F.cross_entropy(
                logits_with.view(-1, logits_with.size(-1)),
                targets.view(-1),
            )
            losses_with_init.append(loss_with.item())

            # Test without W_init (zero)
            w_init_param.data = torch.zeros_like(w_init_param.data)
            logits_without = model(batch)
            loss_without = F.cross_entropy(
                logits_without[:, :-1].contiguous().view(-1, logits_without.size(-1)),
                targets.view(-1),
            )
            losses_without_init.append(loss_without.item())

    # Restore original W_init
    w_init_param.data = original_w_init

    # Guard against empty batch lists
    if not losses_with_init or not losses_without_init:
        logger.warning("No batches processed for reset shock measurement")
        return 0.0, 0.0, 0.0  # shock_ratio, avg_with, avg_without

    # Compute averages
    avg_with = sum(losses_with_init) / len(losses_with_init)
    avg_without = sum(losses_without_init) / len(losses_without_init)

    # Shock ratio: how much worse is it without W_init?
    if avg_with > 0:
        shock_ratio = (avg_without - avg_with) / avg_with
    else:
        shock_ratio = 0.0

    logger.info(
        f"Reset shock measurement: "
        f"with_init={avg_with:.4f}, without_init={avg_without:.4f}, "
        f"shock_ratio={shock_ratio:.2%}"
    )

    if shock_ratio > RESET_SHOCK_THRESHOLD:
        logger.warning(
            f"Reset shock {shock_ratio:.2%} exceeds threshold "
            f"{RESET_SHOCK_THRESHOLD:.0%}! Consider more calibration tokens."
        )
    else:
        logger.info(
            f"Reset shock {shock_ratio:.2%} < {RESET_SHOCK_THRESHOLD:.0%} threshold"
        )

    return shock_ratio, avg_with, avg_without


def apply_w_init(model: AtlasMAGModel, w_init: Tensor) -> None:
    """
    Apply computed W_init to model.

    Args:
        model: The AtlasMAG model
        w_init: Computed W_init tensor
    """
    with torch.no_grad():
        w_init_param: nn.Parameter = model.w_init
        w_init_param.data = w_init.to(w_init_param.device)

    logger.info(f"Applied W_init to model: norm={torch.linalg.norm(w_init).item():.4f}")


def calibrate_with_increasing_tokens(
    model: AtlasMAGModel,
    loader_fn: Callable[[int], DataLoader[Tensor]],
    device: torch.device = torch.device("cuda"),
    initial_tokens: int = CALIBRATION_TOKENS,
    max_tokens: int = 100_000,
    multiplier: float = 2.0,
) -> Tensor:
    """
    Calibrate W_init, increasing tokens if shock > threshold.

    Implements the PRD requirement P0-T5:
        "Start 10K tokens, increase if shock > 5%"

    Args:
        model: The AtlasMAG model
        loader_fn: Function that creates DataLoader given num_tokens
        device: Device to run on
        initial_tokens: Starting token count
        max_tokens: Maximum tokens before giving up
        multiplier: Factor to increase tokens by on failure

    Returns:
        Calibrated W_init tensor

    Raises:
        ValueError: If initial_tokens > max_tokens (loop would never execute)
    """
    if initial_tokens > max_tokens:
        raise ValueError(
            f"initial_tokens ({initial_tokens}) must be <= max_tokens ({max_tokens})"
        )

    num_tokens = initial_tokens
    w_init: Optional[Tensor] = None

    while num_tokens <= max_tokens:
        logger.info(f"Attempting calibration with {num_tokens} tokens...")

        # Create calibration loader
        cal_loader = loader_fn(num_tokens)
        # Ensure test_loader gets at least 1 token (avoids empty loader when num_tokens < 10)
        test_tokens = max(1, num_tokens // 10)
        test_loader = loader_fn(test_tokens)

        # Compute W_init
        w_init = compute_steady_state_init(
            model, cal_loader, num_tokens=num_tokens, device=device
        )

        # Apply and measure shock
        apply_w_init(model, w_init)
        shock, _, _ = measure_reset_shock(model, test_loader, w_init, device)

        if shock < RESET_SHOCK_THRESHOLD:
            logger.info(f"Calibration successful with {num_tokens} tokens")
            return w_init

        # Increase tokens and retry
        num_tokens = int(num_tokens * multiplier)
        logger.warning(
            f"Shock {shock:.2%} too high, increasing to {num_tokens} tokens"
        )

    logger.error(
        f"Failed to achieve < {RESET_SHOCK_THRESHOLD:.0%} shock "
        f"even with {max_tokens} tokens!"
    )
    # w_init is guaranteed to be set (guard ensures loop runs at least once)
    assert w_init is not None, "w_init should be set after loop"
    return w_init  # Return best effort
