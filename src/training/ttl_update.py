"""
Test-Time Learning (TTL) Update Step.

Implements the memory update rule from the Atlas paper (arXiv:2505.23735):

    Momentum accumulation (Eq. 33):
        S_t = theta * S_{t-1} + grad(L_omega)

    Memory update (Eq. 32):
        M_t = alpha * M_{t-1} - eta * NewtonSchulz(S_t)

The Newton-Schulz orthogonalization (Muon optimizer) keeps updates on the
Stiefel manifold, preventing magnitude explosion from accumulated momentum.

Reference:
    - Atlas paper Eq. 32-33: Update rules
    - Atlas paper Section 4.2: Muon optimizer discussion
"""

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from src.nn.newton_schulz import newton_schulz

if TYPE_CHECKING:
    from src.model.atlas_memory import AtlasMemoryPoly


def ttl_step(
    memory: "AtlasMemoryPoly",
    loss: Tensor,
    theta: float = 0.9,
    alpha: float = 0.999,
    eta: float = 0.01,
    ns_iterations: int = 5,
    adaptive_eta: bool = False,
) -> dict[str, float]:
    """
    Perform one TTL update step on memory parameters.

    This implements the core Atlas TTL update:
        1. Compute gradients of Omega loss w.r.t. memory parameters
        2. Accumulate into momentum: S_t = theta * S_{t-1} + grad
        3. Orthogonalize via Newton-Schulz: update = NS(S_t)
        4. Update parameters: M_t = alpha * M_{t-1} - eta * update

    Args:
        memory: AtlasMemoryPoly module with momentum buffers
        loss: Omega loss tensor (scalar, with gradients enabled)
        theta: Momentum decay factor (default: 0.9)
        alpha: Weight decay factor (default: 0.999, close to 1 = minimal decay)
        eta: Learning rate for memory updates (default: 0.01)
        ns_iterations: Newton-Schulz iterations (default: 5, per paper)
        adaptive_eta: If True, scale eta by 1/(1 + grad_norm) to stabilize large-gradient steps

    Returns:
        Dictionary of statistics for logging:
        - '{param}_grad_norm': Gradient norm for each parameter
        - '{param}_momentum_norm': Momentum norm after accumulation
        - '{param}_update_norm': Update norm after Newton-Schulz

    Note:
        - Uses create_graph=False since we don't need second-order gradients
        - Newton-Schulz is only applied to 2D+ tensors (skip for bias terms)
        - Updates are applied in-place to memory.parameters()
    """
    # Get list of parameters (need stable ordering for zip with grads)
    param_list = list(memory.named_parameters())

    # Compute gradients of loss w.r.t. memory parameters
    grads = torch.autograd.grad(
        loss,
        [p for _, p in param_list],
        create_graph=False,  # No second-order gradients needed
        retain_graph=False,  # Can release graph after this
    )

    stats: dict[str, float] = {}

    # Compute effective eta (adaptive scaling by inverse gradient norm)
    if adaptive_eta:
        total_grad_norm = torch.sqrt(sum(g.norm() ** 2 for g in grads))
        effective_eta = eta / (1.0 + total_grad_norm.item())
        stats["effective_eta"] = effective_eta
        stats["total_grad_norm"] = total_grad_norm.item()
    else:
        effective_eta = eta

    # Update each parameter
    for (name, param), grad in zip(param_list, grads):
        # Get momentum buffer
        momentum = memory.get_momentum(name)

        # Accumulate momentum: S_t = theta * S_{t-1} + grad
        momentum.mul_(theta).add_(grad)

        # Orthogonalize via Newton-Schulz (only for 2D matrices)
        if momentum.ndim == 2:
            # Newton-Schulz requires exactly 2D input
            update = newton_schulz(momentum, num_iters=ns_iterations)
        else:
            # For 1D tensors (bias), just normalize
            update = momentum / (momentum.norm() + 1e-8)

        # Update parameter: M_t = alpha * M_{t-1} - eta * update
        # Using in-place operations on param.data to avoid autograd issues
        param.data.mul_(alpha).sub_(update, alpha=effective_eta)

        # Record statistics
        stats[f"{name}_grad_norm"] = grad.norm().item()
        stats[f"{name}_momentum_norm"] = momentum.norm().item()
        stats[f"{name}_update_norm"] = update.norm().item()

    return stats


def ttl_step_with_grad_clip(
    memory: "AtlasMemoryPoly",
    loss: Tensor,
    theta: float = 0.9,
    alpha: float = 0.999,
    eta: float = 0.01,
    ns_iterations: int = 5,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """
    TTL update step with gradient clipping for stability.

    Same as ttl_step but clips gradients before momentum accumulation.
    Useful if Omega loss produces large gradients initially.

    Args:
        memory: AtlasMemoryPoly module with momentum buffers
        loss: Omega loss tensor
        theta: Momentum decay factor
        alpha: Weight decay factor
        eta: Learning rate
        ns_iterations: Newton-Schulz iterations
        max_grad_norm: Maximum gradient norm (default: 1.0)

    Returns:
        Statistics dictionary (same as ttl_step, plus 'grad_clip_ratio')
    """
    param_list = list(memory.named_parameters())

    grads = torch.autograd.grad(
        loss,
        [p for _, p in param_list],
        create_graph=False,
        retain_graph=False,
    )

    # Compute total gradient norm for clipping
    total_norm = torch.sqrt(sum(g.norm() ** 2 for g in grads))
    clip_coef = max_grad_norm / (total_norm + 1e-8)
    clip_coef = torch.clamp(clip_coef, max=1.0)

    stats: dict[str, float] = {"grad_clip_ratio": float(clip_coef.item())}

    for (name, param), grad in zip(param_list, grads):
        # Apply gradient clipping
        clipped_grad = grad * clip_coef

        momentum = memory.get_momentum(name)
        momentum.mul_(theta).add_(clipped_grad)

        if momentum.ndim >= 2:
            update = newton_schulz(momentum, num_iters=ns_iterations)
        else:
            update = momentum / (momentum.norm() + 1e-8)

        param.data.mul_(alpha).sub_(update, alpha=eta)

        stats[f"{name}_grad_norm"] = grad.norm().item()
        stats[f"{name}_clipped_grad_norm"] = clipped_grad.norm().item()
        stats[f"{name}_momentum_norm"] = momentum.norm().item()
        stats[f"{name}_update_norm"] = update.norm().item()

    return stats


class TTLUpdater:
    """
    Stateful TTL updater with configurable reset modes.

    Wraps the ttl_step function with state management for:
    - Tracking update counts
    - Handling reset at sequence/batch boundaries
    - Aggregating statistics across steps

    Args:
        theta: Momentum decay factor
        alpha: Weight decay factor
        eta: Learning rate
        ns_iterations: Newton-Schulz iterations
        reset_mode: When to reset momentum ("sequence", "batch", "never")
    """

    def __init__(
        self,
        theta: float = 0.9,
        alpha: float = 0.999,
        eta: float = 0.01,
        ns_iterations: int = 5,
        reset_mode: str = "sequence",
        adaptive_eta: bool = False,
    ):
        self.theta = theta
        self.alpha = alpha
        self.eta = eta
        self.ns_iterations = ns_iterations
        self.reset_mode = reset_mode
        self.adaptive_eta = adaptive_eta

        self._step_count = 0
        self._sequence_count = 0

    def step(
        self,
        memory: "AtlasMemoryPoly",
        loss: Tensor,
    ) -> dict[str, float]:
        """
        Perform TTL update step.

        Args:
            memory: AtlasMemoryPoly module
            loss: Omega loss tensor

        Returns:
            Statistics dictionary
        """
        stats = ttl_step(
            memory=memory,
            loss=loss,
            theta=self.theta,
            alpha=self.alpha,
            eta=self.eta,
            ns_iterations=self.ns_iterations,
            adaptive_eta=self.adaptive_eta,
        )

        self._step_count += 1
        stats["ttl_step_count"] = self._step_count

        return stats

    def on_sequence_start(self, memory: "AtlasMemoryPoly") -> None:
        """
        Called at the start of a new sequence.

        Resets momentum if reset_mode is "sequence".
        """
        self._sequence_count += 1
        if self.reset_mode == "sequence":
            memory.reset_momentum()

    def on_batch_start(self, memory: "AtlasMemoryPoly") -> None:
        """
        Called at the start of a new batch.

        Resets momentum if reset_mode is "batch".
        """
        if self.reset_mode == "batch":
            memory.reset_momentum()

    @property
    def step_count(self) -> int:
        """Total number of TTL steps performed."""
        return self._step_count

    @property
    def sequence_count(self) -> int:
        """Total number of sequences processed."""
        return self._sequence_count
