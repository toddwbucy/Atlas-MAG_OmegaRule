"""
Muon Optimizer with Newton-Schulz Orthogonalization.

REQ-P3-T1: Use K=5 for all memory modules

The Muon optimizer orthogonalizes gradients before applying updates.
This improves conditioning and training stability for memory networks.

Memory update equation:
    M_t = α_t × M_{t-1} - η_t × NewtonSchulz_K(∇L)

Reference:
- Muon optimizer (https://github.com/KellerJordan/Muon)
- PRD Section 6.2: Newton-Schulz Implementation
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Iterator

import torch
from torch import Tensor
from torch.optim import Optimizer

from src_clean.config import K
from src_clean.nn.newton_schulz import newton_schulz

logger = logging.getLogger(__name__)


@dataclass
class MuonState:
    """State for a single parameter in Muon optimizer."""

    momentum_buffer: Optional[Tensor] = None
    step: int = 0


class Muon(Optimizer):
    """
    Muon optimizer with Newton-Schulz orthogonalization.

    Uses K=5 Newton-Schulz iterations to orthogonalize gradients
    before applying the update. This improves conditioning for
    memory-heavy architectures like AtlasMAG.

    Key features:
    - Orthogonalizes 2D gradients (weight matrices) only
    - 1D gradients (biases, layer norms) use standard momentum
    - K=5 everywhere (no decoupling per PRD Round 8)

    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 1e-3)
        momentum: Momentum coefficient α (default: 0.95)
        k: Newton-Schulz iterations (default: 5, from config)
        weight_decay: L2 regularization (default: 0.0)
        orthogonalize_2d_only: Only orthogonalize 2D tensors (default: True)

    Example:
        >>> model = AtlasMAGSkeleton(...)
        >>> optimizer = Muon(model.parameters(), lr=1e-3, momentum=0.95)
        >>> for batch in dataloader:
        ...     loss = model(batch).mean()
        ...     loss.backward()
        ...     optimizer.step()
        ...     optimizer.zero_grad()
    """

    def __init__(
        self,
        params: Iterator[Tensor],
        lr: float = 1e-3,
        momentum: float = 0.95,
        k: int = K,  # Default from config (5)
        weight_decay: float = 0.0,
        orthogonalize_2d_only: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if k < 1:
            raise ValueError(f"Invalid Newton-Schulz iterations: {k}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            k=k,
            weight_decay=weight_decay,
            orthogonalize_2d_only=orthogonalize_2d_only,
        )
        super().__init__(params, defaults)

        logger.info(
            f"Muon optimizer initialized: lr={lr}, momentum={momentum}, "
            f"k={k}, weight_decay={weight_decay}"
        )

    @torch.no_grad()
    def step(  # type: ignore[override]
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """
        Perform a single optimization step.

        Args:
            closure: Optional closure for loss re-evaluation

        Returns:
            Loss value if closure provided, else None
        """
        loss: Optional[float] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            k = group['k']
            weight_decay = group['weight_decay']
            orthogonalize_2d_only = group['orthogonalize_2d_only']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay (L2 regularization)
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Orthogonalize gradients (2D only if flag set, otherwise all)
                if grad.ndim == 2 or not orthogonalize_2d_only:
                    grad = newton_schulz(grad, num_iters=k)

                # Get or create state
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['step'] = 0

                state['step'] += 1
                buf = state['momentum_buffer']

                # Momentum update: buf = momentum * buf + grad
                buf.mul_(momentum).add_(grad)

                # Parameter update: p = p - lr * buf
                p.add_(buf, alpha=-lr)

        return loss

    def get_k(self) -> int:
        """Get the Newton-Schulz iteration count (should be 5)."""
        if self.param_groups:
            return int(self.param_groups[0]['k'])
        return K

    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of optimizer state for debugging."""
        total_params = 0
        total_2d_params = 0
        total_steps = 0

        for group in self.param_groups:
            for p in group['params']:
                total_params += 1
                if p.ndim == 2:
                    total_2d_params += 1
                if p in self.state and 'step' in self.state[p]:
                    total_steps = max(total_steps, self.state[p]['step'])

        return {
            "total_params": total_params,
            "orthogonalized_params": total_2d_params,
            "current_step": total_steps,
            "k": self.get_k(),
            "lr": self.param_groups[0]['lr'] if self.param_groups else 0,
            "momentum": self.param_groups[0]['momentum'] if self.param_groups else 0,
        }


class MuonMemory(Optimizer):
    """
    Specialized Muon variant for memory matrix updates.

    This optimizer is designed specifically for the M_t memory matrix
    in AtlasMAG, using the full Muon update equation:

        S_t = θ × S_{t-1} + ∇L
        M_t = α × M_{t-1} - η × NewtonSchulz_K(S_t)

    Where S is the gradient accumulator (similar to momentum).

    Args:
        params: Memory matrix parameters
        lr: Learning rate η (default: 1e-3)
        alpha: Memory decay α (default: 0.95)
        theta: Gradient accumulator decay θ (default: 0.9)
        k: Newton-Schulz iterations (default: 5)
    """

    def __init__(
        self,
        params: Iterator[Tensor],
        lr: float = 1e-3,
        alpha: float = 0.95,
        theta: float = 0.9,
        k: int = K,
    ):
        defaults = dict(lr=lr, alpha=alpha, theta=theta, k=k)
        super().__init__(params, defaults)

        logger.info(
            f"MuonMemory initialized: lr={lr}, alpha={alpha}, "
            f"theta={theta}, k={k}"
        )

    @torch.no_grad()
    def step(  # type: ignore[override]
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """Perform memory update step."""
        loss: Optional[float] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            theta = group['theta']
            k = group['k']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['S'] = torch.zeros_like(p)
                    state['step'] = 0

                state['step'] += 1
                S = state['S']

                # Update gradient accumulator: S = θ × S + ∇L
                S.mul_(theta).add_(grad)

                # Orthogonalize accumulated gradient
                if S.ndim == 2:
                    S_orth = newton_schulz(S, num_iters=k)
                else:
                    S_orth = S

                # Memory update: M = α × M - η × NS(S)
                p.mul_(alpha).add_(S_orth, alpha=-lr)

        return loss
