"""
Atlas-MAG Skeleton Model with Polynomial Memory.

This is the minimal testable model implementing the Atlas MAG architecture
with output-level combination of SWA (attention) and memory branches.

Key features from Atlas paper (arXiv:2505.23735):
- Output-level combination: SWA and memory run as parallel branches
- Polynomial features (phi_2): Essential for memory capacity O(d_k^2)
- Input-dependent gamma gates: Per-position decay modulation
- Per-layer gate initialization: Spread across layers for differentiation

Memory capacity (Section 3.1, Propositions 1 & 2):
- Without polynomial features: O(d_k) ~ 64 associations
- With phi_2 polynomial features: O(d_k^2) ~ 4,096 associations

Note: TNT (Titans-in-Titans) hierarchical memory is NOT implemented.
This model uses single-layer memory with polynomial feature expansion.

Reference: Atlas paper (arXiv:2505.23735)
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import (
    GAMMA_GATE_HIDDEN_DIM,
    MEMORY_EXPANSION,
    N_HEADS,
    OMEGA_CONTEXT_WINDOW,
    OMEGA_DECAY_BASE,
    POLY_DEGREE,
    USE_POLY_MEMORY,
    VOCAB_SIZE,
    D,
)

from src.model.atlas_memory import AtlasMemory, AtlasMemoryPoly
from src.model.projections import QKVProjection, RotaryEmbedding
from src.nn.rmsnorm import RMSNorm
from src.nn.swiglu import SwiGLU
from src.training.omega_loss import compute_omega_loss
from src.training.ttl_update import ttl_step

# Re-export for external access
__all__ = ["AtlasMAGBlock", "AtlasMAGSkeleton", "GammaGate"]

logger = logging.getLogger(__name__)


class GammaGate(nn.Module):
    """
    Input-dependent gamma gate for Omega Rule context pruning.

    Produces per-position decay multipliers that modulate the base
    exponential decay in the memory context window. This allows the
    model to selectively forget irrelevant past context based on content.

    Architecture: x -> Linear -> SiLU -> Linear -> Sigmoid -> gamma

    Reference: Atlas paper (arXiv:2505.23735) Eq. 9
    """

    def __init__(self, dim: int, hidden_dim: int = GAMMA_GATE_HIDDEN_DIM):
        """
        Initialize gamma gate module.

        Args:
            dim: Input dimension
            hidden_dim: Hidden layer dimension for the gate MLP
        """
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize to output ~0.5 initially (neutral modulation)."""
        # First linear: small random
        nn.init.normal_(self.gate[0].weight, std=0.02)
        # Second linear: near-zero so sigmoid(0) ~ 0.5
        nn.init.zeros_(self.gate[2].weight)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute per-position gamma gates.

        Args:
            x: Input tensor (batch, seq_len, dim)

        Returns:
            Gamma gates (batch, seq_len, 1) in range [0, 1]
        """
        return self.gate(x)


class AtlasMAGBlock(nn.Module):
    """
    Single Atlas-MAG block combining attention and memory.

    Architecture (Figure 3 from Atlas paper):
        Input → [SWA Branch] ──────┐
              → [Memory Branch] ───┼──► Add → Output

    Memory branch uses AtlasMemoryPoly with polynomial features (phi_2):
        - Capacity without phi: O(d_k) ~ 64 associations
        - Capacity with phi_2: O(d_k^2) ~ 4,096 associations

    Key features from Atlas paper:
    - Output-level combination (not Q-level blending)
    - Polynomial features for increased memory capacity (ESSENTIAL)
    - Input-dependent gamma gates for context pruning
    - Per-layer gate initialization for differentiation

    Args:
        dim: Model dimension
        n_heads: Number of attention heads
        memory_expansion: Memory hidden dimension multiplier
        ffn_expansion: FFN hidden dimension multiplier
        layer_idx: Layer index for per-layer gate initialization
        n_layers: Total number of layers (for initialization scaling)
    """

    def __init__(
        self,
        dim: int = D,
        n_heads: int = N_HEADS,
        memory_expansion: int = MEMORY_EXPANSION,
        ffn_expansion: int = 4,
        disable_memory: bool = False,
        layer_idx: int = 0,
        n_layers: int = 12,
        poly_degree: int = POLY_DEGREE,
        # TTL (Test-Time Learning) configuration
        ttl_enabled: bool = True,
        ttl_theta: float = 0.9,  # Momentum decay
        ttl_alpha: float = 0.999,  # Weight decay
        ttl_eta: float = 0.01,  # Learning rate
        ttl_ns_iters: int = 5,  # Newton-Schulz iterations
        ttl_adaptive_eta: bool = False,  # Scale eta by inverse gradient norm
        # Memory dropout: randomly disable memory for this fraction of training steps
        memory_dropout_rate: float = 0.0,
        # Polynomial low-rank compression
        poly_rank: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.disable_memory = disable_memory
        self.layer_idx = layer_idx
        self.memory_dropout_rate = memory_dropout_rate

        # TTL configuration
        self.ttl_enabled = ttl_enabled and not disable_memory  # TTL requires memory
        self.ttl_theta = ttl_theta
        self.ttl_alpha = ttl_alpha
        self.ttl_eta = ttl_eta
        self.ttl_ns_iters = ttl_ns_iters
        self.ttl_adaptive_eta = ttl_adaptive_eta

        # Normalization layers
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        # Q, K, V projections
        self.qkv = QKVProjection(dim, n_heads, qk_norm=True)

        # Rotary embeddings
        self.rope = RotaryEmbedding(self.head_dim)

        # Input-dependent gamma gate for Omega Rule decay modulation
        # Used for TTL Omega loss weighting
        self.gamma_gate: Optional[GammaGate] = None
        if not disable_memory:
            self.gamma_gate = GammaGate(dim)

        # Memory module: AtlasMemoryPoly (with polynomial features) or AtlasMemory
        # Polynomial features are ESSENTIAL for memory capacity (Atlas paper Section 3.1)
        # Without: O(d_k) capacity. With phi_2: O(d_k^2) capacity (64x improvement for d_k=64)
        self.use_poly_memory = USE_POLY_MEMORY
        if USE_POLY_MEMORY:
            # key_dim = head_dim for polynomial features (capacity theorem uses d_k)
            self.memory: nn.Module = AtlasMemoryPoly(
                dim=dim,
                key_dim=self.head_dim,  # Use head_dim for O(d_k^p) capacity
                expansion=memory_expansion,
                poly_degree=poly_degree,
                poly_rank=poly_rank,
            )
        else:
            self.memory = AtlasMemory(dim, memory_expansion)

        # Memory gate: controls Q vs Q' blending
        # PER-LAYER INITIALIZATION: Spread gates across layers
        # Early layers (idx=0): more attention-focused (gate ~ 0.1)
        # Later layers (idx=n-1): more memory-focused (gate ~ 0.3)
        # This encourages layer specialization during training
        gate_init = self._compute_layer_gate_init(layer_idx, n_layers)
        self.memory_gate = nn.Parameter(torch.tensor([gate_init]))

        # Output projection
        self.w_o = nn.Linear(dim, dim, bias=False)

        # FFN (SwiGLU)
        self.ffn = SwiGLU(dim, int(dim * ffn_expansion * 2 / 3))

        self._init_weights()

        logger.debug(
            f"AtlasMAGBlock layer {layer_idx}: gate_init={gate_init:.3f} "
            f"(sigmoid={torch.sigmoid(torch.tensor(gate_init)).item():.3f})"
        )

    def _compute_layer_gate_init(self, layer_idx: int, n_layers: int) -> float:
        """
        Compute per-layer gate initialization.

        Committee feedback: Previous aggressive init [0.50, 0.62] biased all
        layers toward memory from step 1, preventing attention from developing
        useful patterns. Combined with high polarization lambda, gates were
        pushed further toward memory before attention had a chance.

        New initialization maps to sigmoid values:
        - Layer 0: sigmoid(-1.73) ≈ 0.15 (15% memory, attention-favored)
        - Layer n-1: sigmoid(-0.41) ≈ 0.40 (40% memory, balanced)

        This gives attention a head start in early layers while allowing
        later layers to develop memory usage organically.

        Args:
            layer_idx: Current layer index (0-indexed)
            n_layers: Total number of layers

        Returns:
            Gate initialization value (pre-sigmoid)
        """
        # Linear interpolation from -1.73 (early) to -0.41 (late)
        # sigmoid range [0.15, 0.40] — attention-favored for early layers
        gate_start = -1.73  # sigmoid ≈ 0.15
        gate_end = -0.41    # sigmoid ≈ 0.40

        if n_layers <= 1:
            return (gate_start + gate_end) / 2

        t = layer_idx / (n_layers - 1)  # 0 to 1
        return gate_start + t * (gate_end - gate_start)

    def _init_weights(self):
        """Initialize output projection."""
        nn.init.normal_(self.w_o.weight, std=0.02 / (2**0.5))

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[dict]]:
        """
        Forward pass through the block with output-level memory combination and TTL.

        Architecture (Atlas paper Figure 3 - MAG):
            Input → [SWA Branch] ──────┐
                  → [Memory Branch] ───┼──► Add → Output

        The SWA (Sliding Window Attention) and Memory branches run in parallel,
        and their outputs are combined at the output level (after both computations).
        This is different from Q-level blending which modifies Q before attention.

        TTL Update (if enabled):
            Before using memory output, performs test-time learning update:
            1. Compute Omega loss: L = sum(gamma_i * ||M(h_i) - v_i||^2)
            2. Update memory: S_t = theta*S_{t-1} + grad, M -= eta*NS(S_t)

        Memory contribution is gated:
            output = attn_out + gate * mem_out

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            attention_mask: Optional causal mask

        Returns:
            Tuple of:
                - Output tensor of shape (batch, seq_len, dim)
                - TTL stats dict (or None if TTL disabled/not run)
        """
        batch, seq_len, dim = x.shape
        ttl_stats: Optional[dict] = None

        # Pre-norm
        h = self.norm1(x)

        # Q, K, V projections (with reshape to heads)
        q, k, v = self.qkv(h, reshape_heads=True)

        # Apply rotary embeddings
        q, k = self.rope(q, k)

        # === TTL Update (before using memory) ===
        # Updates memory MLP parameters based on Omega Rule loss (Atlas Eq. 9).
        # Uses h as keys and v as targets: L = Σ γ_i * ||M(h_i) - v_i||²
        if self.ttl_enabled and not self.disable_memory and self.use_poly_memory:
            from typing import cast as type_cast

            memory_poly = type_cast(AtlasMemoryPoly, self.memory)

            # Flatten v from (batch, n_heads, seq_len, head_dim) to (batch, seq_len, dim)
            v_flat = v.transpose(1, 2).contiguous().view(batch, seq_len, dim)

            # Get gamma gates for TTL decay weighting
            gamma_ttl = self.gamma_gate(h) if self.gamma_gate is not None else None

            # Compute Omega loss: trains memory MLP to predict values from hidden states
            with torch.enable_grad():
                omega_loss = compute_omega_loss(
                    memory=memory_poly,
                    keys=h,  # Hidden states as keys (direct input to memory MLP)
                    values=v_flat,  # QKV values as targets
                    gamma=gamma_ttl,
                    context_window=OMEGA_CONTEXT_WINDOW,
                    decay_base=OMEGA_DECAY_BASE,
                )

                # Perform TTL update step
                ttl_stats = ttl_step(
                    memory=memory_poly,
                    loss=omega_loss,
                    theta=self.ttl_theta,
                    alpha=self.ttl_alpha,
                    eta=self.ttl_eta,
                    ns_iterations=self.ttl_ns_iters,
                    adaptive_eta=self.ttl_adaptive_eta,
                )
                ttl_stats["omega_loss"] = omega_loss.item()
                ttl_stats["layer_idx"] = self.layer_idx

        # === SWA Branch: Standard scaled dot-product attention ===
        scale = self.head_dim**-0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
        else:
            attn_weights = attn_weights.masked_fill(~attention_mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        # Reshape back: (batch, n_heads, seq_len, head_dim) -> (batch, seq_len, dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        attn_out = self.w_o(attn_out)

        # === Memory Branch: Direct MLP forward pass ===
        # Feed hidden states h directly through AtlasMemoryPoly.
        # The memory MLP is trained via TTL to predict useful associations.
        mem_out: Optional[Tensor] = None
        # Memory dropout: randomly skip memory to force attention to develop independently
        memory_dropped = (
            self.training
            and self.memory_dropout_rate > 0
            and torch.rand(1).item() < self.memory_dropout_rate
        )
        if not self.disable_memory and not memory_dropped:
            mem_out = self.memory(h, return_contribution=True)

            # Apply learned gate: controls memory vs attention balance
            gate = torch.sigmoid(self.memory_gate)
            mem_out = gate * mem_out

        # === Output-level combination ===
        if mem_out is not None:
            x = x + attn_out + mem_out
        else:
            x = x + attn_out

        # FFN with pre-norm and residual
        x = x + self.ffn(self.norm2(x))

        return x, ttl_stats

    def get_gate_value(self) -> float:
        """Get current gate value for monitoring."""
        return torch.sigmoid(self.memory_gate).item()

    def get_gate_param(self) -> Tensor:
        """Get raw gate parameter tensor (preserves gradients)."""
        return self.memory_gate


class AtlasMAGSkeleton(nn.Module):
    """
    Phase 0 Skeleton Model.

    A minimal but complete model for validating:
    - Persistent memory computation
    - W_init steady-state calibration
    - Forward pass execution

    Args:
        vocab_size: Vocabulary size for embeddings
        dim: Model dimension
        n_layers: Number of transformer blocks
        n_heads: Number of attention heads
        memory_expansion: Memory hidden dimension multiplier
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        dim: int = D,
        n_layers: int = 6,
        n_heads: int = N_HEADS,
        memory_expansion: int = MEMORY_EXPANSION,
        disable_memory: bool = False,
        poly_degree: int = POLY_DEGREE,
        poly_rank: int = 0,
        memory_dropout_rate: float = 0.0,
        # TTL (Test-Time Learning) configuration
        ttl_enabled: bool = True,
        ttl_theta: float = 0.9,
        ttl_alpha: float = 0.999,
        ttl_eta: float = 0.01,
        ttl_ns_iters: int = 5,
        ttl_adaptive_eta: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.disable_memory = disable_memory
        self.ttl_enabled = ttl_enabled

        # Token embeddings
        self.tok_emb = nn.Embedding(vocab_size, dim)

        # Transformer blocks with output-level memory combination
        # Each block gets layer index for per-layer gate initialization
        self.blocks = nn.ModuleList(
            [
                AtlasMAGBlock(
                    dim=dim,
                    n_heads=n_heads,
                    memory_expansion=memory_expansion,
                    disable_memory=disable_memory,
                    layer_idx=i,
                    n_layers=n_layers,
                    poly_degree=poly_degree,
                    poly_rank=poly_rank,
                    memory_dropout_rate=memory_dropout_rate,
                    # TTL configuration
                    ttl_enabled=ttl_enabled,
                    ttl_theta=ttl_theta,
                    ttl_alpha=ttl_alpha,
                    ttl_eta=ttl_eta,
                    ttl_ns_iters=ttl_ns_iters,
                    ttl_adaptive_eta=ttl_adaptive_eta,
                )
                for i in range(n_layers)
            ]
        )

        # Final normalization and output projection
        self.norm_f = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Tie embeddings and output projection
        self.lm_head.weight = self.tok_emb.weight

        # W_init: learned initialization for memory state
        # This will be calibrated in Phase 0
        self.w_init = nn.Parameter(torch.zeros(dim))

        self._init_weights()
        self._audit_output_scales()

        logger.info(
            f"AtlasMAGSkeleton initialized: "
            f"vocab_size={vocab_size}, dim={dim}, n_layers={n_layers}, "
            f"n_heads={n_heads}, poly_degree={poly_degree}"
        )

    def _audit_output_scales(self) -> None:
        """
        Audit memory vs attention output magnitudes at initialization.

        If memory output has significantly different magnitude than attention,
        the effective gate contribution will be skewed regardless of gate value.
        Logs a warning if the ratio exceeds 2x in either direction.
        """
        if self.disable_memory:
            return

        with torch.no_grad():
            # Create random input to measure output scales
            test_input = torch.randn(1, 32, self.dim)
            block: AtlasMAGBlock = self.blocks[0]  # type: ignore[assignment]

            h = block.norm1(test_input)

            # Attention output scale (through w_o)
            _q, _k, v = block.qkv(h, reshape_heads=True)
            attn_rms = block.w_o(v.transpose(1, 2).contiguous().view(1, 32, self.dim)).pow(2).mean().sqrt()

            # Memory output scale
            mem_out = block.memory(h, return_contribution=True)
            mem_rms = mem_out.pow(2).mean().sqrt()

            ratio = mem_rms / attn_rms if attn_rms > 0 else float("inf")

            logger.info(
                f"Output scale audit: attn_rms={attn_rms:.4f}, mem_rms={mem_rms:.4f}, "
                f"ratio(mem/attn)={ratio:.2f}"
            )
            if ratio > 2.0 or ratio < 0.5:
                logger.warning(
                    f"Memory/attention output scale mismatch: ratio={ratio:.2f}. "
                    f"Effective gate contribution will be skewed."
                )

    def _init_weights(self):
        """Initialize model weights."""
        # Embedding initialization
        nn.init.normal_(self.tok_emb.weight, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_ttl_stats: bool = False,
    ) -> Tensor | Tuple[Tensor, list[dict]]:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Optional attention mask
            return_ttl_stats: If True, also return TTL statistics from each layer

        Returns:
            If return_ttl_stats=False: Logits of shape (batch, seq_len, vocab_size)
            If return_ttl_stats=True: Tuple of (logits, list of TTL stats per layer)
        """
        # Token embeddings
        x = self.tok_emb(input_ids)

        # Add W_init to initial hidden state (broadcast across positions)
        x = x + self.w_init

        # Forward through blocks, collecting TTL stats
        ttl_stats_list: list[dict] = []
        for block in self.blocks:
            x, ttl_stats = block(x, attention_mask)
            if ttl_stats is not None:
                ttl_stats_list.append(ttl_stats)

        # Final norm and output
        x = self.norm_f(x)
        logits: Tensor = self.lm_head(x)

        if return_ttl_stats:
            return logits, ttl_stats_list
        return logits

    def forward_memory_only(
        self,
        input_ids: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass that also returns memory state.

        Used for W_init calibration - we need to observe the
        memory module weights after processing calibration data.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)

        Returns:
            Tuple of:
                - logits: Output logits
                - memory_state: Flattened memory module weights
        """
        result = self.forward(input_ids, return_ttl_stats=False)
        # Handle both return types (with and without TTL stats)
        logits = result if isinstance(result, Tensor) else result[0]

        # Collect memory module states from all blocks
        memory_states: list[Tensor] = []
        for block in self.blocks:
            # Cast to AtlasMAGBlock to access memory attribute
            mag_block: AtlasMAGBlock = block  # type: ignore[assignment]
            mem = mag_block.memory
            memory_states.extend(
                [
                    mem.w1.weight.flatten(),
                    mem.w2.weight.flatten(),
                    mem.w3.weight.flatten(),
                ]
            )
            # Include projection layers for AtlasMemoryPoly
            # Note: poly_norm is intentionally excluded - it's normalization
            # infrastructure, not learned memory content (like a sound engineer's
            # mixer settings vs. the actual music being recorded)
            if hasattr(mem, "proj_down"):
                memory_states.append(mem.proj_down.weight.flatten())
            if hasattr(mem, "proj_up"):
                memory_states.append(mem.proj_up.weight.flatten())
            if hasattr(mem, "poly_compress") and mem.poly_compress is not None:
                memory_states.append(mem.poly_compress.weight.flatten())

        memory_state = torch.cat(memory_states)

        return logits, memory_state

    def get_gate_values(self) -> list[float]:
        """Get gate values from all blocks for monitoring."""
        gate_values: list[float] = []
        for block in self.blocks:
            mag_block: AtlasMAGBlock = block  # type: ignore[assignment]
            gate_values.append(mag_block.get_gate_value())
        return gate_values

    def get_gate_params(self) -> list[Tensor]:
        """
        Get raw gate parameter tensors from all blocks.

        Unlike get_gate_values() which returns detached floats, this
        preserves gradient connections for loss backpropagation.

        Returns:
            List of gate parameter tensors (one per layer)
        """
        gate_params: list[Tensor] = []
        for block in self.blocks:
            mag_block: AtlasMAGBlock = block  # type: ignore[assignment]
            gate_params.append(mag_block.get_gate_param())
        return gate_params

    def get_gate_stats(self) -> dict:
        """
        Get comprehensive gate statistics for monitoring.

        Returns dict with:
            - memory_gates: Per-layer memory gate values (Q vs Q' blend)
            - memory_gate_std: Standard deviation across layers
            - memory_gate_range: [min, max] values
        """
        gate_values = self.get_gate_values()
        gate_tensor = torch.tensor(gate_values)

        return {
            "memory_gates": gate_values,
            "memory_gate_mean": float(gate_tensor.mean().item()),
            "memory_gate_std": float(gate_tensor.std().item()),
            "memory_gate_min": float(gate_tensor.min().item()),
            "memory_gate_max": float(gate_tensor.max().item()),
        }

    def reset_ttl_momentum(self) -> None:
        """
        Reset TTL momentum buffers in all layers.

        Call this at sequence/batch boundaries depending on TTL_RESET_MODE.
        """
        for block in self.blocks:
            mag_block: AtlasMAGBlock = block  # type: ignore[assignment]
            if hasattr(mag_block.memory, "reset_momentum"):
                mag_block.memory.reset_momentum()

    def set_ttl_enabled(self, enabled: bool) -> None:
        """
        Enable or disable TTL for all layers.

        Useful for ablation studies or switching behavior between
        training and inference.

        Args:
            enabled: Whether TTL should be enabled
        """
        self.ttl_enabled = enabled
        for block in self.blocks:
            mag_block: AtlasMAGBlock = block  # type: ignore[assignment]
            mag_block.ttl_enabled = enabled and not mag_block.disable_memory

    def get_ttl_stats_summary(self, ttl_stats_list: list[dict]) -> dict:
        """
        Aggregate TTL statistics across all layers.

        Args:
            ttl_stats_list: List of TTL stats dicts from forward pass

        Returns:
            Aggregated statistics:
            - omega_loss_mean: Mean Omega loss across layers
            - omega_loss_max: Maximum Omega loss
            - grad_norm_mean: Mean gradient norm across all params
            - update_norm_mean: Mean update norm after Newton-Schulz
        """
        if not ttl_stats_list:
            return {}

        omega_losses = [s["omega_loss"] for s in ttl_stats_list if "omega_loss" in s]

        # Collect all grad norms and update norms
        grad_norms: list[float] = []
        update_norms: list[float] = []
        for stats in ttl_stats_list:
            for key, value in stats.items():
                if key.endswith("_grad_norm"):
                    grad_norms.append(value)
                elif key.endswith("_update_norm"):
                    update_norms.append(value)

        return {
            "omega_loss_mean": sum(omega_losses) / len(omega_losses) if omega_losses else 0.0,
            "omega_loss_max": max(omega_losses) if omega_losses else 0.0,
            "grad_norm_mean": sum(grad_norms) / len(grad_norms) if grad_norms else 0.0,
            "update_norm_mean": sum(update_norms) / len(update_norms) if update_norms else 0.0,
            "n_layers_with_ttl": len(ttl_stats_list),
        }

    def count_parameters(self) -> dict:
        """Count parameters by component."""
        embedding = sum(p.numel() for p in self.tok_emb.parameters())
        blocks = sum(sum(p.numel() for p in block.parameters()) for block in self.blocks)
        head = 0  # Tied with embeddings

        total = sum(p.numel() for p in self.parameters())

        return {
            "embedding": embedding,
            "blocks": blocks,
            "lm_head": head,
            "total": total,
            "total_millions": total / 1e6,
        }

    def extra_repr(self) -> str:
        """Return a string with extra module information for repr()."""
        params = self.count_parameters()
        return (
            f"vocab_size={self.vocab_size}, dim={self.dim}, "
            f"n_layers={self.n_layers}, params={params['total_millions']:.1f}M"
        )
