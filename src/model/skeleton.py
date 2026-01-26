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
    N_PERSISTENT,
    POLY_DEGREE,
    USE_POLY_MEMORY,
    VOCAB_SIZE,
    D,
)
from src.model.atlas_memory import AtlasMemory, AtlasMemoryPoly
from src.model.persistent_memory import PersistentMemory
from src.model.projections import QKVProjection, RotaryEmbedding
from src.model.qk_projection import CausalQKMemoryProjection
from src.nn.rmsnorm import RMSNorm
from src.nn.swiglu import SwiGLU

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
        persistent_memory: Optional[PersistentMemory] = None,
        layer_idx: int = 0,
        n_layers: int = 12,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.disable_memory = disable_memory
        self.layer_idx = layer_idx

        # Normalization layers
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        # Q, K, V projections
        self.qkv = QKVProjection(dim, n_heads, qk_norm=True)

        # Rotary embeddings
        self.rope = RotaryEmbedding(self.head_dim)

        # Q-K memory projection with Omega Rule
        # Projects queries through accumulated key outer products
        self.qk_memory: Optional[CausalQKMemoryProjection] = None
        self.gamma_gate: Optional[GammaGate] = None
        if not disable_memory:
            self.qk_memory = CausalQKMemoryProjection(
                dim=dim,
                n_heads=n_heads,
                persistent_memory=persistent_memory,
            )
            # Input-dependent gamma gate for context pruning
            self.gamma_gate = GammaGate(dim)

        # Memory module: AtlasMemoryPoly (with polynomial features) or AtlasMemory
        # Polynomial features are ESSENTIAL for memory capacity (Atlas paper Section 3.1)
        # Without: O(d_k) capacity. With phi_2: O(d_k^2) capacity (64x improvement for d_k=64)
        self.use_poly_memory = USE_POLY_MEMORY
        if USE_POLY_MEMORY:
            # key_dim = head_dim for polynomial features (capacity theorem uses d_k)
            self.memory: nn.Module = AtlasMemoryPoly(
                dim=dim,
                key_dim=self.head_dim,  # Use head_dim for O(d_k^2) capacity
                expansion=memory_expansion,
                poly_degree=POLY_DEGREE,
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

        Spreads gate initialization across layers to encourage differentiation:
        - Early layers: more attention-focused (lower gate values)
        - Later layers: more memory-focused (higher gate values)

        The initialization maps to sigmoid values:
        - Layer 0: sigmoid(-3.0) ≈ 0.047 (5% memory)
        - Layer n-1: sigmoid(-1.0) ≈ 0.269 (27% memory)

        This encourages early layers to learn local patterns via attention
        while later layers can leverage long-range memory retrieval.

        Args:
            layer_idx: Current layer index (0-indexed)
            n_layers: Total number of layers

        Returns:
            Gate initialization value (pre-sigmoid)
        """
        # Linear interpolation from -3.0 (early) to -1.0 (late)
        # This gives sigmoid range [0.047, 0.269]
        gate_start = -3.0  # sigmoid ≈ 0.047
        gate_end = -1.0    # sigmoid ≈ 0.269

        if n_layers <= 1:
            return (gate_start + gate_end) / 2

        t = layer_idx / (n_layers - 1)  # 0 to 1
        return gate_start + t * (gate_end - gate_start)

    def _init_weights(self):
        """Initialize output projection."""
        nn.init.normal_(self.w_o.weight, std=0.02 / (2 ** 0.5))

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through the block with output-level memory combination.

        Architecture (Atlas paper Figure 3 - MAG):
            Input → [SWA Branch] ──────┐
                  → [Memory Branch] ───┼──► Add → Output

        The SWA (Sliding Window Attention) and Memory branches run in parallel,
        and their outputs are combined at the output level (after both computations).
        This is different from Q-level blending which modifies Q before attention.

        Memory contribution is gated:
            output = attn_out + gate * mem_out

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            attention_mask: Optional causal mask

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        batch, seq_len, dim = x.shape

        # Pre-norm
        h = self.norm1(x)

        # Q, K, V projections (with reshape to heads)
        q, k, v = self.qkv(h, reshape_heads=True)

        # Apply rotary embeddings
        q, k = self.rope(q, k)

        # === SWA Branch: Standard scaled dot-product attention ===
        # Shape: (batch, n_heads, seq_len, seq_len)
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask
        if attention_mask is None:
            # Create causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
        else:
            attn_weights = attn_weights.masked_fill(~attention_mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Attention output
        attn_out = torch.matmul(attn_weights, v)

        # Reshape back: (batch, n_heads, seq_len, head_dim) -> (batch, seq_len, dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, dim)

        # Project attention output
        attn_out = self.w_o(attn_out)

        # === Memory Branch: Gated MLP memory (parallel to attention) ===
        mem_out: Optional[Tensor] = None
        if not self.disable_memory:
            # Get memory contribution (without residual - we handle combination here)
            mem_out = self.memory(h, return_contribution=True)

            # Apply learned gate: controls memory vs attention balance
            # gate ≈ 0: pure attention, gate ≈ 1: pure memory
            gate = torch.sigmoid(self.memory_gate)
            mem_out = gate * mem_out

        # === Output-level combination ===
        # Combine SWA and Memory branches with residual connection
        if mem_out is not None:
            x = x + attn_out + mem_out
        else:
            x = x + attn_out

        # FFN with pre-norm and residual
        x = x + self.ffn(self.norm2(x))

        return x

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
        n_persistent: Number of persistent memory tokens
        memory_expansion: Memory hidden dimension multiplier
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        dim: int = D,
        n_layers: int = 6,
        n_heads: int = N_HEADS,
        n_persistent: int = N_PERSISTENT,
        memory_expansion: int = MEMORY_EXPANSION,
        disable_memory: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.disable_memory = disable_memory

        # Token embeddings
        self.tok_emb = nn.Embedding(vocab_size, dim)

        # Persistent memory module (skip if memory disabled)
        self.persistent_memory: Optional[PersistentMemory] = None
        if not disable_memory:
            self.persistent_memory = PersistentMemory(dim, n_persistent)

        # Transformer blocks (pass persistent_memory for Q-K projection)
        # Each block gets its layer index for per-layer gate initialization
        self.blocks = nn.ModuleList([
            AtlasMAGBlock(
                dim=dim,
                n_heads=n_heads,
                memory_expansion=memory_expansion,
                disable_memory=disable_memory,
                persistent_memory=self.persistent_memory,
                layer_idx=i,
                n_layers=n_layers,
            )
            for i in range(n_layers)
        ])

        # Final normalization and output projection
        self.norm_f = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Tie embeddings and output projection
        self.lm_head.weight = self.tok_emb.weight

        # W_init: learned initialization for memory state
        # This will be calibrated in Phase 0
        self.w_init = nn.Parameter(torch.zeros(dim))

        self._init_weights()

        logger.info(
            f"AtlasMAGSkeleton initialized: "
            f"vocab_size={vocab_size}, dim={dim}, n_layers={n_layers}, "
            f"n_heads={n_heads}, n_persistent={n_persistent}"
        )

    def _init_weights(self):
        """Initialize model weights."""
        # Embedding initialization
        nn.init.normal_(self.tok_emb.weight, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Optional attention mask

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        # Token embeddings
        x = self.tok_emb(input_ids)

        # Add W_init to initial hidden state (broadcast across positions)
        x = x + self.w_init

        # Forward through blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        # Final norm and output
        x = self.norm_f(x)
        logits: Tensor = self.lm_head(x)

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
        logits = self.forward(input_ids)

        # Collect memory module states from all blocks
        memory_states: list[Tensor] = []
        for block in self.blocks:
            # Cast to AtlasMAGBlock to access memory attribute
            mag_block: AtlasMAGBlock = block  # type: ignore[assignment]
            mem = mag_block.memory
            memory_states.extend([
                mem.w1.weight.flatten(),
                mem.w2.weight.flatten(),
                mem.w3.weight.flatten(),
            ])
            # Include projection layers for AtlasMemoryPoly
            # Note: poly_norm is intentionally excluded - it's normalization
            # infrastructure, not learned memory content (like a sound engineer's
            # mixer settings vs. the actual music being recorded)
            if hasattr(mem, 'proj_down'):
                memory_states.append(mem.proj_down.weight.flatten())
            if hasattr(mem, 'proj_up'):
                memory_states.append(mem.proj_up.weight.flatten())

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

    def count_parameters(self) -> dict:
        """Count parameters by component."""
        embedding = sum(p.numel() for p in self.tok_emb.parameters())
        persistent = (
            sum(p.numel() for p in self.persistent_memory.parameters())
            if self.persistent_memory is not None
            else 0
        )
        blocks = sum(
            sum(p.numel() for p in block.parameters())
            for block in self.blocks
        )
        head = 0  # Tied with embeddings

        total = sum(p.numel() for p in self.parameters())

        return {
            "embedding": embedding,
            "persistent_memory": persistent,
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
