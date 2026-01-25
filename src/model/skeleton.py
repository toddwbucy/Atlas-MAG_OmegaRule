"""
Atlas-MAG Skeleton Model with Omega Rule.

This is the minimal testable model that can validate:
- Persistent memory computation (M_persistent, norm_persistent)
- W_init calibration via steady-state
- Forward pass works end-to-end

Key features from Atlas paper (arXiv:2505.23735):
- Omega Rule: Sliding context window with exponential decay
- Input-dependent γ gates: Per-position decay modulation
- Per-layer gate initialization: Spread across layers for differentiation

Note: TNT (Titans-in-Titans) hierarchical memory is NOT implemented.
This model uses single-layer memory with the Omega Rule context window.

Reference: Atlas paper (arXiv:2505.23735)
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import D, N_HEADS, N_PERSISTENT, MEMORY_EXPANSION, VOCAB_SIZE, GAMMA_GATE_HIDDEN_DIM
from src.model.atlas_memory import AtlasMemory  # Keep for Option B fallback
from src.model.persistent_memory import PersistentMemory
from src.model.qk_projection import CausalQKMemoryProjection
from src.model.projections import QKVProjection, RotaryEmbedding
from src.nn.rmsnorm import RMSNorm
from src.nn.swiglu import SwiGLU

logger = logging.getLogger(__name__)


class GammaGate(nn.Module):
    """
    Input-dependent γ gate for Omega Rule context pruning.

    Produces per-position decay multipliers that modulate the base
    exponential decay in the memory context window. This allows the
    model to selectively forget irrelevant past context based on content.

    Architecture: x -> Linear -> SiLU -> Linear -> Sigmoid -> γ

    Reference: Atlas paper (arXiv:2505.23735) Eq. 9
    """

    def __init__(self, dim: int, hidden_dim: int = GAMMA_GATE_HIDDEN_DIM):
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
        # Second linear: near-zero so sigmoid(0) ≈ 0.5
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

    Architecture:
        x -> norm1 -> [attention + memory_gate * memory] -> + x
        x -> norm2 -> ffn -> + x

    Key features from Atlas paper:
    - Input-dependent γ gates for context pruning
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
            # Input-dependent γ gate for context pruning
            self.gamma_gate = GammaGate(dim)

        # Keep AtlasMemory as Option B fallback (not used in forward for now)
        self.memory = AtlasMemory(dim, memory_expansion)

        # Memory gate: controls Q vs Q' blending
        # PER-LAYER INITIALIZATION: Spread gates across layers
        # Early layers (idx=0): more attention-focused (gate ≈ 0.1)
        # Later layers (idx=n-1): more memory-focused (gate ≈ 0.3)
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
        Forward pass through the block with Omega Rule memory.

        The memory mechanism works by projecting queries through accumulated
        key outer products with context window and decay (Omega Rule):
            M_t = M_persistent + Σ(i=t-c+1 to t) γ_i * (k_i ⊗ k_i)
            Q' = M_t @ Q / norm_sum

        Input-dependent γ gates modulate the decay weights, allowing the
        model to selectively forget irrelevant past context.

        A learned gate blends Q and Q':
            Q_final = (1-gate)*Q + gate*Q'

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

        # Memory: Project Q through accumulated key memory with Omega Rule
        if not self.disable_memory and self.qk_memory is not None:
            # Compute input-dependent γ gates for context pruning
            gamma_gates: Optional[Tensor] = None
            if self.gamma_gate is not None:
                gamma_gates = self.gamma_gate(h)  # (batch, seq_len, 1)

            # Q' = M_t @ Q / norm_sum (memory-enhanced queries)
            # With Omega Rule: context window + decay + γ gates
            q_projected = self.qk_memory(q, k, gamma_gates=gamma_gates)

            # Gate blends original Q with memory-projected Q'
            # gate ≈ 0: pure attention (Q), gate ≈ 1: pure memory (Q')
            gate = torch.sigmoid(self.memory_gate)
            q = (1 - gate) * q + gate * q_projected

        # Scaled dot-product attention (with potentially memory-enhanced Q)
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

        # Output projection and residual
        x = x + self.w_o(attn_out)

        # FFN with pre-norm and residual
        x = x + self.ffn(self.norm2(x))

        return x

    def get_gate_value(self) -> float:
        """Get current gate value for monitoring."""
        return torch.sigmoid(self.memory_gate).item()


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
            memory_states.extend([
                mag_block.memory.w1.weight.flatten(),
                mag_block.memory.w2.weight.flatten(),
                mag_block.memory.w3.weight.flatten(),
            ])

        memory_state = torch.cat(memory_states)

        return logits, memory_state

    def get_gate_values(self) -> list[float]:
        """Get gate values from all blocks for monitoring."""
        gate_values: list[float] = []
        for block in self.blocks:
            mag_block: AtlasMAGBlock = block  # type: ignore[assignment]
            gate_values.append(mag_block.get_gate_value())
        return gate_values

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
        params = self.count_parameters()
        return (
            f"vocab_size={self.vocab_size}, dim={self.dim}, "
            f"n_layers={self.n_layers}, params={params['total_millions']:.1f}M"
        )
