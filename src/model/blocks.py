"""
MAG Block Implementation (Memory-Augmented Generation).

Paper Reference:
    Atlas: Learning to Optimally Memorize the Context at Test Time
    arXiv:2505.23735, Section 4 "DeepTransformers" and Section 5.1

    Also references:
    Titans: Learning to Memorize at Test Time
    arXiv:2501.00663 (Behrouz, Zhong et al. 2024)

Architecture Overview:
    Atlas combines "Memory-Augmented Generation" (MAG) from Titans with the
    Omega Rule memory update. The key insight from Section 4:

    "Not only attention is a non-parametric solution (contrary to the parametric
    nature of recurrent models), it globally optimizes its internal objective,
    while most recent modern recurrent models are online learners."

    MAG addresses this by running attention and memory in parallel, using the
    memory output to gate/modulate the attention output.

MAG (Memory as Gate):
    Two parallel branches combined via element-wise gating:
    - Branch 1: Sliding Window Attention → attn_out
    - Branch 2: Deep Memory (with Omega Rule) → mem_out
    - Output: x + attn_out * sigmoid(mem_out)

    The memory gates the attention, allowing the model to selectively suppress
    or enhance attention based on memory retrieval. This provides the benefits
    of both local attention (efficient, parallelizable) and long-range memory
    (associative retrieval beyond the attention window).

Components:
    - SlidingWindowAttention: Local context via masked attention (Section 4.1)
    - AtlasMemoryPoly: Deep polynomial memory (Section 3.1)
    - CausalQKMemoryProjection: Omega Rule Q-K projection (Section 3.2)
    - GammaGate: Input-dependent decay gates for context pruning

Test-Time Learning:
    MAGBlock supports TTL (inner loop optimization) where memory parameters
    are updated based on Omega loss during forward pass.
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
    OMEGA_CONTEXT_WINDOW,
    OMEGA_DECAY_BASE,
    POLY_DEGREE,
    WINDOW_SIZE,
)
from src.model.atlas_memory import AtlasMemoryPoly
from src.model.projections import QKVProjection, RotaryEmbedding
from src.model.qk_projection import CausalQKMemoryProjection
from src.nn.rmsnorm import RMSNorm
from src.nn.swiglu import SwiGLU
from src.training.omega_loss import compute_omega_loss
from src.training.ttl_update import ttl_step

logger = logging.getLogger(__name__)


def create_sliding_window_mask(seq_len: int, window_size: int, device: torch.device) -> Tensor:
    """Create a sliding window causal attention mask.

    Returns a boolean mask where True = masked (don't attend).
    """
    positions = torch.arange(seq_len, device=device)
    row_pos = positions.unsqueeze(1)  # (seq_len, 1)
    col_pos = positions.unsqueeze(0)  # (1, seq_len)

    # Mask if: col > row (future) OR col < row - window_size + 1 (too far back)
    mask = (col_pos > row_pos) | (col_pos < row_pos - window_size + 1)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


class GammaGate(nn.Module):
    """
    Input-dependent gamma gate for Omega Rule context pruning.

    Paper Reference:
        Atlas paper Section 3.2: "In our design, we use input-dependent parameters
        for γ_i^(t), providing in-context pruning ability."

    The γ (gamma) parameters act as "hard (direct) gates for the past tokens":
        - γ_i → 0: prunes token i from the local context optimization
        - γ_i → 1: fully incorporates token i in memory optimization

    This allows the model to learn which past tokens are relevant for the
    current prediction, enabling adaptive context pruning within the sliding
    window. The paper notes this is efficient because "for each token we need
    a constant number of gates; i.e., {γ_i^(t)}_{i=1}^c" where c is the
    context window size.
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
        nn.init.normal_(self.gate[0].weight, std=0.02)
        nn.init.zeros_(self.gate[2].weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.gate(x)


class MAGBlock(nn.Module):
    """
    Memory-as-Gate Block (Titans/Atlas architecture).

    Two parallel branches combined via element-wise gating:
    - Branch 1: Sliding Window Attention → attn_out
    - Branch 2: Memory (with Omega Rule Q-K projection) → mem_out
    - Output: x + attn_out * sigmoid(mem_out)

    The memory output gates/modulates the attention output, allowing the model
    to selectively suppress or enhance attention based on memory retrieval.

    Reference: Titans paper (arXiv:2501.00663), Atlas paper Section 5.1
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        memory_expansion: int = MEMORY_EXPANSION,
        poly_degree: int = POLY_DEGREE,
        poly_rank: int = 0,
        attn_window_size: int = WINDOW_SIZE,
        # TTL configuration
        ttl_enabled: bool = True,
        ttl_theta: float = 0.9,
        ttl_alpha: float = 0.999,
        ttl_eta: float = 0.01,
        ttl_ns_iters: int = 5,
        ttl_adaptive_eta: bool = False,
        # Eval mode: disable memory for ablation comparisons
        disable_memory: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.attn_window_size = attn_window_size
        self.disable_memory = disable_memory

        # TTL configuration
        self.ttl_enabled = ttl_enabled
        self.ttl_theta = ttl_theta
        self.ttl_alpha = ttl_alpha
        self.ttl_eta = ttl_eta
        self.ttl_ns_iters = ttl_ns_iters
        self.ttl_adaptive_eta = ttl_adaptive_eta

        # Normalization layers
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        # QKV projection with K/Q normalization (paper: "normalization on keys and queries")
        self.qkv = QKVProjection(dim, n_heads, qk_norm=True)

        # Rotary embeddings
        self.rope = RotaryEmbedding(self.head_dim)

        # Gamma gate for Omega Rule decay
        self.gamma_gate = GammaGate(dim)

        # Q-K Memory Projection (Omega Rule)
        self.qk_memory = CausalQKMemoryProjection(
            dim=dim,
            n_heads=n_heads,
            context_window=OMEGA_CONTEXT_WINDOW,
            decay_base=OMEGA_DECAY_BASE,
        )

        # Memory MLP with polynomial features
        self.memory = AtlasMemoryPoly(
            dim=dim,
            key_dim=self.head_dim,
            expansion=memory_expansion,
            poly_degree=poly_degree,
            poly_rank=poly_rank,
        )

        # Output projection for attention
        self.w_o = nn.Linear(dim, dim, bias=False)

        # FFN (SwiGLU)
        self.ffn = SwiGLU(dim, int(dim * memory_expansion * 2 / 3))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.w_o.weight, std=0.02 / (2**0.5))

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[dict]]:
        """
        MAG forward pass: parallel attention and memory with gating combination.

        Architecture:
            x → norm → QKV
                ├── [SWA] → attn_out ────────────┐
                └── [QK-Proj → Memory] → mem_out ─┼→ attn_out * sigmoid(mem_out)
                                                  └→ x + gated_output + FFN
        """
        batch, seq_len, dim = x.shape
        ttl_stats: Optional[dict] = None

        # Pre-norm
        h = self.norm1(x)

        # QKV projection
        q, k, v = self.qkv(h, reshape_heads=True)
        q, k = self.rope(q, k)

        # === Memory Processing (skipped if disable_memory=True) ===
        q_mem_flat = None
        if not self.disable_memory:
            # Q-K Memory Projection (Omega Rule)
            gamma = self.gamma_gate(h)
            q_mem = self.qk_memory(q, k, gamma)
            q_mem_flat = q_mem.transpose(1, 2).contiguous().view(batch, seq_len, dim)

            # TTL Update
            if self.ttl_enabled and self.training:
                v_flat = v.transpose(1, 2).contiguous().view(batch, seq_len, dim)
                gamma_ttl = self.gamma_gate(h)

                with torch.enable_grad():
                    omega_loss = compute_omega_loss(
                        memory=self.memory,
                        keys=q_mem_flat,
                        values=v_flat,
                        gamma=gamma_ttl,
                        context_window=OMEGA_CONTEXT_WINDOW,
                        decay_base=OMEGA_DECAY_BASE,
                    )
                    ttl_stats = ttl_step(
                        memory=self.memory,
                        loss=omega_loss,
                        theta=self.ttl_theta,
                        alpha=self.ttl_alpha,
                        eta=self.ttl_eta,
                        ns_iterations=self.ttl_ns_iters,
                        adaptive_eta=self.ttl_adaptive_eta,
                    )
                    ttl_stats["omega_loss"] = omega_loss.item()

        # === Attention Branch (SWA) ===
        scale = self.head_dim**-0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        swa_mask = create_sliding_window_mask(seq_len, self.attn_window_size, x.device)
        if attention_mask is not None:
            combined_mask = swa_mask | ~attention_mask
        else:
            combined_mask = swa_mask
        attn_weights = attn_weights.masked_fill(combined_mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        attn_out = self.w_o(attn_out)

        # === Memory Branch (skipped if disable_memory=True) ===
        if self.disable_memory or q_mem_flat is None:
            # Ablation mode: use attention output directly, no memory gating
            x = x + attn_out
        else:
            mem_out = self.memory(q_mem_flat, return_contribution=True)

            # === MAG Combination: attention gated by memory ===
            # Paper: o = y ⊙ M(x̃) - memory output gates attention
            # We use sigmoid to bound the gating to [0, 1]
            gated_output = attn_out * torch.sigmoid(mem_out)

            # Residual connection
            x = x + gated_output

        # FFN
        x = x + self.ffn(self.norm2(x))

        return x, ttl_stats


class AttentionOnlyBlock(nn.Module):
    """
    Attention-only block (no memory) for ablation studies.

    This is a standard transformer block with sliding window attention,
    used when disable_memory=True.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_expansion: int = MEMORY_EXPANSION,
        attn_window_size: int = WINDOW_SIZE,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.attn_window_size = attn_window_size

        # Normalization layers
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        # QKV projection
        self.qkv = QKVProjection(dim, n_heads, qk_norm=True)

        # Rotary embeddings
        self.rope = RotaryEmbedding(self.head_dim)

        # Output projection
        self.w_o = nn.Linear(dim, dim, bias=False)

        # FFN
        self.ffn = SwiGLU(dim, int(dim * ffn_expansion * 2 / 3))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.w_o.weight, std=0.02 / (2**0.5))

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, None]:
        """Standard transformer block forward pass."""
        batch, seq_len, dim = x.shape

        h = self.norm1(x)
        q, k, v = self.qkv(h, reshape_heads=True)
        q, k = self.rope(q, k)

        scale = self.head_dim**-0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        swa_mask = create_sliding_window_mask(seq_len, self.attn_window_size, x.device)
        if attention_mask is not None:
            combined_mask = swa_mask | ~attention_mask
        else:
            combined_mask = swa_mask
        attn_weights = attn_weights.masked_fill(combined_mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        attn_out = self.w_o(attn_out)

        x = x + attn_out
        x = x + self.ffn(self.norm2(x))

        return x, None
