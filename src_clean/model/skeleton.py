"""
Atlas-MAG Skeleton Model with Polynomial Memory.

This implements the Atlas MAG architecture from arXiv:2505.23735.
MAG (Memory-as-Gate) is the best performing variant according to paper ablations.

Key features from Atlas paper:
- MAG architecture: Parallel attention+memory branches with gated combination
- Polynomial features (phi_2): Essential for memory capacity O(d_k^2)
- Input-dependent gamma gates: Per-position decay modulation
- Sliding window attention: Memory captures beyond-window context

Memory capacity (Section 3.1, Propositions 1 & 2):
- Without polynomial features: O(d_k) ~ 64 associations
- With phi_2 polynomial features: O(d_k^2) ~ 4,096 associations

Reference: Atlas paper (arXiv:2505.23735)
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src_clean.config import (
    MEMORY_EXPANSION,
    N_HEADS,
    POLY_DEGREE,
    VOCAB_SIZE,
    D,
)

from src_clean.model.blocks import MAGBlock, AttentionOnlyBlock
from src_clean.nn.rmsnorm import RMSNorm

logger = logging.getLogger(__name__)

# Re-export for external access
__all__ = ["AtlasMAGSkeleton", "MAGBlock"]


class AtlasMAGSkeleton(nn.Module):
    """
    Atlas Model with MAG Architecture (Paper-Faithful Implementation).

    Uses pure MAG (Memory-as-Gate) blocks which are the best performing
    according to ablations in both Atlas and Titans papers.

    Args:
        vocab_size: Vocabulary size for embeddings
        dim: Model dimension
        n_layers: Number of transformer blocks
        n_heads: Number of attention heads
        memory_expansion: Memory hidden dimension multiplier
        disable_memory: If True, use attention-only blocks (ablation)
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
        # TTL (Test-Time Learning) configuration
        ttl_enabled: bool = True,
        ttl_theta: float = 0.9,
        ttl_alpha: float = 0.999,
        ttl_eta: float = 0.01,
        ttl_ns_iters: int = 5,
        ttl_adaptive_eta: bool = False,
        # Legacy parameter - ignored, kept for checkpoint compatibility
        layer_pattern: str = "mag",
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

        # Build MAG blocks (or AttentionOnly for ablation)
        self.blocks = nn.ModuleList()
        for layer_idx in range(n_layers):
            if disable_memory:
                self.blocks.append(
                    AttentionOnlyBlock(
                        dim=dim,
                        n_heads=n_heads,
                        ffn_expansion=memory_expansion,
                    )
                )
            else:
                self.blocks.append(
                    MAGBlock(
                        dim=dim,
                        n_heads=n_heads,
                        memory_expansion=memory_expansion,
                        poly_degree=poly_degree,
                        poly_rank=poly_rank,
                        ttl_enabled=ttl_enabled,
                        ttl_theta=ttl_theta,
                        ttl_alpha=ttl_alpha,
                        ttl_eta=ttl_eta,
                        ttl_ns_iters=ttl_ns_iters,
                        ttl_adaptive_eta=ttl_adaptive_eta,
                    )
                )

        # Final normalization and output projection
        self.norm_f = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Tie embeddings and output projection
        self.lm_head.weight = self.tok_emb.weight

        # W_init: learned initialization for memory state
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
        """
        if self.disable_memory:
            return

        block = self.blocks[0]
        if not hasattr(block, "memory"):
            return

        with torch.no_grad():
            test_input = torch.randn(1, 32, self.dim)
            h = block.norm1(test_input)

            # Attention output scale
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
        # Token embeddings + W_init
        x = self.tok_emb(input_ids) + self.w_init

        # Forward through blocks
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

    def forward_memory_only(self, input_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass that also returns memory state for W_init calibration.
        """
        result = self.forward(input_ids, return_ttl_stats=False)
        logits = result if isinstance(result, Tensor) else result[0]

        # Collect memory module states
        memory_states: list[Tensor] = []
        for block in self.blocks:
            if not hasattr(block, "memory"):
                continue
            mem = block.memory
            memory_states.extend([
                mem.w1.weight.flatten(),
                mem.w2.weight.flatten(),
                mem.w3.weight.flatten(),
            ])
            if hasattr(mem, "proj_down"):
                memory_states.append(mem.proj_down.weight.flatten())
            if hasattr(mem, "proj_up"):
                memory_states.append(mem.proj_up.weight.flatten())
            if hasattr(mem, "poly_compress") and mem.poly_compress is not None:
                memory_states.append(mem.poly_compress.weight.flatten())

        memory_state = torch.cat(memory_states) if memory_states else torch.zeros(1)
        return logits, memory_state

    def get_gate_values(self) -> list[float]:
        """
        Get gate values from all MAG blocks for monitoring.

        MAG blocks use sigmoid(memory_gate) for gating.
        """
        gate_values: list[float] = []
        for block in self.blocks:
            if hasattr(block, "memory_gate"):
                gate_values.append(torch.sigmoid(block.memory_gate).item())
        return gate_values

    def reset_ttl_momentum(self) -> None:
        """Reset TTL momentum buffers in all layers."""
        for block in self.blocks:
            if hasattr(block, "memory") and hasattr(block.memory, "reset_momentum"):
                block.memory.reset_momentum()

    def set_ttl_enabled(self, enabled: bool) -> None:
        """Enable or disable TTL for all layers."""
        self.ttl_enabled = enabled
        for block in self.blocks:
            if hasattr(block, "ttl_enabled"):
                block.ttl_enabled = enabled

    def count_parameters(self) -> dict:
        """Count parameters by component."""
        embedding = sum(p.numel() for p in self.tok_emb.parameters())
        blocks = sum(sum(p.numel() for p in block.parameters()) for block in self.blocks)
        total = sum(p.numel() for p in self.parameters())

        return {
            "embedding": embedding,
            "blocks": blocks,
            "lm_head": 0,  # Tied with embeddings
            "total": total,
            "total_millions": total / 1e6,
        }
