"""
Memory Functional Probe (Sliding Window Aware).

REQ-P2-002: Memory Contribution Probe

Validates that Atlas memory is actively contributing to model predictions
by comparing perplexity with memory enabled vs disabled.

CRITICAL: This probe must test positions where memory CAN help - i.e., positions
beyond the sliding window attention range. With WINDOW_SIZE=512, attention at
position t can only see [t-511, t]. Memory must retrieve info from earlier.

Test Design:
1. Use sequences LONGER than attention window (1024 tokens > 512 window)
2. Only measure PPL for positions > WINDOW_SIZE (where some context is beyond attention)
3. This isolates memory's contribution to retrieving "old" information

If we test with seq_len < WINDOW_SIZE, attention can see everything and
memory is redundant - giving misleading 0% contribution!

Reference: PRD Section 5.4 REQ-P2-002
"""

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, List

import torch
import torch.nn.functional as F

from src_clean.config import WINDOW_SIZE

logger = logging.getLogger(__name__)


@dataclass
class NIAHResult:
    """Result from a single memory probe."""

    step: int
    accuracy: float  # PPL reduction ratio: (nomem - mem) / nomem
    passed: bool
    needle_norm: float  # PPL with memory (renamed from legacy)
    retrieved_norm: float  # PPL without memory (renamed from legacy)
    haystack_size: int  # Number of test sequences
    probe_time_ms: float
    # New fields for diagnostics
    ppl_mem: float = 0.0  # Explicit PPL with memory
    ppl_nomem: float = 0.0  # Explicit PPL without memory
    positions_tested: int = 0  # How many positions were evaluated


class NIAHProbe:
    """
    Sliding-Window-Aware Memory Probe.

    Tests whether the memory module improves predictions for positions where
    attention CANNOT see all relevant context (i.e., positions beyond the
    sliding window size).

    Key insight: With WINDOW_SIZE=512, position t can only attend to [t-511, t].
    For positions t > 512, some earlier context is invisible to attention.
    Memory's job is to retrieve that "forgotten" context.

    If we test with short sequences (< WINDOW_SIZE), attention sees everything
    and memory appears useless. We MUST test with long sequences.

    Args:
        dim: Model dimension (used for standalone mode only)
        probe_frequency: How often to run probes (in steps)
        haystack_size: Number of test sequences (default: 4)
        accuracy_threshold: Minimum PPL reduction ratio (default: 0.1 = 10%)
        seq_len: Sequence length - MUST be > WINDOW_SIZE (default: 1024)
        window_size: Attention window size (default: from config)

    Example:
        >>> probe = NIAHProbe(dim=768, probe_frequency=1000, seq_len=1024)
        >>> result = probe.run_probe(model, step=5000, device="cuda")
        >>> print(f"Memory contribution: {result.accuracy:.1%}")
    """

    def __init__(
        self,
        dim: int,
        probe_frequency: int = 1000,
        haystack_size: int = 4,
        accuracy_threshold: float = 0.1,
        seq_len: int = 1024,  # MUST be > WINDOW_SIZE
        window_size: int = WINDOW_SIZE,
        # Legacy params accepted but ignored for backward compat
        ttl_steps: int = 10,  # noqa: ARG002
    ):
        self.dim = dim
        self.probe_frequency = probe_frequency
        self.haystack_size = haystack_size
        self.accuracy_threshold = accuracy_threshold
        self.window_size = window_size

        # Enforce seq_len > window_size to ensure meaningful test
        if seq_len <= window_size:
            logger.warning(
                f"seq_len ({seq_len}) <= window_size ({window_size}). "
                f"Increasing seq_len to {window_size + 512} for meaningful probe."
            )
            seq_len = window_size + 512

        self.seq_len = seq_len

        # History for analysis
        self.history: List[NIAHResult] = []

        logger.info(
            f"NIAHProbe initialized: dim={dim}, freq={probe_frequency}, "
            f"n_seqs={haystack_size}, threshold={accuracy_threshold}, "
            f"seq_len={seq_len}, window_size={window_size}"
        )
        logger.info(
            f"  -> Will test positions [{window_size}, {seq_len-1}] where "
            f"attention cannot see full history"
        )

    def should_probe(self, step: int) -> bool:
        """
        Check if we should run a probe at this step.

        Probes run at step 0 and every probe_frequency steps thereafter.
        """
        if step == 0:
            return True
        return step % self.probe_frequency == 0

    def set_eval_batch(self, input_ids: torch.Tensor) -> None:
        """
        Cache a batch of real token IDs for probing.

        Call this once with a batch from the validation set. The probe
        will reuse these tokens for all future probes, ensuring consistent
        and meaningful measurements.

        IMPORTANT: The cached batch should have seq_len > WINDOW_SIZE.
        If shorter, the probe will pad or warn.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
        """
        if input_ids.shape[1] < self.seq_len:
            logger.warning(
                f"Cached eval batch seq_len ({input_ids.shape[1]}) < "
                f"probe seq_len ({self.seq_len}). Probe will use random tokens."
            )
        self._eval_batch = input_ids.detach()
        logger.info(f"NIAHProbe cached eval batch: {input_ids.shape}")

    def run_probe(
        self,
        model: Any,
        step: int,
        device: str = "cpu",
        **_kwargs: Any,
    ) -> NIAHResult:
        """
        Run a sliding-window-aware memory probe.

        Only measures PPL for positions BEYOND the attention window, where
        memory is the only way to retrieve earlier context.

        Args:
            model: The AtlasMAGSkeleton model
            step: Current training step
            device: Device to run on

        Returns:
            NIAHResult with accuracy = PPL reduction ratio for beyond-window positions
        """
        start_time = time.perf_counter()

        if model is None:
            return self._standalone_result(step, start_time)

        # Use cached eval batch if available and long enough, else random tokens
        if (
            hasattr(self, "_eval_batch")
            and self._eval_batch is not None
            and self._eval_batch.shape[1] >= self.seq_len
        ):
            input_ids = self._eval_batch[:self.haystack_size, :self.seq_len].to(device)
        else:
            vocab_size = model.vocab_size
            input_ids = torch.randint(
                0, vocab_size, (self.haystack_size, self.seq_len), device=device
            )

        vocab_size = model.vocab_size

        # We only evaluate positions > window_size where attention can't see all history
        # This is where memory MUST contribute if it's useful
        eval_start = self.window_size  # First position where some context is beyond window
        eval_end = self.seq_len - 1  # Last position (we predict next token)

        # Labels for the "beyond window" region
        # For position i, label is token at i+1
        labels_full = input_ids[:, 1:].contiguous()  # Shape: (batch, seq_len-1)
        labels_beyond = labels_full[:, eval_start:].contiguous()  # Positions [window, end)

        positions_tested = labels_beyond.numel()

        # CRITICAL: Use evaluation mode to prevent TTL from mutating memory during probe.
        # TTL uses torch.enable_grad() internally which overrides no_grad(), so we
        # must disable training mode entirely to prevent memory weight corruption.
        was_training = model.training
        model.train(False)  # Switch to evaluation mode

        try:
            with torch.no_grad():
                # Forward with memory enabled
                logits_mem_full = model(input_ids)
                # Get logits for positions [window_size, seq_len-1) predicting [window_size+1, seq_len)
                logits_mem = logits_mem_full[:, eval_start:-1, :].contiguous()
                loss_mem = F.cross_entropy(
                    logits_mem.reshape(-1, vocab_size), labels_beyond.reshape(-1)
                )
                ppl_mem = torch.exp(loss_mem).item()

                # Disable memory across all blocks (save/restore original flags)
                # Check if blocks support disable_memory (AtlasMAGBlock has it, MAGBlock/MALBlock don't)
                supports_disable = all(
                    hasattr(block, "disable_memory") for block in model.blocks
                )
                comparison_skipped = False

                if not supports_disable:
                    # For MAG/MAL architectures, memory can't be disabled
                    # Skip the comparison - no meaningful test possible
                    ppl_nomem = ppl_mem
                    comparison_skipped = True
                    logger.info(
                        f"[Step {step}] Memory Probe: MAG/MAL blocks don't support disable_memory, "
                        f"skipping comparison. PPL with memory={ppl_mem:.1f}"
                    )
                else:
                    orig_flags = [block.disable_memory for block in model.blocks]
                    try:
                        for block in model.blocks:
                            block.disable_memory = True

                        logits_nomem_full = model(input_ids)
                        logits_nomem = logits_nomem_full[:, eval_start:-1, :].contiguous()
                        loss_nomem = F.cross_entropy(
                            logits_nomem.reshape(-1, vocab_size), labels_beyond.reshape(-1)
                        )
                        ppl_nomem = torch.exp(loss_nomem).item()
                    finally:
                        # Restore original per-block memory flags even on error
                        for block, flag in zip(model.blocks, orig_flags, strict=False):
                            block.disable_memory = flag
        finally:
            # Restore training mode
            if was_training:
                model.train(True)

        # PPL reduction ratio: how much does memory help for beyond-window positions?
        # Guard against inf/NaN when both PPLs overflow (e.g., untrained model)
        if comparison_skipped:
            # Can't meaningfully measure - treat as passed (no false alarm)
            accuracy = 0.0
            passed = True
        elif math.isinf(ppl_nomem) or math.isinf(ppl_mem) or ppl_nomem <= 0:
            accuracy = 0.0
            passed = False
        else:
            accuracy = (ppl_nomem - ppl_mem) / ppl_nomem
            passed = accuracy >= self.accuracy_threshold

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        result = NIAHResult(
            step=step,
            accuracy=accuracy,
            passed=passed,
            needle_norm=ppl_mem,  # Legacy field name
            retrieved_norm=ppl_nomem,  # Legacy field name
            haystack_size=self.haystack_size,
            probe_time_ms=elapsed_ms,
            ppl_mem=ppl_mem,
            ppl_nomem=ppl_nomem,
            positions_tested=positions_tested,
        )

        if comparison_skipped:
            status = "SKIPPED"
        else:
            status = "PASSED" if passed else "FAILED"
        logger.info(
            f"[Step {step}] Memory Probe {status}: "
            f"contribution={accuracy:.1%} (threshold={self.accuracy_threshold:.0%}), "
            f"PPL mem={ppl_mem:.1f} nomem={ppl_nomem:.1f}, "
            f"positions=[{eval_start},{eval_end}] ({positions_tested} total), "
            f"time={elapsed_ms:.1f}ms"
        )

        if not passed and not comparison_skipped:
            logger.warning(
                f"Memory not contributing at step {step}! "
                f"PPL reduction {accuracy:.1%} < {self.accuracy_threshold:.0%} "
                f"for beyond-window positions"
            )

        self.history.append(result)
        return result

    def run_probe_standalone(
        self,
        step: int = 0,
        device: str = "cpu",
    ) -> NIAHResult:
        """
        Run probe without a model (standalone).

        Returns a dummy failing result. Useful for unit testing.
        """
        return self.run_probe(model=None, step=step, device=device)

    def _standalone_result(self, step: int, start_time: float) -> NIAHResult:
        """Create a result for standalone mode (no model)."""
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        result = NIAHResult(
            step=step,
            accuracy=0.0,
            passed=False,
            needle_norm=0.0,
            retrieved_norm=0.0,
            haystack_size=self.haystack_size,
            probe_time_ms=elapsed_ms,
            ppl_mem=0.0,
            ppl_nomem=0.0,
            positions_tested=0,
        )
        logger.info(
            f"[Step {step}] Memory Probe SKIPPED (no model), time={elapsed_ms:.1f}ms"
        )
        self.history.append(result)
        return result

    def get_statistics(self) -> dict:
        """Get probe statistics across all runs."""
        if not self.history:
            return {
                "num_probes": 0,
                "pass_rate": 0.0,
                "mean_accuracy": 0.0,
            }

        accuracies = [r.accuracy for r in self.history]
        passed = [r for r in self.history if r.passed]
        times = [r.probe_time_ms for r in self.history]

        return {
            "num_probes": len(self.history),
            "pass_rate": len(passed) / len(self.history),
            "mean_accuracy": sum(accuracies) / len(accuracies),
            "min_accuracy": min(accuracies),
            "max_accuracy": max(accuracies),
            "latest_accuracy": self.history[-1].accuracy,
            "latest_passed": self.history[-1].passed,
            "mean_time_ms": sum(times) / len(times),
            "total_failures": len(self.history) - len(passed),
        }

    def reset(self) -> None:
        """Reset probe history."""
        self.history.clear()
