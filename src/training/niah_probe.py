"""
Memory Functional Probe.

REQ-P2-002: Memory Contribution Probe

Validates that Atlas memory is actively contributing to model predictions
by comparing perplexity with memory enabled vs disabled.

This is a functional test of what the memory *actually does* during inference,
not a proxy metric. The memory module is an MLP trained via the outer loop
(backprop) to transform hidden states — it's a learned function, not an
associative lookup table. The correct test is: does it improve predictions?

Steps:
1. Generate random token sequences (no dataset dependency)
2. Forward pass with memory enabled → PPL_mem
3. Forward pass with memory disabled → PPL_nomem
4. Measure PPL reduction: (PPL_nomem - PPL_mem) / PPL_nomem

Target: memory should reduce PPL (positive contribution)

Reference: PRD Section 5.4 REQ-P2-002
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, List

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class NIAHResult:
    """Result from a single memory probe."""

    step: int
    accuracy: float  # PPL reduction ratio: (nomem - mem) / nomem
    passed: bool
    needle_norm: float  # PPL with memory
    retrieved_norm: float  # PPL without memory
    haystack_size: int  # Number of test sequences
    probe_time_ms: float


class NIAHProbe:
    """
    Functional memory probe.

    Tests whether the memory module improves model predictions by comparing
    perplexity with and without memory. This validates the outer loop has
    trained M₀ to be a useful function.

    The "accuracy" metric is the PPL reduction ratio:
        accuracy = (PPL_nomem - PPL_mem) / PPL_nomem

    A value of 0.9 means memory reduces PPL by 90%. The threshold determines
    the minimum acceptable contribution.

    Args:
        dim: Model dimension (used for standalone mode only)
        probe_frequency: How often to run probes (in steps)
        haystack_size: Number of test sequences (default: 4)
        accuracy_threshold: Minimum PPL reduction ratio (default: 0.1 = 10%)
        seq_len: Sequence length for test inputs (default: 128)

    Example:
        >>> probe = NIAHProbe(dim=768, probe_frequency=1000)
        >>> for step in range(10000):
        ...     if probe.should_probe(step):
        ...         result = probe.run_probe(model, step, "cuda")
        ...         if not result.passed:
        ...             print(f"Memory not contributing at step {step}!")
    """

    def __init__(
        self,
        dim: int,
        probe_frequency: int = 1000,
        haystack_size: int = 4,
        accuracy_threshold: float = 0.1,
        seq_len: int = 128,
        # Legacy params accepted but ignored for backward compat
        ttl_steps: int = 10,
    ):
        self.dim = dim
        self.probe_frequency = probe_frequency
        self.haystack_size = haystack_size
        self.accuracy_threshold = accuracy_threshold
        self.seq_len = seq_len

        # History for analysis
        self.history: List[NIAHResult] = []

        logger.info(
            f"MemoryProbe initialized: dim={dim}, freq={probe_frequency}, "
            f"n_seqs={haystack_size}, threshold={accuracy_threshold}"
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

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
        """
        self._eval_batch = input_ids.detach()
        logger.info(
            f"MemoryProbe cached eval batch: {input_ids.shape}"
        )

    def run_probe(
        self,
        model: Any,
        step: int,
        device: str = "cpu",
        **kwargs: Any,
    ) -> NIAHResult:
        """
        Run a functional memory probe.

        Compares model PPL with memory enabled vs disabled. Uses cached
        eval batch if available (real tokens), otherwise falls back to
        random tokens (which may give misleading results on trained models).

        Args:
            model: The AtlasMAGSkeleton model
            step: Current training step
            device: Device to run on

        Returns:
            NIAHResult with accuracy = PPL reduction ratio
        """
        start_time = time.perf_counter()

        if model is None:
            return self._standalone_result(step, start_time)

        # Use cached eval batch if available, else random tokens
        if hasattr(self, "_eval_batch") and self._eval_batch is not None:
            input_ids = self._eval_batch.to(device)
        else:
            vocab_size = model.vocab_size
            input_ids = torch.randint(
                0, vocab_size, (self.haystack_size, self.seq_len), device=device
            )

        labels = input_ids[:, 1:].contiguous()
        vocab_size = model.vocab_size

        with torch.no_grad():
            # Forward with memory enabled
            logits_mem = model(input_ids)
            logits_mem = logits_mem[:, :-1, :].contiguous()
            loss_mem = F.cross_entropy(
                logits_mem.view(-1, vocab_size), labels.view(-1)
            )
            ppl_mem = torch.exp(loss_mem).item()

            # Disable memory across all blocks
            for block in model.blocks:
                block.disable_memory = True

            logits_nomem = model(input_ids)
            logits_nomem = logits_nomem[:, :-1, :].contiguous()
            loss_nomem = F.cross_entropy(
                logits_nomem.view(-1, vocab_size), labels.view(-1)
            )
            ppl_nomem = torch.exp(loss_nomem).item()

            # Re-enable memory
            for block in model.blocks:
                block.disable_memory = False

        # PPL reduction ratio: how much does memory help?
        if ppl_nomem > 0:
            accuracy = (ppl_nomem - ppl_mem) / ppl_nomem
        else:
            accuracy = 0.0

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        passed = accuracy >= self.accuracy_threshold

        result = NIAHResult(
            step=step,
            accuracy=accuracy,
            passed=passed,
            needle_norm=ppl_mem,
            retrieved_norm=ppl_nomem,
            haystack_size=self.haystack_size,
            probe_time_ms=elapsed_ms,
        )

        status = "PASSED" if passed else "FAILED"
        logger.info(
            f"[Step {step}] Memory Probe {status}: "
            f"contribution={accuracy:.1%} (threshold={self.accuracy_threshold:.0%}), "
            f"PPL mem={ppl_mem:.1f} nomem={ppl_nomem:.1f}, "
            f"time={elapsed_ms:.1f}ms"
        )

        if not passed:
            logger.warning(
                f"Memory not contributing at step {step}! "
                f"PPL reduction {accuracy:.1%} < {self.accuracy_threshold:.0%}"
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
