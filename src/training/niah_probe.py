"""
Needle-in-a-Haystack (NIAH) Retrieval Probes.

REQ-P2-002: NIAH Retrieval Probe

Validates that Atlas memory actually WORKS by:
1. Injecting a known key-value pair (the "needle")
2. Running haystack (random keys to bury the needle)
3. Querying with the original key
4. Measuring retrieval accuracy via cosine similarity

Target: > 80% accuracy

This replaces vanity metrics (like gradient norms) with a functional
test of memory capability. If NIAH accuracy drops, the memory is broken.

Reference: PRD Section 5.4 REQ-P2-002
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, Any, List

import torch
import torch.nn.functional as F

from src.model.qk_projection import QKProjection, create_qk_projection_for_model

logger = logging.getLogger(__name__)


@dataclass
class NIAHResult:
    """Result from a single NIAH probe."""

    step: int
    accuracy: float
    passed: bool
    needle_norm: float
    retrieved_norm: float
    haystack_size: int
    probe_time_ms: float


class NIAHProbe:
    """
    Needle-in-a-Haystack probe for memory validation.

    Periodically injects a "needle" (key-value pair) into memory,
    buries it under a "haystack" of random keys, then queries to
    see if the needle can be retrieved.

    The accuracy is measured via cosine similarity between the
    retrieved value and the original value.

    Args:
        dim: Model dimension
        probe_frequency: How often to run probes (in steps)
        haystack_size: Number of random keys in haystack
        accuracy_threshold: Minimum passing accuracy (default: 0.8)

    Example:
        >>> probe = NIAHProbe(dim=768, probe_frequency=1000)
        >>> for step in range(10000):
        ...     if probe.should_probe(step):
        ...         result = probe.run_probe(model, step, "cuda")
        ...         if not result.passed:
        ...             print(f"Memory degraded at step {step}!")
    """

    def __init__(
        self,
        dim: int,
        probe_frequency: int = 1000,
        haystack_size: int = 100,
        accuracy_threshold: float = 0.8,
    ):
        self.dim = dim
        self.probe_frequency = probe_frequency
        self.haystack_size = haystack_size
        self.accuracy_threshold = accuracy_threshold

        # History for analysis
        self.history: List[NIAHResult] = []

        # Cached QKProjection (created on first probe)
        self._qk_proj: Optional[QKProjection] = None

        logger.info(
            f"NIAHProbe initialized: dim={dim}, freq={probe_frequency}, "
            f"haystack={haystack_size}, threshold={accuracy_threshold}"
        )

    def should_probe(self, step: int) -> bool:
        """
        Check if we should run a probe at this step.

        Probes run at step 0 and every probe_frequency steps thereafter.
        """
        if step == 0:
            return True
        return step % self.probe_frequency == 0

    def run_probe(
        self,
        model: Any,
        step: int,
        device: str = "cpu",
        qk_proj: Optional[QKProjection] = None,
    ) -> NIAHResult:
        """
        Run a NIAH retrieval probe.

        Args:
            model: Model with persistent memory (or None if qk_proj provided)
            step: Current training step
            device: Device to run on
            qk_proj: Optional pre-created QKProjection

        Returns:
            NIAHResult with accuracy and pass/fail status
        """
        start_time = time.perf_counter()

        # Get or create QKProjection
        proj: Optional[QKProjection] = None
        if qk_proj is not None:
            proj = qk_proj
        elif self._qk_proj is not None:
            proj = self._qk_proj
        else:
            proj = create_qk_projection_for_model(model)
            if proj is not None:
                self._qk_proj = proj

        if proj is None:
            logger.warning("No QKProjection available for NIAH probe")
            return NIAHResult(
                step=step,
                accuracy=0.0,
                passed=False,
                needle_norm=0.0,
                retrieved_norm=0.0,
                haystack_size=self.haystack_size,
                probe_time_ms=0.0,
            )

        # Move to device if needed
        if hasattr(proj, "to"):
            proj = proj.to(device)

        with torch.no_grad():
            # Reset to clean state
            proj.reset_at_shard_boundary()
            proj.clear_stored_values()

            # Generate random needle (normalized for stable similarity)
            needle_key = torch.randn(self.dim, device=device)
            needle_value = torch.randn(self.dim, device=device)
            needle_key = F.normalize(needle_key, dim=0)
            needle_value = F.normalize(needle_value, dim=0)

            # Inject needle
            proj.inject_memory(needle_key, needle_value)

            # Bury under haystack (random keys)
            for _ in range(self.haystack_size):
                hay_key = torch.randn(self.dim, device=device)
                hay_key = F.normalize(hay_key, dim=0)
                proj.update(hay_key)

            # Query for needle
            retrieved = proj.query_memory(needle_key)
            retrieved = F.normalize(retrieved, dim=0)

            # Measure accuracy via cosine similarity
            accuracy = F.cosine_similarity(
                retrieved.unsqueeze(0),
                needle_value.unsqueeze(0),
            ).item()

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Determine pass/fail
        passed = accuracy >= self.accuracy_threshold

        result = NIAHResult(
            step=step,
            accuracy=accuracy,
            passed=passed,
            needle_norm=needle_key.norm().item(),
            retrieved_norm=retrieved.norm().item(),
            haystack_size=self.haystack_size,
            probe_time_ms=elapsed_ms,
        )

        # Log result
        status = "PASSED" if passed else "FAILED"
        logger.info(
            f"[Step {step}] NIAH Probe {status}: "
            f"accuracy={accuracy:.3f} (threshold={self.accuracy_threshold}), "
            f"time={elapsed_ms:.1f}ms"
        )

        if not passed:
            logger.warning(
                f"Atlas memory retrieval degraded at step {step}! "
                f"Accuracy {accuracy:.3f} < {self.accuracy_threshold}"
            )

        self.history.append(result)
        return result

    def run_probe_standalone(
        self,
        qk_proj: QKProjection,
        step: int = 0,
        device: str = "cpu",
    ) -> NIAHResult:
        """
        Run probe with a standalone QKProjection (no model needed).

        Useful for unit testing.
        """
        return self.run_probe(model=None, step=step, device=device, qk_proj=qk_proj)

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
        self._qk_proj = None


def measure_niah_overhead(
    model: Any,
    num_steps: int = 100,
    probe_frequencies: Optional[List[int]] = None,
    device: str = "cpu",
) -> dict:
    """
    Measure throughput overhead of NIAH probes at different frequencies.

    Used to determine optimal probe frequency (P2-T2).
    Target: < 1% overhead.

    Args:
        model: Model to test
        num_steps: Number of steps to simulate
        probe_frequencies: List of frequencies to test
        device: Device to run on

    Returns:
        Dictionary with timing measurements for each frequency
    """
    if probe_frequencies is None:
        probe_frequencies = [500, 1000, 2000]

    dim = model.dim if hasattr(model, "dim") else 768
    results = {}

    # Baseline (no probes)
    start = time.perf_counter()
    for _ in range(num_steps):
        # Simulate forward pass (tensor created for potential future use)
        _x = torch.randn(4, 64, dim, device=device)
        if hasattr(model, "forward"):
            with torch.no_grad():
                _ = model(torch.randint(0, 1000, (4, 64), device=device))
    baseline_time = time.perf_counter() - start

    results["baseline"] = {
        "time": baseline_time,
        "steps": num_steps,
    }

    # Test each frequency
    for freq in probe_frequencies:
        probe = NIAHProbe(dim=dim, probe_frequency=freq)

        start = time.perf_counter()
        for step in range(num_steps):
            _x = torch.randn(4, 64, dim, device=device)
            if hasattr(model, "forward"):
                with torch.no_grad():
                    _ = model(torch.randint(0, 1000, (4, 64), device=device))
            if probe.should_probe(step):
                probe.run_probe(model, step, device)
        probed_time = time.perf_counter() - start

        overhead = (probed_time - baseline_time) / baseline_time * 100
        num_probes = len(probe.history)

        results[f"freq_{freq}"] = {
            "time": probed_time,
            "overhead_percent": overhead,
            "num_probes": num_probes,
            "passes_threshold": overhead < 1.0,
        }

        logger.info(
            f"NIAH overhead at freq={freq}: {overhead:.2f}% "
            f"({num_probes} probes in {num_steps} steps)"
        )

    return results
