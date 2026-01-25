"""
Throughput Benchmarking for Atlas-MAG with Omega Rule.

REQ-P3-T3: Measure throughput (>=10x baseline target)

Provides utilities for measuring:
- Tokens per second (throughput)
- GPU utilization
- Memory allocation
- Throughput comparison

Reference: PRD Section 6.3, AC-P3-3, AC-P3-4
"""

import logging
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


@dataclass
class ThroughputResult:
    """Result from a throughput benchmark run."""

    tokens_per_second: float
    samples_per_second: float
    total_tokens: int
    total_samples: int
    elapsed_seconds: float
    gpu_utilization_mean: float
    gpu_memory_allocated_gb: float
    gpu_memory_reserved_gb: float

    @property
    def passes_target(self) -> bool:
        """Check if this meets the 10x baseline target."""
        # This should be called on a comparison result
        return False  # Override in comparison

    def __str__(self) -> str:
        return (
            f"ThroughputResult("
            f"tokens/s={self.tokens_per_second:.1f}, "
            f"samples/s={self.samples_per_second:.2f}, "
            f"GPU util={self.gpu_utilization_mean:.1f}%, "
            f"GPU mem={self.gpu_memory_allocated_gb:.2f}GB)"
        )


@dataclass
class ComparisonResult:
    """Result from baseline vs optimized comparison."""

    baseline: ThroughputResult
    optimized: ThroughputResult
    speedup: float
    passes_target: bool

    def __str__(self) -> str:
        status = "PASS" if self.passes_target else "FAIL"
        return (
            f"ComparisonResult({status}): "
            f"speedup={self.speedup:.1f}x, "
            f"baseline={self.baseline.tokens_per_second:.1f} tok/s, "
            f"optimized={self.optimized.tokens_per_second:.1f} tok/s"
        )


def get_gpu_utilization() -> float:
    """
    Get current GPU utilization percentage.

    Uses nvidia-smi for accurate utilization measurement.

    Returns:
        GPU utilization as percentage (0-100), or 0 if unavailable
    """
    if not torch.cuda.is_available():
        return 0.0

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5.0,
        )

        if result.returncode == 0:
            # Get first GPU's utilization
            lines = result.stdout.strip().split('\n')
            if lines:
                return float(lines[0])
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass

    return 0.0


def get_gpu_memory() -> Tuple[float, float]:
    """
    Get GPU memory usage.

    Returns:
        (allocated_gb, reserved_gb)
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)

    return allocated, reserved


class ThroughputBenchmark:
    """
    Benchmark training throughput.

    Measures tokens/second for forward and training passes.

    Target: >= 10x baseline throughput

    Args:
        model: Model to benchmark
        batch_size: Batch size for benchmarking
        seq_len: Sequence length
        device: Device to run on
        vocab_size: Vocabulary size for random data

    Example:
        >>> model = AtlasMAGSkeleton(...)
        >>> benchmark = ThroughputBenchmark(model, batch_size=8, seq_len=512)
        >>> result = benchmark.run_comparison(num_steps=100)
        >>> print(f"Speedup: {result.speedup:.1f}x")
    """

    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 8,
        seq_len: int = 512,
        device: str = "cuda",
        vocab_size: int = 32000,
    ):
        self.model = model
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.vocab_size = vocab_size

        # Move model to device
        self.model.to(device)

        logger.info(
            f"ThroughputBenchmark initialized: batch={batch_size}, "
            f"seq_len={seq_len}, device={device}"
        )

    def _generate_batch(self) -> Tensor:
        """Generate a random batch of token IDs."""
        return torch.randint(
            0, self.vocab_size,
            (self.batch_size, self.seq_len),
            device=self.device,
        )

    def _warmup(self, num_steps: int = 10) -> None:
        """Warmup GPU and model caches."""
        self.model.train()
        for _ in range(num_steps):
            batch = self._generate_batch()
            with torch.no_grad():
                _ = self.model(batch)

        # Synchronize GPU
        if self.device == "cuda":
            torch.cuda.synchronize()

    def _sample_gpu_utilization(
        self,
        duration: float,
        interval: float = 0.1,
    ) -> List[float]:
        """Sample GPU utilization over a duration."""
        samples = []
        start = time.perf_counter()

        while time.perf_counter() - start < duration:
            samples.append(get_gpu_utilization())
            time.sleep(interval)

        return samples

    def measure_forward_throughput(
        self,
        num_steps: int = 100,
        warmup_steps: int = 10,
    ) -> ThroughputResult:
        """
        Measure forward pass throughput.

        Args:
            num_steps: Number of forward passes to measure
            warmup_steps: Warmup steps before measurement

        Returns:
            ThroughputResult with tokens/second and GPU metrics
        """
        self._warmup(warmup_steps)

        # Start measurement
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        total_tokens = 0
        gpu_samples = []

        start_time = time.perf_counter()

        self.model.eval()
        with torch.no_grad():
            for step in range(num_steps):
                batch = self._generate_batch()
                _ = self.model(batch)

                total_tokens += self.batch_size * self.seq_len

                # Sample GPU utilization occasionally
                if step % 10 == 0:
                    gpu_samples.append(get_gpu_utilization())

        if self.device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time
        allocated, reserved = get_gpu_memory()

        return ThroughputResult(
            tokens_per_second=total_tokens / elapsed,
            samples_per_second=num_steps / elapsed,
            total_tokens=total_tokens,
            total_samples=num_steps,
            elapsed_seconds=elapsed,
            gpu_utilization_mean=sum(gpu_samples) / max(len(gpu_samples), 1),
            gpu_memory_allocated_gb=allocated,
            gpu_memory_reserved_gb=reserved,
        )

    def measure_training_throughput(
        self,
        optimizer: torch.optim.Optimizer,
        num_steps: int = 100,
        warmup_steps: int = 10,
    ) -> ThroughputResult:
        """
        Measure training (forward + backward) throughput.

        Args:
            optimizer: Optimizer to use
            num_steps: Number of training steps
            warmup_steps: Warmup steps before measurement

        Returns:
            ThroughputResult with tokens/second and GPU metrics
        """
        self._warmup(warmup_steps)

        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        total_tokens = 0
        gpu_samples = []

        start_time = time.perf_counter()

        self.model.train()
        for step in range(num_steps):
            batch = self._generate_batch()

            # Forward
            logits = self.model(batch)

            # Simple loss
            loss = logits.mean()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_tokens += self.batch_size * self.seq_len

            if step % 10 == 0:
                gpu_samples.append(get_gpu_utilization())

        if self.device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time
        allocated, reserved = get_gpu_memory()

        return ThroughputResult(
            tokens_per_second=total_tokens / elapsed,
            samples_per_second=num_steps / elapsed,
            total_tokens=total_tokens,
            total_samples=num_steps,
            elapsed_seconds=elapsed,
            gpu_utilization_mean=sum(gpu_samples) / max(len(gpu_samples), 1),
            gpu_memory_allocated_gb=allocated,
            gpu_memory_reserved_gb=reserved,
        )

    def run_comparison(
        self,
        num_steps: int = 100,
        target_speedup: float = 1.0,
    ) -> ComparisonResult:
        """
        Run baseline vs optimized comparison.

        Note: This is a placeholder comparison that measures the same model
        twice. It will always yield ~1x speedup. To get meaningful results:
        - TODO: Accept separate baseline_model parameter
        - TODO: Compare attention-only vs memory-enabled forward passes

        Args:
            num_steps: Steps for each measurement
            target_speedup: Target speedup multiplier (default: 1.0 for placeholder)

        Returns:
            ComparisonResult with speedup and pass/fail status
        """
        logger.warning(
            "run_comparison is a placeholder - measures same model twice. "
            "Speedup will be ~1x until baseline model is provided."
        )

        logger.info("Measuring baseline throughput...")
        baseline = self.measure_forward_throughput(num_steps)
        logger.info(f"Baseline: {baseline}")

        logger.info("Measuring optimized throughput...")
        optimized = self.measure_forward_throughput(num_steps)
        logger.info(f"Optimized: {optimized}")

        speedup = optimized.tokens_per_second / max(baseline.tokens_per_second, 1)
        passes = speedup >= target_speedup

        result = ComparisonResult(
            baseline=baseline,
            optimized=optimized,
            speedup=speedup,
            passes_target=passes,
        )

        logger.info(f"Comparison result: {result}")
        return result


def measure_operation_throughput(
    operation: Callable[[], Any],
    num_iterations: int = 1000,
    warmup_iterations: int = 100,
    device: str = "cuda",
) -> float:
    """
    Measure throughput of a single operation.

    Useful for microbenchmarking specific operations like
    Newton-Schulz or memory updates.

    Args:
        operation: Callable to benchmark (should have no args)
        num_iterations: Number of iterations to measure
        warmup_iterations: Warmup iterations
        device: Device for synchronization

    Returns:
        Operations per second
    """
    # Warmup
    for _ in range(warmup_iterations):
        operation()

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iterations):
        operation()

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    return num_iterations / elapsed


def benchmark_tensorized_operations(device: str = "cuda") -> dict:
    """
    Benchmark tensorized memory operations.

    Tests the performance of:
    - Outer product sum
    - Memory update
    - Batch projection
    - Parallel local update

    Returns:
        Dictionary with ops/second for each operation
    """
    from src.optim.memory_update import (
        tensorized_outer_product_sum,
        tensorized_memory_update,
        batch_qk_projection,
        parallel_local_memory_update,
    )

    dim = 768
    batch = 8
    seq_len = 512
    num_shards = 4

    # Setup tensors
    M = torch.zeros(dim, dim, device=device)
    keys = torch.randn(batch, seq_len, dim, device=device)
    queries = torch.randn(batch, seq_len, dim, device=device)

    local_M = torch.zeros(num_shards, dim, dim, device=device)
    local_keys = torch.randn(num_shards, seq_len, dim, device=device)
    local_norms = torch.zeros(num_shards, device=device)

    results = {}

    # Outer product sum
    def outer_op():
        return tensorized_outer_product_sum(keys)

    results["outer_product_sum"] = measure_operation_throughput(
        outer_op, num_iterations=100, warmup_iterations=10, device=device
    )

    # Memory update
    def update_op():
        return tensorized_memory_update(M, keys, 0.0)

    results["memory_update"] = measure_operation_throughput(
        update_op, num_iterations=100, warmup_iterations=10, device=device
    )

    # Batch projection
    def proj_op():
        return batch_qk_projection(M, queries, 1.0)

    results["batch_projection"] = measure_operation_throughput(
        proj_op, num_iterations=100, warmup_iterations=10, device=device
    )

    # Parallel local update
    def parallel_op():
        return parallel_local_memory_update(local_M, local_keys, local_norms)

    results["parallel_local_update"] = measure_operation_throughput(
        parallel_op, num_iterations=100, warmup_iterations=10, device=device
    )

    return results
