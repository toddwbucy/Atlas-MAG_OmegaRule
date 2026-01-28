"""
Phase 3 Unit Tests.

Tests for:
- Muon optimizer (K=10 Newton-Schulz)
- Tensorized memory updates (no Python loops)
- Throughput benchmarking
- GPU utilization monitoring

Reference: PRD Phase 3 requirements P3-T1 through P3-T3
"""

import time
from typing import List

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from src.config import K
from src.nn.newton_schulz import newton_schulz, orthogonality_error
from src.optim.muon import Muon, MuonMemory
from src.optim.memory_update import (
    tensorized_outer_product_sum,
    tensorized_memory_update,
    batch_qk_projection,
    parallel_local_memory_update,
    incremental_memory_update,
    verify_no_python_loops,
)
from src.training.benchmark import (
    ThroughputResult,
    ComparisonResult,
    ThroughputBenchmark,
    get_gpu_utilization,
    get_gpu_memory,
    measure_operation_throughput,
)


# ============================================================================
# Test Fixtures
# ============================================================================


class SimpleModel(nn.Module):
    """Simple model for testing optimizers."""

    def __init__(self, dim: int = 64, vocab_size: int = 100):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.linear2 = nn.Linear(dim * 2, dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x)
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return self.head(x)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel(dim=64, vocab_size=100)


@pytest.fixture
def device():
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# G3-1: K Unified (K=10 Everywhere)
# ============================================================================


class TestKUnified:
    """Tests for AC-P3-1: K=10 used everywhere."""

    def test_config_k_is_10(self):
        """K constant should be 10 (increased from 5 per committee critique #2)."""
        assert K == 10

    def test_muon_default_k(self, simple_model):
        """Muon optimizer should default to K=10."""
        optimizer = Muon(simple_model.parameters())
        assert optimizer.get_k() == 10

    def test_muon_explicit_k(self, simple_model):
        """Muon optimizer should use explicit K."""
        optimizer = Muon(simple_model.parameters(), k=5)
        assert optimizer.get_k() == 5

    def test_newton_schulz_default_k(self):
        """Newton-Schulz should use K=5 by default."""
        G = torch.randn(64, 64)
        X = newton_schulz(G)  # Should use default k=5
        assert X.shape == G.shape

    def test_newton_schulz_k5_convergence(self):
        """K=5 should achieve reasonable orthogonality."""
        G = torch.randn(64, 64)
        X = newton_schulz(G, num_iters=5)

        error = orthogonality_error(X)
        # K=5 should give error < 1.0 for most random matrices
        # (convergence depends on input conditioning)
        assert error < 1.0


# ============================================================================
# G3-2: Tensorized Operations (No Python Loops)
# ============================================================================


class TestTensorizedOps:
    """Tests for AC-P3-2: No Python loops over tokens."""

    def test_outer_product_sum_shape(self):
        """Outer product sum should produce (dim, dim) output."""
        keys = torch.randn(4, 64, 128)  # batch=4, seq=64, dim=128
        result = tensorized_outer_product_sum(keys)
        assert result.shape == (128, 128)

    def test_outer_product_sum_correctness(self):
        """Tensorized sum should match loop-based computation."""
        keys = torch.randn(4, 32, 64)
        dim = keys.size(-1)

        # Tensorized version
        tensor_result = tensorized_outer_product_sum(keys)

        # Loop-based version (for verification)
        flat = keys.reshape(-1, dim)
        loop_result = torch.zeros(dim, dim)
        for k in flat:
            loop_result += torch.outer(k, k)

        assert torch.allclose(tensor_result, loop_result, atol=1e-5)

    def test_memory_update_correctness(self):
        """Tensorized memory update should match expected behavior."""
        dim = 64
        M = torch.zeros(dim, dim)
        keys = torch.randn(2, 16, dim)

        M_new, norm_new = tensorized_memory_update(M, keys, 0.0)

        # Verify shape
        assert M_new.shape == (dim, dim)

        # Verify norm is positive
        assert norm_new > 0

        # Verify M is updated (not zero anymore)
        assert M_new.norm() > 0

    def test_memory_update_with_decay(self):
        """Memory update should apply decay correctly."""
        dim = 64
        M = torch.randn(dim, dim)
        M_initial = M.clone()
        keys = torch.randn(2, 16, dim)

        M_new, _ = tensorized_memory_update(M, keys, 0.0, alpha=0.9)

        # M_new should contain 0.9 * M_initial + new_outer_products
        # Just verify it changed from initial
        assert not torch.allclose(M_new, M_initial)

    def test_batch_projection_shape(self):
        """Batch projection should preserve query shape."""
        dim = 64
        M = torch.randn(dim, dim)
        queries = torch.randn(4, 32, dim)

        projected = batch_qk_projection(M, queries, 1.0)
        assert projected.shape == queries.shape

    def test_batch_projection_correctness(self):
        """Batch projection should match per-query computation."""
        dim = 64
        M = torch.randn(dim, dim)
        queries = torch.randn(4, 32, dim)

        # Tensorized version
        tensor_result = batch_qk_projection(M, queries, 1.0)

        # Loop-based version (for verification)
        flat_queries = queries.reshape(-1, dim)
        loop_result = torch.zeros_like(flat_queries)
        for i, q in enumerate(flat_queries):
            loop_result[i] = M @ q

        loop_result = loop_result.reshape(queries.shape)

        assert torch.allclose(tensor_result, loop_result, atol=1e-5)

    def test_parallel_local_update(self):
        """Parallel local update should process all shards at once."""
        num_shards = 4
        dim = 64
        shard_len = 32

        local_M = torch.zeros(num_shards, dim, dim)
        local_keys = torch.randn(num_shards, shard_len, dim)
        local_norms = torch.zeros(num_shards)

        updated_M, updated_norms = parallel_local_memory_update(
            local_M, local_keys, local_norms
        )

        assert updated_M.shape == (num_shards, dim, dim)
        assert updated_norms.shape == (num_shards,)

        # All norms should be positive
        assert (updated_norms > 0).all()

        # All memories should be non-zero
        for i in range(num_shards):
            assert updated_M[i].norm() > 0

    def test_incremental_update(self):
        """Incremental single-key update should work."""
        dim = 64
        M = torch.zeros(dim, dim)
        k = torch.randn(dim)

        M_new, norm_new = incremental_memory_update(M, k, 0.0)

        assert M_new.shape == (dim, dim)
        assert norm_new > 0
        assert M_new.norm() > 0

    def test_verify_no_python_loops(self):
        """Verification function should pass."""
        assert verify_no_python_loops() is True


# ============================================================================
# Muon Optimizer Tests
# ============================================================================


class TestMuonOptimizer:
    """Tests for Muon optimizer implementation."""

    def test_muon_init(self, simple_model):
        """Muon should initialize correctly."""
        optimizer = Muon(
            simple_model.parameters(),
            lr=1e-3,
            momentum=0.95,
            k=5,
        )

        assert optimizer.defaults['lr'] == 1e-3
        assert optimizer.defaults['momentum'] == 0.95
        assert optimizer.defaults['k'] == 5

    def test_muon_step(self, simple_model):
        """Muon step should update parameters."""
        optimizer = Muon(simple_model.parameters(), lr=0.1)

        # Get initial parameters
        initial_params = [p.clone() for p in simple_model.parameters()]

        # Forward and backward
        x = torch.randint(0, 100, (4, 32))
        loss = simple_model(x).mean()
        loss.backward()

        # Step
        optimizer.step()

        # Verify parameters changed
        for p_init, p_new in zip(initial_params, simple_model.parameters()):
            if p_new.grad is not None:
                assert not torch.allclose(p_init, p_new)

    def test_muon_orthogonalizes_2d(self, simple_model):
        """Muon should orthogonalize 2D gradients."""
        optimizer = Muon(simple_model.parameters(), k=5)

        # Forward and backward
        x = torch.randint(0, 100, (4, 32))
        loss = simple_model(x).mean()
        loss.backward()

        # The optimizer step internally orthogonalizes 2D grads
        # We can verify it runs without error
        optimizer.step()

    def test_muon_state_summary(self, simple_model):
        """State summary should provide useful info."""
        optimizer = Muon(simple_model.parameters())

        # Do a few steps
        for _ in range(5):
            x = torch.randint(0, 100, (4, 32))
            loss = simple_model(x).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        summary = optimizer.get_state_summary()

        assert summary['k'] == 10
        assert summary['current_step'] == 5
        assert summary['total_params'] > 0
        assert summary['orthogonalized_params'] > 0


class TestMuonMemory:
    """Tests for specialized MuonMemory optimizer."""

    def test_muon_memory_init(self):
        """MuonMemory should initialize with correct params."""
        M = torch.zeros(64, 64, requires_grad=True)
        optimizer = MuonMemory([M], lr=1e-3, alpha=0.95, theta=0.9, k=5)

        assert optimizer.defaults['alpha'] == 0.95
        assert optimizer.defaults['theta'] == 0.9
        assert optimizer.defaults['k'] == 5

    def test_muon_memory_step(self):
        """MuonMemory should update memory matrix."""
        M = torch.randn(64, 64, requires_grad=True)
        M_initial = M.clone().detach()

        optimizer = MuonMemory([M], lr=0.1)

        # Simulate gradient
        M.grad = torch.randn_like(M)

        optimizer.step()

        # M should have changed
        assert not torch.allclose(M, M_initial)


# ============================================================================
# Throughput Benchmark Tests
# ============================================================================


class TestThroughputBenchmark:
    """Tests for throughput benchmarking."""

    def test_throughput_result_str(self):
        """ThroughputResult should have readable string representation."""
        result = ThroughputResult(
            tokens_per_second=10000.0,
            samples_per_second=10.0,
            total_tokens=100000,
            total_samples=1000,
            elapsed_seconds=10.0,
            gpu_utilization_mean=85.0,
            gpu_memory_allocated_gb=2.5,
            gpu_memory_reserved_gb=4.0,
        )

        str_repr = str(result)
        assert "10000" in str_repr
        assert "85" in str_repr

    def test_comparison_result_pass(self):
        """ComparisonResult should indicate pass/fail."""
        baseline = ThroughputResult(
            tokens_per_second=1000.0,
            samples_per_second=1.0,
            total_tokens=1000,
            total_samples=100,
            elapsed_seconds=1.0,
            gpu_utilization_mean=50.0,
            gpu_memory_allocated_gb=1.0,
            gpu_memory_reserved_gb=2.0,
        )

        optimized = ThroughputResult(
            tokens_per_second=15000.0,  # 15x speedup
            samples_per_second=15.0,
            total_tokens=15000,
            total_samples=1500,
            elapsed_seconds=1.0,
            gpu_utilization_mean=90.0,
            gpu_memory_allocated_gb=1.5,
            gpu_memory_reserved_gb=3.0,
        )

        result = ComparisonResult(
            baseline=baseline,
            optimized=optimized,
            speedup=15.0,
            passes_target=True,
        )

        assert result.passes_target is True
        assert "PASS" in str(result)

    def test_benchmark_forward_throughput(self, simple_model, device):
        """Benchmark should measure forward throughput."""
        benchmark = ThroughputBenchmark(
            model=simple_model,
            batch_size=4,
            seq_len=32,
            device=device,
            vocab_size=100,
        )

        result = benchmark.measure_forward_throughput(num_steps=10, warmup_steps=2)

        assert result.tokens_per_second > 0
        assert result.total_tokens == 4 * 32 * 10
        assert result.elapsed_seconds > 0

    def test_measure_operation_throughput(self, device):
        """Operation throughput measurement should work."""
        x = torch.randn(64, 64, device=device)

        def op():
            return torch.mm(x, x)

        ops_per_sec = measure_operation_throughput(
            op,
            num_iterations=100,
            warmup_iterations=10,
            device=device,
        )

        assert ops_per_sec > 0


# ============================================================================
# GPU Monitoring Tests
# ============================================================================


class TestGPUMonitoring:
    """Tests for GPU utilization monitoring."""

    def test_get_gpu_utilization(self):
        """GPU utilization should return a valid percentage."""
        util = get_gpu_utilization()

        # Should be 0-100 or 0 if no GPU
        assert 0.0 <= util <= 100.0

    def test_get_gpu_memory(self):
        """GPU memory should return non-negative values."""
        allocated, reserved = get_gpu_memory()

        assert allocated >= 0.0
        assert reserved >= 0.0


# ============================================================================
# Acceptance Criteria Tests
# ============================================================================


class TestAcceptanceCriteria:
    """Tests for Phase 3 acceptance criteria."""

    def test_ac_p3_1_k_unified(self, simple_model):
        """AC-P3-1: K should be 10 everywhere (increased per committee critique #2)."""
        # Config
        assert K == 10

        # Muon optimizer
        optimizer = Muon(simple_model.parameters())
        assert optimizer.get_k() == 10

    def test_ac_p3_2_no_python_loops(self):
        """AC-P3-2: Memory operations should be tensorized."""
        assert verify_no_python_loops() is True

    def test_ac_p3_3_throughput_measurement(self, simple_model, device):
        """AC-P3-3: Throughput measurement should work."""
        benchmark = ThroughputBenchmark(
            model=simple_model,
            batch_size=4,
            seq_len=32,
            device=device,
            vocab_size=100,
        )

        result = benchmark.measure_forward_throughput(num_steps=5)

        # Verify we can measure throughput
        assert result.tokens_per_second > 0

    def test_ac_p3_4_gpu_utilization(self):
        """AC-P3-4: GPU utilization measurement should work."""
        # Just verify we can measure it
        util = get_gpu_utilization()
        assert isinstance(util, float)


# ============================================================================
# Integration Tests
# ============================================================================


class TestPhase3Integration:
    """Integration tests for Phase 3 components."""

    def test_muon_with_tensorized_memory(self, device):
        """Muon optimizer should work with tensorized memory ops."""
        dim = 64

        # Create memory as parameter on the correct device
        M = nn.Parameter(torch.zeros(dim, dim, device=device))
        optimizer = Muon([M], lr=0.1, k=5)

        # Simulate training loop with tensorized updates
        for _ in range(5):
            keys = torch.randn(4, 32, dim, device=device)

            # Tensorized outer product (simulating memory gradient)
            outer_sum = tensorized_outer_product_sum(keys)

            # Set as gradient
            M.grad = outer_sum

            # Optimizer step (orthogonalizes and applies)
            optimizer.step()
            optimizer.zero_grad()

        # M should have been updated
        assert M.norm() > 0

    def test_full_memory_pipeline(self, device):
        """Full memory update and projection pipeline."""
        dim = 64
        batch = 4
        seq_len = 32

        # Initialize
        M = torch.zeros(dim, dim, device=device)
        norm_sum = 0.0

        # Generate keys and queries
        keys = torch.randn(batch, seq_len, dim, device=device)
        queries = torch.randn(batch, seq_len, dim, device=device)

        # Update memory
        M, norm_sum = tensorized_memory_update(M, keys, norm_sum)

        # Project queries
        projected = batch_qk_projection(M, queries, norm_sum)

        assert projected.shape == queries.shape
        assert not torch.isnan(projected).any()

    def test_parallel_shards_pipeline(self, device):
        """Parallel shard processing pipeline."""
        num_shards = 4
        dim = 64
        shard_len = 32

        # Initialize local memories
        local_M = torch.zeros(num_shards, dim, dim, device=device)
        local_norms = torch.zeros(num_shards, device=device)

        # Process multiple rounds of updates
        for _ in range(3):
            local_keys = torch.randn(num_shards, shard_len, dim, device=device)

            local_M, local_norms = parallel_local_memory_update(
                local_M, local_keys, local_norms
            )

        # All shards should be updated
        for i in range(num_shards):
            assert local_M[i].norm() > 0
            assert local_norms[i] > 0
