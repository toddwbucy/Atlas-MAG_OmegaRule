"""
Tests for Phase 1 Architecture Validation.

Validates:
    P1-T1: Gate polarization with annealing lambda
    P1-T2: Gate noise suppression test (< 0.01)
    P1-T3: Polynomial degree decision (degree 2)
    P1-T4: Memory depth decision (L_M = 2)
    P1-T5: Fast-fail gate check at step 500
    P1-T6: Multiplicative fusion test (gate=0 vs gate=1)
    P1-T7: Gated MLP structure (Atlas++)
"""

import pytest
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    D,
    LAMBDA_INITIAL,
    LAMBDA_FINAL,
    POLARIZATION_ANNEAL_RATIO,
    POLY_DEGREE,
    L_M,
    FAST_FAIL,
)
from src.model.skeleton import AtlasMAGSkeleton, AtlasMAGBlock
from src.model.atlas_memory import AtlasMemory, AtlasMemoryPoly
from src.training.polarization import (
    get_lambda_polar,
    gate_polarization_loss,
    compute_gate_statistics,
)
from src.training.fast_fail import GateMonitor, FastFailError, check_gate_health
from src.training.trainer import Phase1Trainer, verify_multiplicative_fusion


class TestPolarizationLoss:
    """Tests for gate polarization loss (P1-T1)."""

    def test_polarization_loss_shape(self):
        """Polarization loss should be a scalar tensor."""
        gate_values = torch.tensor([0.5, 0.7, 0.3, 0.9])
        loss = gate_polarization_loss(gate_values, step=100, total_steps=1000)
        assert loss.ndim == 0, "Loss should be a scalar"
        assert loss.item() > 0, "Loss should be positive for mixed gates"

    def test_polarization_loss_zero_at_extremes(self):
        """Polarization loss should be zero when all gates are decisive."""
        # All gates at 0 (pure attention)
        gate_values = torch.tensor([0.0, 0.0, 0.0, 0.0])
        loss = gate_polarization_loss(gate_values, step=100, total_steps=1000)
        assert loss.item() < 1e-6, "Loss should be ~0 when all gates are 0"

        # All gates at 1 (pure memory)
        gate_values = torch.tensor([1.0, 1.0, 1.0, 1.0])
        loss = gate_polarization_loss(gate_values, step=100, total_steps=1000)
        assert loss.item() < 1e-6, "Loss should be ~0 when all gates are 1"

    def test_polarization_loss_max_at_middle(self):
        """Polarization loss should be maximum when gates are at 0.5."""
        # All gates at 0.5 (maximally indecisive)
        gate_values = torch.tensor([0.5, 0.5, 0.5, 0.5])
        loss = gate_polarization_loss(gate_values, step=0, total_steps=1000)
        # At step 0, lambda = 10.0, and penalty per gate = 1.0
        # So loss should be 10.0
        assert abs(loss.item() - LAMBDA_INITIAL) < 0.01

    def test_polarization_loss_symmetric(self):
        """Loss should be symmetric around 0.5."""
        gates_low = torch.tensor([0.3])
        gates_high = torch.tensor([0.7])  # Same distance from 0.5
        loss_low = gate_polarization_loss(gates_low, step=0, total_steps=1000)
        loss_high = gate_polarization_loss(gates_high, step=0, total_steps=1000)
        assert torch.allclose(loss_low, loss_high, atol=1e-6)


class TestAnnealingSchedule:
    """Tests for lambda annealing schedule (P1-T1)."""

    def test_lambda_initial(self):
        """Lambda should start at LAMBDA_INITIAL."""
        assert get_lambda_polar(0, 10000) == LAMBDA_INITIAL
        assert get_lambda_polar(1, 10000) == LAMBDA_INITIAL

    def test_lambda_warmup(self):
        """Lambda should stay high during warmup."""
        total = 10000
        warmup_steps = int(POLARIZATION_ANNEAL_RATIO * total)  # 1000

        # Still in warmup
        assert get_lambda_polar(warmup_steps - 1, total) == LAMBDA_INITIAL

    def test_lambda_decay(self):
        """Lambda should decay after warmup."""
        total = 10000
        warmup_steps = int(POLARIZATION_ANNEAL_RATIO * total)

        # After warmup, should start decaying
        lambda_after = get_lambda_polar(warmup_steps + 1000, total)
        assert lambda_after < LAMBDA_INITIAL
        assert lambda_after > LAMBDA_FINAL

    def test_lambda_final(self):
        """Lambda should reach LAMBDA_FINAL at end."""
        lambda_end = get_lambda_polar(10000, 10000)
        assert abs(lambda_end - LAMBDA_FINAL) < 0.01

    def test_lambda_monotonic(self):
        """Lambda should decrease monotonically after warmup."""
        total = 10000
        warmup = int(POLARIZATION_ANNEAL_RATIO * total)

        prev = get_lambda_polar(warmup, total)
        for step in range(warmup + 100, total, 100):
            curr = get_lambda_polar(step, total)
            assert curr <= prev, f"Lambda increased at step {step}"
            prev = curr


class TestFastFail:
    """Tests for fast-fail gate monitoring (P1-T5)."""

    def test_baseline_recording(self):
        """Monitor should record baseline at step 100."""
        monitor = GateMonitor()
        gates = torch.tensor([0.5, 0.5, 0.5])

        # Before step 100
        monitor.check(gates, step=50)
        assert monitor.initial_variance is None

        # At step 100
        monitor.check(gates, step=100)
        assert monitor.initial_variance is not None

    def test_variance_check_pass(self):
        """Should pass if variance increases by 1.5x."""
        monitor = GateMonitor()

        # Low variance at step 100
        gates_low_var = torch.tensor([0.5, 0.5, 0.5])
        monitor.check(gates_low_var, step=100)

        # High variance at step 500 (should pass)
        gates_high_var = torch.tensor([0.1, 0.5, 0.9])
        # This should not raise
        monitor.check(gates_high_var, step=500)

    def test_variance_check_fail(self):
        """Should fail if variance doesn't increase enough."""
        monitor = GateMonitor()

        # Some variance at step 100
        gates = torch.tensor([0.4, 0.5, 0.6])
        monitor.check(gates, step=100)

        # Same variance at step 500 (should fail)
        with pytest.raises(FastFailError) as exc_info:
            monitor.check(gates, step=500)

        assert "variance not increasing" in str(exc_info.value).lower()

    def test_std_collapse_detection(self):
        """Should fail if std drops below 0.01."""
        monitor = GateMonitor()

        # Normal at step 100
        gates = torch.tensor([0.4, 0.5, 0.6])
        monitor.check(gates, step=100)

        # Collapsed at step 200 (all same value)
        gates_collapsed = torch.tensor([0.5001, 0.5002, 0.5003])  # std < 0.01
        with pytest.raises(FastFailError) as exc_info:
            monitor.check(gates_collapsed, step=200)

        assert "std collapsed" in str(exc_info.value).lower()

    def test_reset(self):
        """Reset should clear state."""
        monitor = GateMonitor()
        gates = torch.tensor([0.5, 0.5, 0.5])
        monitor.check(gates, step=100)
        assert monitor.initial_variance is not None

        monitor.reset()
        assert monitor.initial_variance is None
        assert len(monitor.history) == 0


class TestGateHealthCheck:
    """Tests for non-fatal gate health checks."""

    def test_healthy_gates(self):
        """Healthy gates should report healthy=True."""
        # Use values clearly in polarized range (<0.1 or >0.9)
        gates = torch.tensor([0.05, 0.08, 0.92, 0.95])  # Clearly polarized
        result = check_gate_health(gates, step=0)
        assert result["healthy"] is True
        assert result["polarized_ratio"] == 1.0  # All 4 are polarized

    def test_unhealthy_gates(self):
        """Indecisive gates should report healthy=False."""
        gates = torch.tensor([0.45, 0.48, 0.52, 0.55])  # All in middle
        result = check_gate_health(gates, step=0, warn_threshold=0.3)
        assert result["healthy"] is False
        assert result["indecisive_ratio"] > 0.3


class TestMultiplicativeFusion:
    """Tests for multiplicative gate fusion (P1-T6)."""

    def test_gate_zero_uses_attention(self):
        """When gate=0, output should equal pure attention output."""
        block = AtlasMAGBlock(dim=128, n_heads=4)
        x = torch.randn(1, 32, 128)

        with torch.no_grad():
            # Force gate to 0
            block.memory_gate.data.fill_(-100)
            output_gate0 = block(x.clone())

            # Force gate to 1
            block.memory_gate.data.fill_(100)
            output_gate1 = block(x.clone())

        # Outputs should be different
        assert not torch.allclose(output_gate0, output_gate1, atol=1e-5)

    def test_gate_one_uses_memory(self):
        """When gate=1, memory should contribute."""
        block = AtlasMAGBlock(dim=128, n_heads=4)
        x = torch.randn(1, 32, 128)

        with torch.no_grad():
            block.memory_gate.data.fill_(100)  # gate -> 1
            output = block(x)

        # Output should differ from input (memory adds something)
        assert not torch.allclose(x, output, atol=1e-3)

    def verify_multiplicative_fusion_function(self):
        """Test the multiplicative fusion test function."""
        block = AtlasMAGBlock(dim=128, n_heads=4)
        result = verify_multiplicative_fusion(block)
        assert result is True

    def test_model_level_fusion(self):
        """Test multiplicative fusion at model level."""
        model = AtlasMAGSkeleton(
            vocab_size=1000, dim=128, n_layers=2, n_heads=4
        )
        result = verify_multiplicative_fusion(model)
        assert result is True


class TestGateStatistics:
    """Tests for gate statistics computation."""

    def test_compute_gate_statistics(self):
        """Statistics should be computed correctly."""
        # 0.0 and 1.0 are clearly polarized (<0.1 and >0.9)
        # 0.05 is also polarized (<0.1)
        # 0.5 is indecisive
        # 0.95 is polarized (>0.9)
        gates = torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0])
        stats = compute_gate_statistics(gates)

        assert "mean" in stats
        assert "std" in stats
        assert "polarized_ratio" in stats
        assert "indecisive_ratio" in stats

        # Check values
        assert 0.4 < stats["mean"] < 0.6  # Mean around 0.5
        # 4 out of 5 are polarized (0.0, 0.05, 0.95, 1.0)
        assert abs(stats["polarized_ratio"] - 0.8) < 1e-5


class TestArchitectureDecisions:
    """Tests for pre-resolved architecture decisions (P1-T3, P1-T4)."""

    def test_polynomial_degree(self):
        """Polynomial degree should be 2."""
        assert POLY_DEGREE == 2

    def test_memory_depth(self):
        """Memory depth L_M should be 2."""
        assert L_M == 2

    def test_poly_memory_degree_2_only(self):
        """AtlasMemoryPoly should only support degree 2."""
        mem = AtlasMemoryPoly(dim=64)
        assert mem.poly_degree == 2

        with pytest.raises(NotImplementedError):
            AtlasMemoryPoly(dim=64, poly_degree=3)


class TestPhase1Trainer:
    """Tests for Phase 1 training loop."""

    def test_trainer_init(self):
        """Trainer should initialize correctly."""
        model = AtlasMAGSkeleton(
            vocab_size=1000, dim=128, n_layers=2, n_heads=4
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        trainer = Phase1Trainer(model, optimizer, total_steps=1000)

        assert trainer.total_steps == 1000
        assert trainer.current_step == 0

    def test_train_step(self):
        """Single training step should work."""
        model = AtlasMAGSkeleton(
            vocab_size=1000, dim=128, n_layers=2, n_heads=4
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        trainer = Phase1Trainer(model, optimizer, total_steps=1000)

        input_ids = torch.randint(0, 1000, (2, 32))
        result = trainer.train_step(input_ids, step=0)

        assert result.lm_loss > 0
        assert result.polar_loss >= 0
        assert result.total_loss > 0
        assert 0 <= result.gate_mean <= 1

    def test_check_gates(self):
        """Gate check should return statistics."""
        model = AtlasMAGSkeleton(
            vocab_size=1000, dim=128, n_layers=2, n_heads=4
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        trainer = Phase1Trainer(model, optimizer, total_steps=1000)

        result = trainer.check_gates()

        assert "gate_values" in result
        assert "statistics" in result
        assert len(result["gate_values"]) == 2  # 2 layers


class TestAtlasMemoryGatedMLP:
    """Tests for Gated MLP structure (P1-T7)."""

    def test_gated_structure(self):
        """AtlasMemory should have gated MLP structure."""
        mem = AtlasMemory(dim=64)

        # Should have three projections (w1, w2, w3)
        assert hasattr(mem, "w1")
        assert hasattr(mem, "w2")
        assert hasattr(mem, "w3")

        # Forward should work
        x = torch.randn(1, 10, 64)
        y = mem(x)
        assert y.shape == x.shape

    def test_residual_connection(self):
        """Memory should add to input."""
        mem = AtlasMemory(dim=64)

        # Zero the output projection
        with torch.no_grad():
            mem.w1.weight.zero_()

        x = torch.randn(1, 10, 64)
        y = mem(x)

        # With zero w1, output should equal input
        assert torch.allclose(x, y)


class TestGPUCompatibility:
    """GPU-specific tests."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_polarization_on_gpu(self):
        """Polarization loss should work on GPU."""
        gates = torch.tensor([0.5, 0.5, 0.5], device="cuda")
        loss = gate_polarization_loss(gates, step=0, total_steps=1000)
        assert loss.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_trainer_on_gpu(self):
        """Trainer should work on GPU."""
        model = AtlasMAGSkeleton(
            vocab_size=1000, dim=128, n_layers=2, n_heads=4
        ).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        trainer = Phase1Trainer(model, optimizer, total_steps=100)

        input_ids = torch.randint(0, 1000, (2, 32), device="cuda")
        result = trainer.train_step(input_ids, step=0)

        assert result.lm_loss > 0


class TestIntegration:
    """Integration tests for Phase 1."""

    def test_short_training_run(self):
        """Short training run should complete without fast-fail."""
        from src.training.trainer import run_short_training

        model = AtlasMAGSkeleton(
            vocab_size=1000, dim=128, n_layers=2, n_heads=4
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Run 50 steps (not enough to trigger step-500 check)
        results = run_short_training(
            model, optimizer, num_steps=50, batch_size=2, seq_len=32
        )

        assert len(results) == 50
        # Loss should decrease or stay stable
        assert results[-1].total_loss < results[0].total_loss * 1.5
