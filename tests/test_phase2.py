"""
Phase 2 Unit Tests.

Tests for:
- NIAH retrieval probes (REQ-P2-002)
- PPL delta telemetry (REQ-P2-003)
- Checkpoint and rollback (REQ-P2-003)
- Phase 2 training loop integration

Reference: PRD Phase 2 requirements P2-T1 through P2-T6
"""

import tempfile
import shutil
from pathlib import Path
from typing import List, Optional

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from src.training.niah_probe import NIAHProbe, NIAHResult
from src.training.telemetry import TelemetryLogger, PPLDeltaTracker, StepMetrics
from src.training.checkpoint import CheckpointManager, CheckpointMetadata, verify_rollback_trigger
from src.training.phase2_trainer import Phase2Trainer, Phase2StepResult


# ============================================================================
# Test Fixtures
# ============================================================================


class MockModel(nn.Module):
    """Mock model for testing Phase 2 trainer."""

    def __init__(self, dim: int = 64, vocab_size: int = 100, num_layers: int = 2):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_layers)
        ])
        self.head = nn.Linear(dim, vocab_size)

        # Mock gate values
        self._gate_values: List[float] = [0.5] * num_layers

        # Mock blocks with disable_memory attribute (for NIAH probe)
        class _MockBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.disable_memory = False

            def forward(self, x: Tensor) -> Tensor:
                return x

        self.blocks = nn.ModuleList([_MockBlock() for _ in range(num_layers)])

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.head(x)

    def get_gate_values(self) -> List[float]:
        return self._gate_values

    def set_gate_values(self, values: List[float]) -> None:
        self._gate_values = values


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    return MockModel(dim=64, vocab_size=100)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path, ignore_errors=True)


# ============================================================================
# NIAH Probe Tests (REQ-P2-002)
# ============================================================================


class TestNIAHProbe:
    """Tests for Needle-in-a-Haystack retrieval probes."""

    def test_niah_probe_init(self):
        """NIAHProbe should initialize with correct settings."""
        probe = NIAHProbe(dim=64, probe_frequency=100, haystack_size=50)

        assert probe.dim == 64
        assert probe.probe_frequency == 100
        assert probe.haystack_size == 50
        assert probe.accuracy_threshold == 0.1  # Default: 10% PPL reduction
        assert len(probe.history) == 0

    def test_should_probe_step_zero(self):
        """Should probe at step 0."""
        probe = NIAHProbe(dim=64, probe_frequency=100)
        assert probe.should_probe(0) is True

    def test_should_probe_frequency(self):
        """Should probe at multiples of frequency."""
        probe = NIAHProbe(dim=64, probe_frequency=100)

        assert probe.should_probe(100) is True
        assert probe.should_probe(200) is True
        assert probe.should_probe(50) is False
        assert probe.should_probe(150) is False

    def test_run_probe_standalone(self):
        """Run probe should return NIAHResult."""
        probe = NIAHProbe(dim=64, probe_frequency=100, haystack_size=5, ttl_steps=3)

        result = probe.run_probe_standalone(step=0)

        assert isinstance(result, NIAHResult)
        assert result.step == 0
        # Standalone returns dummy result (no model)
        assert result.accuracy == 0.0
        assert result.passed is False
        assert result.haystack_size == 5
        assert result.probe_time_ms > 0

    def test_probe_history_tracking(self):
        """Probe should track history."""
        probe = NIAHProbe(dim=64, probe_frequency=100, haystack_size=3, ttl_steps=2)

        probe.run_probe_standalone(step=0)
        probe.run_probe_standalone(step=100)
        probe.run_probe_standalone(step=200)

        assert len(probe.history) == 3
        assert [r.step for r in probe.history] == [0, 100, 200]

    def test_get_statistics(self):
        """Statistics should aggregate results."""
        probe = NIAHProbe(dim=64, probe_frequency=100, haystack_size=3, ttl_steps=2)

        # Run a few probes
        for step in [0, 100, 200]:
            probe.run_probe_standalone(step=step)

        stats = probe.get_statistics()

        assert stats["num_probes"] == 3
        assert stats["pass_rate"] == 0.0  # Standalone always fails
        assert stats["mean_accuracy"] == 0.0
        assert "latest_accuracy" in stats

    def test_probe_reset(self):
        """Reset should clear history."""
        probe = NIAHProbe(dim=64, haystack_size=3, ttl_steps=2)

        probe.run_probe_standalone(step=0)
        assert len(probe.history) == 1

        probe.reset()
        assert len(probe.history) == 0


# ============================================================================
# PPL Delta Tracker Tests
# ============================================================================


class TestPPLDeltaTracker:
    """Tests for perplexity delta tracking."""

    def test_tracker_init(self):
        """Tracker should initialize correctly."""
        tracker = PPLDeltaTracker(window_size=100, spike_threshold=0.05)

        assert tracker.window_size == 100
        assert tracker.spike_threshold == 0.05
        assert tracker.ema is None
        assert tracker.spike_count == 0

    def test_first_update(self):
        """First update should set EMA without spike."""
        tracker = PPLDeltaTracker()

        delta, is_spike = tracker.update(100.0)

        assert delta == 0.0
        assert is_spike is False
        assert tracker.ema == 100.0

    def test_normal_updates(self):
        """Normal updates should track EMA correctly."""
        tracker = PPLDeltaTracker(window_size=10)  # Faster EMA

        tracker.update(100.0)
        delta, is_spike = tracker.update(95.0)

        # Delta should be negative (improvement)
        assert delta < 0
        assert is_spike is False

    def test_spike_detection(self):
        """Large increase should trigger spike."""
        tracker = PPLDeltaTracker(spike_threshold=0.05)

        # Initialize with stable value
        tracker.update(100.0)
        for _ in range(10):
            tracker.update(100.0)

        # 10% spike should trigger
        delta, is_spike = tracker.update(110.0)

        assert delta > 0.05
        assert is_spike is True
        assert tracker.spike_count == 1

    def test_statistics(self):
        """Statistics should aggregate history."""
        tracker = PPLDeltaTracker()

        for ppl in [100.0, 95.0, 90.0, 85.0]:
            tracker.update(ppl)

        stats = tracker.get_statistics()

        assert stats["count"] == 4
        assert stats["min_ppl"] == 85.0
        assert stats["max_ppl"] == 100.0

    def test_reset(self):
        """Reset should clear state."""
        tracker = PPLDeltaTracker()

        tracker.update(100.0)
        tracker.reset()

        assert tracker.ema is None
        assert len(tracker.history) == 0


# ============================================================================
# Telemetry Logger Tests
# ============================================================================


class TestTelemetryLogger:
    """Tests for training telemetry."""

    def test_logger_init(self, temp_dir):
        """Logger should initialize correctly."""
        logger = TelemetryLogger(output_dir=temp_dir, log_frequency=10)

        assert logger.output_dir == temp_dir
        assert logger.log_frequency == 10
        assert len(logger.all_metrics) == 0

    def test_log_step(self, temp_dir):
        """Log step should record metrics."""
        logger = TelemetryLogger(output_dir=temp_dir)

        metrics = logger.log_step(
            step=0,
            lm_loss=5.0,
            polar_loss=0.1,
            total_loss=5.1,
            gate_mean=0.5,
            gate_std=0.1,
            polarized_ratio=0.2,
            learning_rate=1e-4,
        )

        assert isinstance(metrics, StepMetrics)
        assert metrics.step == 0
        assert metrics.lm_loss == 5.0
        assert len(logger.all_metrics) == 1

    def test_log_with_niah(self, temp_dir):
        """Log step should include NIAH accuracy."""
        logger = TelemetryLogger(output_dir=temp_dir)

        metrics = logger.log_step(
            step=0,
            lm_loss=5.0,
            polar_loss=0.1,
            total_loss=5.1,
            gate_mean=0.5,
            gate_std=0.1,
            polarized_ratio=0.2,
            learning_rate=1e-4,
            niah_accuracy=0.85,
        )

        assert metrics.niah_accuracy == 0.85

    def test_ppl_delta_tracked(self, temp_dir):
        """PPL delta should be computed."""
        logger = TelemetryLogger(output_dir=temp_dir)

        # Log multiple steps
        for i in range(5):
            metrics = logger.log_step(
                step=i,
                lm_loss=5.0 - i * 0.1,  # Decreasing loss
                polar_loss=0.1,
                total_loss=5.1 - i * 0.1,
                gate_mean=0.5,
                gate_std=0.1,
                polarized_ratio=0.2,
                learning_rate=1e-4,
            )

        # Later steps should have ppl_delta
        assert logger.all_metrics[-1].ppl_delta is not None

    def test_get_summary(self, temp_dir):
        """Summary should aggregate metrics."""
        logger = TelemetryLogger(output_dir=temp_dir)

        for i in range(10):
            logger.log_step(
                step=i,
                lm_loss=5.0 - i * 0.1,
                polar_loss=0.1,
                total_loss=5.1 - i * 0.1,
                gate_mean=0.5,
                gate_std=0.1,
                polarized_ratio=0.2,
                learning_rate=1e-4,
            )

        summary = logger.get_summary()

        assert summary["num_steps"] == 10
        assert "final_loss" in summary
        assert "min_loss" in summary

    def test_flush_writes_file(self, temp_dir):
        """Flush should write metrics to JSONL."""
        logger = TelemetryLogger(output_dir=temp_dir, log_frequency=100)

        for i in range(5):
            logger.log_step(
                step=i,
                lm_loss=5.0,
                polar_loss=0.1,
                total_loss=5.1,
                gate_mean=0.5,
                gate_std=0.1,
                polarized_ratio=0.2,
                learning_rate=1e-4,
            )

        logger.flush()

        # Check file was written
        assert logger.metrics_file.exists()

    def test_check_ppl_delta_visible(self, temp_dir):
        """Should verify ppl_delta in JSONL."""
        logger = TelemetryLogger(output_dir=temp_dir, log_frequency=1)

        logger.log_step(
            step=0,
            lm_loss=5.0,
            polar_loss=0.1,
            total_loss=5.1,
            gate_mean=0.5,
            gate_std=0.1,
            polarized_ratio=0.2,
            learning_rate=1e-4,
        )

        # Should find ppl_delta in file
        assert logger.check_ppl_delta_visible() is True


# ============================================================================
# Checkpoint Manager Tests
# ============================================================================


class TestCheckpointManager:
    """Tests for checkpoint and rollback."""

    def test_manager_init(self, temp_dir):
        """Manager should initialize correctly."""
        manager = CheckpointManager(output_dir=temp_dir, keep_last=3)

        assert manager.output_dir == temp_dir
        assert manager.keep_last == 3
        assert len(manager.checkpoints) == 0

    def test_should_save(self):
        """Should save at correct intervals."""
        manager = CheckpointManager(
            output_dir=Path(tempfile.mkdtemp()),
            save_frequency=100
        )

        assert manager.should_save(0) is True
        assert manager.should_save(100) is True
        assert manager.should_save(50) is False

    def test_save_checkpoint(self, temp_dir, mock_model):
        """Save should create checkpoint file."""
        manager = CheckpointManager(output_dir=temp_dir)
        optimizer = torch.optim.Adam(mock_model.parameters())

        metadata = manager.save(
            step=100,
            model=mock_model,
            optimizer=optimizer,
            loss=5.0,
            perplexity=150.0,
        )

        assert isinstance(metadata, CheckpointMetadata)
        assert metadata.step == 100
        assert metadata.path.exists()
        assert len(manager.checkpoints) == 1

    def test_rollback(self, temp_dir, mock_model):
        """Rollback should restore model state."""
        manager = CheckpointManager(output_dir=temp_dir)
        optimizer = torch.optim.Adam(mock_model.parameters())

        # Get initial state
        initial_param = next(mock_model.parameters()).clone()

        # Save checkpoint
        manager.save(step=100, model=mock_model, optimizer=optimizer, loss=5.0, perplexity=150.0)

        # Modify model
        with torch.no_grad():
            for p in mock_model.parameters():
                p.add_(torch.randn_like(p) * 0.5)

        # Verify model changed
        modified_param = next(mock_model.parameters())
        assert not torch.allclose(initial_param, modified_param)

        # Rollback
        restored_step = manager.rollback(mock_model, optimizer)

        assert restored_step == 100
        assert manager.rollback_count == 1

        # Verify model restored
        restored_param = next(mock_model.parameters())
        assert torch.allclose(initial_param, restored_param, atol=1e-6)

    def test_rollback_to_second_last(self, temp_dir, mock_model):
        """Rollback should use second-most-recent by default."""
        manager = CheckpointManager(output_dir=temp_dir)
        optimizer = torch.optim.Adam(mock_model.parameters())

        # Save two checkpoints
        manager.save(step=100, model=mock_model, optimizer=optimizer, loss=5.0, perplexity=150.0)
        manager.save(step=200, model=mock_model, optimizer=optimizer, loss=4.0, perplexity=100.0)

        # Rollback should go to step 100
        restored_step = manager.rollback(mock_model, optimizer)
        assert restored_step == 100

    def test_cleanup_old_checkpoints(self, temp_dir, mock_model):
        """Should cleanup checkpoints beyond keep_last."""
        manager = CheckpointManager(output_dir=temp_dir, keep_last=2)
        optimizer = torch.optim.Adam(mock_model.parameters())

        # Save 5 checkpoints
        for step in [100, 200, 300, 400, 500]:
            manager.save(step=step, model=mock_model, optimizer=optimizer, loss=5.0, perplexity=150.0)

        # Should only keep last 2
        assert len(manager.checkpoints) == 2
        assert manager.checkpoints[0].step == 400
        assert manager.checkpoints[1].step == 500

    def test_get_statistics(self, temp_dir, mock_model):
        """Statistics should report state."""
        manager = CheckpointManager(output_dir=temp_dir)
        optimizer = torch.optim.Adam(mock_model.parameters())

        manager.save(step=100, model=mock_model, optimizer=optimizer, loss=5.0, perplexity=150.0)

        stats = manager.get_statistics()

        assert stats["num_checkpoints"] == 1
        assert stats["latest_step"] == 100
        assert stats["rollback_count"] == 0

    def verify_rollback_trigger_function(self, mock_model, temp_dir):
        """verify_rollback_trigger should pass."""
        optimizer = torch.optim.Adam(mock_model.parameters())
        result = verify_rollback_trigger(mock_model, optimizer)
        assert result is True


# ============================================================================
# Phase 2 Trainer Tests
# ============================================================================


class TestPhase2Trainer:
    """Tests for Phase 2 training loop."""

    def test_trainer_init(self, mock_model, temp_dir):
        """Trainer should initialize correctly."""
        optimizer = torch.optim.Adam(mock_model.parameters())
        trainer = Phase2Trainer(
            model=mock_model,
            optimizer=optimizer,
            total_steps=1000,
            output_dir=temp_dir,
        )

        assert trainer.total_steps == 1000
        assert trainer.current_step == 0

    def test_single_train_step(self, mock_model, temp_dir):
        """Single training step should work."""
        optimizer = torch.optim.Adam(mock_model.parameters())
        trainer = Phase2Trainer(
            model=mock_model,
            optimizer=optimizer,
            total_steps=100,
            output_dir=temp_dir,
        )

        input_ids = torch.randint(0, 100, (4, 32))
        result = trainer.train_step(input_ids, step=0)

        assert isinstance(result, Phase2StepResult)
        assert result.step == 0
        assert result.lm_loss > 0
        assert result.total_loss > 0

    def test_multiple_steps(self, mock_model, temp_dir):
        """Multiple training steps should accumulate."""
        optimizer = torch.optim.Adam(mock_model.parameters())
        trainer = Phase2Trainer(
            model=mock_model,
            optimizer=optimizer,
            total_steps=100,
            output_dir=temp_dir,
            niah_frequency=50,
            checkpoint_frequency=25,
            log_interval=10,
        )

        for step in range(10):
            input_ids = torch.randint(0, 100, (4, 32))
            result = trainer.train_step(input_ids, step=step)

        assert len(trainer.results) == 10
        assert trainer.current_step == 10

    def test_build_summary(self, mock_model, temp_dir):
        """Summary should aggregate training results."""
        optimizer = torch.optim.Adam(mock_model.parameters())
        trainer = Phase2Trainer(
            model=mock_model,
            optimizer=optimizer,
            total_steps=100,
            output_dir=temp_dir,
        )

        for step in range(5):
            input_ids = torch.randint(0, 100, (4, 32))
            trainer.train_step(input_ids, step=step)

        summary = trainer._build_summary()

        assert summary["completed_steps"] == 5
        assert "final_loss" in summary
        assert "niah_stats" in summary


# ============================================================================
# Acceptance Criteria Tests
# ============================================================================


class TestAcceptanceCriteria:
    """Tests for Phase 2 acceptance criteria."""

    def test_ac_p2_5_niah_threshold(self):
        """AC-P2-5: NIAH threshold is 80%."""
        probe = NIAHProbe(dim=64, accuracy_threshold=0.8, haystack_size=3, ttl_steps=2)

        # Verify threshold is set correctly
        assert probe.accuracy_threshold == 0.8

        # Run probe and check pass/fail logic
        result = probe.run_probe_standalone(step=0)

        if result.accuracy >= 0.8:
            assert result.passed is True
        else:
            assert result.passed is False

    def test_ac_p2_6_rollback_tested(self, mock_model, temp_dir):
        """AC-P2-6: Rollback must be testable."""
        optimizer = torch.optim.Adam(mock_model.parameters())
        result = verify_rollback_trigger(mock_model, optimizer)
        assert result is True

    def test_ac_p2_7_ppl_delta_visible(self, temp_dir):
        """AC-P2-7: PPL delta must appear in telemetry."""
        logger = TelemetryLogger(output_dir=temp_dir, log_frequency=1)

        # Log a step
        logger.log_step(
            step=0,
            lm_loss=5.0,
            polar_loss=0.1,
            total_loss=5.1,
            gate_mean=0.5,
            gate_std=0.1,
            polarized_ratio=0.2,
            learning_rate=1e-4,
        )

        # Verify ppl_delta is in the logged data
        assert logger.check_ppl_delta_visible() is True


# ============================================================================
# Integration Tests
# ============================================================================


class TestPhase2Integration:
    """Integration tests for Phase 2 components."""

    def test_full_training_loop(self, mock_model, temp_dir):
        """Full training loop should run without errors."""
        optimizer = torch.optim.Adam(mock_model.parameters())
        trainer = Phase2Trainer(
            model=mock_model,
            optimizer=optimizer,
            total_steps=20,
            output_dir=temp_dir,
            niah_frequency=10,
            checkpoint_frequency=10,
            log_interval=5,
        )

        def data_generator():
            for _ in range(25):
                yield torch.randint(0, 100, (4, 32))

        summary = trainer.run_training(data_generator(), device="cpu")

        assert summary["completed_steps"] == 20
        assert "niah_stats" in summary
        assert "checkpoint_stats" in summary

    def test_spike_detection_integration(self, temp_dir):
        """PPL spike should be detected in telemetry."""
        logger = TelemetryLogger(output_dir=temp_dir, spike_threshold=0.1)

        # Simulate stable training
        for i in range(10):
            logger.log_step(
                step=i,
                lm_loss=5.0,
                polar_loss=0.1,
                total_loss=5.1,
                gate_mean=0.5,
                gate_std=0.1,
                polarized_ratio=0.2,
                learning_rate=1e-4,
            )

        # Simulate spike (25% increase in loss)
        spike_metrics = logger.log_step(
            step=10,
            lm_loss=6.25,  # 25% increase
            polar_loss=0.1,
            total_loss=6.35,
            gate_mean=0.5,
            gate_std=0.1,
            polarized_ratio=0.2,
            learning_rate=1e-4,
        )

        assert spike_metrics.is_spike is True


# ============================================================================
# Memory Wiring Tests (Updated for direct AtlasMemoryPoly)
# ============================================================================


class TestMemoryWiring:
    """Tests for memory wiring in AtlasMAGBlock.

    Verifies that AtlasMemoryPoly is directly wired to the block
    without CausalQKMemoryProjection intermediary.
    """

    def test_block_has_no_qk_memory(self):
        """AtlasMAGBlock should NOT have qk_memory attribute."""
        from src.model.skeleton import AtlasMAGBlock

        block = AtlasMAGBlock(
            dim=128,
            n_heads=4,
            disable_memory=False,
        )

        assert not hasattr(block, 'qk_memory') or block.qk_memory is None

    def test_block_has_memory_module(self):
        """AtlasMAGBlock should have memory module when enabled."""
        from src.model.skeleton import AtlasMAGBlock

        block = AtlasMAGBlock(
            dim=128,
            n_heads=4,
            disable_memory=False,
        )

        assert block.memory is not None

    def test_memory_output_contributes(self):
        """Memory should contribute to output."""
        from src.model.skeleton import AtlasMAGBlock

        torch.manual_seed(42)

        dim = 128
        n_heads = 4

        block = AtlasMAGBlock(
            dim=dim,
            n_heads=n_heads,
            disable_memory=False,
            ttl_enabled=False,
        )

        batch = 2
        seq_len = 64
        x = torch.randn(batch, seq_len, dim)

        # Forward pass with memory
        with torch.no_grad():
            out_with_memory, _ = block(x)

        # For comparison, run without memory
        block_no_mem = AtlasMAGBlock(
            dim=dim,
            n_heads=n_heads,
            disable_memory=True,
        )
        # Copy attention weights to ensure only memory differs
        block_no_mem.qkv.load_state_dict(block.qkv.state_dict())
        block_no_mem.w_o.load_state_dict(block.w_o.state_dict())
        block_no_mem.ffn.load_state_dict(block.ffn.state_dict())
        block_no_mem.norm1.load_state_dict(block.norm1.state_dict())
        block_no_mem.norm2.load_state_dict(block.norm2.state_dict())

        with torch.no_grad():
            out_no_memory, _ = block_no_mem(x)

        # Memory should make a difference
        diff = (out_with_memory - out_no_memory).abs().mean()
        assert diff > 1e-6, f"Memory should contribute to output, diff={diff}"

    def test_low_rank_poly_compress(self):
        """Low-rank compression should reduce MLP input dim from poly_dim to poly_rank."""
        from src.model.atlas_memory import AtlasMemoryPoly

        mem = AtlasMemoryPoly(dim=128, key_dim=64, poly_degree=2, poly_rank=128)

        # poly_dim for key_dim=64, degree=2 is 64 + 64*65/2 = 2144
        assert mem.poly_dim == 2144
        assert mem.poly_compress is not None
        assert mem.poly_compress.weight.shape == (128, 2144)
        # w2/w3 should use compressed dim
        assert mem.w2.weight.shape[1] == 128

        # Forward pass should work
        x = torch.randn(1, 8, 128)
        out = mem(x, return_contribution=True)
        assert out.shape == (1, 8, 128)

    def test_no_compression_when_rank_zero(self):
        """poly_rank=0 should disable compression."""
        from src.model.atlas_memory import AtlasMemoryPoly

        mem = AtlasMemoryPoly(dim=128, key_dim=64, poly_degree=2, poly_rank=0)
        assert mem.poly_compress is None
        # w2 input should be full poly_dim
        assert mem.w2.weight.shape[1] == mem.poly_dim

    def test_freeze_unfreeze_static_weights(self):
        """freeze/unfreeze should toggle requires_grad on all params."""
        from src.model.atlas_memory import AtlasMemoryPoly

        mem = AtlasMemoryPoly(dim=64, key_dim=32, poly_degree=2, poly_rank=64)

        # All params should start as requires_grad=True
        for p in mem.parameters():
            assert p.requires_grad is True

        mem.freeze_static_weights()
        for p in mem.parameters():
            assert p.requires_grad is False

        mem.unfreeze_static_weights()
        for p in mem.parameters():
            assert p.requires_grad is True

    def test_full_model_has_memory(self):
        """Full AtlasMAGSkeleton should have memory in all blocks."""
        from src.model.skeleton import AtlasMAGSkeleton

        model = AtlasMAGSkeleton(
            vocab_size=1000,
            dim=128,
            n_layers=2,
            n_heads=4,
            disable_memory=False,
        )

        for i, block in enumerate(model.blocks):
            assert block.memory is not None, f"Block {i} should have memory"
            assert block.gamma_gate is not None, f"Block {i} should have gamma_gate"
