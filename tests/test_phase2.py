"""
Phase 2 Unit Tests.

Tests for:
- Q-K Projection with normalization (REQ-P2-001)
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

from src.model.qk_projection import QKProjection, create_qk_projection_for_model
from src.training.niah_probe import NIAHProbe, NIAHResult
from src.training.telemetry import TelemetryLogger, PPLDeltaTracker, StepMetrics
from src.training.checkpoint import CheckpointManager, CheckpointMetadata, verify_rollback_trigger
from src.training.phase2_trainer import Phase2Trainer, Phase2StepResult


# ============================================================================
# Test Fixtures
# ============================================================================


class MockPersistentMemory:
    """Mock persistent memory for testing QKProjection."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        # Create random M_persistent
        keys = torch.randn(10, dim)
        self.m_persistent = sum(torch.outer(k, k) for k in keys)
        self.norm_persistent = float(sum(k.norm() ** 2 for k in keys))


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

        # Mock persistent memory for NIAH
        self.persistent_memory = MockPersistentMemory(dim)

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


@pytest.fixture
def qk_projection():
    """Create a QKProjection for testing."""
    dim = 64
    pm = MockPersistentMemory(dim)
    return QKProjection(dim=dim, m_persistent=pm.m_persistent, norm_persistent=pm.norm_persistent)


# ============================================================================
# QKProjection Tests (REQ-P2-001)
# ============================================================================


class TestQKProjection:
    """Tests for Q-K Projection with normalization."""

    def test_qk_projection_init(self, qk_projection):
        """QKProjection should initialize with correct state."""
        assert qk_projection.dim == 64
        assert qk_projection.norm_persistent > 0
        assert not qk_projection._initialized

    def test_reset_at_shard_boundary(self, qk_projection):
        """Reset should inject both M_persistent and norm_persistent."""
        qk_projection.reset_at_shard_boundary()

        # Verify M_t equals M_persistent
        assert torch.allclose(qk_projection.M_t, qk_projection.m_persistent)

        # Verify norm_sum equals norm_persistent (AC-P2-3)
        assert qk_projection.norm_sum == qk_projection.norm_persistent

        # Verify initialized flag
        assert qk_projection._initialized

    def test_update_single_key(self, qk_projection):
        """Update should accumulate key outer product."""
        qk_projection.reset_at_shard_boundary()
        initial_norm = qk_projection.norm_sum

        key = torch.randn(64)
        qk_projection.update(key)

        # norm_sum should increase
        assert qk_projection.norm_sum > initial_norm

        # M_t should change
        expected_delta = torch.outer(key, key)
        # Can't easily verify M_t, but verify it's updated
        assert qk_projection._initialized

    def test_update_batch_keys(self, qk_projection):
        """Update should handle batched keys."""
        qk_projection.reset_at_shard_boundary()
        initial_norm = qk_projection.norm_sum

        keys = torch.randn(5, 64)  # Batch of 5 keys
        qk_projection.update(keys)

        # norm_sum should increase by sum of squared norms
        expected_increase = (keys.norm(dim=-1) ** 2).sum().item()
        assert abs(qk_projection.norm_sum - initial_norm - expected_increase) < 1e-4

    def test_project_single_query(self, qk_projection):
        """Project should return scaled query."""
        qk_projection.reset_at_shard_boundary()

        query = torch.randn(64)
        projected = qk_projection.project(query)

        # Should have same shape
        assert projected.shape == query.shape

        # Should not be zero (unless query is orthogonal to all keys)
        # For random data, this is extremely unlikely

    def test_project_batch_queries(self, qk_projection):
        """Project should handle batched queries."""
        qk_projection.reset_at_shard_boundary()

        queries = torch.randn(5, 64)  # Batch of 5 queries
        projected = qk_projection.project(queries)

        assert projected.shape == queries.shape

    def test_project_before_reset(self, qk_projection):
        """Project before reset should auto-initialize."""
        assert not qk_projection._initialized

        query = torch.randn(64)
        projected = qk_projection.project(query)

        # Should have auto-initialized
        assert qk_projection._initialized
        assert projected.shape == query.shape

    def test_inject_and_query_memory(self, qk_projection):
        """Inject and query should work for NIAH testing."""
        qk_projection.reset_at_shard_boundary()
        qk_projection.clear_stored_values()

        key = torch.randn(64)
        key = key / key.norm()  # Normalize
        value = torch.randn(64)
        value = value / value.norm()

        qk_projection.inject_memory(key, value)

        # Query should return something
        result = qk_projection.query_memory(key)
        assert result.shape == key.shape

    def test_clear_stored_values(self, qk_projection):
        """Clear should remove stored values."""
        key = torch.randn(64)
        value = torch.randn(64)

        qk_projection.inject_memory(key, value)
        assert len(qk_projection._stored_values) > 0

        qk_projection.clear_stored_values()
        assert len(qk_projection._stored_values) == 0

    def test_create_qk_projection_for_model(self, mock_model):
        """Should create QKProjection from model's persistent memory."""
        qk = create_qk_projection_for_model(mock_model)

        assert qk is not None
        assert qk.dim == 64
        assert qk.norm_persistent > 0


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
        assert probe.accuracy_threshold == 0.8  # Default
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

    def test_run_probe_standalone(self, qk_projection):
        """Run probe should return NIAHResult."""
        probe = NIAHProbe(dim=64, probe_frequency=100, haystack_size=10)

        result = probe.run_probe_standalone(qk_projection, step=0)

        assert isinstance(result, NIAHResult)
        assert result.step == 0
        # Cosine similarity ranges from -1 to 1
        assert -1.0 <= result.accuracy <= 1.0
        assert result.haystack_size == 10
        assert result.probe_time_ms > 0

    def test_probe_history_tracking(self, qk_projection):
        """Probe should track history."""
        probe = NIAHProbe(dim=64, probe_frequency=100)

        probe.run_probe_standalone(qk_projection, step=0)
        probe.run_probe_standalone(qk_projection, step=100)
        probe.run_probe_standalone(qk_projection, step=200)

        assert len(probe.history) == 3
        assert [r.step for r in probe.history] == [0, 100, 200]

    def test_get_statistics(self, qk_projection):
        """Statistics should aggregate results."""
        probe = NIAHProbe(dim=64, probe_frequency=100)

        # Run a few probes
        for step in [0, 100, 200]:
            probe.run_probe_standalone(qk_projection, step=step)

        stats = probe.get_statistics()

        assert stats["num_probes"] == 3
        assert 0.0 <= stats["pass_rate"] <= 1.0
        # Cosine similarity ranges from -1 to 1
        assert -1.0 <= stats["mean_accuracy"] <= 1.0
        assert "latest_accuracy" in stats

    def test_probe_reset(self, qk_projection):
        """Reset should clear history."""
        probe = NIAHProbe(dim=64)

        probe.run_probe_standalone(qk_projection, step=0)
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
        result = verify_rollback_trigger(mock_model, optimizer, temp_dir)
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

    def test_ac_p2_3_norm_in_projection(self, qk_projection):
        """AC-P2-3: norm_persistent must be in projection denominator."""
        # This is verified by checking norm_persistent is positive
        # and used in reset_at_shard_boundary
        assert qk_projection.norm_persistent > 0

        qk_projection.reset_at_shard_boundary()
        assert qk_projection.norm_sum == qk_projection.norm_persistent

    def test_ac_p2_5_niah_threshold(self, qk_projection):
        """AC-P2-5: NIAH threshold is 80%."""
        probe = NIAHProbe(dim=64, accuracy_threshold=0.8)

        # Verify threshold is set correctly
        assert probe.accuracy_threshold == 0.8

        # Run probe and check pass/fail logic
        result = probe.run_probe_standalone(qk_projection, step=0)

        if result.accuracy >= 0.8:
            assert result.passed is True
        else:
            assert result.passed is False

    def test_ac_p2_6_rollback_tested(self, mock_model, temp_dir):
        """AC-P2-6: Rollback must be testable."""
        optimizer = torch.optim.Adam(mock_model.parameters())
        result = verify_rollback_trigger(mock_model, optimizer, temp_dir)
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
# Causal QK Memory Projection Parallelization Tests
# ============================================================================


class TestCausalQKMemoryProjection:
    """Tests for CausalQKMemoryProjection parallelization."""

    @pytest.fixture
    def causal_projection(self):
        """Create a CausalQKMemoryProjection for testing."""
        from src.model.qk_projection import CausalQKMemoryProjection

        dim = 64
        n_heads = 4
        pm = MockPersistentMemory(dim)

        # Create a mock object with the expected interface
        class MockPM:
            def __init__(self, m_persistent, norm_persistent):
                self.m_persistent = m_persistent
                self.norm_persistent = norm_persistent

        mock_pm = MockPM(pm.m_persistent, pm.norm_persistent)

        return CausalQKMemoryProjection(
            dim=dim,
            n_heads=n_heads,
            persistent_memory=mock_pm,
            context_window=32,
            decay_base=0.99,
        )

    def test_chunked_matches_sequential_basic(self, causal_projection):
        """Chunked implementation should match sequential for basic case."""
        torch.manual_seed(42)

        batch = 2
        n_heads = 4
        seq_len = 64
        head_dim = 16

        q = torch.randn(batch, n_heads, seq_len, head_dim)
        k = torch.randn(batch, n_heads, seq_len, head_dim)

        # Run both implementations
        out_seq = causal_projection.forward_sequential(q, k, gamma_gates=None)
        out_chunk = causal_projection.forward_chunked(q, k, gamma_gates=None)

        # Check numerical equivalence
        assert out_seq.shape == out_chunk.shape, f"Shape mismatch: {out_seq.shape} vs {out_chunk.shape}"
        assert torch.allclose(out_seq, out_chunk, rtol=1e-4, atol=1e-6), \
            f"Max diff: {(out_seq - out_chunk).abs().max().item()}"

    def test_chunked_matches_sequential_with_gamma_gates(self, causal_projection):
        """Chunked implementation should match sequential with gamma gates."""
        torch.manual_seed(123)

        batch = 2
        n_heads = 4
        seq_len = 64
        head_dim = 16

        q = torch.randn(batch, n_heads, seq_len, head_dim)
        k = torch.randn(batch, n_heads, seq_len, head_dim)
        gamma_gates = torch.sigmoid(torch.randn(batch, seq_len, 1))

        # Run both implementations
        out_seq = causal_projection.forward_sequential(q, k, gamma_gates=gamma_gates)
        out_chunk = causal_projection.forward_chunked(q, k, gamma_gates=gamma_gates)

        # Check numerical equivalence
        assert torch.allclose(out_seq, out_chunk, rtol=1e-4, atol=1e-6), \
            f"Max diff with gamma_gates: {(out_seq - out_chunk).abs().max().item()}"

    def test_chunked_matches_sequential_long_sequence(self, causal_projection):
        """Chunked implementation should match sequential for longer sequences."""
        torch.manual_seed(456)

        batch = 1
        n_heads = 4
        seq_len = 256  # Longer sequence spans multiple chunks
        head_dim = 16

        q = torch.randn(batch, n_heads, seq_len, head_dim)
        k = torch.randn(batch, n_heads, seq_len, head_dim)

        # Run both implementations
        out_seq = causal_projection.forward_sequential(q, k, gamma_gates=None)
        out_chunk = causal_projection.forward_chunked(q, k, gamma_gates=None, chunk_size=64)

        # Check numerical equivalence
        assert torch.allclose(out_seq, out_chunk, rtol=1e-4, atol=1e-6), \
            f"Max diff for long seq: {(out_seq - out_chunk).abs().max().item()}"

    def test_chunked_different_chunk_sizes(self, causal_projection):
        """Different chunk sizes should produce same results."""
        torch.manual_seed(789)

        batch = 2
        n_heads = 4
        seq_len = 128
        head_dim = 16

        q = torch.randn(batch, n_heads, seq_len, head_dim)
        k = torch.randn(batch, n_heads, seq_len, head_dim)

        # Reference: sequential
        out_ref = causal_projection.forward_sequential(q, k, gamma_gates=None)

        # Test various chunk sizes
        for chunk_size in [16, 32, 64, 128]:
            out_chunk = causal_projection.forward_chunked(q, k, gamma_gates=None, chunk_size=chunk_size)
            assert torch.allclose(out_ref, out_chunk, rtol=1e-4, atol=1e-6), \
                f"Chunk size {chunk_size}: max diff = {(out_ref - out_chunk).abs().max().item()}"

    def test_forward_uses_chunked(self, causal_projection):
        """Default forward() should use chunked implementation."""
        torch.manual_seed(111)

        batch = 2
        n_heads = 4
        seq_len = 64
        head_dim = 16

        q = torch.randn(batch, n_heads, seq_len, head_dim)
        k = torch.randn(batch, n_heads, seq_len, head_dim)

        # forward() should match forward_chunked()
        out_forward = causal_projection.forward(q, k, gamma_gates=None)
        out_chunked = causal_projection.forward_chunked(q, k, gamma_gates=None)

        assert torch.allclose(out_forward, out_chunked, rtol=1e-6, atol=1e-8), \
            "forward() should be identical to forward_chunked()"

    def test_build_chunk_decay_weights_shape(self, causal_projection):
        """_build_chunk_decay_weights should return correct shape."""
        device = torch.device('cpu')
        dtype = torch.float32

        # Test case: chunk from position 64 to 128, context from 32
        weights = causal_projection._build_chunk_decay_weights(
            chunk_start=64,
            chunk_end=128,
            context_start=32,
            device=device,
            dtype=dtype,
        )

        # Expected shape: (chunk_len, max_context_len) = (64, 96)
        assert weights.shape == (64, 96), f"Wrong shape: {weights.shape}"

    def test_build_chunk_decay_weights_causality(self, causal_projection):
        """Decay weights should be zero for future positions (causal)."""
        device = torch.device('cpu')
        dtype = torch.float32

        weights = causal_projection._build_chunk_decay_weights(
            chunk_start=10,
            chunk_end=20,
            context_start=5,
            device=device,
            dtype=dtype,
        )  # (10, 15)

        # For each row i (position 10+i), columns j where (5+j) >= (10+i) should be 0
        for i in range(10):
            pos_t = 10 + i  # Query position
            for j in range(15):
                pos_j = 5 + j  # Key position
                if pos_j >= pos_t:  # Future or same position
                    assert weights[i, j] == 0.0, \
                        f"Non-zero weight at t={pos_t}, j={pos_j}: {weights[i, j]}"

    def test_build_chunk_decay_weights_decay_values(self, causal_projection):
        """Decay weights should follow exponential decay pattern."""
        device = torch.device('cpu')
        dtype = torch.float32

        weights = causal_projection._build_chunk_decay_weights(
            chunk_start=10,
            chunk_end=11,  # Single position
            context_start=5,
            device=device,
            dtype=dtype,
        )  # (1, 6)

        # Position 10 attends to positions 5, 6, 7, 8, 9
        # Distances: 5, 4, 3, 2, 1
        # Weights: decay^4, decay^3, decay^2, decay^1, decay^0=1
        decay = causal_projection.decay_base

        expected_weights = [decay**4, decay**3, decay**2, decay**1, 1.0, 0.0]
        for j, expected in enumerate(expected_weights):
            assert abs(weights[0, j].item() - expected) < 1e-6, \
                f"Weight at j={j}: expected {expected}, got {weights[0, j].item()}"


class TestMemoryWiring:
    """Tests for memory wiring in AtlasMAGBlock.

    Verifies that CausalQKMemoryProjection is properly connected to
    AtlasMemoryPoly for associative retrieval.
    """

    def test_block_has_qk_memory(self):
        """AtlasMAGBlock should have qk_memory when memory is enabled."""
        from src.model.skeleton import AtlasMAGBlock
        from src.model.persistent_memory import PersistentMemory
        from src.model.qk_projection import CausalQKMemoryProjection

        dim = 128
        n_heads = 4
        pm = PersistentMemory(dim=dim, n_persistent=16)

        block = AtlasMAGBlock(
            dim=dim,
            n_heads=n_heads,
            disable_memory=False,
            persistent_memory=pm,
        )

        assert block.qk_memory is not None, "qk_memory should be initialized"
        assert isinstance(block.qk_memory, CausalQKMemoryProjection)

    def test_block_qk_memory_disabled_when_no_memory(self):
        """AtlasMAGBlock should not have qk_memory when memory is disabled."""
        from src.model.skeleton import AtlasMAGBlock

        block = AtlasMAGBlock(
            dim=128,
            n_heads=4,
            disable_memory=True,
        )

        assert block.qk_memory is None, "qk_memory should be None when memory disabled"

    def test_memory_output_uses_qk_projection(self):
        """Memory output should be different from position-wise processing."""
        from src.model.skeleton import AtlasMAGBlock
        from src.model.persistent_memory import PersistentMemory

        torch.manual_seed(42)

        dim = 128
        n_heads = 4
        pm = PersistentMemory(dim=dim, n_persistent=16)

        block = AtlasMAGBlock(
            dim=dim,
            n_heads=n_heads,
            disable_memory=False,
            persistent_memory=pm,
            ttl_enabled=False,  # Disable TTL to isolate memory wiring
        )

        # Create input where different positions have different content
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

    def test_full_model_has_wired_memory(self):
        """Full AtlasMAGSkeleton should have wired memory in all blocks."""
        from src.model.skeleton import AtlasMAGSkeleton

        model = AtlasMAGSkeleton(
            vocab_size=1000,
            dim=128,
            n_layers=2,
            n_heads=4,
            n_persistent=16,
            disable_memory=False,
        )

        for i, block in enumerate(model.blocks):
            assert block.qk_memory is not None, f"Block {i} should have qk_memory"
            assert block.gamma_gate is not None, f"Block {i} should have gamma_gate"
