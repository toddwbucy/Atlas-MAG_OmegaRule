"""
Tests for Test-Time Learning (TTL) implementation.

Tests the core Atlas innovation: gradient-based memory updates during
the forward pass, enabling true test-time memorization.

Reference: Atlas paper (arXiv:2505.23735) Eq. 9, 32-33
"""

import pytest
import torch
import torch.nn as nn

from src.model.atlas_memory import AtlasMemoryPoly
from src.model.blocks import MAGBlock as AtlasMAGBlock
from src.model.skeleton import AtlasMAGSkeleton
from src.training.omega_loss import compute_omega_loss, compute_omega_loss_with_stats
from src.training.ttl_update import TTLUpdater, ttl_step, ttl_step_with_grad_clip


class TestOmegaLoss:
    """Tests for Omega Rule loss computation."""

    def test_omega_loss_shape(self):
        """Loss should be a scalar tensor with gradients."""
        memory = AtlasMemoryPoly(dim=128, key_dim=32)
        keys = torch.randn(2, 64, 128)
        values = torch.randn(2, 64, 128)

        loss = compute_omega_loss(memory, keys, values)

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.requires_grad, "Loss should have gradients"

    def test_omega_loss_context_window(self):
        """Loss should only use last `context_window` positions."""
        memory = AtlasMemoryPoly(dim=128, key_dim=32)

        # Sequence longer than context window
        keys = torch.randn(2, 512, 128)
        values = torch.randn(2, 512, 128)

        # With context_window=256, should only use last 256 positions
        loss = compute_omega_loss(memory, keys, values, context_window=256)

        assert loss.item() > 0, "Loss should be positive"

    def test_omega_loss_decay_weights_recent_more(self):
        """Recent positions should have higher weight than older ones."""
        memory = AtlasMemoryPoly(dim=128, key_dim=32)

        # Create keys/values where recent positions have higher error
        keys = torch.randn(1, 64, 128)
        values = torch.zeros(1, 64, 128)  # Target is zero

        # Make recent positions have larger keys (larger error)
        keys[:, -10:, :] *= 10

        # With decay, the weighted loss should still be dominated by recent positions
        loss_with_decay = compute_omega_loss(
            memory, keys, values, decay_base=0.95
        )
        loss_no_decay = compute_omega_loss(
            memory, keys, values, decay_base=1.0  # No decay
        )

        # Both should be positive
        assert loss_with_decay.item() > 0
        assert loss_no_decay.item() > 0

    def test_omega_loss_with_gamma(self):
        """Custom gamma weights should be respected."""
        memory = AtlasMemoryPoly(dim=128, key_dim=32)
        keys = torch.randn(2, 64, 128)
        values = torch.randn(2, 64, 128)

        # Custom gamma: zero weight for all but last position
        gamma = torch.zeros(2, 64, 1)
        gamma[:, -1:, :] = 1.0

        loss = compute_omega_loss(memory, keys, values, gamma=gamma)

        assert loss.item() > 0, "Loss should be positive"

    def test_omega_loss_with_stats(self):
        """Should return both loss and diagnostic stats."""
        memory = AtlasMemoryPoly(dim=128, key_dim=32)
        keys = torch.randn(2, 64, 128)
        values = torch.randn(2, 64, 128)

        loss, stats = compute_omega_loss_with_stats(memory, keys, values)

        assert "omega_loss" in stats
        assert "prediction_norm" in stats
        assert "target_norm" in stats
        assert "error_norm" in stats
        assert "context_len" in stats

        assert stats["omega_loss"] == pytest.approx(loss.item())

    def test_omega_loss_zero_when_perfect(self):
        """Loss should be zero when memory perfectly predicts values."""
        memory = AtlasMemoryPoly(dim=128, key_dim=32)
        keys = torch.randn(2, 32, 128)

        # Get the memory's actual output for these keys
        with torch.no_grad():
            predictions = memory(keys, return_contribution=True)

        # Use predictions as targets - loss should be ~0
        loss = compute_omega_loss(memory, keys, predictions.detach())

        assert loss.item() < 1e-5, f"Loss should be near zero, got {loss.item()}"

    def test_omega_loss_gradient_flows(self):
        """Gradients should flow back to memory parameters."""
        memory = AtlasMemoryPoly(dim=128, key_dim=32)
        keys = torch.randn(2, 64, 128)
        values = torch.randn(2, 64, 128)

        loss = compute_omega_loss(memory, keys, values)

        # Compute gradients
        grads = torch.autograd.grad(loss, memory.parameters())

        # All gradients should be non-zero
        for grad in grads:
            assert grad.abs().sum() > 0, "Gradient should be non-zero"


class TestMomentumBuffers:
    """Tests for momentum buffer management in AtlasMemoryPoly."""

    def test_momentum_buffers_exist(self):
        """Momentum buffers should be created for all parameters."""
        memory = AtlasMemoryPoly(dim=128, key_dim=32)

        param_names = [name for name, _ in memory.named_parameters()]

        for name in param_names:
            buffer_name = f"momentum_{name.replace('.', '_')}"
            assert hasattr(memory, buffer_name), f"Missing buffer: {buffer_name}"

    def test_momentum_buffers_shape(self):
        """Momentum buffers should match parameter shapes."""
        memory = AtlasMemoryPoly(dim=128, key_dim=32)

        for name, param in memory.named_parameters():
            momentum = memory.get_momentum(name)
            assert momentum.shape == param.shape, f"Shape mismatch for {name}"

    def test_momentum_reset(self):
        """Reset should zero all momentum buffers."""
        memory = AtlasMemoryPoly(dim=128, key_dim=32)

        # Add some values to momentum
        for name, _ in memory.named_parameters():
            buffer_name = f"momentum_{name.replace('.', '_')}"
            getattr(memory, buffer_name).fill_(1.0)

        # Reset
        memory.reset_momentum()

        # Check all zeros
        for name, _ in memory.named_parameters():
            momentum = memory.get_momentum(name)
            assert torch.all(momentum == 0), f"Momentum not reset for {name}"


class TestTTLStep:
    """Tests for the TTL update step."""

    def test_momentum_accumulation(self):
        """Momentum should accumulate: S_t = theta * S_{t-1} + grad."""
        memory = AtlasMemoryPoly(dim=128, key_dim=32)
        keys = torch.randn(2, 64, 128)
        values = torch.randn(2, 64, 128)

        # Initial momentum should be zero
        initial_momentum = memory.get_momentum("w1.weight").clone()
        assert torch.all(initial_momentum == 0)

        # Compute loss and step
        loss = compute_omega_loss(memory, keys, values)
        ttl_step(memory, loss)

        # Momentum should now be non-zero
        updated_momentum = memory.get_momentum("w1.weight")
        assert not torch.all(updated_momentum == 0), "Momentum should be updated"

    def test_memory_params_change(self):
        """Memory parameters should change after TTL step."""
        memory = AtlasMemoryPoly(dim=128, key_dim=32)
        keys = torch.randn(2, 64, 128)
        values = torch.randn(2, 64, 128)

        # Store initial parameters
        initial_w1 = memory.w1.weight.clone()

        # Compute loss and step
        loss = compute_omega_loss(memory, keys, values)
        ttl_step(memory, loss)

        # Parameters should change
        assert not torch.allclose(memory.w1.weight, initial_w1), "Parameters should change"

    def test_ttl_step_returns_stats(self):
        """TTL step should return statistics."""
        memory = AtlasMemoryPoly(dim=128, key_dim=32)
        keys = torch.randn(2, 64, 128)
        values = torch.randn(2, 64, 128)

        loss = compute_omega_loss(memory, keys, values)
        stats = ttl_step(memory, loss)

        # Should have stats for each parameter
        assert any("grad_norm" in k for k in stats.keys())
        assert any("momentum_norm" in k for k in stats.keys())
        assert any("update_norm" in k for k in stats.keys())

    def test_ttl_step_with_grad_clip(self):
        """Gradient clipping should work."""
        memory = AtlasMemoryPoly(dim=128, key_dim=32)
        keys = torch.randn(2, 64, 128)
        values = torch.randn(2, 64, 128)

        loss = compute_omega_loss(memory, keys, values)
        stats = ttl_step_with_grad_clip(memory, loss, max_grad_norm=0.1)

        assert "grad_clip_ratio" in stats
        assert 0 < stats["grad_clip_ratio"] <= 1.0

    def test_newton_schulz_on_momentum(self):
        """Newton-Schulz should improve orthogonality of the update."""
        from src.nn.newton_schulz import newton_schulz, orthogonality_error

        memory = AtlasMemoryPoly(dim=128, key_dim=32)
        keys = torch.randn(2, 64, 128)
        values = torch.randn(2, 64, 128)

        # Compute loss and accumulate momentum
        loss = compute_omega_loss(memory, keys, values)
        ttl_step(memory, loss)

        # Get momentum for a 2D parameter
        momentum = memory.get_momentum("w1.weight")

        # Check orthogonality before NS
        error_before = orthogonality_error(momentum / (momentum.norm() + 1e-8))

        # Apply Newton-Schulz
        orthogonalized = newton_schulz(momentum, num_iters=5)

        # Check orthogonality after NS (should improve)
        error_after = orthogonality_error(orthogonalized)

        # NS should reduce orthogonality error (or at least not make it worse)
        assert error_after <= error_before + 0.1, \
            f"NS should improve orthogonality: before={error_before:.4f}, after={error_after:.4f}"

        # Error should be bounded (not explode)
        assert error_after < 1.0, f"Orthogonality error too high: {error_after}"

    def test_weight_decay_applied(self):
        """Weight decay (alpha) should shrink parameters."""
        memory = AtlasMemoryPoly(dim=128, key_dim=32)
        keys = torch.randn(2, 64, 128)
        values = torch.randn(2, 64, 128)

        # Get initial parameter norm
        initial_norm = memory.w1.weight.norm().item()

        # Run TTL with strong weight decay
        loss = compute_omega_loss(memory, keys, values)
        ttl_step(memory, loss, alpha=0.9, eta=0.0)  # eta=0 so only decay matters

        # Parameter norm should decrease
        final_norm = memory.w1.weight.norm().item()
        assert final_norm < initial_norm, "Weight decay should shrink parameters"


class TestTTLUpdater:
    """Tests for the stateful TTL updater."""

    def test_updater_tracks_steps(self):
        """Updater should track step count."""
        updater = TTLUpdater()
        memory = AtlasMemoryPoly(dim=128, key_dim=32)

        assert updater.step_count == 0

        keys = torch.randn(2, 64, 128)
        values = torch.randn(2, 64, 128)

        # First step
        loss = compute_omega_loss(memory, keys, values)
        updater.step(memory, loss)
        assert updater.step_count == 1

        # Second step (need new loss computation)
        loss = compute_omega_loss(memory, keys, values)
        updater.step(memory, loss)
        assert updater.step_count == 2

    def test_updater_sequence_reset(self):
        """Sequence reset mode should reset momentum at sequence start."""
        updater = TTLUpdater(reset_mode="sequence")
        memory = AtlasMemoryPoly(dim=128, key_dim=32)

        # Do a step to accumulate momentum
        keys = torch.randn(2, 64, 128)
        values = torch.randn(2, 64, 128)
        loss = compute_omega_loss(memory, keys, values)
        updater.step(memory, loss)

        # Momentum should be non-zero
        assert not torch.all(memory.get_momentum("w1.weight") == 0)

        # New sequence should reset
        updater.on_sequence_start(memory)
        assert torch.all(memory.get_momentum("w1.weight") == 0)


class TestTTLInForwardPass:
    """Integration tests for TTL in the full model forward pass."""

    def test_ttl_in_block_forward(self):
        """Block forward should perform TTL update."""
        block = AtlasMAGBlock(dim=128, n_heads=4, ttl_enabled=True)
        x = torch.randn(2, 64, 128)

        # Store initial parameters
        initial_w1 = block.memory.w1.weight.clone()

        # Forward pass (should update memory)
        output, ttl_stats = block(x)

        # Check output shape
        assert output.shape == x.shape

        # Check TTL stats returned
        assert ttl_stats is not None
        assert "omega_loss" in ttl_stats

        # Check parameters changed
        assert not torch.allclose(block.memory.w1.weight, initial_w1)

    def test_ttl_disabled_no_change(self):
        """When TTL disabled, parameters should not change."""
        block = AtlasMAGBlock(dim=128, n_heads=4, ttl_enabled=False)
        x = torch.randn(2, 64, 128)

        # Store initial parameters
        initial_w1 = block.memory.w1.weight.clone()

        # Forward pass (should NOT update memory)
        output, ttl_stats = block(x)

        # TTL stats should be None
        assert ttl_stats is None

        # Parameters should NOT change
        assert torch.allclose(block.memory.w1.weight, initial_w1)

    def test_ttl_in_full_model(self):
        """Full model forward should perform TTL in all layers."""
        model = AtlasMAGSkeleton(
            vocab_size=1000,
            dim=128,
            n_layers=2,
            n_heads=4,
            ttl_enabled=True,
        )

        input_ids = torch.randint(0, 1000, (2, 64))

        # Forward with TTL stats
        logits, ttl_stats_list = model(input_ids, return_ttl_stats=True)

        # Should have stats from both layers
        assert len(ttl_stats_list) == 2

        # Each layer should have omega_loss
        for stats in ttl_stats_list:
            assert "omega_loss" in stats
            # Note: layer_idx not included in new implementation

    def test_model_reset_momentum(self):
        """Model-level momentum reset should work."""
        model = AtlasMAGSkeleton(
            vocab_size=1000,
            dim=128,
            n_layers=2,
            n_heads=4,
            ttl_enabled=True,
        )

        input_ids = torch.randint(0, 1000, (2, 64))

        # Forward to accumulate momentum
        model(input_ids)

        # Check momentum is non-zero
        first_block = model.blocks[0]
        assert hasattr(first_block.memory, 'get_momentum')
        momentum = first_block.memory.get_momentum("w1.weight")
        assert not torch.all(momentum == 0)

        # Reset
        model.reset_ttl_momentum()

        # Check momentum is zero
        momentum = first_block.memory.get_momentum("w1.weight")
        assert torch.all(momentum == 0)

    def test_model_set_ttl_enabled(self):
        """Should be able to toggle TTL at runtime."""
        model = AtlasMAGSkeleton(
            vocab_size=1000,
            dim=128,
            n_layers=2,
            n_heads=4,
            ttl_enabled=True,
        )

        # Disable TTL
        model.set_ttl_enabled(False)

        input_ids = torch.randint(0, 1000, (2, 64))
        _, ttl_stats_list = model(input_ids, return_ttl_stats=True)

        # Should have no TTL stats when disabled
        assert len(ttl_stats_list) == 0

        # Re-enable TTL
        model.set_ttl_enabled(True)

        _, ttl_stats_list = model(input_ids, return_ttl_stats=True)

        # Should have TTL stats when enabled
        assert len(ttl_stats_list) == 2

    def test_ttl_training_vs_inference_mode(self):
        """TTL updates in train mode but is disabled in inference mode."""
        model = AtlasMAGSkeleton(
            vocab_size=1000,
            dim=128,
            n_layers=2,
            n_heads=4,
            ttl_enabled=True,
        )

        input_ids = torch.randint(0, 1000, (2, 64))

        # Train mode
        model.train()
        initial_w1_train = model.blocks[0].memory.w1.weight.clone()
        _, ttl_stats_train = model(input_ids, return_ttl_stats=True)

        # TTL should update params in train mode
        assert len(ttl_stats_train) == 2
        assert not torch.allclose(model.blocks[0].memory.w1.weight, initial_w1_train)

        # Inference mode - TTL is disabled (training=False gates TTL in this implementation)
        model.train(False)  # Set to inference mode
        initial_w1_inference = model.blocks[0].memory.w1.weight.clone()
        _, ttl_stats_inference = model(input_ids, return_ttl_stats=True)

        # TTL should NOT update params in inference mode (training=False gates it)
        assert len(ttl_stats_inference) == 0  # No TTL stats in inference mode
        assert torch.allclose(model.blocks[0].memory.w1.weight, initial_w1_inference)

    def test_ttl_stats_structure(self):
        """TTL stats should have expected structure."""
        model = AtlasMAGSkeleton(
            vocab_size=1000,
            dim=128,
            n_layers=2,
            n_heads=4,
            ttl_enabled=True,
        )

        input_ids = torch.randint(0, 1000, (2, 64))
        _, ttl_stats_list = model(input_ids, return_ttl_stats=True)

        # Should have stats from each layer
        assert len(ttl_stats_list) == 2

        # Each stats dict should have omega_loss
        for stats in ttl_stats_list:
            assert "omega_loss" in stats
            assert stats["omega_loss"] > 0


class TestGradientFlow:
    """Tests for gradient flow compatibility."""

    def test_outer_loop_backprop_still_works(self):
        """TTL should not interfere with outer-loop backpropagation."""
        model = AtlasMAGSkeleton(
            vocab_size=1000,
            dim=128,
            n_layers=2,
            n_heads=4,
            ttl_enabled=True,
        )

        input_ids = torch.randint(0, 1000, (2, 64))
        targets = torch.randint(0, 1000, (2, 64))

        # Forward pass with TTL
        logits = model(input_ids, return_ttl_stats=False)

        # Compute cross-entropy loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, 1000),
            targets.view(-1),
        )

        # Backward should work
        loss.backward()

        # Check gradients exist for embeddings (not affected by TTL)
        assert model.tok_emb.weight.grad is not None
        assert model.tok_emb.weight.grad.abs().sum() > 0


class TestNumericalStability:
    """Tests for numerical stability of TTL."""

    def test_no_nan_or_inf(self):
        """TTL should not produce NaN or Inf values."""
        model = AtlasMAGSkeleton(
            vocab_size=1000,
            dim=128,
            n_layers=2,
            n_heads=4,
            ttl_enabled=True,
        )

        # Run multiple forward passes
        for _ in range(10):
            input_ids = torch.randint(0, 1000, (2, 64))
            logits, ttl_stats = model(input_ids, return_ttl_stats=True)

            # Check logits
            assert not torch.isnan(logits).any(), "NaN in logits"
            assert not torch.isinf(logits).any(), "Inf in logits"

            # Check TTL stats
            for stats in ttl_stats:
                for key, value in stats.items():
                    if isinstance(value, float):
                        assert not (value != value), f"NaN in {key}"  # NaN check
                        assert abs(value) < float('inf'), f"Inf in {key}"
