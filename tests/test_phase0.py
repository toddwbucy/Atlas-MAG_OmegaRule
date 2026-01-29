"""
Tests for Phase 0 requirements.

Validates:
    P0-T1: W_init via steady-state calibration
    P0-T2: M_persistent from 64 persistent keys
    P0-T3: norm_persistent scalar
    P0-T4: Hash verification infrastructure
    P0-T5: Calibration batch size handling
"""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import N_PERSISTENT, D
from src.data.calibration import SimpleCalibrationDataset, create_calibration_loader
from src.initialization.hash_verify import (
    compute_tensor_hash,
    verify_m_persistent_consistency,
)
from src.model.atlas_memory import AtlasMemory, AtlasMemoryPoly
from src.model.persistent_memory import (
    PersistentMemory,
    compute_m_persistent,
    compute_norm_persistent,
)
from src.model.projections import CausalConv1d, QKVProjection, RotaryEmbedding
from src.model.skeleton import AtlasMAGBlock, AtlasMAGSkeleton


class TestMPersistent:
    """Tests for M_persistent computation (P0-T2)."""

    def test_shape(self):
        """M_persistent should be (dim, dim)."""
        keys = torch.randn(N_PERSISTENT, D)
        m = compute_m_persistent(keys)
        assert m.shape == (D, D)

    def test_symmetric(self):
        """M_persistent should be symmetric (sum of outer products)."""
        keys = torch.randn(N_PERSISTENT, D)
        m = compute_m_persistent(keys)
        assert torch.allclose(m, m.T, atol=1e-6)

    def test_positive_semidefinite(self):
        """M_persistent should be positive semi-definite."""
        keys = torch.randn(N_PERSISTENT, D)
        m = compute_m_persistent(keys)

        # All eigenvalues should be >= 0 (with floating point tolerance)
        eigenvalues = torch.linalg.eigvalsh(m)
        # Allow small negative values due to floating point precision
        assert (eigenvalues >= -1e-3).all(), f"Min eigenvalue: {eigenvalues.min()}"

    def test_deterministic(self):
        """Same keys should give same M_persistent."""
        keys = torch.randn(N_PERSISTENT, D)
        m1 = compute_m_persistent(keys)
        m2 = compute_m_persistent(keys)
        assert torch.allclose(m1, m2)


class TestNormPersistent:
    """Tests for norm_persistent computation (P0-T3)."""

    def test_positive(self):
        """norm_persistent should be positive."""
        keys = torch.randn(N_PERSISTENT, D)
        norm = compute_norm_persistent(keys)
        assert norm > 0

    def test_is_float(self):
        """norm_persistent should be a Python float."""
        keys = torch.randn(N_PERSISTENT, D)
        norm = compute_norm_persistent(keys)
        assert isinstance(norm, float)

    def test_sum_of_squared_norms(self):
        """norm_persistent should equal sum of squared norms."""
        keys = torch.randn(N_PERSISTENT, D)
        norm = compute_norm_persistent(keys)

        expected = sum(k.norm().item() ** 2 for k in keys)
        # Relative tolerance for floating point accumulation
        rel_error = abs(norm - expected) / max(norm, expected)
        assert rel_error < 1e-4, f"Relative error: {rel_error}"


class TestHashVerification:
    """Tests for hash verification (P0-T4)."""

    def test_deterministic_hash(self):
        """Same tensor should give same hash."""
        t = torch.randn(64, 64)
        h1 = compute_tensor_hash(t)
        h2 = compute_tensor_hash(t)
        assert h1 == h2

    def test_different_tensor_different_hash(self):
        """Different tensors should give different hashes."""
        t1 = torch.randn(64, 64)
        t2 = torch.randn(64, 64)
        h1 = compute_tensor_hash(t1)
        h2 = compute_tensor_hash(t2)
        assert h1 != h2

    def test_hash_format(self):
        """Hash should be 64-character hex string."""
        t = torch.randn(64, 64)
        h = compute_tensor_hash(t)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_verify_single_gpu(self):
        """Verification should pass for single GPU."""
        m = torch.randn(D, D)
        assert verify_m_persistent_consistency(m, world_size=1, rank=0)


class TestPersistentMemoryModule:
    """Tests for PersistentMemory module."""

    def test_initialization(self):
        """Module should initialize correctly."""
        pm = PersistentMemory(dim=D, n_persistent=N_PERSISTENT)
        assert pm.persistent_keys.shape == (N_PERSISTENT, D)
        assert pm.m_persistent.shape == (D, D)
        assert pm.norm_persistent > 0

    def test_recompute(self):
        """Recompute should update cached values."""
        pm = PersistentMemory(dim=D, n_persistent=N_PERSISTENT)
        old_norm = pm.norm_persistent

        # Modify keys
        with torch.no_grad():
            pm.persistent_keys.data *= 2

        pm.recompute()

        # Norm should have changed (doubled keys = 4x norm)
        assert abs(pm.norm_persistent - old_norm * 4) < 1


class TestAtlasMemory:
    """Tests for Atlas memory module."""

    def test_output_shape(self):
        """Output should match input shape."""
        mem = AtlasMemory(dim=D)
        x = torch.randn(2, 512, D)
        y = mem(x)
        assert y.shape == x.shape

    def test_residual_connection(self):
        """Memory should add to input (residual)."""
        mem = AtlasMemory(dim=64)
        x = torch.randn(1, 10, 64)

        # With zero weights, output should equal input
        with torch.no_grad():
            mem.w1.weight.zero_()
        y = mem(x)
        assert torch.allclose(x, y)

    def test_poly_features(self):
        """Polynomial memory should expand features."""
        mem = AtlasMemoryPoly(dim=64)
        x = torch.randn(1, 10, 64)
        y = mem(x)
        assert y.shape == x.shape  # Output is still (batch, seq, dim)


class TestProjections:
    """Tests for Q, K, V projections."""

    def test_qkv_shapes(self):
        """Q, K, V should have correct shapes."""
        proj = QKVProjection(dim=D, n_heads=12)
        x = torch.randn(2, 512, D)
        q, k, v = proj(x, reshape_heads=True)

        assert q.shape == (2, 12, 512, 64)  # (batch, heads, seq, head_dim)
        assert k.shape == (2, 12, 512, 64)
        assert v.shape == (2, 12, 512, 64)

    def test_causal_conv(self):
        """Causal conv should preserve causality."""
        conv = CausalConv1d(dim=64, kernel_size=4)
        x = torch.randn(1, 10, 64)
        y = conv(x)
        assert y.shape == x.shape

    def test_rotary_embedding(self):
        """RoPE should apply to Q and K."""
        rope = RotaryEmbedding(dim=64)
        q = torch.randn(2, 12, 512, 64)
        k = torch.randn(2, 12, 512, 64)
        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

        # Values should have changed
        assert not torch.allclose(q, q_rot)


class TestAtlasMAGBlock:
    """Tests for single AtlasMAG block."""

    def test_forward(self):
        """Block forward should work."""
        block = AtlasMAGBlock(dim=D, n_heads=12)
        x = torch.randn(2, 512, D)
        y, ttl_stats = block(x)
        assert y.shape == x.shape
        # TTL stats should be returned when TTL is enabled
        assert ttl_stats is not None or not block.ttl_enabled

    def test_gate_value(self):
        """Gate should be accessible with per-layer initialization.

        Symmetric initialization (0.15-0.85) spanning both sides of 0.5.
        This ensures polarization pushes early layers toward attention (0)
        and later layers toward memory (1), giving both branches a chance.
        """
        # Test layer 0: sigmoid(-1.73) ≈ 0.15 (attention-favored)
        block_0 = AtlasMAGBlock(dim=D, n_heads=12, layer_idx=0, n_layers=12)
        gate_0 = block_0.get_gate_value()
        assert 0 <= gate_0 <= 1
        # Layer 0 should start at ~15% memory (attention-favored)
        assert 0.13 <= gate_0 <= 0.18

        # Test last layer: sigmoid(+1.73) ≈ 0.85 (memory-favored)
        block_11 = AtlasMAGBlock(dim=D, n_heads=12, layer_idx=11, n_layers=12)
        gate_11 = block_11.get_gate_value()
        # Layer 11 should be memory-favored: ~85% memory
        assert 0.83 <= gate_11 <= 0.88

        # Verify later layers have higher gates (more memory contribution)
        assert gate_11 > gate_0


class TestAtlasMAGSkeleton:
    """Tests for full skeleton model."""

    def test_forward(self):
        """Model forward should work."""
        model = AtlasMAGSkeleton(
            vocab_size=1000,
            dim=128,
            n_layers=2,
            n_heads=4,
        )
        input_ids = torch.randint(0, 1000, (2, 64))
        logits = model(input_ids)
        assert logits.shape == (2, 64, 1000)

    def test_forward_memory_only(self):
        """Memory-only forward should return state."""
        model = AtlasMAGSkeleton(
            vocab_size=1000,
            dim=128,
            n_layers=2,
            n_heads=4,
        )
        input_ids = torch.randint(0, 1000, (1, 64))
        logits, memory_state = model.forward_memory_only(input_ids)

        assert logits.shape == (1, 64, 1000)
        assert memory_state.ndim == 1  # Flattened

    def test_parameter_count(self):
        """Parameter count should be reasonable."""
        model = AtlasMAGSkeleton(
            vocab_size=32000,
            dim=768,
            n_layers=6,
            n_heads=12,
        )
        params = model.count_parameters()
        assert "total_millions" in params
        # ~110M for this config (6 layers with attention + memory + FFN)
        assert 50 < params["total_millions"] < 150


class TestCalibrationData:
    """Tests for calibration data pipeline."""

    def test_simple_dataset(self):
        """Simple dataset should work."""
        dataset = SimpleCalibrationDataset(
            vocab_size=1000,
            seq_len=64,
            num_samples=10,
        )
        assert len(dataset) == 10
        sample = dataset[0]
        assert sample.shape == (64,)
        assert sample.dtype == torch.long

    def test_calibration_loader(self):
        """Calibration loader should work."""
        loader = create_calibration_loader(
            num_tokens=1000,
            batch_size=4,
            seq_len=64,
            use_wikitext=False,
        )

        batch = next(iter(loader))
        assert batch.shape == (4, 64)


class TestMemoryDropout:
    """Tests for memory dropout (committee critique #1)."""

    def test_memory_dropout_skips_memory(self):
        """When memory dropout triggers, mem_out should be None (added to nothing)."""
        # Set dropout to 100% so it always triggers
        block = AtlasMAGBlock(
            dim=128, n_heads=4, memory_dropout_rate=1.0, ttl_enabled=False,
        )
        block.train()
        x = torch.randn(1, 16, 128)
        out, _ = block(x)
        # Output should still be valid (attention-only path)
        assert out.shape == x.shape

    def test_memory_dropout_zero_rate_never_skips(self):
        """With dropout=0, memory always runs."""
        block = AtlasMAGBlock(
            dim=128, n_heads=4, memory_dropout_rate=0.0, ttl_enabled=False,
        )
        block.train()
        x = torch.randn(1, 16, 128)
        for _ in range(5):
            out, _ = block(x)
            assert out.shape == x.shape

    def test_memory_dropout_not_in_inference(self):
        """Memory dropout should NOT apply in inference mode.

        With 100% dropout rate: training mode skips memory, eval mode runs it.
        We compare outputs of the SAME block in train vs eval mode.
        """
        torch.manual_seed(42)
        block = AtlasMAGBlock(
            dim=128, n_heads=4, memory_dropout_rate=1.0, ttl_enabled=False,
        )
        x = torch.randn(1, 16, 128)

        # Training mode with 100% dropout — memory skipped
        block.train(True)
        with torch.no_grad():
            out_train, _ = block(x)

        # Eval mode — memory should run despite dropout rate
        block.train(False)
        with torch.no_grad():
            out_eval, _ = block(x)

        # Outputs should differ because memory runs in eval but not train
        diff = (out_train - out_eval).abs().mean()
        assert diff > 1e-6, "Memory should be active in eval mode despite high dropout rate"


class TestSlidingWindowAttention:
    """Tests for sliding window attention mask (Atlas SWA)."""

    def test_sliding_window_mask_shape(self):
        """Mask should be (seq_len, seq_len)."""
        from src.model.skeleton import create_sliding_window_mask

        mask = create_sliding_window_mask(seq_len=16, window_size=4, device=torch.device("cpu"))
        assert mask.shape == (16, 16)
        assert mask.dtype == torch.bool

    def test_sliding_window_mask_causal(self):
        """Upper triangle should always be masked (causal constraint)."""
        from src.model.skeleton import create_sliding_window_mask

        mask = create_sliding_window_mask(seq_len=8, window_size=8, device=torch.device("cpu"))
        # With window=seq_len, it's just a causal mask (no sliding window constraint)
        expected_causal = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
        assert torch.equal(mask, expected_causal)

    def test_sliding_window_mask_window_constraint(self):
        """Positions beyond window should be masked."""
        from src.model.skeleton import create_sliding_window_mask

        mask = create_sliding_window_mask(seq_len=6, window_size=3, device=torch.device("cpu"))
        # Position 3 should NOT see position 0 (0 < 3 - 3 + 1 = 1)
        assert mask[3, 0] == True  # masked
        assert mask[3, 1] == False  # can see
        assert mask[3, 2] == False  # can see
        assert mask[3, 3] == False  # can see (self)
        # Position 5 should NOT see positions 0, 1, 2
        assert mask[5, 0] == True  # masked
        assert mask[5, 1] == True  # masked
        assert mask[5, 2] == True  # masked
        assert mask[5, 3] == False  # can see
        assert mask[5, 4] == False  # can see
        assert mask[5, 5] == False  # can see (self)

    def test_sliding_window_mask_early_positions(self):
        """Early positions should see all available (no sliding yet)."""
        from src.model.skeleton import create_sliding_window_mask

        mask = create_sliding_window_mask(seq_len=6, window_size=3, device=torch.device("cpu"))
        # Position 0: only sees self
        assert mask[0, 0] == False  # can see
        assert mask[0, 1] == True   # future - masked
        # Position 2: sees 0, 1, 2 (window not full yet counts from 0)
        assert mask[2, 0] == False  # can see
        assert mask[2, 1] == False  # can see
        assert mask[2, 2] == False  # can see

    def test_block_uses_sliding_window(self):
        """AtlasMAGBlock should use sliding window by default."""
        block = AtlasMAGBlock(dim=128, n_heads=4, ttl_enabled=False)
        # WINDOW_SIZE from config is 512
        from src.config import WINDOW_SIZE
        assert block.attn_window_size == WINDOW_SIZE


class TestGPUCompatibility:
    """Tests that require GPU."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_on_gpu(self):
        """Model should work on GPU."""
        model = AtlasMAGSkeleton(
            vocab_size=1000,
            dim=128,
            n_layers=2,
            n_heads=4,
        ).cuda()

        input_ids = torch.randint(0, 1000, (1, 64), device="cuda")
        logits = model(input_ids)
        assert logits.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_persistent_memory_on_gpu(self):
        """Persistent memory should work on GPU."""
        pm = PersistentMemory(dim=128, n_persistent=16).cuda()
        assert pm.m_persistent.device.type == "cuda"
        assert pm.norm_persistent > 0
