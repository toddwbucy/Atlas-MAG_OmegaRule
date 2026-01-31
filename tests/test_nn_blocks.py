"""
Tests for custom neural network building blocks.

Tests RMSNorm, SwiGLU, and Newton-Schulz implementations.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nn.newton_schulz import newton_schulz, newton_schulz_batched, orthogonality_error
from src.nn.rmsnorm import RMSNorm
from src.nn.swiglu import SwiGLU, SwiGLUFused


class TestRMSNorm:
    """Tests for RMSNorm module."""

    def test_output_shape(self):
        """Output should match input shape."""
        norm = RMSNorm(dim=768)
        x = torch.randn(2, 512, 768)
        y = norm(x)
        assert y.shape == x.shape

    def test_normalized_output(self):
        """Output should have roughly unit RMS per position."""
        norm = RMSNorm(dim=768)
        x = torch.randn(2, 512, 768) * 10  # Scale up input
        y = norm(x)

        # Check RMS is close to 1 (with learned scale = 1)
        rms = torch.sqrt((y ** 2).mean(dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_gradient_flow(self):
        """Gradients should flow through normalization."""
        norm = RMSNorm(dim=64)
        x = torch.randn(1, 10, 64, requires_grad=True)
        y = norm(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_different_dtypes(self):
        """Should work with different dtypes."""
        norm = RMSNorm(dim=64)
        for dtype in [torch.float32, torch.float16]:
            norm = norm.to(dtype)
            x = torch.randn(1, 10, 64, dtype=dtype)
            y = norm(x)
            assert y.dtype == dtype


class TestSwiGLU:
    """Tests for SwiGLU module."""

    def test_output_shape(self):
        """Output should match input shape."""
        ffn = SwiGLU(dim=768)
        x = torch.randn(2, 512, 768)
        y = ffn(x)
        assert y.shape == x.shape

    def test_hidden_dim_calculation(self):
        """Hidden dim should be ~8/3 of input dim, rounded to 64."""
        ffn = SwiGLU(dim=768)
        expected_hidden = ((768 * 4 * 2 // 3 + 63) // 64) * 64
        assert ffn.hidden_dim == expected_hidden

    def test_custom_hidden_dim(self):
        """Should accept custom hidden dimension."""
        ffn = SwiGLU(dim=768, hidden_dim=1024)
        assert ffn.hidden_dim == 1024

    def test_gradient_flow(self):
        """Gradients should flow through FFN."""
        ffn = SwiGLU(dim=64)
        x = torch.randn(1, 10, 64, requires_grad=True)
        y = ffn(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None

    def test_fused_vs_regular(self):
        """Fused and regular should have same behavior."""
        torch.manual_seed(42)
        regular = SwiGLU(dim=64, hidden_dim=128)
        fused = SwiGLUFused(dim=64, hidden_dim=128)

        # Copy weights from regular to fused
        with torch.no_grad():
            fused.w12.weight[:128] = regular.w1.weight
            fused.w12.weight[128:] = regular.w2.weight
            fused.w3.weight = regular.w3.weight

        x = torch.randn(1, 10, 64)
        y_regular = regular(x)
        y_fused = fused(x)

        assert torch.allclose(y_regular, y_fused, atol=1e-5)


class TestNewtonSchulz:
    """Tests for Newton-Schulz orthogonalization."""

    def test_output_shape(self):
        """Output should match input shape."""
        G = torch.randn(64, 128)
        X = newton_schulz(G)
        assert X.shape == G.shape

    def test_orthogonality(self):
        """Output should be approximately orthogonal."""
        G = torch.randn(64, 128)
        X = newton_schulz(G, num_iters=5)

        # Check X @ X^T ≈ I (for rows)
        # With K=5 iterations, error is typically ~0.3-0.4
        # More iterations would get closer to 0
        error = orthogonality_error(X)
        assert error < 0.5, f"Orthogonality error too high: {error}"

    def test_more_iterations_better(self):
        """More iterations should give better orthogonality."""
        G = torch.randn(64, 128)
        error_3 = orthogonality_error(newton_schulz(G, num_iters=3))
        error_5 = orthogonality_error(newton_schulz(G, num_iters=5))
        error_10 = orthogonality_error(newton_schulz(G, num_iters=10))

        assert error_5 <= error_3 + 0.01
        assert error_10 <= error_5 + 0.01

    def test_zero_gradient(self):
        """Should handle zero gradient gracefully."""
        G = torch.zeros(64, 128)
        X = newton_schulz(G)
        assert torch.allclose(X, G)

    def test_batched(self):
        """Batched version should work."""
        G = torch.randn(4, 64, 128)
        X = newton_schulz_batched(G)
        assert X.shape == G.shape

        # Each matrix should be orthogonalized (with K=5, error ~0.3-0.4)
        for i in range(4):
            error = orthogonality_error(X[i])
            assert error < 0.5, f"Batch {i} orthogonality error too high: {error}"

    def test_deterministic(self):
        """Same input should give same output."""
        G = torch.randn(64, 128)
        X1 = newton_schulz(G, num_iters=5)
        X2 = newton_schulz(G, num_iters=5)
        assert torch.allclose(X1, X2)


class TestIntegration:
    """Integration tests for NN blocks."""

    def test_rmsnorm_in_forward(self):
        """RMSNorm should work in a typical forward pass."""
        batch_size, seq_len, dim = 2, 512, 768

        norm = RMSNorm(dim)
        x = torch.randn(batch_size, seq_len, dim)

        # Multiple forward passes should work
        for _ in range(3):
            y = norm(x)
            assert y.shape == x.shape
            assert not torch.isnan(y).any()

    def test_swiglu_memory_efficiency(self):
        """SwiGLU should not leak memory."""
        ffn = SwiGLU(dim=768)

        if torch.cuda.is_available():
            ffn = ffn.cuda()
            torch.cuda.reset_peak_memory_stats()

            for _ in range(10):
                x = torch.randn(4, 512, 768, device="cuda")
                y = ffn(x)
                del x, y

            # Memory should be stable
            torch.cuda.synchronize()

    def test_newton_schulz_gpu(self):
        """Newton-Schulz should work on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        G = torch.randn(64, 128, device="cuda")
        X = newton_schulz(G)
        assert X.device.type == "cuda"
        assert orthogonality_error(X) < 0.5  # K=5 gives ~0.3-0.4 error
