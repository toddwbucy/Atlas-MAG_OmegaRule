"""
Newton-Schulz Orthogonalization for the Muon Optimizer.

The Muon optimizer uses Newton-Schulz iteration to orthogonalize gradients
before applying updates. This improves conditioning and training stability.

The iteration computes the orthogonalized matrix (G @ G^T)^(-1/2) @ G,
which is the orthogonal part of the polar decomposition.

Reference: Muon optimizer (https://github.com/KellerJordan/Muon)
"""

import torch
from torch import Tensor

# Optimal coefficients for Newton-Schulz iteration
# These values give cubic convergence for the matrix sign function iteration
_NS_A: float = 3.4445
_NS_B: float = -4.7750
_NS_C: float = 2.0315


def newton_schulz(G: Tensor, num_iters: int = 5, eps: float = 1e-7) -> Tensor:
    """
    Apply Newton-Schulz iteration to orthogonalize a matrix.

    This computes an approximation to (G @ G^T)^(-1/2) @ G, which orthogonalizes
    the rows of G. The resulting matrix X satisfies X @ X^T ≈ I.

    The iteration is:
        X_{k+1} = a*X_k + (b*A + c*A^2) @ X_k
        where A = X_k @ X_k^T

    Args:
        G: Input gradient matrix of shape (out_features, in_features)
        num_iters: Number of Newton-Schulz iterations (default: 5, per PRD)
        eps: Small constant for numerical stability

    Returns:
        Orthogonalized matrix of same shape (rows are orthonormal)

    Note:
        - K=5 iterations is sufficient for most cases (PRD requirement)
        - Works for any shape matrix
    """
    assert G.ndim == 2, f"Expected 2D tensor, got {G.ndim}D"

    # Normalize to ensure top singular value <= 1 (required for convergence)
    norm = G.norm()
    if norm < eps:
        return G  # Near-zero gradient, nothing to do

    X = G / (norm + eps)

    for _ in range(num_iters):
        # A = X @ X^T (Gram matrix for rows)
        A = X @ X.transpose(-2, -1)
        # B = b*A + c*A^2 (polynomial approximation)
        B = _NS_B * A + _NS_C * (A @ A)
        # X = a*X + B @ X
        X = _NS_A * X + B @ X

    result: Tensor = X
    return result


def newton_schulz_batched(G: Tensor, num_iters: int = 5, eps: float = 1e-7) -> Tensor:
    """
    Batched Newton-Schulz iteration for multiple matrices.

    Useful when orthogonalizing gradients for multiple layers in parallel.

    Args:
        G: Batch of gradient matrices, shape (batch, out_features, in_features)
        num_iters: Number of iterations (default: 5)
        eps: Small constant for numerical stability

    Returns:
        Batch of orthogonalized matrices, same shape
    """
    assert G.ndim == 3, f"Expected 3D tensor (batch, out, in), got {G.ndim}D"

    # Per-matrix normalization
    norms = torch.linalg.norm(G, dim=(-2, -1), keepdim=True)
    X = G / (norms + eps)

    for _ in range(num_iters):
        # Batched: A = X @ X^T
        A = torch.bmm(X, X.transpose(-2, -1))
        B = _NS_B * A + _NS_C * torch.bmm(A, A)
        X = _NS_A * X + torch.bmm(B, X)

    result: Tensor = X
    return result


def orthogonality_error(X: Tensor) -> float:
    """
    Compute how far a matrix is from having orthonormal rows.

    Returns ||X @ X^T - I||_F / sqrt(m) where m is the number of rows.
    Perfect orthogonality gives 0.

    Useful for debugging and validating Newton-Schulz convergence.
    """
    if X.ndim == 2:
        XXT = X @ X.transpose(-2, -1)
        m = X.shape[0]
    else:
        XXT = torch.bmm(X, X.transpose(-2, -1))
        m = X.shape[1]

    identity = torch.eye(m, device=X.device, dtype=X.dtype)
    if XXT.ndim == 3:
        identity = identity.unsqueeze(0).expand_as(XXT)

    error = torch.linalg.norm(XXT - identity) / (m ** 0.5)
    error_val: float = error.item() if error.ndim == 0 else float(error.mean().item())
    return error_val
