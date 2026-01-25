"""
Optimizers for Atlas-MAG.

Phase 3: Optimization
- Muon optimizer with Newton-Schulz orthogonalization
- Tensorized memory updates (no Python loops)
"""

from src.optim.muon import Muon, MuonState
from src.optim.memory_update import (
    tensorized_memory_update,
    batch_qk_projection,
    tensorized_outer_product_sum,
)

__all__ = [
    # Muon optimizer
    "Muon",
    "MuonState",
    # Tensorized updates
    "tensorized_memory_update",
    "batch_qk_projection",
    "tensorized_outer_product_sum",
]
