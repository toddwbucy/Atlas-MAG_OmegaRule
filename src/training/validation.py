"""
Shared validation utilities for Atlas-MAG.

Extracted from train_smollm.py so both the training script and the
async eval worker can use the same validation logic.

Committee v6 feedback: Added functional PPL probes (memory ON vs OFF)
to measure actual memory contribution. This replaces brittle unit tests
that were testing the wrong module.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


@torch.no_grad()
def run_validation(model: nn.Module, loader, device: str) -> dict:
    """Run validation on model.

    Args:
        model: The model to evaluate.
        loader: DataLoader yielding dicts with 'input_ids' and 'labels'.
        device: Device string (e.g. 'cuda:0', 'cpu').

    Returns:
        Dict with keys: loss, ppl, num_batches, num_tokens.
    """
    model_was_training = model.training
    model.train(False)

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        total_loss += loss.item() * labels.numel()
        total_tokens += labels.numel()
        num_batches += 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(min(avg_loss, 20))  # Cap at exp(20) to avoid overflow

    model.train(model_was_training)
    return {"loss": avg_loss, "ppl": ppl, "num_batches": num_batches, "num_tokens": total_tokens}


@torch.no_grad()
def compute_memory_contribution(
    model: nn.Module,
    loader,
    device: str,
    max_batches: Optional[int] = 4,
) -> dict:
    """
    Compute memory contribution by comparing PPL with memory enabled vs disabled.

    This is a functional end-to-end probe that measures whether memory is actually
    helping the model. Committee v6 feedback: This replaces brittle unit tests
    that were testing the wrong module (QK projection instead of AtlasMemoryPoly).

    Formula: memory_contribution_pct = (ppl_nomem - ppl_mem) / ppl_nomem * 100
    - Positive: memory is helping (lower PPL with memory)
    - Negative: memory is hurting (higher PPL with memory)

    Args:
        model: AtlasMAGSkeleton model with .blocks attribute
        loader: DataLoader yielding dicts with 'input_ids' and 'labels'
        device: Device string (e.g. 'cuda:0', 'cpu')
        max_batches: Maximum batches to evaluate (for speed). None = all batches.

    Returns:
        Dict with keys:
            - ppl_with_memory: PPL with memory enabled
            - ppl_without_memory: PPL with memory disabled
            - memory_contribution_pct: Percentage improvement from memory (positive = helping)
    """
    model_was_training = model.training
    model.train(False)

    # Check if model has blocks attribute (AtlasMAGSkeleton)
    if not hasattr(model, "blocks"):
        model.train(model_was_training)
        return {
            "ppl_with_memory": 0.0,
            "ppl_without_memory": 0.0,
            "memory_contribution_pct": 0.0,
            "error": "Model does not have .blocks attribute",
        }

    # Collect batches for consistent comparison
    batches = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batches.append({
            "input_ids": batch["input_ids"].to(device),
            "labels": batch["labels"].to(device),
        })

    if not batches:
        model.train(model_was_training)
        return {
            "ppl_with_memory": 0.0,
            "ppl_without_memory": 0.0,
            "memory_contribution_pct": 0.0,
            "error": "No batches available",
        }

    # === PPL with memory enabled ===
    total_loss_mem = 0.0
    total_tokens = 0

    for batch in batches:
        logits = model(batch["input_ids"])
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch["labels"].view(-1),
        )
        total_loss_mem += loss.item() * batch["labels"].numel()
        total_tokens += batch["labels"].numel()

    avg_loss_mem = total_loss_mem / total_tokens if total_tokens > 0 else float("inf")
    ppl_with_memory = math.exp(min(avg_loss_mem, 20))

    # === PPL with memory disabled ===
    # Save original disable_memory flags
    orig_flags = [block.disable_memory for block in model.blocks]

    try:
        # Disable memory in all blocks
        for block in model.blocks:
            block.disable_memory = True

        total_loss_nomem = 0.0

        for batch in batches:
            logits = model(batch["input_ids"])
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch["labels"].view(-1),
            )
            total_loss_nomem += loss.item() * batch["labels"].numel()

        avg_loss_nomem = total_loss_nomem / total_tokens if total_tokens > 0 else float("inf")
        ppl_without_memory = math.exp(min(avg_loss_nomem, 20))

    finally:
        # Restore original flags
        for block, flag in zip(model.blocks, orig_flags):
            block.disable_memory = flag

    # Compute memory contribution
    # Positive = memory helping (lower PPL), Negative = memory hurting
    if ppl_without_memory > 0:
        memory_contribution_pct = (ppl_without_memory - ppl_with_memory) / ppl_without_memory * 100
    else:
        memory_contribution_pct = 0.0

    model.train(model_was_training)

    return {
        "ppl_with_memory": ppl_with_memory,
        "ppl_without_memory": ppl_without_memory,
        "memory_contribution_pct": memory_contribution_pct,
    }
