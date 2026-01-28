"""
Shared validation utilities for Atlas-MAG.

Extracted from train_smollm.py so both the training script and the
async eval worker can use the same validation logic.
"""

import math

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
