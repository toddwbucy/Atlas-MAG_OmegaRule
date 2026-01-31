#!/usr/bin/env python3
"""
Atlas-MAG Training Script for SmolLM-Corpus Dataset.

Train on the SmolLM-Corpus dataset for proper pre-training.
Uses curated data from cosmopedia-v2, fineweb-edu-dedup, and python-edu-cleaned.

Usage:
    # Standard training (all subsets, default weights)
    python scripts/train_smollm.py --epochs 1

    # Quick test
    python scripts/train_smollm.py --epochs 1 --max-steps 500

    # Custom subset weights (more code)
    python scripts/train_smollm.py --subset-weights cosmopedia-v2:0.3 fineweb-edu-dedup:0.4 python-edu-cleaned:0.3

    # Only cosmopedia (synthetic textbooks)
    python scripts/train_smollm.py --subsets cosmopedia-v2

    # Ablation: attention-only (no memory module)
    python scripts/train_smollm.py --disable-memory

Default runs on GPU 0 with 195M model configuration.

Evaluation is decoupled - use scripts/eval_worker.py to evaluate checkpoints on GPU1.
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src_clean.data.smollm_dataset import create_smollm_dataloader
from src_clean.data.tokenizer import load_tokenizer
from src_clean.model.skeleton import AtlasMAGSkeleton
# DEADCODE: MAGBlock has no explicit gate params, these produce meaningless output
# from src_clean.training.gate_monitor import GateMonitor
# from src_clean.training.polarization import compute_gate_statistics
from src_clean.utils.logging import get_logger, setup_logging

# Logger will be configured in main() after parsing args
logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration for SmolLM dataset."""

    # Model (195M default: dim=768, layers=12, heads=12)
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12

    # Training
    batch_size: int = 8
    seq_len: int = 2048  # Increased from 512 to exceed attention window (512)
    # This forces the model to use memory for long-range dependencies.
    # With seq_len <= attention window, attention can see everything and memory is redundant.
    epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.01
    grad_clip: float = 1.0
    max_steps: Optional[int] = None  # Optional step limit for quick tests
    gradient_accumulation_steps: int = 1  # Accumulate gradients over N micro-batches

    # Note: Polarization loss was removed - it was NOT part of the Atlas paper.
    # Gates learn their values purely through the main training objective.

    # Memory architecture
    poly_degree: int = 2  # Polynomial feature degree (1 for small models, 2 for 195M+)
    poly_rank: int = 512  # Low-rank compression dim for polynomial features (0=none)
    frozen_m0_ratio: float = 0.0  # Fraction of steps with static memory frozen

    # TTL tuning
    ttl_eta: float = 0.01  # Inner loop learning rate
    ttl_adaptive_eta: bool = True  # Scale eta by inverse gradient norm
    ns_iterations: int = 10  # Newton-Schulz iterations

    # Ablation
    disable_memory: bool = False  # For attention-only ablation

    # Data
    tokenizer_path: str = "data/tokenizer_smollm.json"
    subsets: list[str] = field(
        default_factory=lambda: ["cosmopedia-v2", "fineweb-edu-dedup", "python-edu-cleaned"]
    )
    subset_weights: dict[str, float] = field(
        default_factory=lambda: {
            "cosmopedia-v2": 0.4,
            "fineweb-edu-dedup": 0.5,
            "python-edu-cleaned": 0.1,
        }
    )
    num_workers: int = 4

    # Checkpoints
    output_dir: str = "runs/smollm_195m"
    save_every: int = 500
    log_every: int = 10

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def get_lr_schedule(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay."""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(config: TrainingConfig):
    """Main training loop."""

    # Setup output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.txt", "w") as f:
        for k, v in vars(config).items():
            f.write(f"{k}: {v}\n")

    logger.info("=" * 70)
    logger.info("Atlas-MAG Training on SmolLM-Corpus")
    logger.info("=" * 70)
    logger.info(f"Config: dim={config.dim}, layers={config.n_layers}, heads={config.n_heads}")
    effective_batch = config.batch_size * config.gradient_accumulation_steps
    logger.info(
        f"Training: batch={config.batch_size} x {config.gradient_accumulation_steps} accum = {effective_batch} effective, seq_len={config.seq_len}, epochs={config.epochs}"
    )
    logger.info(f"Data subsets: {config.subsets}")
    logger.info(f"Subset weights: {config.subset_weights}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Output: {output_dir}")

    # Check tokenizer exists
    if not Path(config.tokenizer_path).exists():
        logger.error(f"Tokenizer not found: {config.tokenizer_path}")
        logger.error("Run: python scripts/train_tokenizer_smollm.py")
        sys.exit(1)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {config.tokenizer_path}...")
    tokenizer = load_tokenizer(config.tokenizer_path)
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Create model (use tokenizer's vocab size for consistency)
    logger.info("Creating model...")
    model = AtlasMAGSkeleton(
        vocab_size=tokenizer.vocab_size,
        dim=config.dim,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        disable_memory=config.disable_memory,
        poly_degree=config.poly_degree,
        poly_rank=config.poly_rank,
        ttl_eta=config.ttl_eta,
        ttl_ns_iters=config.ns_iterations,
        ttl_adaptive_eta=config.ttl_adaptive_eta,
    ).to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mode = "ATTENTION-ONLY (ablation)" if config.disable_memory else "MEMORY+ATTENTION"
    logger.info(
        f"Model: {total_params / 1e6:.1f}M total params, {trainable_params / 1e6:.1f}M trainable"
    )
    logger.info(f"Mode: {mode}")

    # Verify GPU placement
    param_device = next(model.parameters()).device
    logger.info(f"Model device: {param_device}")
    if "cuda" in str(param_device):
        gpu_idx = param_device.index if param_device.index is not None else 0
        gpu_mem_mb = torch.cuda.memory_allocated(gpu_idx) / 1e6
        logger.info(f"GPU {gpu_idx} memory allocated: {gpu_mem_mb:.1f} MB")

    # Create data loader
    logger.info("Creating data loader...")
    train_loader = create_smollm_dataloader(
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        subsets=config.subsets,
        subset_weights=config.subset_weights,
        num_workers=config.num_workers,
    )

    # Estimate total steps (account for gradient accumulation)
    # SmolLM is ~237M samples, each ~800 tokens average
    est_tokens_total = 200_000_000_000  # 200B tokens rough estimate
    est_seqs = est_tokens_total // config.seq_len
    effective_batch = config.batch_size * config.gradient_accumulation_steps
    est_steps_per_epoch = est_seqs // effective_batch
    est_steps = est_steps_per_epoch * config.epochs

    if config.max_steps:
        est_steps = min(est_steps, config.max_steps)
        logger.info(f"Max steps limited to: {config.max_steps}")

    logger.info(f"Estimated training: ~{est_steps:,} steps per epoch")

    warmup_steps = max(100, int(est_steps * config.warmup_ratio))
    logger.info(f"Warmup steps: {warmup_steps}")

    # Optimizer and scheduler
    # CRITICAL: Exclude memory parameters from AdamW to prevent optimizer mismatch.
    # Memory is updated via TTL (Test-Time Learning) which uses Newton-Schulz
    # orthogonalization - the correct optimizer for memory's non-convex landscape.
    # AdamW (first-order) treats memory gradients as noise and collapses gates.
    memory_params: list[nn.Parameter] = []  # For zeroing gradients after backward
    if config.disable_memory:
        # No memory to exclude
        optimizer_params = list(model.parameters())
        logger.info("Optimizer: AdamW for all parameters (memory disabled)")
    else:
        # Collect memory parameters to exclude from AdamW
        # We need both IDs (for filtering) and the params themselves (for zeroing grads)
        memory_param_ids: set[int] = set()
        for block in model.blocks:
            if not hasattr(block, "memory"):
                continue  # Skip attention-only blocks
            for p in block.memory.parameters():
                memory_param_ids.add(id(p))
                memory_params.append(p)

        # Separate parameters: AdamW handles non-memory, TTL handles memory
        optimizer_params = [p for p in model.parameters() if id(p) not in memory_param_ids]
        n_adamw_params = sum(p.numel() for p in optimizer_params)
        n_memory_params = sum(p.numel() for p in memory_params)
        logger.info(
            f"Optimizer: AdamW for {n_adamw_params / 1e6:.1f}M params "
            f"(excluding {n_memory_params / 1e6:.1f}M memory params, handled by TTL)"
        )

    optimizer = AdamW(
        optimizer_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    scheduler = get_lr_schedule(optimizer, warmup_steps, est_steps)

    # DEADCODE: Gate monitor was for old AtlasMAGBlock with explicit gates
    # MAGBlock uses element-wise gating (sigmoid(mem_out)), no explicit gate params
    # gate_monitor: GateMonitor | None = None
    # if not config.disable_memory:
    #     gate_monitor = GateMonitor()
    #     logger.info("GateMonitor enabled: tracking gate health at steps 100, 500")

    # Training loop
    logger.info("Starting training...")
    logger.info(f"Training will use device: {config.device}")
    model.train()

    global_step = 0
    micro_step = 0  # Counts individual forward passes
    start_time = time.time()
    tokens_seen = 0
    accum_steps = config.gradient_accumulation_steps

    # For accumulating losses across micro-batches
    accum_lm_loss = 0.0
    accum_tokens = 0

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        epoch_tokens = 0

        for batch in train_loader:
            if config.max_steps and global_step >= config.max_steps:
                logger.info(f"Reached max_steps ({config.max_steps}), stopping.")
                break

            input_ids = batch["input_ids"].to(config.device)
            labels = batch["labels"].to(config.device)

            # Frozen M0: occasionally freeze static memory weights so only TTL can reduce loss
            # NOTE: We disable TTL on frozen steps because freeze_static_weights()
            # sets requires_grad=False, which breaks torch.autograd.grad() in ttl_step.
            is_frozen_step = (
                config.frozen_m0_ratio > 0
                and not config.disable_memory
                and torch.rand(1).item() < config.frozen_m0_ratio
            )
            if is_frozen_step:
                for block in model.blocks:
                    block.memory.freeze_static_weights()
                    block.ttl_enabled = False

            # Forward pass
            logits = model(input_ids)

            # Language modeling loss (scaled for accumulation)
            lm_loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

            # Total loss (no auxiliary losses - paper fidelity)
            total_loss = lm_loss / accum_steps  # Scale for accumulation

            # Backward pass (accumulates gradients)
            total_loss.backward()

            # Unfreeze memory if it was frozen this step
            if is_frozen_step:
                for block in model.blocks:
                    block.memory.unfreeze_static_weights()
                    block.ttl_enabled = True

            # Zero memory gradients immediately after backward.
            # Memory params receive gradients from LM loss, but these are never used
            # (TTL handles memory updates separately via omega_loss). Without zeroing,
            # these orphaned gradients accumulate indefinitely and inflate clip_grad_norm_.
            for p in memory_params:
                if p.grad is not None:
                    p.grad.zero_()

            # Track stats for this micro-batch
            batch_tokens = labels.numel()
            accum_lm_loss += lm_loss.item() * batch_tokens
            accum_tokens += batch_tokens
            tokens_seen += batch_tokens
            micro_step += 1

            # Only step optimizer after accumulation is complete
            if micro_step % accum_steps == 0:
                # Gradient clipping (only optimizer-managed params, not memory)
                grad_norm = nn.utils.clip_grad_norm_(optimizer_params, config.grad_clip)

                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Compute averaged losses for logging (before resetting)
                lm_loss_for_log = accum_lm_loss / accum_tokens if accum_tokens > 0 else 0

                # Update epoch stats
                epoch_loss += accum_lm_loss
                epoch_tokens += accum_tokens

                global_step += 1

                # Reset accumulators
                accum_lm_loss = 0.0
                accum_tokens = 0

                # DEADCODE: Gate monitoring was for old AtlasMAGBlock
                # if gate_monitor is not None:
                #     gate_values_tensor = torch.tensor(model.get_gate_values())
                #     monitor_stats = gate_monitor.check(gate_values_tensor, global_step)
                #     if monitor_stats.get("variance_check_passed") is False:
                #         logger.warning(
                #             f"[Step {global_step}] GATE HEALTH WARNING: "
                #             f"Variance check failed - gates may be collapsing!"
                #         )

                # Logging (only after optimizer step)
                if global_step % config.log_every == 0 and global_step > 0:
                    elapsed = time.time() - start_time
                    tokens_per_sec = tokens_seen / elapsed if elapsed > 0 else 0

                    logger.info(
                        f"[Epoch {epoch + 1}/{config.epochs}] "
                        f"Step {global_step} | "
                        f"Tokens: {tokens_seen / 1e6:.1f}M | "
                        f"LM Loss: {lm_loss_for_log:.4f} | "
                        f"PPL: {math.exp(min(lm_loss_for_log, 20)):.2f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                        f"GradNorm: {grad_norm:.2f} | "
                        f"Tok/s: {tokens_per_sec:.0f}"
                    )

                # Regular checkpoint
                if global_step % config.save_every == 0 and global_step > 0:
                    torch.save(
                        {
                            "step": global_step,
                            "epoch": epoch,
                            "tokens_seen": tokens_seen,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "config": vars(config),
                        },
                        output_dir / f"checkpoint_step{global_step:06d}.pt",
                    )
                    logger.info(f"  Saved checkpoint: checkpoint_step{global_step:06d}.pt")

        # Check max_steps again for epoch break
        if config.max_steps and global_step >= config.max_steps:
            break

        # End of epoch
        if epoch_tokens > 0:
            avg_epoch_loss = epoch_loss / epoch_tokens
            logger.info(
                f"[Epoch {epoch + 1} Complete] "
                f"Avg Loss: {avg_epoch_loss:.4f}, "
                f"PPL: {math.exp(min(avg_epoch_loss, 20)):.2f}, "
                f"Tokens: {tokens_seen / 1e6:.1f}M"
            )

    logger.info("=" * 70)
    logger.info("Training complete - run eval_worker.py to evaluate checkpoints")

    # Save final model
    torch.save(
        {
            "step": global_step,
            "tokens_seen": tokens_seen,
            "model_state_dict": model.state_dict(),
            "config": vars(config),
        },
        output_dir / "final_model.pt",
    )

    total_time = time.time() - start_time
    logger.info(f"Training complete in {total_time / 60:.1f} minutes")
    logger.info(f"Total tokens: {tokens_seen / 1e6:.1f}M")
    logger.info(f"Models saved to {output_dir}")

    return {
        "total_steps": global_step,
        "tokens_seen": tokens_seen,
    }


def parse_subset_weights(weight_strs: list[str]) -> dict[str, float]:
    """Parse subset weights from CLI arguments like 'cosmopedia-v2:0.4'."""
    weights = {}
    for s in weight_strs:
        if ":" in s:
            name, weight = s.split(":")
            weights[name.strip()] = float(weight)
        else:
            logger.warning(f"Invalid weight format '{s}', expected 'name:weight'")
    return weights


def main():
    parser = argparse.ArgumentParser(description="Train Atlas-MAG on SmolLM-Corpus")

    # Model (195M default)
    parser.add_argument("--dim", type=int, default=768, help="Model dimension")
    parser.add_argument("--layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--heads", type=int, default=12, help="Number of attention heads")

    # Training
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Max training steps (for quick tests)"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Accumulate gradients over N micro-batches (effective batch = batch_size * N)",
    )

    # Data
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        default=["cosmopedia-v2", "fineweb-edu-dedup", "python-edu-cleaned"],
        help="Subsets to use (e.g., cosmopedia-v2 fineweb-edu-dedup python-edu-cleaned)",
    )
    parser.add_argument(
        "--subset-weights",
        type=str,
        nargs="+",
        default=[],
        help="Subset weights (e.g., 'cosmopedia-v2:0.4 fineweb-edu-dedup:0.5 python-edu-cleaned:0.1')",
    )

    # Other
    parser.add_argument(
        "--output-dir", type=str, default="runs/smollm_195m", help="Output directory"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="data/tokenizer_smollm.json",
        help="Path to tokenizer",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu), auto-detects if not specified",
    )
    parser.add_argument(
        "--poly-degree", type=int, default=2,
        help="Polynomial feature degree for memory (1 for <100M models, 2 for 195M+)",
    )
    parser.add_argument("--ns-iterations", type=int, default=10, help="Newton-Schulz iterations (default: 10)")
    parser.add_argument("--ttl-eta", type=float, default=0.01, help="Inner loop learning rate (default: 0.01)")
    parser.add_argument("--no-ttl-adaptive-eta", action="store_true", help="Disable adaptive eta scaling")
    parser.add_argument("--poly-rank", type=int, default=512, help="Low-rank compression dim for poly features (0=none, default: 512)")
    parser.add_argument("--frozen-m0-ratio", type=float, default=0.0, help="Fraction of steps with M0 frozen (default: 0.0)")
    parser.add_argument(
        "--disable-memory",
        action="store_true",
        help="Ablation: attention-only (no memory module)",
    )
    parser.add_argument("--log-every", type=int, default=10, help="Log frequency")
    parser.add_argument("--save-every", type=int, default=500, help="Checkpoint frequency")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--log-file", type=str, default=None, help="Log file path (default: {output-dir}/train.log)"
    )

    args = parser.parse_args()

    # Setup logging (before any other logging calls)
    log_file = Path(args.log_file) if args.log_file else Path(args.output_dir) / "train.log"
    setup_logging(log_file=log_file)

    # Parse subset weights
    subset_weights = {"cosmopedia-v2": 0.4, "fineweb-edu-dedup": 0.5, "python-edu-cleaned": 0.1}
    if args.subset_weights:
        subset_weights.update(parse_subset_weights(args.subset_weights))

    config = TrainingConfig(
        dim=args.dim,
        n_layers=args.layers,
        n_heads=args.heads,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        seq_len=args.seq_len,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        subsets=args.subsets,
        subset_weights=subset_weights,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer,
        device=args.device,
        poly_degree=args.poly_degree,
        poly_rank=args.poly_rank,
        frozen_m0_ratio=args.frozen_m0_ratio,
        ns_iterations=args.ns_iterations,
        ttl_eta=args.ttl_eta,
        ttl_adaptive_eta=not args.no_ttl_adaptive_eta,
        disable_memory=args.disable_memory,
        log_every=args.log_every,
        save_every=args.save_every,
        num_workers=args.num_workers,
    )

    train(config)


if __name__ == "__main__":
    main()
