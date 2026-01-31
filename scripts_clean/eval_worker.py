#!/usr/bin/env python3
"""
Async Evaluation Worker for Atlas-MAG.

Watches a checkpoint directory for new checkpoint_step*.pt files,
loads each onto a separate GPU, and runs validation + NIAH probes.
Results are appended to eval_results.jsonl in the checkpoint directory.

Usage:
    # On GPU1 while training runs on GPU0:
    poetry run python scripts/eval_worker.py \
        --checkpoint-dir runs/committee_v1 \
        --device cuda:1

    # Dry-run on CPU:
    poetry run python scripts/eval_worker.py \
        --checkpoint-dir runs/committee_v1 \
        --device cpu --poll-interval 5
"""

import argparse
import json
import logging
import re
import signal
import sys
import time
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src_clean.data.smollm_dataset import create_smollm_val_dataloader
from src_clean.data.tokenizer import load_tokenizer
from src_clean.model.skeleton import AtlasMAGSkeleton
from src_clean.training.niah_probe import NIAHProbe
# DEADCODE: from src_clean.training.polarization import compute_gate_statistics
from src_clean.training.validation import run_validation

logger = logging.getLogger(__name__)

# Graceful shutdown flag
_shutdown = False


def _handle_sigint(sig, frame):
    global _shutdown
    logger.info("SIGINT received, finishing current assessment then shutting down...")
    _shutdown = True


def _extract_step(filename: str) -> int | None:
    """Extract step number from checkpoint_step000500.pt style filenames."""
    m = re.search(r"checkpoint_step(\d+)\.pt", filename)
    return int(m.group(1)) if m else None


def _build_model_from_config(config: dict, device: str) -> AtlasMAGSkeleton:
    """Reconstruct model from saved config dict."""
    model = AtlasMAGSkeleton(
        vocab_size=config.get("vocab_size", 32000),
        dim=config["dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        disable_memory=config.get("disable_memory", False),
        poly_degree=config.get("poly_degree", 2),
        poly_rank=config.get("poly_rank", 0),
        ttl_eta=config.get("ttl_eta", 0.01),
        ttl_ns_iters=config.get("ns_iterations", 10),
        ttl_adaptive_eta=config.get("ttl_adaptive_eta", True),
    )
    return model.to(device)


def _discover_vocab_size(config: dict, checkpoint: dict) -> int:
    """Determine vocab size from checkpoint state_dict or config."""
    sd = checkpoint.get("model_state_dict", {})
    if "tok_emb.weight" in sd:
        return sd["tok_emb.weight"].shape[0]
    return config.get("vocab_size", 32000)


def run_worker_loop(args):
    """Main worker loop: poll for checkpoints and assess each one."""
    checkpoint_dir = Path(args.checkpoint_dir)
    results_file = checkpoint_dir / "eval_results.jsonl"

    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory does not exist: {checkpoint_dir}")
        sys.exit(1)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = load_tokenizer(args.tokenizer)

    # Build val dataloader (reused across all checkpoints)
    logger.info(f"Creating validation dataloader ({args.val_samples} samples)...")
    val_loader = create_smollm_val_dataloader(
        tokenizer=tokenizer,
        batch_size=8,
        seq_len=2048,
        num_samples=args.val_samples,
        subsets=["cosmopedia-v2", "fineweb-edu-dedup", "python-edu-cleaned"],
    )

    # Cache a batch for NIAH probe
    probe_batch = next(iter(val_loader))["input_ids"]

    # Track which checkpoints we've already assessed
    assessed: set[str] = set()

    # Load existing results to avoid re-assessing
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if "checkpoint" in entry:
                        assessed.add(entry["checkpoint"])
                except json.JSONDecodeError:
                    continue
        logger.info(f"Loaded {len(assessed)} already-assessed checkpoints from {results_file}")

    logger.info(f"Watching {checkpoint_dir} for new checkpoints (poll every {args.poll_interval}s)")
    logger.info(f"Device: {args.device}")

    while not _shutdown:
        # Scan for checkpoint files
        ckpt_files = sorted(
            checkpoint_dir.glob("checkpoint_step*.pt"),
            key=lambda p: _extract_step(p.name) or 0,
        )

        new_ckpts = [f for f in ckpt_files if f.name not in assessed]

        if not new_ckpts:
            time.sleep(args.poll_interval)
            continue

        for ckpt_path in new_ckpts:
            if _shutdown:
                break

            step = _extract_step(ckpt_path.name)
            logger.info(f"Assessing {ckpt_path.name} (step {step})...")

            try:
                # Load checkpoint
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                config = checkpoint.get("config", {})

                # Discover vocab size from the actual weights
                vocab_size = _discover_vocab_size(config, checkpoint)
                config["vocab_size"] = vocab_size

                # Build model from config and load weights
                model = _build_model_from_config(config, args.device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.train(False)

                # Run validation
                val_metrics = run_validation(model, val_loader, args.device)

                # Run NIAH probe
                niah_result = None
                if not config.get("disable_memory", False) and step is not None:
                    if (step % args.niah_frequency) == 0 or step == 0:
                        niah = NIAHProbe(
                            dim=config["dim"],
                            probe_frequency=1,
                            haystack_size=4,
                            accuracy_threshold=0.1,
                            seq_len=1024,  # Must be > WINDOW_SIZE (512) to test beyond-window retrieval
                        )
                        niah.set_eval_batch(probe_batch)
                        niah_result = niah.run_probe(model, step or 0, args.device)

                # DEADCODE: Gate statistics meaningless for MAGBlock (no explicit gates)
                # MAGBlock uses element-wise gating via sigmoid(mem_out), not learnable gates
                # gate_stats = {}
                # if not config.get("disable_memory", False):
                #     gate_values = model.get_gate_values()
                #     gate_stats = compute_gate_statistics(torch.tensor(gate_values))
                #     gate_stats = {k: float(v) if hasattr(v, 'item') else v for k, v in gate_stats.items()}
                #     gate_stats["per_layer"] = gate_values
                gate_stats = {}  # Empty - MAGBlock has no explicit gates

                # Build result entry
                result_entry = {
                    "checkpoint": ckpt_path.name,
                    "step": step,
                    "tokens_seen": checkpoint.get("tokens_seen", 0),
                    "timestamp": time.time(),
                    "val_loss": val_metrics["loss"],
                    "val_ppl": val_metrics["ppl"],
                    "val_tokens": val_metrics["num_tokens"],
                    "gate_stats": gate_stats,
                }

                if niah_result is not None:
                    result_entry["niah"] = {
                        "accuracy": niah_result.accuracy,
                        "passed": niah_result.passed,
                        "ppl_mem": niah_result.ppl_mem,
                        "ppl_nomem": niah_result.ppl_nomem,
                        "positions_tested": niah_result.positions_tested,
                        "probe_time_ms": niah_result.probe_time_ms,
                    }

                # Append to results file
                with open(results_file, "a") as f:
                    f.write(json.dumps(result_entry) + "\n")

                assessed.add(ckpt_path.name)

                logger.info(
                    f"  {ckpt_path.name}: val_ppl={val_metrics['ppl']:.2f}, "
                    f"val_loss={val_metrics['loss']:.4f}"
                    + (
                        f", niah={niah_result.accuracy:.1%}"
                        if niah_result
                        else ""
                    )
                )

                # Free GPU memory
                del model, checkpoint
                if "cuda" in args.device:
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error assessing {ckpt_path.name}: {e}", exc_info=True)
                assessed.add(ckpt_path.name)

        # In one-shot mode, exit after processing all current checkpoints
        if args.once:
            logger.info(f"One-shot mode: assessed {len(assessed)} checkpoints total. Exiting.")
            break

        # Wait before next poll
        if not _shutdown:
            time.sleep(args.poll_interval)

    logger.info("Worker shut down.")


def main():
    parser = argparse.ArgumentParser(description="Async checkpoint assessment worker for Atlas-MAG")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory to watch for .pt files")
    parser.add_argument("--device", default="cuda:1", help="GPU for assessment (default: cuda:1)")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between directory scans")
    parser.add_argument("--tokenizer", default="data/tokenizer_smollm.json", help="Tokenizer path")
    parser.add_argument("--val-samples", type=int, default=2000, help="Validation set size")
    parser.add_argument("--niah-frequency", type=int, default=1, help="Run NIAH every Nth checkpoint")
    parser.add_argument("--once", action="store_true", help="Process all current checkpoints then exit (no polling)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    signal.signal(signal.SIGINT, _handle_sigint)

    run_worker_loop(args)


if __name__ == "__main__":
    main()
