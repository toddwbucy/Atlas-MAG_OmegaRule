#!/usr/bin/env python3
"""
Benchmark: Needle-In-A-Haystack Memory Probe (TTL ON vs TTL OFF)

This script measures how much the Atlas-MAG memory module contributes to
predictions at positions BEYOND the sliding window attention range, under
three conditions:

  1. ATTENTION ONLY  — Memory disabled entirely. Baseline.
  2. TTL OFF         — Memory exists but is static (standard serving mode).
                       This is what every serving framework gives you.
  3. TTL ON          — Memory adapts to input during the forward pass.
                       This is how the model was designed to run.

The test uses sequences longer than the attention window (default: 512).
For positions beyond the window, attention cannot see the full history.
Memory is the ONLY way to retrieve earlier context. If TTL is silenced,
the memory cannot adapt to the specific input being processed.

The gap between TTL OFF and TTL ON is what the serving stack costs you.

Paper: Atlas — Learning to Optimally Memorize the Context at Test Time
       arXiv:2505.23735 (Behrouz et al., 2025)

Usage:
    python scripts/benchmark_niah.py --checkpoint runs/atlas_54m_gelu/checkpoint_step008800.pt
    python scripts/benchmark_niah.py --checkpoint path/to/ckpt --seq-lengths 512 1024 2048
    python scripts/benchmark_niah.py --checkpoint path/to/ckpt --n-sequences 8 --device cuda:1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import List

import torch
import torch.nn.functional as F

from src.config import WINDOW_SIZE
from src.data.tokenizer import load_tokenizer
from src.model.skeleton import AtlasMAGSkeleton


@dataclass
class ProbeResult:
    """Result from a single NIAH probe condition."""

    condition: str
    seq_len: int
    ppl: float
    loss: float
    positions_tested: int
    time_s: float


@dataclass
class ComparisonRow:
    """Side-by-side comparison for one sequence length."""

    seq_len: int
    ppl_baseline: float
    ppl_ttl_off: float
    ppl_ttl_on: float
    # Memory contribution: (baseline - condition) / baseline
    contrib_ttl_off: float
    contrib_ttl_on: float
    # The gap: what the serving stack costs you
    ttl_gap: float


def load_model(checkpoint_path: str, device: str):
    """Load Atlas-MAG from checkpoint with full config restoration."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    vocab_size = checkpoint["model_state_dict"]["tok_emb.weight"].shape[0]

    model = AtlasMAGSkeleton(
        vocab_size=vocab_size,
        dim=config.get("dim", 768),
        n_layers=config.get("n_layers", 12),
        n_heads=config.get("n_heads", 12),
        disable_memory=config.get("disable_memory", False),
        poly_degree=config.get("poly_degree", 2),
        poly_rank=config.get("poly_rank", 512),
        ttl_enabled=config.get("ttl_enabled", True),
        ttl_theta=config.get("ttl_theta", 0.9),
        ttl_alpha=config.get("ttl_alpha", 0.999),
        ttl_eta=config.get("ttl_eta", 0.01),
        ttl_ns_iters=config.get("ttl_ns_iters", 5),
        ttl_adaptive_eta=config.get("ttl_adaptive_eta", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    return model, config, param_count, vocab_size


def compute_beyond_window_ppl(
    model, input_ids: torch.Tensor, window_size: int
) -> tuple:
    """
    Compute perplexity ONLY for positions beyond the attention window.

    At position t, sliding window attention can see [t-window+1, t].
    For t >= window_size, some earlier context is invisible to attention.
    Memory is the only mechanism that can retrieve it.

    Returns: (loss, ppl, positions_tested)
    """
    seq_len = input_ids.shape[1]
    vocab_size = model.vocab_size

    logits = model(input_ids)

    # Only measure positions beyond the window
    boundary = min(window_size, seq_len - 2)
    labels = input_ids[:, 1:].contiguous()
    labels_beyond = labels[:, boundary:].contiguous()
    logits_beyond = logits[:, boundary:-1, :].contiguous()

    positions_tested = labels_beyond.numel()
    if positions_tested == 0:
        return float("inf"), float("inf"), 0

    loss = F.cross_entropy(
        logits_beyond.reshape(-1, vocab_size), labels_beyond.reshape(-1)
    )
    ppl = torch.exp(loss).item()
    return loss.item(), ppl, positions_tested


def run_condition(
    model,
    input_ids: torch.Tensor,
    window_size: int,
    condition: str,
    disable_memory: bool = False,
    use_ttl: bool = False,
) -> ProbeResult:
    """Run a single probe condition."""
    seq_len = input_ids.shape[1]

    # Save and set memory flags
    orig_flags = []
    supports_disable = all(
        hasattr(block, "disable_memory") for block in model.blocks
    )

    if disable_memory and supports_disable:
        orig_flags = [block.disable_memory for block in model.blocks]
        for block in model.blocks:
            block.disable_memory = True

    try:
        if use_ttl:
            model.train()
            if hasattr(model, "reset_ttl_momentum"):
                model.reset_ttl_momentum()
            t0 = time.time()
            loss, ppl, positions = compute_beyond_window_ppl(
                model, input_ids, window_size
            )
            elapsed = time.time() - t0
        else:
            model.train(False)
            with torch.no_grad():
                t0 = time.time()
                loss, ppl, positions = compute_beyond_window_ppl(
                    model, input_ids, window_size
                )
                elapsed = time.time() - t0
    finally:
        # Restore memory flags
        if disable_memory and supports_disable and orig_flags:
            for block, flag in zip(model.blocks, orig_flags, strict=False):
                block.disable_memory = flag
        # Leave model in inference mode
        model.train(False)

    return ProbeResult(
        condition=condition,
        seq_len=seq_len,
        ppl=ppl,
        loss=loss,
        positions_tested=positions,
        time_s=elapsed,
    )


def run_benchmark(
    model,
    vocab_size: int,
    seq_lengths: List[int],
    n_sequences: int,
    window_size: int,
    device: str,
    seed: int = 42,
) -> List[ComparisonRow]:
    """Run the full NIAH benchmark across all sequence lengths."""
    results = []

    for seq_len in seq_lengths:
        if seq_len <= window_size:
            print(
                f"  Skipping seq_len={seq_len} (must be > window_size={window_size})"
            )
            continue

        beyond = seq_len - window_size
        print(f"\n  seq_len={seq_len} ({beyond} positions beyond window)")

        # Generate deterministic random sequences
        gen = torch.Generator(device="cpu").manual_seed(seed)
        input_ids = torch.randint(
            0, vocab_size, (n_sequences, seq_len), generator=gen
        ).to(device)

        # Condition 1: Attention only (memory disabled)
        baseline = run_condition(
            model, input_ids, window_size,
            condition="attention_only", disable_memory=True, use_ttl=False,
        )
        print(f"    Attention only : PPL={baseline.ppl:>10.2f}  ({baseline.time_s:.2f}s)")

        # Condition 2: TTL OFF (memory enabled, inference mode)
        ttl_off = run_condition(
            model, input_ids, window_size,
            condition="ttl_off", disable_memory=False, use_ttl=False,
        )
        print(f"    TTL OFF (serve): PPL={ttl_off.ppl:>10.2f}  ({ttl_off.time_s:.2f}s)")

        # Condition 3: TTL ON (memory enabled, training mode = active memory)
        ttl_on = run_condition(
            model, input_ids, window_size,
            condition="ttl_on", disable_memory=False, use_ttl=True,
        )
        print(f"    TTL ON (design): PPL={ttl_on.ppl:>10.2f}  ({ttl_on.time_s:.2f}s)")

        # Compute contributions
        if math.isinf(baseline.ppl) or baseline.ppl <= 0:
            contrib_off = 0.0
            contrib_on = 0.0
            gap = 0.0
        else:
            contrib_off = (baseline.ppl - ttl_off.ppl) / baseline.ppl
            contrib_on = (baseline.ppl - ttl_on.ppl) / baseline.ppl
            gap = contrib_on - contrib_off

        results.append(ComparisonRow(
            seq_len=seq_len,
            ppl_baseline=baseline.ppl,
            ppl_ttl_off=ttl_off.ppl,
            ppl_ttl_on=ttl_on.ppl,
            contrib_ttl_off=contrib_off,
            contrib_ttl_on=contrib_on,
            ttl_gap=gap,
        ))

    return results


def print_results_table(rows: List[ComparisonRow], window_size: int):
    """Print results as a formatted comparison table."""
    print()
    print("=" * 90)
    print("NIAH BENCHMARK RESULTS")
    print(f"Attention window: {window_size} tokens")
    print("Positions tested: beyond window only (where memory is the sole retrieval path)")
    print("=" * 90)
    print()

    # Header
    print(f"{'Seq Len':>8}  {'Attn Only':>10}  {'TTL OFF':>10}  {'TTL ON':>10}  "
          f"{'Contrib':>8}  {'Contrib':>8}  {'TTL Gap':>8}")
    print(f"{'':>8}  {'(baseline)':>10}  {'(serving)':>10}  {'(design)':>10}  "
          f"{'OFF':>8}  {'ON':>8}  {'':>8}")
    print("-" * 90)

    for r in rows:
        def fmt_ppl(v):
            if math.isinf(v):
                return "inf"
            if v > 99999:
                return f"{v:.0f}"
            return f"{v:.1f}"

        print(
            f"{r.seq_len:>8}  "
            f"{fmt_ppl(r.ppl_baseline):>10}  "
            f"{fmt_ppl(r.ppl_ttl_off):>10}  "
            f"{fmt_ppl(r.ppl_ttl_on):>10}  "
            f"{r.contrib_ttl_off:>7.1%}  "
            f"{r.contrib_ttl_on:>7.1%}  "
            f"{r.ttl_gap:>+7.1%}"
        )

    print("-" * 90)
    print()

    # Summary
    if rows:
        avg_gap = sum(r.ttl_gap for r in rows) / len(rows)
        avg_contrib_on = sum(r.contrib_ttl_on for r in rows) / len(rows)
        avg_contrib_off = sum(r.contrib_ttl_off for r in rows) / len(rows)

        print("Column Guide:")
        print("  Attn Only  = Memory disabled entirely (attention-only baseline)")
        print("  TTL OFF    = Memory exists but is static (standard serving mode)")
        print("               This is what vLLM, TGI, TensorRT-LLM give you")
        print("  TTL ON     = Memory adapts to input during the forward pass")
        print("               This is how the model was designed to run")
        print("  Contrib    = PPL reduction vs baseline: (baseline - condition) / baseline")
        print("  TTL Gap    = What the serving stack costs: Contrib ON - Contrib OFF")
        print()
        print(f"  Average memory contribution (TTL OFF): {avg_contrib_off:>7.1%}")
        print(f"  Average memory contribution (TTL ON):  {avg_contrib_on:>7.1%}")
        print(f"  Average TTL gap (serving cost):        {avg_gap:>+7.1%}")
        print()

        if avg_gap > 0.01:
            print("  The model performs better when TTL is active. Standard serving")
            print("  frameworks silence TTL by calling model.eval(). The gap above")
            print("  is what that decision costs.")
        elif avg_gap < -0.01:
            print("  TTL did not improve beyond-window predictions in this test.")
            print("  This may indicate the model was not trained long enough for")
            print("  TTL to learn meaningful memory updates.")
        else:
            print("  TTL ON and TTL OFF produced similar results. The memory module")
            print("  may already capture most information through trained weights.")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="NIAH benchmark: memory probe with TTL ON vs OFF comparison",
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to Atlas-MAG checkpoint (.pt file)"
    )
    parser.add_argument(
        "--tokenizer",
        default="data/tokenizer_smollm.json",
        help="Path to tokenizer (default: data/tokenizer_smollm.json)",
    )
    parser.add_argument(
        "--device", default="cuda:0", help="Device (default: cuda:0)"
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[768, 1024, 1536, 2048],
        help="Sequence lengths to test (default: 768 1024 1536 2048)",
    )
    parser.add_argument(
        "--n-sequences",
        type=int,
        default=4,
        help="Number of test sequences per length (default: 4)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=WINDOW_SIZE,
        help=f"Attention window size (default: {WINDOW_SIZE})",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file (optional)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Atlas-MAG: NIAH Memory Benchmark (TTL ON vs TTL OFF)")
    print("=" * 70)
    print(f"Checkpoint    : {args.checkpoint}")
    print(f"Device        : {args.device}")
    print(f"Window size   : {args.window_size}")
    print(f"Seq lengths   : {args.seq_lengths}")
    print(f"N sequences   : {args.n_sequences}")
    print(f"Seed          : {args.seed}")
    print()

    model, config, param_count, vocab_size = load_model(args.checkpoint, args.device)

    print(f"Model         : {param_count:.1f}M params, {config.get('n_layers', '?')} layers")
    print(f"Vocab         : {vocab_size}")
    print(f"Poly memory   : degree={config.get('poly_degree')}, rank={config.get('poly_rank')}")
    print(f"TTL config    : theta={config.get('ttl_theta')}, eta={config.get('ttl_eta')}, "
          f"ns_iters={config.get('ttl_ns_iters')}")
    print()
    print("Running benchmark...")

    rows = run_benchmark(
        model=model,
        vocab_size=vocab_size,
        seq_lengths=args.seq_lengths,
        n_sequences=args.n_sequences,
        window_size=args.window_size,
        device=args.device,
        seed=args.seed,
    )

    print_results_table(rows, args.window_size)

    if args.output:
        output_data = {
            "checkpoint": args.checkpoint,
            "config": config,
            "param_count_m": param_count,
            "window_size": args.window_size,
            "n_sequences": args.n_sequences,
            "seed": args.seed,
            "results": [asdict(r) for r in rows],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
