#!/usr/bin/env python3
"""
Stress Test: TTL Memory Stability Under Increasing Context Length

This script loads the Atlas-MAG model with test-time learning (TTL) active
and feeds it progressively longer sequences. At each context length it
measures:

  1. Memory parameter norms (L2) — are they bounded or diverging?
  2. Momentum buffer norms — is accumulated momentum stable?
  3. Per-layer gradient norms — are gradients healthy?
  4. Omega loss per layer — is the inner-loop optimization converging?
  5. Perplexity — does prediction quality improve or degrade?
  6. NaN/Inf detection — any numerical blowups?

The key question: does the memory module remain numerically stable as
context grows far beyond the 512-token attention window?

If it does, the model is LEARNING from the extended context — the inner
loop is doing its job. If it diverges, we've found the stability boundary.

Paper: Atlas — Learning to Optimally Memorize the Context at Test Time
       arXiv:2505.23735 (Behrouz et al., 2025)

Usage:
    python scripts/stress_test_memory.py                              # defaults: 256 to 8192 tokens
    python scripts/stress_test_memory.py --seq-lengths 512 1024 2048 4096 8192
    python scripts/stress_test_memory.py --device cuda:1 --output results.json
    python scripts/stress_test_memory.py --no-reset                   # don't reset momentum between lengths
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from src.config import WINDOW_SIZE
from src.data.tokenizer import load_tokenizer
from src.model.skeleton import AtlasMAGSkeleton

HF_REPO = "r3d91ll/Atlas-MAG_OmegaRule"
HF_CHECKPOINT = "checkpoint_step008800.pt"
HF_TOKENIZER = "tokenizer_smollm.json"


def download_from_hf(filename: str) -> str:
    """Download a file from HuggingFace Hub, returning the cached path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        print("  Or provide --checkpoint and --tokenizer paths manually.")
        sys.exit(1)

    print(f"Downloading {filename} from {HF_REPO}...")
    path = hf_hub_download(repo_id=HF_REPO, filename=filename)
    print(f"  Cached at: {path}")
    return path


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


@dataclass
class MemorySnapshot:
    """Snapshot of memory module state at one layer."""
    layer: int
    # Parameter norms (L2)
    param_norms: Dict[str, float] = field(default_factory=dict)
    # Momentum buffer norms
    momentum_norms: Dict[str, float] = field(default_factory=dict)
    # Gradient norms (from TTL stats)
    grad_norms: Dict[str, float] = field(default_factory=dict)
    # Update norms (post Newton-Schulz)
    update_norms: Dict[str, float] = field(default_factory=dict)
    # Omega loss for this layer
    omega_loss: float = 0.0


@dataclass
class StressResult:
    """Result from one context length probe."""
    seq_len: int
    beyond_window: int
    ppl_full: float
    ppl_beyond_window: float
    loss_full: float
    loss_beyond_window: float
    time_s: float
    has_nan: bool
    has_inf: bool
    layer_snapshots: List[MemorySnapshot] = field(default_factory=list)
    # Aggregate stats
    max_param_norm: float = 0.0
    max_momentum_norm: float = 0.0
    max_grad_norm: float = 0.0
    mean_omega_loss: float = 0.0


def snapshot_memory(model: AtlasMAGSkeleton) -> List[MemorySnapshot]:
    """Take a snapshot of all memory module states."""
    snapshots = []
    for i, block in enumerate(model.blocks):
        if not hasattr(block, "memory"):
            continue

        snap = MemorySnapshot(layer=i)
        memory = block.memory

        # Parameter norms
        for name, param in memory.named_parameters():
            snap.param_norms[name] = param.data.norm().item()

        # Momentum buffer norms
        for name, _ in memory.named_parameters():
            buffer_name = f"momentum_{name.replace('.', '_')}"
            if hasattr(memory, buffer_name):
                buf = getattr(memory, buffer_name)
                snap.momentum_norms[name] = buf.norm().item()

        snapshots.append(snap)

    return snapshots


def check_numerical_health(model: AtlasMAGSkeleton) -> tuple:
    """Check for NaN or Inf in model parameters and buffers."""
    has_nan = False
    has_inf = False

    for name, param in model.named_parameters():
        if torch.isnan(param.data).any():
            has_nan = True
        if torch.isinf(param.data).any():
            has_inf = True

    for name, buf in model.named_buffers():
        if torch.isnan(buf).any():
            has_nan = True
        if torch.isinf(buf).any():
            has_inf = True

    return has_nan, has_inf


def run_probe(
    model: AtlasMAGSkeleton,
    input_ids: torch.Tensor,
    window_size: int,
) -> StressResult:
    """Run a single probe at one context length with TTL active."""
    seq_len = input_ids.shape[1]
    beyond_window = max(0, seq_len - window_size)

    model.train()  # TTL active

    t0 = time.time()

    # Forward with TTL stats
    output = model(input_ids, return_ttl_stats=True)
    logits, ttl_stats_list = output

    elapsed = time.time() - t0

    # Full-sequence perplexity
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss_full = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    )
    ppl_full = torch.exp(loss_full).item()

    # Beyond-window perplexity (where only memory can help)
    if beyond_window > 1:
        boundary = min(window_size, seq_len - 2)
        labels_beyond = shift_labels[:, boundary:].contiguous()
        logits_beyond = shift_logits[:, boundary:, :].contiguous()
        if labels_beyond.numel() > 0:
            loss_bw = F.cross_entropy(
                logits_beyond.reshape(-1, logits_beyond.size(-1)),
                labels_beyond.reshape(-1),
                reduction="mean",
            )
            ppl_bw = torch.exp(loss_bw).item()
            loss_bw_val = loss_bw.item()
        else:
            ppl_bw = float("inf")
            loss_bw_val = float("inf")
    else:
        ppl_bw = float("nan")
        loss_bw_val = float("nan")

    # Snapshot memory state
    snapshots = snapshot_memory(model)

    # Merge TTL stats into snapshots
    for snap, stats in zip(snapshots, ttl_stats_list):
        if stats:
            snap.omega_loss = stats.get("omega_loss", 0.0)
            for key, val in stats.items():
                if key.endswith("_grad_norm"):
                    snap.grad_norms[key] = val
                elif key.endswith("_update_norm"):
                    snap.update_norms[key] = val

    # Check numerical health
    has_nan, has_inf = check_numerical_health(model)

    # Aggregate stats
    all_param_norms = [v for s in snapshots for v in s.param_norms.values()]
    all_momentum_norms = [v for s in snapshots for v in s.momentum_norms.values()]
    all_grad_norms = [v for s in snapshots for v in s.grad_norms.values()]
    omega_losses = [s.omega_loss for s in snapshots if s.omega_loss > 0]

    result = StressResult(
        seq_len=seq_len,
        beyond_window=beyond_window,
        ppl_full=ppl_full,
        ppl_beyond_window=ppl_bw,
        loss_full=loss_full.item(),
        loss_beyond_window=loss_bw_val,
        time_s=elapsed,
        has_nan=has_nan,
        has_inf=has_inf,
        layer_snapshots=snapshots,
        max_param_norm=max(all_param_norms) if all_param_norms else 0.0,
        max_momentum_norm=max(all_momentum_norms) if all_momentum_norms else 0.0,
        max_grad_norm=max(all_grad_norms) if all_grad_norms else 0.0,
        mean_omega_loss=sum(omega_losses) / len(omega_losses) if omega_losses else 0.0,
    )

    return result


def print_results(results: List[StressResult], window_size: int, reset_between: bool):
    """Print results as a formatted table."""
    print()
    print("=" * 100)
    print("TTL MEMORY STABILITY STRESS TEST")
    print(f"Attention window: {window_size} tokens")
    print(f"Momentum reset between lengths: {'yes' if reset_between else 'NO (accumulating)'}")
    print("=" * 100)
    print()

    # Summary table
    print(f"{'Seq Len':>8}  {'Beyond':>6}  {'PPL':>10}  {'PPL>Win':>10}  "
          f"{'Omega':>8}  {'Param‖':>10}  {'Mom‖':>10}  {'Grad‖':>10}  "
          f"{'NaN':>4}  {'Inf':>4}  {'Time':>6}")
    print(f"{'':>8}  {'Window':>6}  {'(full)':>10}  {'(memory)':>10}  "
          f"{'Loss':>8}  {'(max)':>10}  {'(max)':>10}  {'(max)':>10}  "
          f"{'':>4}  {'':>4}  {'(s)':>6}")
    print("-" * 100)

    for r in results:
        def fmt(v, width=10):
            if math.isnan(v):
                return "n/a".rjust(width)
            if math.isinf(v):
                return "inf".rjust(width)
            if v > 99999:
                return f"{v:.0f}".rjust(width)
            if v > 100:
                return f"{v:.1f}".rjust(width)
            return f"{v:.4f}".rjust(width)

        nan_flag = "YES" if r.has_nan else "-"
        inf_flag = "YES" if r.has_inf else "-"

        print(
            f"{r.seq_len:>8}  "
            f"{r.beyond_window:>6}  "
            f"{fmt(r.ppl_full)}  "
            f"{fmt(r.ppl_beyond_window)}  "
            f"{fmt(r.mean_omega_loss, 8)}  "
            f"{fmt(r.max_param_norm)}  "
            f"{fmt(r.max_momentum_norm)}  "
            f"{fmt(r.max_grad_norm)}  "
            f"{nan_flag:>4}  "
            f"{inf_flag:>4}  "
            f"{r.time_s:>5.1f}s"
        )

    print("-" * 100)
    print()

    # Stability assessment
    if not results:
        return

    first = results[0]
    last = results[-1]

    print("STABILITY ASSESSMENT")
    print("-" * 50)

    # Parameter norm drift
    if first.max_param_norm > 0:
        drift = (last.max_param_norm - first.max_param_norm) / first.max_param_norm
        print(f"  Parameter norm drift:  {drift:>+.2%}  "
              f"({first.max_param_norm:.4f} → {last.max_param_norm:.4f})")
    else:
        print(f"  Parameter norm drift:  n/a")

    # Momentum growth
    if first.max_momentum_norm > 0:
        growth = last.max_momentum_norm / first.max_momentum_norm
        print(f"  Momentum growth:      {growth:>6.2f}x  "
              f"({first.max_momentum_norm:.4f} → {last.max_momentum_norm:.4f})")
    else:
        print(f"  Momentum growth:      n/a")

    # PPL trend
    ppl_trend = last.ppl_full - first.ppl_full
    print(f"  Perplexity trend:     {ppl_trend:>+.2f}  "
          f"({first.ppl_full:.2f} → {last.ppl_full:.2f})")

    # NaN/Inf check
    any_nan = any(r.has_nan for r in results)
    any_inf = any(r.has_inf for r in results)
    if any_nan or any_inf:
        print(f"  Numerical issues:     {'NaN detected!' if any_nan else ''} "
              f"{'Inf detected!' if any_inf else ''}")
        for r in results:
            if r.has_nan or r.has_inf:
                print(f"    → First at seq_len={r.seq_len}")
                break
    else:
        print(f"  Numerical issues:     None (clean)")

    print()

    # Verdict
    stable = (
        not any_nan
        and not any_inf
        and (first.max_param_norm == 0 or abs((last.max_param_norm - first.max_param_norm) / first.max_param_norm) < 0.5)
    )

    if stable:
        print("  VERDICT: Memory parameters remain STABLE across context lengths.")
        print("  The TTL inner loop is functioning as designed — the model learns")
        print("  from extended context without numerical divergence.")
    else:
        print("  VERDICT: Memory shows INSTABILITY at longer contexts.")
        print("  The TTL inner loop may need learning rate tuning or gradient")
        print("  clipping for safe deployment at these context lengths.")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Stress test: TTL memory stability under increasing context",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to checkpoint (.pt). If omitted, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--tokenizer", default=None,
        help="Path to tokenizer. If omitted, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--device", default="cuda:0", help="Device (default: cuda:0)"
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192],
        help="Sequence lengths to test (default: 256 to 8192)",
    )
    parser.add_argument(
        "--n-sequences",
        type=int,
        default=2,
        help="Number of sequences per length (default: 2, averaged)",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Don't reset momentum between context lengths (test accumulation)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results to JSON file (optional)",
    )
    args = parser.parse_args()

    # Resolve paths
    checkpoint_path = args.checkpoint or download_from_hf(HF_CHECKPOINT)

    print("=" * 70)
    print("Atlas-MAG: TTL Memory Stability Stress Test")
    print("=" * 70)
    print(f"Checkpoint     : {checkpoint_path}")
    print(f"Device         : {args.device}")
    print(f"Sequence lengths: {args.seq_lengths}")
    print(f"N sequences    : {args.n_sequences}")
    print(f"Reset momentum : {'between each length' if not args.no_reset else 'NEVER (accumulating)'}")
    print(f"Seed           : {args.seed}")
    print()

    model, config, param_count, vocab_size = load_model(checkpoint_path, args.device)

    print(f"Model          : {param_count:.1f}M params, {config.get('n_layers', '?')} layers")
    print(f"Vocab          : {vocab_size}")
    print(f"Poly memory    : degree={config.get('poly_degree')}, rank={config.get('poly_rank')}")
    print(f"TTL config     : theta={config.get('ttl_theta')}, alpha={config.get('ttl_alpha')}, "
          f"eta={config.get('ttl_eta')}, ns_iters={config.get('ttl_ns_iters')}")
    print(f"Window size    : {WINDOW_SIZE}")
    print()

    # Generate deterministic input
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    max_len = max(args.seq_lengths)
    # Pre-generate one long sequence, slice for each length
    full_ids = torch.randint(
        0, vocab_size, (args.n_sequences, max_len), generator=gen
    ).to(args.device)

    results: List[StressResult] = []

    print("Running stress test...")
    print()

    for seq_len in args.seq_lengths:
        input_ids = full_ids[:, :seq_len]

        # Reset momentum between lengths (unless --no-reset)
        if not args.no_reset:
            model.reset_ttl_momentum()

        # Run probe
        result = run_probe(model, input_ids, WINDOW_SIZE)
        results.append(result)

        # Live progress
        status = ""
        if result.has_nan:
            status = " [NaN!]"
        elif result.has_inf:
            status = " [Inf!]"

        beyond = max(0, seq_len - WINDOW_SIZE)
        print(f"  seq_len={seq_len:>5}  beyond_window={beyond:>4}  "
              f"ppl={result.ppl_full:>8.2f}  "
              f"omega={result.mean_omega_loss:.4f}  "
              f"param‖={result.max_param_norm:.4f}  "
              f"mom‖={result.max_momentum_norm:.4f}  "
              f"{result.time_s:.1f}s{status}")

        # Early exit on numerical failure
        if result.has_nan:
            print(f"\n  ABORT: NaN detected at seq_len={seq_len}. Stopping.")
            break

    print_results(results, WINDOW_SIZE, reset_between=not args.no_reset)

    # Save JSON
    if args.output:
        output_data = {
            "config": config,
            "param_count_m": param_count,
            "window_size": WINDOW_SIZE,
            "n_sequences": args.n_sequences,
            "reset_between": not args.no_reset,
            "seed": args.seed,
            "results": [],
        }
        for r in results:
            row = {
                "seq_len": r.seq_len,
                "beyond_window": r.beyond_window,
                "ppl_full": r.ppl_full,
                "ppl_beyond_window": r.ppl_beyond_window,
                "loss_full": r.loss_full,
                "loss_beyond_window": r.loss_beyond_window,
                "time_s": r.time_s,
                "has_nan": r.has_nan,
                "has_inf": r.has_inf,
                "max_param_norm": r.max_param_norm,
                "max_momentum_norm": r.max_momentum_norm,
                "max_grad_norm": r.max_grad_norm,
                "mean_omega_loss": r.mean_omega_loss,
                "layer_details": [
                    {
                        "layer": s.layer,
                        "param_norms": s.param_norms,
                        "momentum_norms": s.momentum_norms,
                        "omega_loss": s.omega_loss,
                    }
                    for s in r.layer_snapshots
                ],
            }
            output_data["results"].append(row)

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
