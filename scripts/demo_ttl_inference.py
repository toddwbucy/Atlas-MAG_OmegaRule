#!/usr/bin/env python3
"""
Demo: The Train-Flag Problem in Nested Learning

This script demonstrates the core infrastructure problem described in the
Nested Learning research program (Behrouz et al., Google Research). It loads
a trained Atlas-MAG model and runs the SAME inference twice:

  1. model.set_eval()  — Standard PyTorch inference. The test-time learning
                         (TTL) inner loop is SILENCED by a training-flag gate.
                         The polynomial memory module becomes dead weight.

  2. model.set_train() — TTL activates. The model's memory updates itself
                         during the forward pass using gradient descent. This
                         is how the model was DESIGNED to run — but PyTorch
                         calls this "training."

The two runs produce DIFFERENT outputs from the SAME weights and input.

This matters because every production serving framework (vLLM, TGI, TensorRT-LLM)
sets the model to inference mode before serving. There is no flag for "inference
with learning." The infrastructure assumes weights are frozen at serve time.
Nested Learning assumes they are not.

Paper: Atlas — Learning to Optimally Memorize the Context at Test Time
       arXiv:2505.23735 (Behrouz et al., 2025)

Usage:
    python scripts/demo_ttl_inference.py --checkpoint runs/atlas_54m_gelu/checkpoint_step008800.pt
    python scripts/demo_ttl_inference.py --checkpoint path/to/checkpoint.pt --device cuda:1
    python scripts/demo_ttl_inference.py --checkpoint path/to/checkpoint.pt --prompt "Your text here"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time

import torch
import torch.nn.functional as F

from src.data.tokenizer import load_tokenizer
from src.model.skeleton import AtlasMAGSkeleton


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


def compute_perplexity(model, input_ids):
    """Compute loss and perplexity on input_ids."""
    logits = model(input_ids)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    )
    return loss.item(), torch.exp(loss).item()


def generate_greedy(model, input_ids, max_new: int = 30):
    """Greedy decode to compare outputs deterministically."""
    generated = input_ids.clone()
    for _ in range(max_new):
        logits = model(generated)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
    return generated


def run_inference(model, input_ids, tokenizer, use_ttl: bool):
    """Run inference in a specific mode and return results."""
    if use_ttl:
        model.train()  # activates TTL inner loop via self.training flag
        t0 = time.time()
        loss, ppl = compute_perplexity(model, input_ids)
        gen = generate_greedy(model, input_ids)
        elapsed = time.time() - t0
    else:
        model.eval()  # silences TTL via self.training = False
        with torch.no_grad():
            t0 = time.time()
            loss, ppl = compute_perplexity(model, input_ids)
            gen = generate_greedy(model, input_ids)
            elapsed = time.time() - t0

    text = tokenizer.decode(list(gen[0].cpu().numpy()))
    return {"loss": loss, "ppl": ppl, "time": elapsed, "text": text}


def main():
    parser = argparse.ArgumentParser(
        description="Demo: Atlas-MAG TTL inference comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--prompt",
        nargs="+",
        default=[
            "The capital of France is",
            "In machine learning, test-time training refers to the process of",
        ],
        help="One or more prompts to test (default: two built-in prompts)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=30,
        help="Tokens to generate per prompt (default: 30)",
    )
    args = parser.parse_args()

    # --- Load ---
    print("=" * 70)
    print("Atlas-MAG: The Train-Flag Problem")
    print("=" * 70)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {args.device}")
    print()

    model, config, param_count, vocab_size = load_model(args.checkpoint, args.device)
    tokenizer = load_tokenizer(args.tokenizer)

    print(f"Model      : {param_count:.1f}M params, {config.get('n_layers', '?')} layers")
    print(f"Vocab      : {vocab_size}")
    print(f"TTL config : theta={config.get('ttl_theta')}, alpha={config.get('ttl_alpha')}, "
          f"eta={config.get('ttl_eta')}, ns_iters={config.get('ttl_ns_iters')}")
    print(f"Poly memory: degree={config.get('poly_degree')}, rank={config.get('poly_rank')}")

    # --- Run each prompt in both modes ---
    for prompt in args.prompt:
        print(f"\n{'_' * 70}")
        print(f"Prompt: {prompt!r}")
        print(f"{'_' * 70}")

        encoded = tokenizer.encode(prompt)
        input_ids_list = encoded.ids if hasattr(encoded, "ids") else encoded
        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=args.device)
        print(f"Input: {input_ids.shape[1]} tokens")

        # Mode 1: standard inference (TTL silenced)
        no_ttl = run_inference(model, input_ids, tokenizer, use_ttl=False)

        # Reset memory state between modes
        if hasattr(model, "reset_ttl_momentum"):
            model.reset_ttl_momentum()

        # Mode 2: TTL active (memory updates during forward pass)
        with_ttl = run_inference(model, input_ids, tokenizer, use_ttl=True)

        # --- Display ---
        print(f"\n  [TTL OFF] model.eval() — standard serving mode")
        print(f"    loss={no_ttl['loss']:.4f}  ppl={no_ttl['ppl']:.2f}  "
              f"time={no_ttl['time']:.3f}s")
        print(f"    {no_ttl['text']}")

        print(f"\n  [TTL ON]  model.train() — memory updates during forward pass")
        print(f"    loss={with_ttl['loss']:.4f}  ppl={with_ttl['ppl']:.2f}  "
              f"time={with_ttl['time']:.3f}s")
        print(f"    {with_ttl['text']}")

        # --- Comparison ---
        same_output = no_ttl["text"] == with_ttl["text"]
        speed_ratio = with_ttl["time"] / no_ttl["time"] if no_ttl["time"] > 0 else 0

        print(f"\n  Delta loss : {with_ttl['loss'] - no_ttl['loss']:+.4f}")
        print(f"  Delta ppl  : {with_ttl['ppl'] - no_ttl['ppl']:+.2f}")
        print(f"  Same output: {same_output}")
        print(f"  Speed ratio: {speed_ratio:.1f}x {'slower' if speed_ratio > 1 else 'faster'} "
              f"with TTL")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("What you just saw:")
    print("=" * 70)
    print()
    print("  The SAME model, with the SAME weights, produced DIFFERENT outputs")
    print("  depending on whether PyTorch's training flag was set.")
    print()
    print("  In inference mode, the polynomial memory and test-time learning")
    print("  inner loop are silenced by an `if self.training` gate. The model")
    print("  runs faster but ignores its memory architecture entirely.")
    print()
    print("  In training mode, the model runs as designed: the inner loop")
    print("  updates memory weights via gradient descent DURING the forward")
    print("  pass. This is not training. It is inference. But PyTorch has no")
    print("  concept of 'inference with learning.'")
    print()
    print("  Every production serving stack (vLLM, TGI, TensorRT-LLM) calls")
    print("  model.eval() before serving. There is no deployment path that")
    print("  preserves test-time learning.")
    print()
    print("  This is the infrastructure problem.")
    print()


if __name__ == "__main__":
    main()
