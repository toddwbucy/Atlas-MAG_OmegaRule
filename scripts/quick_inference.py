#!/usr/bin/env python3
"""Quick inference script for testing checkpoints."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from src.model.skeleton import AtlasMAGSkeleton
from src.data.tokenizer import load_tokenizer


def load_model(model_path: str, device: str):
    """Load model from checkpoint, respecting disable_memory setting."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    # Infer vocab size from checkpoint weights
    vocab_size = checkpoint["model_state_dict"]["tok_emb.weight"].shape[0]

    model = AtlasMAGSkeleton(
        vocab_size=vocab_size,
        dim=config.get("dim", 768),
        n_layers=config.get("n_layers", 12),
        n_heads=config.get("n_heads", 12),
        disable_memory=config.get("disable_memory", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config, vocab_size


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_tokens: int = 100,
             temperature: float = 0.7, top_k: int = 50, device: str = "cuda"):
    """Generate text from prompt."""

    # Encode prompt
    encoded = tokenizer.encode(prompt)
    # Handle both tokenizers (list) and HF tokenizers (object with .ids)
    input_ids = encoded.ids if hasattr(encoded, 'ids') else encoded
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated = list(input_ids[0].cpu().numpy())

    for _ in range(max_tokens):
        # Get logits for last position
        # Determine device type for autocast (cuda vs cpu)
        device_type = "cuda" if "cuda" in device else "cpu"
        with torch.amp.autocast(device_type):
            logits = model(input_ids)

        next_token_logits = logits[0, -1, :] / temperature

        # Top-k filtering (clamp to vocab size to avoid torch.topk error)
        vocab_size = next_token_logits.size(-1)
        effective_k = min(top_k, vocab_size) if top_k > 0 else 0
        if effective_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, effective_k)[0][-1]
            next_token_logits[indices_to_remove] = float('-inf')

        # Sample
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        # Stop on EOS (get from tokenizer, default to 3 for our BPE tokenizer)
        eos_id = getattr(tokenizer, 'eos_id', 3)
        if next_token.item() == eos_id:
            break

    return tokenizer.decode(generated)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/smollm_195m_ablation/best_model.pt")
    parser.add_argument("--tokenizer", default="data/tokenizer_smollm.json")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--prompt", default="The quick brown fox")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=3)
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model, config, vocab_size = load_model(args.model, args.device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {param_count:.1f}M parameters, vocab_size={vocab_size}")
    print(f"Memory disabled: {config.get('disable_memory', False)}")

    tokenizer = load_tokenizer(args.tokenizer)

    print(f"\n{'='*60}")
    print(f"Prompt: {args.prompt}")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}")
    print(f"{'='*60}\n")

    for i in range(args.num_samples):
        print(f"--- Sample {i+1} ---")
        output = generate(
            model, tokenizer, args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
        )
        print(output)
        print()


if __name__ == "__main__":
    main()
