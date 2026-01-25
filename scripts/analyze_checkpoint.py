#!/usr/bin/env python3
"""
Comprehensive checkpoint analysis for Atlas-MAG.

Tests:
1. Gate statistics - are gates actually polarizing?
2. Memory usage - is the Atlas memory being used?
3. Generation quality - is output coherent?
4. Attention vs Memory - which path is the model preferring?
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import VOCAB_SIZE
from src.model.skeleton import AtlasMAGSkeleton
from src.data.tokenizer import load_tokenizer
from src.inference.engine import InferenceEngine
from src.inference.generator import TextGenerator, GenerationConfig


def analyze_gates(model):
    """Analyze gate statistics to verify they're not stuck at 0.5."""
    print("\n" + "=" * 60)
    print("GATE ANALYSIS")
    print("=" * 60)

    gate_values = model.get_gate_values()
    gates = torch.tensor(gate_values)

    print(f"\nGate values per layer:")
    for i, g in enumerate(gate_values):
        bar_len = int(g * 40)
        bar = "#" * bar_len + "." * (40 - bar_len)
        direction = "-> MEMORY" if g > 0.5 else "-> ATTENTION" if g < 0.5 else "-> BALANCED"
        print(f"  Layer {i}: {g:.4f} [{bar}] {direction}")

    print(f"\nAggregate statistics:")
    print(f"  Mean: {gates.mean():.4f}")
    print(f"  Std:  {gates.std():.4f}")
    print(f"  Min:  {gates.min():.4f}")
    print(f"  Max:  {gates.max():.4f}")

    # Check for polarization
    attention_heavy = (gates < 0.3).sum().item()
    memory_heavy = (gates > 0.7).sum().item()
    balanced = len(gates) - attention_heavy - memory_heavy

    print(f"\nPolarization:")
    print(f"  Attention-heavy (<0.3): {attention_heavy}/{len(gates)} layers")
    print(f"  Memory-heavy (>0.7):    {memory_heavy}/{len(gates)} layers")
    print(f"  Balanced (0.3-0.7):     {balanced}/{len(gates)} layers")

    if gates.std() < 0.05:
        print("\n[!] WARNING: Gates have low variance - may not be learning!")
    elif gates.std() > 0.1:
        print("\n[OK] GOOD: Gates show healthy polarization")

    return {
        "mean": gates.mean().item(),
        "std": gates.std().item(),
        "values": gate_values,
    }


def analyze_memory(model, device):
    """Analyze if Atlas memory is being used."""
    print("\n" + "=" * 60)
    print("MEMORY ANALYSIS")
    print("=" * 60)

    # Check persistent memory
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'atlas_memory'):
            mem = block.atlas_memory
            if hasattr(mem, 'M_persistent'):
                m_norm = mem.M_persistent.norm().item()
                print(f"  Layer {i} M_persistent norm: {m_norm:.4f}")
            if hasattr(mem, 'norm_persistent'):
                print(f"  Layer {i} norm_persistent: {mem.norm_persistent:.4f}")

    # Test memory contribution by comparing with/without memory
    print("\nTesting memory contribution...")

    # Get gate values to see memory weight
    gate_values = model.get_gate_values()
    avg_memory_weight = sum(gate_values) / len(gate_values)

    print(f"  Average memory weight (gate value): {avg_memory_weight:.4f}")
    print(f"  Average attention weight: {1 - avg_memory_weight:.4f}")

    if avg_memory_weight > 0.6:
        print("\n[OK] Model is MEMORY-dominant")
    elif avg_memory_weight < 0.4:
        print("\n[OK] Model is ATTENTION-dominant")
    else:
        print("\n[OK] Model uses BALANCED attention/memory mix")


def test_generation(generator, prompts, config):
    """Test generation quality with multiple prompts."""
    print("\n" + "=" * 60)
    print("GENERATION QUALITY TEST")
    print("=" * 60)

    for prompt in prompts:
        print(f"\n--- Prompt: '{prompt}' ---")

        try:
            output = generator.generate(prompt, config)
            # Remove prompt from output for clarity
            generated = output[len(prompt):] if output.startswith(prompt) else output
            print(f"Generated: {generated[:200]}...")

            # Basic quality checks
            if len(set(generated[:50])) < 5:
                print("[!] WARNING: Highly repetitive output")
            elif "@-@" in generated[:100]:
                print("[!] Note: Contains tokenizer artifacts (@-@)")
            else:
                print("[OK] Output appears varied")

        except Exception as e:
            print(f"ERROR: {e}")


def test_perplexity_samples(model, tokenizer, device):
    """Test perplexity on sample sentences."""
    print("\n" + "=" * 60)
    print("PERPLEXITY SAMPLES")
    print("=" * 60)

    samples = [
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning, there was nothing but darkness.",
        "The stock market crashed yesterday causing widespread panic.",
        "askjdhf laksjdhf alskjdhf random noise text here",
    ]

    model.train(False)
    for text in samples:
        tokens = tokenizer.encode(text, add_bos=True, add_eos=False)
        input_ids = torch.tensor([tokens], device=device)

        with torch.no_grad():
            logits = model(input_ids)
            # Compute loss
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1),
            )
            ppl = torch.exp(loss).item()

        print(f"  PPL {ppl:>8.2f}: '{text[:50]}...'")


def main():
    parser = argparse.ArgumentParser(description="Analyze AtlasMAG checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/wikitext103_54m/best_model.pt",
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="data/tokenizer.json",
        help="Path to tokenizer",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",  # Use GPU 1 for testing while training on GPU 0
        help="Device for inference",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ATLASMAG-TNT CHECKPOINT ANALYSIS")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")

    # Load checkpoint
    print("\nLoading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = checkpoint.get("config", {})

    print(f"  Step: {checkpoint.get('step', 'N/A')}")
    print(f"  Val PPL: {checkpoint.get('val_ppl', 'N/A')}")
    print(f"  Config: dim={config.get('dim')}, layers={config.get('n_layers')}")

    # Create model
    model = AtlasMAGSkeleton(
        vocab_size=VOCAB_SIZE,
        dim=config.get("dim", 512),
        n_layers=config.get("n_layers", 6),
        n_heads=config.get("n_heads", 8),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.train(False)

    print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer)

    # Run analyses
    gate_stats = analyze_gates(model)
    analyze_memory(model, args.device)
    test_perplexity_samples(model, tokenizer, args.device)

    # Create generator and test
    engine = InferenceEngine(model=model, device=args.device)
    generator = TextGenerator(engine, tokenizer)

    gen_config = GenerationConfig(
        max_new_tokens=100,
        temperature=args.temperature,
        top_k=50,
        top_p=0.9,
        do_sample=True,
    )

    test_prompts = [
        "The president announced",
        "Scientists discovered that",
        "In a surprising turn of events,",
        "The technology industry",
        "Once upon a time in a land far away,",
    ]

    test_generation(generator, test_prompts, gen_config)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Validation PPL: {checkpoint.get('val_ppl', 'N/A')}")
    print(f"  Gate std: {gate_stats['std']:.4f}")
    print(f"  Gate mean: {gate_stats['mean']:.4f}")

    if gate_stats['std'] > 0.1:
        print("\n[OK] Gates are polarizing - memory/attention routing is learning")
    else:
        print("\n[!] Gates have low variance - model may not be routing effectively")


if __name__ == "__main__":
    main()
