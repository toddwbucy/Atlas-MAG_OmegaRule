#!/usr/bin/env python3
"""
Train a BPE tokenizer on SmolLM-Corpus samples.

Creates a 32K vocabulary tokenizer optimized for the SmolLM-Corpus content:
- Cosmopedia-v2: Synthetic educational textbooks (40%)
- FineWeb-edu-dedup: Educational web content (50%)
- Python-edu-cleaned: Educational Python code (10%)

Usage:
    python scripts/train_tokenizer_smollm.py
    python scripts/train_tokenizer_smollm.py --target-chars 100000000 --output data/tokenizer_smollm.json
"""

import argparse
import random
import sys
import tempfile
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import setup_logging, get_logger

# Logger will be configured in main()
logger = get_logger(__name__)

# SmolLM corpus path
SMOLLM_PATH = Path("/bulk-store/training-datasets/smollm-corpus")

# Subset configurations
SUBSETS = {
    "cosmopedia-v2": {
        "path": SMOLLM_PATH / "cosmopedia-v2" / "data.parquet",
        "text_column": "text",
        "weight": 0.4,  # 40% of training corpus
    },
    "fineweb-edu-dedup": {
        "path": SMOLLM_PATH / "fineweb-edu-dedup" / "data.parquet",
        "text_column": "text",
        "weight": 0.5,  # 50% of training corpus
    },
    "python-edu-cleaned": {
        "path": SMOLLM_PATH / "python-edu-cleaned" / "data.parquet",
        "text_column": "text",
        "weight": 0.1,  # 10% of training corpus
    },
}


def iter_parquet_texts(
    parquet_path: Path,
    text_column: str = "text",
    max_chars: int | None = None,
    seed: int = 42,
) -> Iterator[str]:
    """
    Iterate over text samples from a parquet file using row group streaming.

    Args:
        parquet_path: Path to parquet file
        text_column: Column containing text
        max_chars: Maximum total characters to yield
        seed: Random seed for row group shuffling

    Yields:
        Text strings from the dataset
    """
    pf = pq.ParquetFile(parquet_path)
    num_row_groups = pf.metadata.num_row_groups

    # Shuffle row groups for better diversity
    rng = random.Random(seed)
    row_group_indices = list(range(num_row_groups))
    rng.shuffle(row_group_indices)

    total_chars = 0
    logger.info(f"Reading from {parquet_path.name} ({num_row_groups} row groups)")

    for rg_idx in row_group_indices:
        if max_chars and total_chars >= max_chars:
            break

        try:
            table = pf.read_row_group(rg_idx, columns=[text_column])
            texts = table[text_column].to_pylist()

            for text in texts:
                if text and len(text) > 50:  # Skip very short texts
                    yield text
                    total_chars += len(text)
                    if max_chars and total_chars >= max_chars:
                        break

        except Exception as e:
            logger.warning(f"Error reading row group {rg_idx}: {e}")
            continue


def sample_corpus_for_tokenizer(
    output_path: Path,
    target_chars: int = 50_000_000,
    seed: int = 42,
) -> Path:
    """
    Sample a balanced corpus from all SmolLM subsets for tokenizer training.

    Creates a temporary text file with samples from each subset proportional
    to their configured weights.

    Args:
        output_path: Path to save the corpus file
        target_chars: Target total characters (default: 50M)
        seed: Random seed for reproducibility

    Returns:
        Path to the created corpus file
    """
    logger.info(f"Sampling corpus for tokenizer training (target: {target_chars:,} chars)")

    with open(output_path, "w", encoding="utf-8") as f:
        for subset_name, config in SUBSETS.items():
            subset_chars = int(target_chars * config["weight"])
            logger.info(f"  {subset_name}: targeting {subset_chars:,} chars ({config['weight']*100:.0f}%)")

            chars_written = 0
            for text in iter_parquet_texts(
                config["path"],
                config["text_column"],
                max_chars=subset_chars,
                seed=seed,
            ):
                f.write(text + "\n")
                chars_written += len(text)

            logger.info(f"    -> wrote {chars_written:,} chars")

    file_size = output_path.stat().st_size
    logger.info(f"Corpus saved to {output_path} ({file_size / 1e6:.1f} MB)")

    return output_path


def train_smollm_tokenizer(
    corpus_path: Path,
    output_path: Path,
    vocab_size: int = 32000,
) -> Tokenizer:
    """
    Train a BPE tokenizer on the sampled corpus.

    Uses ByteLevel BPE (like GPT-2/LLaMA) with the following special tokens:
    - <pad>: ID 0
    - <unk>: ID 1
    - <bos>: ID 2
    - <eos>: ID 3

    Args:
        corpus_path: Path to training corpus file
        output_path: Path to save tokenizer.json
        vocab_size: Target vocabulary size

    Returns:
        Trained Tokenizer object
    """
    logger.info(f"Training BPE tokenizer with vocab_size={vocab_size}")

    # Initialize BPE tokenizer with unknown token
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Pre-tokenizer: ByteLevel (like GPT-2, handles any Unicode)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder: ByteLevel for proper detokenization
    tokenizer.decoder = decoders.ByteLevel()

    # Post-processor: ByteLevel for offset handling
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Special tokens (order matters - determines IDs)
    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

    # Trainer configuration
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # Train on corpus file
    logger.info(f"Training on corpus file: {corpus_path}")
    tokenizer.train([str(corpus_path)], trainer=trainer)

    # Verify vocab size
    actual_vocab = tokenizer.get_vocab_size()
    logger.info(f"Vocabulary size: {actual_vocab}")

    # Save tokenizer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    logger.info(f"Saved tokenizer to {output_path}")

    return tokenizer


def validate_tokenizer(tokenizer: Tokenizer) -> dict:
    """
    Validate tokenizer properties and efficiency.

    Returns:
        Dict with validation results
    """
    results = {}

    # Check vocab size
    vocab_size = tokenizer.get_vocab_size()
    results["vocab_size"] = vocab_size

    # Check special tokens
    vocab = tokenizer.get_vocab()
    results["special_tokens"] = {
        "pad": vocab.get("<pad>", -1),
        "unk": vocab.get("<unk>", -1),
        "bos": vocab.get("<bos>", -1),
        "eos": vocab.get("<eos>", -1),
    }

    # Test encode/decode roundtrip
    test_texts = [
        "Hello, world! This is a test.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "The quadratic formula is: x = (-b +/- sqrt(b^2 - 4ac)) / 2a",
        "import numpy as np\nimport torch\nfrom typing import List, Optional",
    ]

    roundtrip_ok = True
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        if decoded.strip() != text.strip():
            logger.warning(f"Roundtrip mismatch:\n  Original: {text[:50]}...\n  Decoded: {decoded[:50]}...")
            roundtrip_ok = False

    results["roundtrip_ok"] = roundtrip_ok

    # Test efficiency (chars per token)
    total_chars = 0
    total_tokens = 0
    for text in test_texts:
        encoded = tokenizer.encode(text)
        total_chars += len(text)
        total_tokens += len(encoded.ids)

    results["chars_per_token"] = total_chars / total_tokens if total_tokens > 0 else 0

    # Test Python indentation preservation
    python_code = "def foo():\n    if True:\n        return 42\n    else:\n        return 0"
    encoded = tokenizer.encode(python_code)
    decoded = tokenizer.decode(encoded.ids)
    results["python_indent_preserved"] = "\n    " in decoded and "\n        " in decoded

    return results


def main():
    parser = argparse.ArgumentParser(description="Train tokenizer on SmolLM-Corpus")
    parser.add_argument(
        "--target-chars",
        type=int,
        default=50_000_000,
        help="Target characters for training corpus (default: 50M)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Vocabulary size (default: 32000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/tokenizer_smollm.json",
        help="Output path for tokenizer",
    )
    parser.add_argument(
        "--keep-corpus",
        action="store_true",
        help="Keep the intermediate corpus file",
    )
    parser.add_argument(
        "--corpus-path",
        type=str,
        default=None,
        help="Custom corpus file path (if --keep-corpus or reusing existing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    logger.info("=" * 70)
    logger.info("SmolLM-Corpus Tokenizer Training")
    logger.info("=" * 70)
    logger.info(f"Target chars: {args.target_chars:,}")
    logger.info(f"Vocab size: {args.vocab_size}")
    logger.info(f"Output: {args.output}")

    # Verify SmolLM path exists
    if not SMOLLM_PATH.exists():
        logger.error(f"SmolLM corpus not found at {SMOLLM_PATH}")
        sys.exit(1)

    # Verify subsets exist
    for name, config in SUBSETS.items():
        if not config["path"].exists():
            logger.error(f"Subset {name} not found at {config['path']}")
            sys.exit(1)
        logger.info(f"Found subset: {name}")

    # Determine corpus path
    if args.corpus_path:
        corpus_path = Path(args.corpus_path)
        if corpus_path.exists():
            logger.info(f"Using existing corpus: {corpus_path}")
        else:
            logger.info(f"Creating corpus at: {corpus_path}")
            sample_corpus_for_tokenizer(corpus_path, args.target_chars, args.seed)
    else:
        # Create temporary corpus file
        if args.keep_corpus:
            corpus_path = Path("data/smollm_corpus_sample.txt")
        else:
            # Use mkstemp instead of deprecated mktemp (S306)
            fd, temp_path = tempfile.mkstemp(suffix=".txt")
            import os
            os.close(fd)  # Close file descriptor, we'll write via Path
            corpus_path = Path(temp_path)

        sample_corpus_for_tokenizer(corpus_path, args.target_chars, args.seed)

    # Train tokenizer
    output_path = Path(args.output)
    tokenizer = train_smollm_tokenizer(corpus_path, output_path, args.vocab_size)

    # Validate
    logger.info("\nValidating tokenizer...")
    results = validate_tokenizer(tokenizer)

    logger.info(f"  Vocab size: {results['vocab_size']}")
    logger.info(f"  Special tokens: {results['special_tokens']}")
    logger.info(f"  Roundtrip OK: {results['roundtrip_ok']}")
    logger.info(f"  Chars/token: {results['chars_per_token']:.2f} (ideal: 3.5-4.5)")
    logger.info(f"  Python indent preserved: {results['python_indent_preserved']}")

    # Check special token IDs
    special = results["special_tokens"]
    if special["pad"] != 0 or special["unk"] != 1 or special["bos"] != 2 or special["eos"] != 3:
        logger.warning("Special token IDs not in expected order!")

    # Cleanup temporary corpus
    if not args.keep_corpus and not args.corpus_path and corpus_path.exists():
        corpus_path.unlink()
        logger.info("Cleaned up temporary corpus file")

    logger.info("\n" + "=" * 70)
    logger.info("Tokenizer training complete!")
    logger.info(f"Saved to: {output_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
