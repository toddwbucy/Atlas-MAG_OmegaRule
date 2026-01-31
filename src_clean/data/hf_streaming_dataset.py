"""
HuggingFace Streaming Dataset for SmolLM-Corpus.

Streams data directly from HuggingFace Hub - no local storage needed.
Perfect for cloud/pod training where you don't want to download 600GB.

Usage:
    from src_clean.data.hf_streaming_dataset import create_hf_streaming_dataloader

    train_loader = create_hf_streaming_dataloader(
        tokenizer=tokenizer,
        batch_size=32,
        seq_len=1024,
        max_tokens=3_000_000_000,  # Stop after 3B tokens (for 124M model)
    )
"""

import logging
from typing import Iterator

import torch
from datasets import interleave_datasets, load_dataset
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)


class HFStreamingDataset(IterableDataset):
    """
    Streaming dataset from HuggingFace SmolLM-Corpus.

    Streams and tokenizes on-the-fly, packing documents into fixed-length sequences.
    Stops after max_tokens to control training budget.

    Args:
        tokenizer: Tokenizer with encode() method
        seq_len: Sequence length for training
        max_tokens: Maximum tokens to yield (for Chinchilla-optimal training)
        subset_weights: Dict of subset name -> weight (default: cosmopedia 0.4, fineweb 0.5, python 0.1)
        seed: Random seed for shuffling
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int = 1024,
        max_tokens: int | None = None,
        subset_weights: dict[str, float] | None = None,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_tokens = max_tokens
        self.seed = seed

        # Default weights matching local dataset
        if subset_weights is None:
            subset_weights = {
                "cosmopedia-v2": 0.4,
                "fineweb-edu-dedup": 0.5,
                "python-edu": 0.1,
            }
        self.subset_weights = subset_weights

        # Normalize weights
        total = sum(subset_weights.values())
        self.subset_weights = {k: v/total for k, v in subset_weights.items()}

        logger.info(f"HFStreamingDataset: seq_len={seq_len}, max_tokens={max_tokens}")
        logger.info(f"  Subsets: {self.subset_weights}")

    def _load_streaming_dataset(self):
        """Load and interleave streaming datasets from HuggingFace."""
        datasets = []
        probs = []

        for subset_name, weight in self.subset_weights.items():
            try:
                ds = load_dataset(
                    "HuggingFaceTB/smollm-corpus",
                    subset_name,
                    split="train",
                    streaming=True,
                    trust_remote_code=True,
                )
                datasets.append(ds)
                probs.append(weight)
                logger.info(f"  Loaded {subset_name} (weight={weight:.2f})")
            except Exception as e:
                logger.warning(f"  Failed to load {subset_name}: {e}")

        if not datasets:
            raise ValueError("No datasets loaded successfully")

        # Interleave with weights
        combined = interleave_datasets(
            datasets,
            probabilities=probs,
            seed=self.seed,
            stopping_strategy="all_exhausted",
        )

        return combined

    def _iterate_texts(self) -> Iterator[str]:
        """Yield text documents from streaming dataset."""
        ds = self._load_streaming_dataset()

        for sample in ds:
            text = sample.get("text", "")
            if text and len(text) > 50:
                yield text

    def _pack_sequences(self) -> Iterator[dict]:
        """Pack documents into fixed-length sequences."""
        token_buffer: list[int] = []
        tokens_yielded = 0

        for text in self._iterate_texts():
            # Check if we've hit the token budget
            if self.max_tokens and tokens_yielded >= self.max_tokens:
                logger.info(f"Reached max_tokens limit: {tokens_yielded:,}")
                return

            # Tokenize with BOS/EOS
            tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            token_buffer.extend(tokens)

            # Yield complete sequences
            while len(token_buffer) >= self.seq_len + 1:
                seq = token_buffer[:self.seq_len + 1]
                token_buffer = token_buffer[self.seq_len + 1:]

                input_ids = torch.tensor(seq[:-1], dtype=torch.long)
                labels = torch.tensor(seq[1:], dtype=torch.long)

                tokens_yielded += self.seq_len
                yield {"input_ids": input_ids, "labels": labels}

                # Check budget after each sequence
                if self.max_tokens and tokens_yielded >= self.max_tokens:
                    return

    def __iter__(self) -> Iterator[dict]:
        return self._pack_sequences()


def create_hf_streaming_dataloader(
    tokenizer,
    batch_size: int = 32,
    seq_len: int = 1024,
    max_tokens: int | None = None,
    subset_weights: dict[str, float] | None = None,
    num_workers: int = 0,  # Streaming doesn't benefit from multi-worker
) -> DataLoader:
    """
    Create a streaming dataloader from HuggingFace SmolLM-Corpus.

    Args:
        tokenizer: Tokenizer instance
        batch_size: Batch size
        seq_len: Sequence length
        max_tokens: Maximum tokens to train on (for Chinchilla budget)
        subset_weights: Optional custom weights per subset
        num_workers: Number of workers (0 recommended for streaming)

    Returns:
        DataLoader for training

    Example:
        # For 124M model (Chinchilla = 20x = 2.56B tokens)
        loader = create_hf_streaming_dataloader(
            tokenizer=tokenizer,
            batch_size=32,
            seq_len=1024,
            max_tokens=2_560_000_000,
        )
    """
    dataset = HFStreamingDataset(
        tokenizer=tokenizer,
        seq_len=seq_len,
        max_tokens=max_tokens,
        subset_weights=subset_weights,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )


# Chinchilla token budgets for common model sizes
CHINCHILLA_TOKENS = {
    "44M": int(44e6 * 20),    # 880M
    "124M": int(128e6 * 20),  # 2.56B
    "350M": int(350e6 * 20),  # 7B
    "760M": int(760e6 * 20),  # 15.2B
}


if __name__ == "__main__":
    # Quick test
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src_clean.data.tokenizer import load_tokenizer

    print("Testing HFStreamingDataset...")

    tokenizer = load_tokenizer("data/tokenizer_smollm.json")

    # Test with small token budget
    loader = create_hf_streaming_dataloader(
        tokenizer=tokenizer,
        batch_size=4,
        seq_len=512,
        max_tokens=50_000,  # Just 50K for quick test
    )

    total_tokens = 0
    for i, batch in enumerate(loader):
        total_tokens += batch["input_ids"].numel()
        if i == 0:
            print(f"Batch shape: {batch['input_ids'].shape}")
        if i >= 10:
            break

    print(f"Got {total_tokens:,} tokens in {i+1} batches")
    print("✅ Test passed!")
