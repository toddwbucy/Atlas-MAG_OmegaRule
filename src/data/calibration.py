"""
Calibration Data Pipeline for Phase 0.

Uses WikiText-2 (or similar) for domain-neutral calibration data.
This data is used to compute W_init via steady-state analysis.

User Decision: WikiText-2 for fast iteration
"""

import logging
from pathlib import Path
from typing import cast, Iterator, List, Optional, Protocol, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class SizedDataset(Protocol):
    """Protocol for datasets with __len__ and __getitem__."""
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Tensor: ...

from src.config import (
    CALIBRATION_TOKENS,
    CALIBRATION_BATCH_SIZE,
    CALIBRATION_SEQ_LEN,
)

logger = logging.getLogger(__name__)


def load_wikitext(
    split: str = "train",
    cache_dir: Optional[str] = None,
) -> List[str]:
    """
    Load WikiText-2 dataset.

    Args:
        split: Dataset split ("train", "validation", "test")
        cache_dir: Optional cache directory for downloaded data

    Returns:
        List of text documents
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required: pip install datasets")

    logger.info(f"Loading WikiText-2 ({split} split)...")

    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split=split,
        cache_dir=cache_dir,
    )

    # Filter out empty lines and section headers
    texts = []
    for item in dataset:
        text = item["text"].strip()
        if text and not text.startswith(" = "):  # Skip headers
            texts.append(text)

    logger.info(f"Loaded {len(texts)} documents from WikiText-2 {split}")

    return texts


class CalibrationDataset(Dataset):
    """
    Dataset for calibration data.

    Tokenizes text and creates fixed-length sequences for
    W_init steady-state calibration.

    Args:
        texts: List of text documents
        tokenizer: Tokenizer instance (or None for char-level)
        seq_len: Sequence length for each sample
        max_tokens: Maximum total tokens to use
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer=None,
        seq_len: int = CALIBRATION_SEQ_LEN,
        max_tokens: int = CALIBRATION_TOKENS,
    ):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        # Tokenize all texts into a single stream
        if tokenizer is not None:
            all_tokens = []
            for text in texts:
                tokens = tokenizer.encode(text, add_bos=False, add_eos=False)
                all_tokens.extend(tokens)
                if len(all_tokens) >= max_tokens:
                    break
            all_tokens = all_tokens[:max_tokens]
        else:
            # Fallback: character-level tokenization for testing
            all_text = " ".join(texts)[:max_tokens]
            all_tokens = [ord(c) % 256 for c in all_text]

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)

        # Calculate number of complete sequences
        self.n_samples = len(self.tokens) // seq_len

        logger.info(
            f"CalibrationDataset: {len(self.tokens)} tokens, "
            f"{self.n_samples} samples of length {seq_len}"
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tensor:
        """Get a single sequence."""
        start = idx * self.seq_len
        end = start + self.seq_len
        return self.tokens[start:end]


class SimpleCalibrationDataset(Dataset):
    """
    Simple dataset using random token IDs.

    Used when we don't have a trained tokenizer yet.
    Creates random sequences for testing the model architecture.

    Args:
        vocab_size: Vocabulary size for random tokens
        seq_len: Sequence length
        num_samples: Number of samples to generate
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        seq_len: int = CALIBRATION_SEQ_LEN,
        num_samples: int = 100,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

        # Pre-generate random tokens for reproducibility
        torch.manual_seed(42)
        self.tokens = torch.randint(
            4,  # Skip special tokens
            vocab_size,
            (num_samples, seq_len),
            dtype=torch.long,
        )

        logger.info(
            f"SimpleCalibrationDataset: {num_samples} random samples "
            f"of length {seq_len}, vocab_size={vocab_size}"
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tensor:
        return self.tokens[idx]


def create_calibration_loader(
    num_tokens: int = CALIBRATION_TOKENS,
    batch_size: int = CALIBRATION_BATCH_SIZE,
    seq_len: int = CALIBRATION_SEQ_LEN,
    tokenizer=None,
    use_wikitext: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for calibration data.

    Args:
        num_tokens: Total number of tokens to use
        batch_size: Batch size
        seq_len: Sequence length
        tokenizer: Optional tokenizer (uses random if None)
        use_wikitext: Whether to load WikiText-2 (requires internet)
        num_workers: Number of data loading workers

    Returns:
        DataLoader yielding batches of token IDs
    """
    dataset: SizedDataset

    if use_wikitext and tokenizer is not None:
        try:
            texts = load_wikitext("train")
            dataset = CalibrationDataset(
                texts,
                tokenizer,
                seq_len=seq_len,
                max_tokens=num_tokens,
            )
        except (ImportError, OSError, RuntimeError) as e:
            logger.warning(f"Failed to load WikiText-2: {e}")
            logger.warning("Falling back to random data")
            use_wikitext = False

    if not use_wikitext or tokenizer is None:
        # Fallback to random data for testing
        num_samples = num_tokens // seq_len
        dataset = SimpleCalibrationDataset(
            vocab_size=32000,
            seq_len=seq_len,
            num_samples=num_samples,
        )

    # Guard against empty datasets - DataLoader crashes with batch_size=0
    if len(dataset) == 0:
        raise ValueError(
            "Calibration dataset is empty. "
            "Ensure num_tokens >= seq_len and input texts are non-empty."
        )

    # Adjust batch size if dataset is too small
    actual_batch_size = min(batch_size, len(dataset))
    if actual_batch_size < batch_size:
        logger.warning(
            f"Dataset too small ({len(dataset)} samples), "
            f"reducing batch_size from {batch_size} to {actual_batch_size}"
        )

    loader: DataLoader[Tensor] = DataLoader(
        cast(Dataset[Tensor], dataset),
        batch_size=actual_batch_size,
        shuffle=False,  # Deterministic for calibration
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,  # Keep all samples for small datasets
    )

    logger.info(
        f"Created calibration loader: {len(dataset)} samples, "
        f"batch_size={batch_size}, {len(loader)} batches"
    )

    return loader


def iterate_calibration_tokens(
    loader: DataLoader,
    max_tokens: int = CALIBRATION_TOKENS,
) -> Iterator[Tensor]:
    """
    Iterate over calibration data up to max_tokens.

    Args:
        loader: Calibration DataLoader
        max_tokens: Maximum tokens to yield

    Yields:
        Batches of token IDs
    """
    tokens_seen = 0
    for batch in loader:
        if tokens_seen >= max_tokens:
            break
        yield batch
        tokens_seen += batch.numel()
        if tokens_seen >= max_tokens:
            break


def save_corpus_for_tokenizer(
    output_path: Union[str, Path],
    max_chars: int = 10_000_000,
) -> Path:
    """
    Save WikiText-2 as a text file for tokenizer training.

    Args:
        output_path: Where to save the corpus file
        max_chars: Maximum characters to save

    Returns:
        Path to saved corpus file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    texts = load_wikitext("train")

    with open(output_path, "w", encoding="utf-8") as f:
        chars_written = 0
        for text in texts:
            if chars_written >= max_chars:
                break
            f.write(text + "\n")
            chars_written += len(text) + 1

    logger.info(f"Saved corpus to {output_path}: {chars_written} characters")

    return output_path
