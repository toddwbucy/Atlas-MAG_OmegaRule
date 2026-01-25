"""
SmolLM-Corpus Dataset Loader.

Loads data from the SmolLM-Corpus dataset for pre-training.
Supports streaming from parquet files via row group iteration.

Dataset Structure:
    /bulk-store/training-datasets/smollm-corpus/
    ├── cosmopedia-v2/data.parquet      (~112GB, 39M samples)
    ├── fineweb-edu-dedup/data.parquet  (~505GB, 190M samples)
    └── python-edu-cleaned/data.parquet (~5.5GB, 7.7M samples)

All subsets have a 'text' column containing the document text.
"""

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)

# Default dataset location
SMOLLM_PATH = Path("/bulk-store/training-datasets/smollm-corpus")

# Subset configurations
SUBSET_CONFIGS = {
    "cosmopedia-v2": {
        "parquet": "cosmopedia-v2/data.parquet",
        "text_column": "text",
        "default_weight": 0.4,
    },
    "fineweb-edu-dedup": {
        "parquet": "fineweb-edu-dedup/data.parquet",
        "text_column": "text",
        "default_weight": 0.5,
    },
    "python-edu-cleaned": {
        "parquet": "python-edu-cleaned/data.parquet",
        "text_column": "text",
        "default_weight": 0.1,
    },
}


class Tokenizer(Protocol):
    """Protocol for tokenizer interface."""

    pad_id: int
    bos_id: int
    eos_id: int

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ...


@dataclass
class SubsetInfo:
    """Information about a dataset subset."""

    name: str
    parquet_path: Path
    text_column: str
    num_row_groups: int
    weight: float


def discover_subsets(
    base_path: Path = SMOLLM_PATH,
    subsets: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> list[SubsetInfo]:
    """
    Discover available subsets and their metadata.

    Args:
        base_path: Root path for SmolLM corpus
        subsets: Optional list of subset names to include
        weights: Optional custom weights per subset

    Returns:
        List of SubsetInfo objects
    """
    # Normalize to Path to avoid TypeError when callers pass string
    base_path = Path(base_path)

    if subsets is None:
        subsets = list(SUBSET_CONFIGS.keys())

    result = []
    for name in subsets:
        if name not in SUBSET_CONFIGS:
            logger.warning(f"Unknown subset: {name}, skipping")
            continue

        config = SUBSET_CONFIGS[name]
        parquet_path = base_path / config["parquet"]

        if not parquet_path.exists():
            logger.warning(f"Subset {name} not found at {parquet_path}")
            continue

        # Get row group count
        pf = pq.ParquetFile(parquet_path)
        num_row_groups = pf.metadata.num_row_groups

        weight = config["default_weight"]
        if weights and name in weights:
            weight = weights[name]

        result.append(
            SubsetInfo(
                name=name,
                parquet_path=parquet_path,
                text_column=config["text_column"],
                num_row_groups=num_row_groups,
                weight=weight,
            )
        )

    # Normalize weights to sum to 1
    total_weight = sum(s.weight for s in result)
    if total_weight > 0:
        for s in result:
            s.weight = s.weight / total_weight

    return result


def split_row_groups_for_validation(
    subset_info: SubsetInfo,
    val_ratio: float = 0.02,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """
    Split row groups into train and validation sets.

    Uses deterministic shuffling and takes the last val_ratio% for validation.
    This ensures validation set is always the same regardless of worker count.

    Args:
        subset_info: Subset metadata
        val_ratio: Fraction of row groups for validation
        seed: Random seed for shuffling

    Returns:
        Tuple of (train_row_groups, val_row_groups)
    """
    all_groups = list(range(subset_info.num_row_groups))

    # Shuffle deterministically
    rng = random.Random(seed)
    shuffled = all_groups.copy()
    rng.shuffle(shuffled)

    # Take last N for validation
    n_val = max(1, int(len(shuffled) * val_ratio))
    train_groups = shuffled[:-n_val]
    val_groups = shuffled[-n_val:]

    return train_groups, val_groups


class SmolLMDataset(IterableDataset):
    """
    Streaming dataset for SmolLM-Corpus.

    Streams documents from parquet files via row group iteration and tokenizes on-the-fly.
    Packs multiple documents into sequences of fixed length for efficient training.

    Multi-worker Support:
        When using with DataLoader + num_workers > 0, row groups are automatically
        sharded across workers to avoid duplicate data. Each worker processes a
        disjoint subset of row groups.

    Args:
        tokenizer: Tokenizer instance (must have encode method)
        seq_len: Sequence length for training
        subsets: List of subset names to include
        subset_weights: Optional dict of custom weights per subset
        base_path: Root path for SmolLM corpus
        exclude_val_groups: If True, exclude validation row groups
        val_ratio: Fraction of row groups reserved for validation
        seed: Random seed for shuffling
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        seq_len: int = 512,
        subsets: list[str] | None = None,
        subset_weights: dict[str, float] | None = None,
        base_path: Path = SMOLLM_PATH,
        exclude_val_groups: bool = True,
        val_ratio: float = 0.02,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.base_path = Path(base_path)
        self.exclude_val_groups = exclude_val_groups
        self.val_ratio = val_ratio
        self.seed = seed
        self.pad_id = tokenizer.pad_id

        # Discover subsets
        self.subsets = discover_subsets(base_path, subsets, subset_weights)
        if not self.subsets:
            raise ValueError(f"No valid subsets found at {base_path}")

        logger.info(f"SmolLMDataset: {len(self.subsets)} subsets")
        for s in self.subsets:
            logger.info(f"  {s.name}: {s.num_row_groups} row groups, weight={s.weight:.2f}")

        # Precompute train/val splits for each subset
        self._train_groups: dict[str, list[int]] = {}
        self._val_groups: dict[str, list[int]] = {}

        for subset in self.subsets:
            train_g, val_g = split_row_groups_for_validation(subset, val_ratio, seed)
            self._train_groups[subset.name] = train_g
            self._val_groups[subset.name] = val_g
            if exclude_val_groups:
                logger.info(
                    f"    {subset.name}: {len(train_g)} train, {len(val_g)} val row groups"
                )

    def _get_worker_row_groups(self, subset: SubsetInfo) -> list[int]:
        """
        Get row groups for current worker, accounting for multi-worker sharding.

        Each worker gets every num_workers-th row group starting from worker_id.
        """
        if self.exclude_val_groups:
            all_groups = self._train_groups[subset.name]
        else:
            all_groups = list(range(subset.num_row_groups))

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return all_groups
        else:
            # Shard across workers
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            return all_groups[worker_id::num_workers]

    def _iterate_texts(self, subset: SubsetInfo) -> Iterator[str]:
        """Yield texts from a subset's row groups (worker-aware)."""
        row_groups = self._get_worker_row_groups(subset)

        # Shuffle row groups for each iteration
        rng = random.Random(self.seed)
        rng.shuffle(row_groups)

        pf = pq.ParquetFile(subset.parquet_path)

        for rg_idx in row_groups:
            try:
                table = pf.read_row_group(rg_idx, columns=[subset.text_column])
                texts = table[subset.text_column].to_pylist()

                for text in texts:
                    if text and len(text) > 50:
                        yield text

            except Exception as e:
                logger.warning(f"Error reading {subset.name} row group {rg_idx}: {e}")
                continue

    def _iterate_all_texts(self) -> Iterator[str]:
        """Yield texts from all subsets with weighted interleaving."""
        # For simplicity, iterate through subsets in weighted rounds
        # Each subset contributes proportionally to its weight
        iterators = {s.name: self._iterate_texts(s) for s in self.subsets}
        weights = {s.name: s.weight for s in self.subsets}

        # Round-robin with weights
        subset_names = list(iterators.keys())
        subset_counts = {name: 0 for name in subset_names}
        total_yielded = 0

        while subset_names:
            # Pick subset with lowest count relative to its weight
            min_ratio = float("inf")
            chosen = subset_names[0]

            for name in subset_names:
                if weights[name] > 0:
                    ratio = subset_counts[name] / (total_yielded * weights[name] + 1e-9)
                    if ratio < min_ratio:
                        min_ratio = ratio
                        chosen = name

            try:
                text = next(iterators[chosen])
                subset_counts[chosen] += 1
                total_yielded += 1
                yield text
            except StopIteration:
                subset_names.remove(chosen)
                continue

    def _pack_sequences(self) -> Iterator[dict]:
        """Pack documents into fixed-length sequences."""
        token_buffer: list[int] = []

        for text in self._iterate_all_texts():
            # Tokenize document with BOS/EOS
            tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            token_buffer.extend(tokens)

            # Yield complete sequences
            while len(token_buffer) >= self.seq_len + 1:
                seq = token_buffer[: self.seq_len + 1]
                token_buffer = token_buffer[self.seq_len + 1 :]

                input_ids = torch.tensor(seq[:-1], dtype=torch.long)
                labels = torch.tensor(seq[1:], dtype=torch.long)

                yield {"input_ids": input_ids, "labels": labels}

    def __iter__(self) -> Iterator[dict]:
        return self._pack_sequences()


class SmolLMValidationDataset(IterableDataset):
    """
    Fixed validation dataset from SmolLM-Corpus.

    Creates a deterministic validation set by sampling from HELD-OUT row groups.
    This ensures validation data never overlaps with training data.

    The validation set is pre-built on initialization for consistency across runs.

    Args:
        tokenizer: Tokenizer instance
        seq_len: Sequence length
        num_samples: Number of validation samples (much larger than Dolmino's 127!)
        subsets: List of subset names to sample from
        base_path: Root path for SmolLM corpus
        seed: Random seed for deterministic selection
        val_ratio: Fraction of row groups held out for validation
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        seq_len: int = 512,
        num_samples: int = 2000,
        subsets: list[str] | None = None,
        base_path: Path = SMOLLM_PATH,
        seed: int = 42,
        val_ratio: float = 0.02,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.seed = seed
        self.pad_id = tokenizer.pad_id

        # Discover subsets
        self.subsets = discover_subsets(base_path, subsets, None)
        if not self.subsets:
            raise ValueError(f"No valid subsets found at {base_path}")

        # Build validation samples
        self.samples = self._build_validation_set(val_ratio)

    def _build_validation_set(self, val_ratio: float) -> list[dict]:
        """Build fixed validation set from held-out row groups."""
        logger.info(f"Building SmolLM validation set (target: {self.num_samples} samples)...")

        samples: list[dict] = []

        # Collect from each subset proportionally
        for subset in self.subsets:
            if len(samples) >= self.num_samples:
                break

            subset_target = int(self.num_samples * subset.weight)
            subset_samples = 0

            # Get validation row groups
            _, val_groups = split_row_groups_for_validation(
                subset, val_ratio, self.seed
            )

            logger.info(f"  {subset.name}: {len(val_groups)} val row groups, target {subset_target} samples")

            pf = pq.ParquetFile(subset.parquet_path)
            token_buffer: list[int] = []

            for rg_idx in val_groups:
                if subset_samples >= subset_target:
                    break

                try:
                    table = pf.read_row_group(rg_idx, columns=[subset.text_column])
                    texts = table[subset.text_column].to_pylist()

                    for text in texts:
                        if not text or len(text) < 50:
                            continue

                        tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
                        token_buffer.extend(tokens)

                        # Create sequences from buffer
                        while len(token_buffer) >= self.seq_len + 1:
                            seq = token_buffer[: self.seq_len + 1]
                            token_buffer = token_buffer[self.seq_len + 1 :]

                            input_ids = torch.tensor(seq[:-1], dtype=torch.long)
                            labels = torch.tensor(seq[1:], dtype=torch.long)
                            samples.append({"input_ids": input_ids, "labels": labels})
                            subset_samples += 1

                            if subset_samples >= subset_target:
                                break
                            if len(samples) >= self.num_samples:
                                break

                        if subset_samples >= subset_target or len(samples) >= self.num_samples:
                            break

                except Exception as e:
                    logger.warning(f"Error reading {subset.name} row group {rg_idx}: {e}")
                    continue

            logger.info(f"    -> collected {subset_samples} samples from {subset.name}")

        # Shuffle samples for variety during validation
        rng = random.Random(self.seed)
        rng.shuffle(samples)

        logger.info(f"Built validation set with {len(samples)} samples")
        return samples

    def __iter__(self) -> Iterator[dict]:
        for sample in self.samples:
            yield sample

    def __len__(self) -> int:
        return len(self.samples)


def create_smollm_dataloader(
    tokenizer: Tokenizer,
    batch_size: int = 8,
    seq_len: int = 512,
    subsets: list[str] | None = None,
    subset_weights: dict[str, float] | None = None,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create training dataloader for SmolLM dataset.

    Args:
        tokenizer: Tokenizer instance
        batch_size: Batch size
        seq_len: Sequence length
        subsets: Optional list of subset names
        subset_weights: Optional custom weights per subset
        num_workers: Number of data loading workers

    Returns:
        DataLoader for training
    """
    dataset = SmolLMDataset(
        tokenizer=tokenizer,
        seq_len=seq_len,
        subsets=subsets,
        subset_weights=subset_weights,
        exclude_val_groups=True,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )


def create_smollm_val_dataloader(
    tokenizer: Tokenizer,
    batch_size: int = 8,
    seq_len: int = 512,
    num_samples: int = 2000,
    subsets: list[str] | None = None,
) -> DataLoader:
    """
    Create validation dataloader for SmolLM dataset.

    Args:
        tokenizer: Tokenizer instance
        batch_size: Batch size
        seq_len: Sequence length
        num_samples: Number of validation samples
        subsets: Optional list of subset names

    Returns:
        DataLoader for validation
    """
    dataset = SmolLMValidationDataset(
        tokenizer=tokenizer,
        seq_len=seq_len,
        num_samples=num_samples,
        subsets=subsets,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # Validation is pre-computed, no workers needed
        pin_memory=True,
    )


if __name__ == "__main__":
    # Test the dataset
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.data.tokenizer import load_tokenizer
    from src.utils.logging import setup_logging

    setup_logging()

    print("Testing SmolLMDataset...")

    # Check if tokenizer exists
    tokenizer_path = Path("data/tokenizer_smollm.json")
    if not tokenizer_path.exists():
        tokenizer_path = Path("data/tokenizer.json")

    if not tokenizer_path.exists():
        print(f"Tokenizer not found at {tokenizer_path}")
        print("Run: python scripts/train_tokenizer_smollm.py")
        sys.exit(1)

    tokenizer = load_tokenizer(tokenizer_path)

    # Test training dataset
    dataset = SmolLMDataset(
        tokenizer=tokenizer,
        seq_len=512,
    )

    count = 0
    for batch in dataset:
        count += 1
        if count == 1:
            print(f"Sample batch: input_ids shape = {batch['input_ids'].shape}")
            print(f"Sample text: {tokenizer.decode(batch['input_ids'].tolist()[:50])}...")
        if count >= 10:
            break

    print(f"Tested {count} batches from training dataset")

    # Test validation dataset
    print("\nTesting SmolLMValidationDataset...")
    val_dataset = SmolLMValidationDataset(
        tokenizer=tokenizer,
        seq_len=512,
        num_samples=100,  # Small for quick test
    )

    print(f"Validation dataset size: {len(val_dataset)}")
    for i, batch in enumerate(val_dataset):
        if i == 0:
            print(f"Val sample: input_ids shape = {batch['input_ids'].shape}")
        if i >= 5:
            break

    print("All tests passed!")
