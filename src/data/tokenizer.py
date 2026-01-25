"""
Custom BPE Tokenizer for Atlas-MAG.

We train our own BPE tokenizer on the calibration corpus to ensure
the vocabulary is well-suited to our data. Uses the HuggingFace
tokenizers library (Rust-based, very fast).

User Decision: Custom BPE trained on WikiText-2
"""

import logging
from pathlib import Path
from typing import List, Union

from torch import Tensor

from src.config import VOCAB_SIZE, SPECIAL_TOKENS

logger = logging.getLogger(__name__)

# Attempt to import tokenizers - it's a required dependency
try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
    from tokenizers.normalizers import NFKC
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False
    logger.warning("tokenizers library not found - tokenizer training disabled")


class BPETokenizer:
    """
    Wrapper around HuggingFace tokenizers BPE model.

    Provides a simple interface for encoding/decoding text with
    special token handling.

    Args:
        tokenizer_path: Path to saved tokenizer.json file
    """

    def __init__(self, tokenizer_path: Union[str, Path]):
        if not HAS_TOKENIZERS:
            raise ImportError("tokenizers library required for BPETokenizer")

        self.path = Path(tokenizer_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {self.path}")

        self.tokenizer = Tokenizer.from_file(str(self.path))
        self.vocab_size = self.tokenizer.get_vocab_size()

        # Get special token IDs
        vocab = self.tokenizer.get_vocab()
        self.pad_id = vocab.get("<pad>", 0)
        self.unk_id = vocab.get("<unk>", 1)
        self.bos_id = vocab.get("<bos>", 2)
        self.eos_id = vocab.get("<eos>", 3)

        logger.info(
            f"Loaded tokenizer from {self.path}: "
            f"vocab_size={self.vocab_size}, "
            f"special_tokens=[pad={self.pad_id}, unk={self.unk_id}, "
            f"bos={self.bos_id}, eos={self.eos_id}]"
        )

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text string
            add_bos: Whether to prepend BOS token
            add_eos: Whether to append EOS token

        Returns:
            List of token IDs
        """
        # Disable automatic special tokens since TemplateProcessing adds BOS/EOS
        # and we handle them manually below based on add_bos/add_eos flags
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids

        result: List[int] = list(ids)
        if add_bos:
            result = [self.bos_id, *result]
        if add_eos:
            result = [*result, self.eos_id]

        return result

    def decode(
        self,
        ids: Union[List[int], Tensor],
        skip_special: bool = True,
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            ids: List or tensor of token IDs
            skip_special: Whether to skip special tokens in output

        Returns:
            Decoded text string
        """
        if isinstance(ids, Tensor):
            ids = ids.tolist()

        ids_list: List[int]
        if isinstance(ids, list):
            ids_list = ids
        else:
            ids_list = list(ids)

        if skip_special:
            special = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
            ids_list = [i for i in ids_list if i not in special]

        result: str = self.tokenizer.decode(ids_list)
        return result

    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[List[int]]:
        """Encode multiple texts."""
        return [self.encode(t, add_bos, add_eos) for t in texts]

    def __call__(
        self,
        text: Union[str, List[str]],
        **kwargs,
    ) -> Union[List[int], List[List[int]]]:
        """Convenience method for encoding."""
        if isinstance(text, str):
            return self.encode(text, **kwargs)
        return self.encode_batch(text, **kwargs)


def train_tokenizer(
    corpus_files: List[Union[str, Path]],
    output_path: Union[str, Path],
    vocab_size: int = VOCAB_SIZE,
    special_tokens: tuple = SPECIAL_TOKENS,
) -> BPETokenizer:
    """
    Train a BPE tokenizer on corpus files.

    Args:
        corpus_files: List of text file paths for training
        output_path: Where to save the tokenizer.json
        vocab_size: Target vocabulary size
        special_tokens: Tuple of special tokens to include

    Returns:
        Trained BPETokenizer instance
    """
    if not HAS_TOKENIZERS:
        raise ImportError("tokenizers library required for training")

    logger.info(f"Training BPE tokenizer: vocab_size={vocab_size}")

    # Initialize BPE model
    tokenizer = Tokenizer(models.BPE())

    # Normalizer: Unicode NFKC normalization
    tokenizer.normalizer = NFKC()

    # Pre-tokenizer: ByteLevel (like GPT-2)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Trainer configuration
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(special_tokens),
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # Convert paths to strings
    files = [str(p) for p in corpus_files]

    # Train
    tokenizer.train(files, trainer)

    # Post-processor: add ByteLevel decoder
    tokenizer.decoder = tokenizers.decoders.ByteLevel()

    # Add template for BOS/EOS handling
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))

    logger.info(
        f"Tokenizer trained and saved to {output_path}: "
        f"vocab_size={tokenizer.get_vocab_size()}"
    )

    return BPETokenizer(output_path)


def load_tokenizer(path: Union[str, Path]) -> BPETokenizer:
    """
    Load a trained tokenizer from file.

    Args:
        path: Path to tokenizer.json

    Returns:
        BPETokenizer instance
    """
    return BPETokenizer(path)


# Import for decoder
if HAS_TOKENIZERS:
    import tokenizers.decoders
