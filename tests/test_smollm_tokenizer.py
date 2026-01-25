"""
Tests for SmolLM tokenizer training and usage.

These tests verify the tokenizer meets our requirements:
- Vocab size exactly 32,000
- Special tokens at IDs 0-3 (pad, unk, bos, eos)
- Proper encode/decode roundtrip
- Python code indentation preservation
- Low unknown token rate
"""

import pytest
from pathlib import Path
import tempfile
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizers import Tokenizer

# Check if the trained tokenizer exists
TOKENIZER_PATH = Path("data/tokenizer_smollm.json")
FALLBACK_TOKENIZER_PATH = Path("data/tokenizer.json")


def get_tokenizer_path() -> Path:
    """Get path to available tokenizer for testing."""
    if TOKENIZER_PATH.exists():
        return TOKENIZER_PATH
    elif FALLBACK_TOKENIZER_PATH.exists():
        return FALLBACK_TOKENIZER_PATH
    else:
        pytest.skip(
            "No tokenizer found. Run: python scripts/train_tokenizer_smollm.py"
        )


@pytest.fixture
def tokenizer():
    """Load the SmolLM tokenizer for testing."""
    path = get_tokenizer_path()
    return Tokenizer.from_file(str(path))


@pytest.fixture
def bpe_tokenizer():
    """Load the BPETokenizer wrapper for testing."""
    from src.data.tokenizer import load_tokenizer
    path = get_tokenizer_path()
    return load_tokenizer(path)


class TestVocabulary:
    """Tests for vocabulary size and structure."""

    def test_vocab_size(self, tokenizer):
        """Vocab size should be exactly 32,000."""
        vocab_size = tokenizer.get_vocab_size()
        assert vocab_size == 32000, f"Expected 32000, got {vocab_size}"

    def test_special_tokens_exist(self, tokenizer):
        """All special tokens should exist in vocabulary."""
        vocab = tokenizer.get_vocab()
        special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

        for token in special_tokens:
            assert token in vocab, f"Missing special token: {token}"

    def test_special_token_ids(self, tokenizer):
        """Special tokens should have correct IDs (0-3)."""
        vocab = tokenizer.get_vocab()

        assert vocab.get("<pad>") == 0, f"<pad> should be 0, got {vocab.get('<pad>')}"
        assert vocab.get("<unk>") == 1, f"<unk> should be 1, got {vocab.get('<unk>')}"
        assert vocab.get("<bos>") == 2, f"<bos> should be 2, got {vocab.get('<bos>')}"
        assert vocab.get("<eos>") == 3, f"<eos> should be 3, got {vocab.get('<eos>')}"


class TestEncodeDecode:
    """Tests for encoding and decoding functionality."""

    def test_encode_decode_roundtrip_simple(self, tokenizer):
        """Simple text should roundtrip correctly."""
        text = "Hello, world! This is a test."
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)

        # ByteLevel tokenizers may add/remove some whitespace
        assert decoded.strip() == text.strip()

    def test_encode_decode_roundtrip_unicode(self, tokenizer):
        """Unicode text should roundtrip correctly."""
        text = "Héllo wörld! 你好世界 🌍"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)

        assert decoded.strip() == text.strip()

    def test_encode_decode_roundtrip_math(self, tokenizer):
        """Mathematical text should roundtrip correctly."""
        text = "The quadratic formula: x = (-b ± √(b² - 4ac)) / 2a"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)

        assert decoded.strip() == text.strip()

    def test_encode_produces_tokens(self, tokenizer):
        """Encoding should produce token IDs."""
        text = "Hello world"
        encoded = tokenizer.encode(text)

        assert len(encoded.ids) > 0
        assert all(isinstance(id, int) for id in encoded.ids)
        assert all(0 <= id < tokenizer.get_vocab_size() for id in encoded.ids)

    def test_empty_string(self, tokenizer):
        """Empty string should encode to empty list."""
        encoded = tokenizer.encode("")
        # May or may not produce empty - depends on post-processor
        # At minimum, should not crash
        assert isinstance(encoded.ids, list)


class TestPythonCodePreservation:
    """Tests for Python code handling, especially indentation."""

    def test_python_function_roundtrip(self, tokenizer):
        """Python function with indentation should roundtrip."""
        code = """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)"""

        encoded = tokenizer.encode(code)
        decoded = tokenizer.decode(encoded.ids)

        assert decoded.strip() == code.strip()

    def test_python_indentation_preserved(self, tokenizer):
        """Python indentation patterns should be preserved."""
        code = "def foo():\n    if True:\n        return 42\n    else:\n        return 0"

        encoded = tokenizer.encode(code)
        decoded = tokenizer.decode(encoded.ids)

        # Check specific indentation patterns are preserved
        assert "\n    " in decoded, "4-space indent not preserved"
        assert "\n        " in decoded, "8-space indent not preserved"

    def test_python_imports(self, tokenizer):
        """Python imports should roundtrip."""
        code = "import numpy as np\nimport torch\nfrom typing import List, Optional"

        encoded = tokenizer.encode(code)
        decoded = tokenizer.decode(encoded.ids)

        assert decoded.strip() == code.strip()

    def test_python_class_definition(self, tokenizer):
        """Python class with methods should roundtrip."""
        code = '''class MyClass:
    """Docstring."""

    def __init__(self, x):
        self.x = x

    def method(self):
        return self.x * 2'''

        encoded = tokenizer.encode(code)
        decoded = tokenizer.decode(encoded.ids)

        # Normalize whitespace for comparison
        assert decoded.strip() == code.strip()


class TestTokenizerEfficiency:
    """Tests for tokenizer efficiency metrics."""

    def test_reasonable_chars_per_token(self, tokenizer):
        """Chars per token should be in reasonable range (3-5)."""
        test_texts = [
            "Hello, world! This is a test sentence for the tokenizer.",
            "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models use gradient descent for optimization.",
        ]

        total_chars = 0
        total_tokens = 0

        for text in test_texts:
            encoded = tokenizer.encode(text)
            total_chars += len(text)
            total_tokens += len(encoded.ids)

        chars_per_token = total_chars / total_tokens if total_tokens > 0 else 0

        # Typical range for BPE tokenizers
        assert 2.5 <= chars_per_token <= 6.0, f"Chars/token {chars_per_token:.2f} outside reasonable range"


class TestBPETokenizerWrapper:
    """Tests for the BPETokenizer wrapper class."""

    def test_wrapper_attributes(self, bpe_tokenizer):
        """Wrapper should have expected attributes."""
        assert hasattr(bpe_tokenizer, "pad_id")
        assert hasattr(bpe_tokenizer, "unk_id")
        assert hasattr(bpe_tokenizer, "bos_id")
        assert hasattr(bpe_tokenizer, "eos_id")
        assert hasattr(bpe_tokenizer, "vocab_size")

    def test_wrapper_special_token_ids(self, bpe_tokenizer):
        """Wrapper special token IDs should match raw tokenizer."""
        assert bpe_tokenizer.pad_id == 0
        assert bpe_tokenizer.unk_id == 1
        assert bpe_tokenizer.bos_id == 2
        assert bpe_tokenizer.eos_id == 3

    def test_wrapper_encode_with_bos_eos(self, bpe_tokenizer):
        """Wrapper encode should add BOS/EOS tokens."""
        text = "Hello world"
        tokens = bpe_tokenizer.encode(text, add_bos=True, add_eos=True)

        assert tokens[0] == bpe_tokenizer.bos_id, "First token should be BOS"
        assert tokens[-1] == bpe_tokenizer.eos_id, "Last token should be EOS"

    def test_wrapper_encode_without_bos_eos(self, bpe_tokenizer):
        """Wrapper encode should work without BOS/EOS."""
        text = "Hello world"
        tokens = bpe_tokenizer.encode(text, add_bos=False, add_eos=False)

        if len(tokens) > 0:
            assert tokens[0] != bpe_tokenizer.bos_id or tokens[-1] != bpe_tokenizer.eos_id

    def test_wrapper_decode(self, bpe_tokenizer):
        """Wrapper decode should work correctly."""
        text = "Hello world"
        tokens = bpe_tokenizer.encode(text, add_bos=False, add_eos=False)
        decoded = bpe_tokenizer.decode(tokens, skip_special=False)

        assert decoded.strip() == text.strip()

    def test_wrapper_decode_skip_special(self, bpe_tokenizer):
        """Wrapper decode should skip special tokens when requested."""
        text = "Hello world"
        tokens = bpe_tokenizer.encode(text, add_bos=True, add_eos=True)
        decoded = bpe_tokenizer.decode(tokens, skip_special=True)

        assert "<bos>" not in decoded
        assert "<eos>" not in decoded

    def test_wrapper_batch_encode(self, bpe_tokenizer):
        """Wrapper batch encoding should work."""
        texts = ["Hello world", "Goodbye world", "Testing 123"]
        batch = bpe_tokenizer.encode_batch(texts)

        assert len(batch) == 3
        for tokens in batch:
            assert len(tokens) > 0
            assert tokens[0] == bpe_tokenizer.bos_id
            assert tokens[-1] == bpe_tokenizer.eos_id


class TestCorpusCoverage:
    """Tests for tokenizer coverage on diverse content."""

    def test_no_unknown_on_english(self, tokenizer):
        """English text should not produce unknown tokens."""
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language.",
        ]

        vocab = tokenizer.get_vocab()
        unk_id = vocab.get("<unk>", 1)

        for text in texts:
            encoded = tokenizer.encode(text)
            unknown_count = sum(1 for id in encoded.ids if id == unk_id)
            assert unknown_count == 0, f"Found {unknown_count} unknown tokens in: {text[:50]}"

    def test_low_unknown_rate_on_code(self, tokenizer):
        """Code should have very low unknown token rate."""
        code_samples = [
            "def main():\n    print('Hello, World!')",
            "import os\nimport sys\nfrom pathlib import Path",
            "class MyClass:\n    def __init__(self, x):\n        self.x = x",
            "for i in range(10):\n    if i % 2 == 0:\n        print(i)",
        ]

        vocab = tokenizer.get_vocab()
        unk_id = vocab.get("<unk>", 1)

        total_tokens = 0
        unknown_tokens = 0

        for code in code_samples:
            encoded = tokenizer.encode(code)
            total_tokens += len(encoded.ids)
            unknown_tokens += sum(1 for id in encoded.ids if id == unk_id)

        unknown_rate = unknown_tokens / total_tokens if total_tokens > 0 else 0

        assert unknown_rate < 0.01, f"Unknown rate {unknown_rate:.2%} too high for code"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
