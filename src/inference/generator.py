"""
Text Generator for Atlas-MAG.

Implements autoregressive text generation using the InferenceEngine.
Supports various sampling strategies and generation configurations.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from src.inference.engine import InferenceEngine, InferenceMode

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    do_sample: bool = True
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None


class TextGenerator:
    """
    Text generator using Atlas-MAG inference engine.

    Implements:
    - Autoregressive generation with prefill/decode modes
    - Temperature sampling
    - Top-k and top-p (nucleus) sampling
    - Repetition penalty
    - Early stopping on EOS

    Args:
        engine: InferenceEngine instance
        tokenizer: Tokenizer for encoding/decoding
    """

    def __init__(
        self,
        engine: InferenceEngine,
        tokenizer,  # BPETokenizer
    ):
        self.engine = engine
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            config: Generation configuration
            callback: Optional callback for streaming tokens

        Returns:
            Generated text (including prompt)
        """
        if config is None:
            config = GenerationConfig()

        # Reset engine state for new generation
        self.engine.reset()

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)

        # Prefill: Process prompt
        logits = self.engine.prefill(input_tensor)

        # Get last token logits for first generation step
        next_token_logits = logits[0, -1, :]

        # Track generated tokens for repetition penalty
        generated_ids = list(input_ids)

        # Generate tokens one at a time
        for step in range(config.max_new_tokens):
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated_ids, config.repetition_penalty
                )

            # Sample next token
            next_token_id = self._sample_token(
                next_token_logits,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                do_sample=config.do_sample,
            )

            # Check for EOS
            if config.eos_token_id is not None and next_token_id == config.eos_token_id:
                break

            # Add to generated sequence
            generated_ids.append(next_token_id)

            # Callback for streaming
            if callback is not None:
                token_text = self.tokenizer.decode([next_token_id])
                callback(token_text)

            # Decode step: Get next logits
            token_tensor = torch.tensor([[next_token_id]], dtype=torch.long)
            logits = self.engine.decode_step(token_tensor)
            next_token_logits = logits[0, -1, :]

        # Decode full sequence
        generated_text = self.tokenizer.decode(generated_ids)

        return generated_text

    def _sample_token(
        self,
        logits: Tensor,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> int:
        """
        Sample a token from logits using specified strategy.

        Args:
            logits: [vocab_size] unnormalized log probabilities
            temperature: Sampling temperature (1.0 = neutral)
            top_k: Keep only top-k tokens
            top_p: Keep tokens with cumulative probability < top_p
            do_sample: If False, use greedy decoding

        Returns:
            Sampled token ID
        """
        if not do_sample:
            # Greedy decoding
            return logits.argmax(dim=-1).item()

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][-1]
            logits[indices_to_remove] = float('-inf')

        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift to keep first token above threshold
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')

        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token.item()

    def _apply_repetition_penalty(
        self,
        logits: Tensor,
        generated_ids: List[int],
        penalty: float,
    ) -> Tensor:
        """
        Apply repetition penalty to discourage repeated tokens.

        For tokens in generated_ids:
        - If logit > 0: divide by penalty
        - If logit < 0: multiply by penalty

        Args:
            logits: [vocab_size] logits
            generated_ids: Previously generated token IDs
            penalty: Repetition penalty factor (> 1.0 discourages repetition)

        Returns:
            Modified logits
        """
        if penalty == 1.0 or not generated_ids:
            return logits

        logits = logits.clone()
        unique_ids = set(generated_ids)

        for token_id in unique_ids:
            if logits[token_id] > 0:
                logits[token_id] = logits[token_id] / penalty
            else:
                logits[token_id] = logits[token_id] * penalty

        return logits

    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
    ) -> List[str]:
        """
        Generate text for multiple prompts.

        Note: Currently processes sequentially. Batch processing
        would require more complex state management.

        Args:
            prompts: List of input prompts
            config: Generation configuration

        Returns:
            List of generated texts
        """
        return [self.generate(prompt, config) for prompt in prompts]

    def get_generation_stats(self) -> dict:
        """Get statistics from the last generation."""
        timing = self.engine.get_timing_stats()
        return {
            "timing": timing,
            "mode": self.engine.state.mode.value,
            "position": self.engine.state.position,
            "no_reset_verified": self.engine.verify_no_reset(),
        }
