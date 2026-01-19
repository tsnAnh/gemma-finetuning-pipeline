"""Keras-based inference for GPU acceleration."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

os.environ["KERAS_BACKEND"] = "jax"

import keras

from ..data.formatter import format_gemma_chat, parse_gemma_response


class KerasInference:
    """Inference using Keras model on GPU."""

    def __init__(
        self,
        model: Any,
        max_length: int = 512,
    ):
        """Initialize inference engine.

        Args:
            model: Keras Hub GemmaCausalLM model.
            max_length: Maximum generation length.
        """
        self.model = model
        self.max_length = max_length

    @classmethod
    def from_checkpoint(
        cls,
        model_path: str | None = None,
        lora_path: str | None = None,
    ) -> KerasInference:
        """Load model from checkpoint with optional LoRA.

        Args:
            model_path: Ignored (uses default model).
            lora_path: Optional path to LoRA weights.

        Returns:
            Configured inference engine.
        """
        from ..models.config import DEFAULT_LORA, DEFAULT_MODEL
        from ..models.loader import load_gemma_model
        from ..models.lora import apply_lora_to_model, load_lora_weights

        model = load_gemma_model(DEFAULT_MODEL)

        if lora_path:
            model = apply_lora_to_model(model, DEFAULT_LORA)
            load_lora_weights(model, lora_path)

        return cls(model)

    @classmethod
    def from_merged_model(cls, model_path: str) -> KerasInference:
        """Load from a merged (LoRA-free) model.

        Args:
            model_path: Path to saved Keras model.

        Returns:
            Configured inference engine.
        """
        model = keras.models.load_model(model_path)
        return cls(model)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        stop_sequences: list[str] | None = None,
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: User prompt text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Top-p (nucleus) sampling parameter.
            stop_sequences: Optional stop sequences.

        Returns:
            Generated response text.
        """
        # Format as Gemma chat
        conversations = [{"from": "human", "value": prompt}]
        formatted = format_gemma_chat(conversations, add_generation_prompt=True)

        # Generate
        output = self.model.generate(
            formatted,
            max_length=min(self.max_length, max_new_tokens + len(formatted.split())),
        )

        # Extract response
        return parse_gemma_response(output)

    def generate_stream(
        self,
        prompt: str,
        callback: Callable[[str], None],
        max_new_tokens: int = 256,
        **kwargs: Any,
    ) -> str:
        """Generate with streaming output.

        Note: Keras Hub doesn't have native streaming, so this
        simulates streaming by generating fully then chunking output.

        Args:
            prompt: User prompt text.
            callback: Function called for each token/word.
            max_new_tokens: Maximum tokens to generate.
            **kwargs: Additional generation parameters.

        Returns:
            Complete generated text.
        """
        response = self.generate(prompt, max_new_tokens, **kwargs)

        # Stream word by word
        words = response.split()
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            callback(token)

        return response

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 256,
        **kwargs: Any,
    ) -> list[str]:
        """Generate for multiple prompts.

        Args:
            prompts: List of prompts.
            max_new_tokens: Maximum tokens per generation.
            **kwargs: Additional generation parameters.

        Returns:
            List of generated responses.
        """
        return [self.generate(p, max_new_tokens, **kwargs) for p in prompts]
