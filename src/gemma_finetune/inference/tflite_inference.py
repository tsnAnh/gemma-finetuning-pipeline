"""TFLite-based inference for CPU/edge deployment."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


class TFLiteInference:
    """Inference using TFLite model on CPU/GPU."""

    def __init__(
        self,
        model_path: str | Path,
        tokenizer_path: str | Path | None = None,
    ):
        """Initialize TFLite inference engine.

        Args:
            model_path: Path to TFLite model file.
            tokenizer_path: Path to tokenizer directory or file.
        """
        self.model_path = Path(model_path)
        self.interpreter = None
        self.tokenizer = None
        self.input_details = None
        self.output_details = None

        self._load_model()
        if tokenizer_path:
            self._load_tokenizer(tokenizer_path)

    def _load_model(self) -> None:
        """Load TFLite interpreter."""
        try:
            import tensorflow as tf

            # Try GPU delegate first (if available)
            try:
                gpu_delegate = tf.lite.experimental.load_delegate("libedgetpu.so.1")
                self.interpreter = tf.lite.Interpreter(
                    model_path=str(self.model_path),
                    experimental_delegates=[gpu_delegate],
                )
                print("TFLite: Loaded with Edge TPU delegate")
            except Exception:
                # Fall back to CPU
                self.interpreter = tf.lite.Interpreter(
                    model_path=str(self.model_path),
                )
                print("TFLite: Loaded with CPU")

            self.interpreter.allocate_tensors()

            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            print(f"TFLite: Model loaded ({len(self.input_details)} inputs)")

        except ImportError as e:
            raise ImportError("TensorFlow Lite not available. Install with: pip install tensorflow") from e

    def _load_tokenizer(self, path: str | Path) -> None:
        """Load tokenizer from path."""
        path = Path(path)

        # Try HuggingFace tokenizer
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(str(path))
            print(f"Tokenizer: Loaded HuggingFace tokenizer from {path}")
            return
        except Exception:
            pass

        # Try sentencepiece directly
        try:
            import sentencepiece as spm

            sp = spm.SentencePieceProcessor()

            # Find tokenizer file
            if path.is_file():
                sp.load(str(path))
            else:
                # Look for common tokenizer filenames
                for filename in ["tokenizer.model", "spiece.model", "gemma.model"]:
                    tokenizer_file = path / filename
                    if tokenizer_file.exists():
                        sp.load(str(tokenizer_file))
                        break
                else:
                    raise FileNotFoundError(f"No tokenizer found in {path}")

            self.tokenizer = sp
            print(f"Tokenizer: Loaded SentencePiece from {path}")

        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Generate text using TFLite model.

        Args:
            prompt: Input prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 for greedy).

        Returns:
            Generated text.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Provide tokenizer_path.")

        # Tokenize
        if hasattr(self.tokenizer, "encode"):
            # SentencePiece
            input_ids = self.tokenizer.encode(prompt)
        else:
            # HuggingFace tokenizer
            encoded = self.tokenizer(prompt, return_tensors="np")
            input_ids = encoded["input_ids"][0].tolist()

        generated = list(input_ids)
        max_seq_len = 512  # Default, adjust based on model

        for _ in range(max_new_tokens):
            # Prepare input (last max_seq_len tokens)
            current_input = generated[-max_seq_len:]

            # Pad to fixed size
            if len(current_input) < max_seq_len:
                current_input = [0] * (max_seq_len - len(current_input)) + current_input

            input_array = np.array([current_input], dtype=np.int32)

            # Run inference
            self.interpreter.set_tensor(self.input_details[0]["index"], input_array)
            self.interpreter.invoke()

            logits = self.interpreter.get_tensor(self.output_details[0]["index"])

            # Get next token
            next_logits = logits[0, -1, :]

            if temperature > 0:
                # Sample with temperature
                probs = self._softmax(next_logits / temperature)
                next_token = np.random.choice(len(probs), p=probs)
            else:
                # Greedy
                next_token = np.argmax(next_logits)

            generated.append(int(next_token))

            # Check for EOS
            if self._is_eos(next_token):
                break

        # Decode
        return self._decode(generated)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def _is_eos(self, token_id: int) -> bool:
        """Check if token is end-of-sequence."""
        if hasattr(self.tokenizer, "eos_id"):
            return token_id == self.tokenizer.eos_id()
        if hasattr(self.tokenizer, "eos_token_id"):
            return token_id == self.tokenizer.eos_token_id
        return False

    def _decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        if hasattr(self.tokenizer, "decode"):
            # HuggingFace
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        elif hasattr(self.tokenizer, "decode_ids"):
            # SentencePiece
            return self.tokenizer.decode_ids(token_ids)
        else:
            return str(token_ids)

    def generate_stream(
        self,
        prompt: str,
        callback: Callable[[str], None],
        max_new_tokens: int = 256,
        **kwargs: Any,
    ) -> str:
        """Generate with streaming output.

        Args:
            prompt: Input prompt.
            callback: Function called for each token.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Complete generated text.
        """
        # TFLite doesn't have native streaming, generate fully then chunk
        response = self.generate(prompt, max_new_tokens, **kwargs)

        words = response.split()
        for i, word in enumerate(words):
            callback(word + (" " if i < len(words) - 1 else ""))

        return response
