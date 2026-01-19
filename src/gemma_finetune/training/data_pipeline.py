"""Data pipeline for training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_training_texts(data_path: str | Path) -> list[str]:
    """Load training texts from JSONL file.

    Expects format: {"text": "formatted conversation"}

    Args:
        data_path: Path to JSONL file.

    Returns:
        List of text strings.
    """
    texts = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            if "text" in item:
                texts.append(item["text"])
    return texts


def create_tf_dataset(
    data_path: str | Path,
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    shuffle: bool = True,
) -> Any:  # Returns tf.data.Dataset
    """Create TensorFlow dataset for Keras training.

    Args:
        data_path: Path to prepared JSONL data.
        tokenizer: Tokenizer with encode method.
        max_length: Maximum sequence length.
        batch_size: Batch size.
        shuffle: Whether to shuffle data.

    Returns:
        tf.data.Dataset ready for training.
    """
    import tensorflow as tf

    # Load texts
    texts = load_training_texts(data_path)

    if not texts:
        raise ValueError(f"No training data found in {data_path}")

    print(f"Loaded {len(texts)} training examples")

    # Tokenize all texts
    encoded = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        {
            "input_ids": encoded["input_ids"],
            "labels": encoded["input_ids"],  # Causal LM uses input as labels
        }
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(texts), 10000))

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_keras_dataset(
    data_path: str | Path,
    model: Any,  # GemmaCausalLM
    max_length: int,
    batch_size: int,
    shuffle: bool = True,
) -> Any:
    """Create dataset using Keras Hub preprocessor.

    Args:
        data_path: Path to prepared JSONL data.
        model: GemmaCausalLM model (uses its preprocessor).
        max_length: Maximum sequence length.
        batch_size: Batch size.
        shuffle: Whether to shuffle data.

    Returns:
        Dataset ready for model.fit().
    """
    import tensorflow as tf

    # Load texts
    texts = load_training_texts(data_path)

    if not texts:
        raise ValueError(f"No training data found in {data_path}")

    print(f"Loaded {len(texts)} training examples")

    # Create tf.data.Dataset from texts
    dataset = tf.data.Dataset.from_tensor_slices(texts)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(texts), 10000))

    dataset = dataset.batch(batch_size)

    return dataset


def estimate_training_steps(
    data_path: str | Path,
    batch_size: int,
    gradient_accumulation_steps: int = 1,
    num_epochs: int = 1,
) -> int:
    """Estimate total training steps.

    Args:
        data_path: Path to training data.
        batch_size: Batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        num_epochs: Number of epochs.

    Returns:
        Estimated total training steps.
    """
    texts = load_training_texts(data_path)
    num_examples = len(texts)

    steps_per_epoch = num_examples // (batch_size * gradient_accumulation_steps)
    total_steps = steps_per_epoch * num_epochs

    return max(1, total_steps)
