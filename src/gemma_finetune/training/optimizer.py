"""Optimizer configuration with learning rate schedules."""

from __future__ import annotations

import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras import optimizers


def create_optimizer(
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    total_steps: int,
) -> keras.Optimizer:
    """Create AdamW optimizer with linear warmup and cosine decay.

    Args:
        learning_rate: Peak learning rate.
        weight_decay: Weight decay coefficient.
        warmup_steps: Number of warmup steps.
        total_steps: Total training steps.

    Returns:
        Configured AdamW optimizer.
    """
    decay_steps = max(1, total_steps - warmup_steps)

    # Cosine decay with warmup
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=decay_steps,
        warmup_target=learning_rate,
        warmup_steps=warmup_steps,
    )

    optimizer = optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=weight_decay,
        clipnorm=1.0,  # Gradient clipping for stability
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
    )

    return optimizer


def create_simple_optimizer(learning_rate: float) -> keras.Optimizer:
    """Create simple AdamW without schedule (for quick tests).

    Args:
        learning_rate: Constant learning rate.

    Returns:
        Configured AdamW optimizer.
    """
    return optimizers.AdamW(
        learning_rate=learning_rate,
        clipnorm=1.0,
    )


def create_sgd_optimizer(learning_rate: float, momentum: float = 0.9) -> keras.Optimizer:
    """Create SGD optimizer (alternative for debugging).

    Args:
        learning_rate: Learning rate.
        momentum: Momentum coefficient.

    Returns:
        Configured SGD optimizer.
    """
    return optimizers.SGD(
        learning_rate=learning_rate,
        momentum=momentum,
        clipnorm=1.0,
    )
