"""Training configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class TrainingConfig:
    """Training hyperparameters.

    Optimized defaults for RTX 3060 12GB VRAM with Gemma 1B + LoRA.

    Attributes:
        learning_rate: Peak learning rate.
        weight_decay: AdamW weight decay.
        warmup_steps: Linear warmup steps.
        max_steps: Total training steps.
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Steps to accumulate before optimizer update.
        max_sequence_length: Maximum sequence length.
        output_dir: Directory for checkpoints and logs.
        save_steps: Save checkpoint every N steps.
        logging_steps: Log metrics every N steps.
        eval_steps: Run evaluation every N steps.
        early_stopping_patience: Epochs to wait before early stopping.
        seed: Random seed for reproducibility.
    """

    # Optimization
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 500

    # Batching
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_sequence_length: int = 512

    # Checkpointing
    output_dir: str = "./checkpoints"
    save_steps: int = 100
    logging_steps: int = 10

    # Validation
    eval_steps: int = 100
    early_stopping_patience: int = 3

    # Reproducibility
    seed: int = 42

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size after gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load training config from YAML file.

    Args:
        path: Path to YAML file.

    Returns:
        TrainingConfig instance.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)

    training_dict = cfg.get("training", cfg)
    return TrainingConfig(**training_dict)


def save_training_config(config: TrainingConfig, path: str | Path) -> None:
    """Save training config to YAML file.

    Args:
        config: Training configuration.
        path: Output path.
    """
    config_dict = {
        "training": {
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "warmup_steps": config.warmup_steps,
            "max_steps": config.max_steps,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "max_sequence_length": config.max_sequence_length,
            "output_dir": config.output_dir,
            "save_steps": config.save_steps,
            "logging_steps": config.logging_steps,
            "eval_steps": config.eval_steps,
            "early_stopping_patience": config.early_stopping_patience,
            "seed": config.seed,
        }
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
