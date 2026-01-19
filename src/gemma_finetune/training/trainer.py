"""Main trainer class for Gemma finetuning."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ["KERAS_BACKEND"] = "jax"

import keras

from ..models.config import LoRAConfig, ModelConfig
from ..models.loader import load_gemma_model, print_model_summary
from ..models.lora import apply_lora_to_model, load_lora_weights, save_lora_weights
from .callbacks import EarlyStoppingOnNaN, LoRASaveCallback, MemoryMonitor, MetricsLogger
from .config import TrainingConfig
from .data_pipeline import create_keras_dataset
from .optimizer import create_optimizer


class GemmaTrainer:
    """Gemma finetuning trainer with LoRA."""

    def __init__(
        self,
        model_config: ModelConfig,
        lora_config: LoRAConfig,
        training_config: TrainingConfig,
    ):
        """Initialize trainer.

        Args:
            model_config: Model configuration.
            lora_config: LoRA configuration.
            training_config: Training configuration.
        """
        self.model_config = model_config
        self.lora_config = lora_config
        self.training_config = training_config
        self.model: Any = None
        self._is_setup = False

    def setup(self) -> None:
        """Initialize model and LoRA."""
        print("Setting up trainer...")

        # Load model
        self.model = load_gemma_model(self.model_config)

        # Apply LoRA
        self.model = apply_lora_to_model(self.model, self.lora_config)

        # Print summary
        print_model_summary(self.model)

        # Create optimizer
        optimizer = create_optimizer(
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            warmup_steps=self.training_config.warmup_steps,
            total_steps=self.training_config.max_steps,
        )

        # Compile model
        # Note: Keras Hub GemmaCausalLM handles loss internally
        self.model.compile(
            optimizer=optimizer,
            weighted_metrics=["accuracy"],
        )

        self._is_setup = True
        print("Trainer setup complete!")

    def train(
        self,
        train_data_path: str | Path,
        val_data_path: str | Path | None = None,
    ) -> keras.callbacks.History:
        """Run training.

        Args:
            train_data_path: Path to training JSONL file.
            val_data_path: Optional path to validation JSONL file.

        Returns:
            Keras training history.
        """
        if not self._is_setup:
            self.setup()

        cfg = self.training_config
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create datasets
        print(f"Loading training data from {train_data_path}...")
        train_dataset = create_keras_dataset(
            train_data_path,
            self.model,
            cfg.max_sequence_length,
            cfg.batch_size,
            shuffle=True,
        )

        val_dataset = None
        if val_data_path:
            print(f"Loading validation data from {val_data_path}...")
            val_dataset = create_keras_dataset(
                val_data_path,
                self.model,
                cfg.max_sequence_length,
                cfg.batch_size,
                shuffle=False,
            )

        # Setup callbacks
        callbacks = [
            LoRASaveCallback(str(output_dir), cfg.save_steps),
            MetricsLogger(str(output_dir / "metrics.json")),
            MemoryMonitor(cfg.logging_steps * 5),
            EarlyStoppingOnNaN(),
            keras.callbacks.TerminateOnNaN(),
        ]

        if val_dataset is not None:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=cfg.early_stopping_patience,
                    restore_best_weights=True,
                )
            )

        # Train
        print(f"\nStarting training for {cfg.max_steps} steps...")
        print(f"Effective batch size: {cfg.effective_batch_size}")

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=1,  # Use steps, not epochs
            steps_per_epoch=cfg.max_steps,
            callbacks=callbacks,
            verbose=1,
        )

        print("Training complete!")
        return history

    def save(self, path: str | Path) -> None:
        """Save LoRA weights.

        Args:
            path: Output path (should end with .npz).
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call setup() first.")
        save_lora_weights(self.model, path)

    def load(self, path: str | Path) -> None:
        """Load LoRA weights.

        Args:
            path: Path to saved weights.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call setup() first.")
        load_lora_weights(self.model, path)

    def generate(
        self,
        prompt: str,
        max_length: int = 256,
    ) -> str:
        """Generate text from prompt (for testing).

        Args:
            prompt: Input prompt.
            max_length: Maximum generation length.

        Returns:
            Generated text.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call setup() first.")
        return self.model.generate(prompt, max_length=max_length)
