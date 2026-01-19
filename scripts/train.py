#!/usr/bin/env python3
"""Train Gemma with LoRA."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["KERAS_BACKEND"] = "jax"

from gemma_finetune.models.config import LoRAConfig, ModelConfig, load_config
from gemma_finetune.training.config import TrainingConfig, load_training_config
from gemma_finetune.training.trainer import GemmaTrainer


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Gemma with LoRA")
    parser.add_argument("--config", type=Path, help="Model/LoRA config YAML")
    parser.add_argument("--training-config", type=Path, help="Training config YAML")
    parser.add_argument("--data", type=Path, required=True, help="Training data JSONL")
    parser.add_argument("--val-data", type=Path, help="Validation data JSONL")
    parser.add_argument("--output", type=Path, default=Path("./checkpoints"), help="Output dir")

    # Override options
    parser.add_argument("--max-steps", type=int, help="Override max training steps")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--lora-rank", type=int, help="Override LoRA rank")

    args = parser.parse_args()

    # Load configs
    if args.config:
        print(f"Loading config from {args.config}")
        model_cfg, lora_cfg = load_config(args.config)
    else:
        print("Using default model/LoRA config")
        model_cfg = ModelConfig()
        lora_cfg = LoRAConfig()

    if args.training_config:
        print(f"Loading training config from {args.training_config}")
        train_cfg = load_training_config(args.training_config)
    else:
        print("Using default training config")
        train_cfg = TrainingConfig(output_dir=str(args.output))

    # Apply overrides
    train_cfg.output_dir = str(args.output)
    if args.max_steps:
        train_cfg.max_steps = args.max_steps
    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    if args.lr:
        train_cfg.learning_rate = args.lr
    if args.lora_rank:
        lora_cfg.rank = args.lora_rank
        lora_cfg.alpha = args.lora_rank * 2  # Keep 2x ratio

    # Print config summary
    print("\n" + "=" * 50)
    print("Configuration Summary")
    print("=" * 50)
    print(f"Model: {model_cfg.model_id}")
    print(f"Dtype: {model_cfg.dtype}")
    print(f"Max seq length: {model_cfg.max_sequence_length}")
    print(f"LoRA rank: {lora_cfg.rank}")
    print(f"LoRA alpha: {lora_cfg.alpha}")
    print(f"Learning rate: {train_cfg.learning_rate}")
    print(f"Batch size: {train_cfg.batch_size}")
    print(f"Gradient accumulation: {train_cfg.gradient_accumulation_steps}")
    print(f"Effective batch: {train_cfg.effective_batch_size}")
    print(f"Max steps: {train_cfg.max_steps}")
    print(f"Output: {train_cfg.output_dir}")
    print("=" * 50 + "\n")

    # Initialize trainer
    trainer = GemmaTrainer(model_cfg, lora_cfg, train_cfg)

    try:
        trainer.setup()
    except Exception as e:
        print(f"Error setting up trainer: {e}")
        return 1

    # Train
    try:
        trainer.train(args.data, args.val_data)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save current state
        final_path = Path(args.output) / "lora_interrupted.npz"
        trainer.save(final_path)
        print(f"Saved interrupted checkpoint to {final_path}")
        return 130
    except Exception as e:
        print(f"Error during training: {e}")
        return 1

    # Save final weights
    final_path = Path(args.output) / "lora_final.npz"
    trainer.save(final_path)
    print(f"\nTraining complete! Final weights saved to {final_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
