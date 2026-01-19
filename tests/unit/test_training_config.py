"""Tests for training configuration."""

from pathlib import Path

import pytest

from gemma_finetune.training.config import (
    TrainingConfig,
    load_training_config,
    save_training_config,
)


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        """Default training config for RTX 3060 12GB."""
        cfg = TrainingConfig()
        assert cfg.batch_size == 2
        assert cfg.gradient_accumulation_steps == 4
        assert cfg.learning_rate == 2e-4
        assert cfg.max_steps == 500
        assert cfg.warmup_steps == 100

    def test_effective_batch_size(self):
        """Effective batch size should be batch_size * grad_accum."""
        cfg = TrainingConfig(batch_size=2, gradient_accumulation_steps=4)
        assert cfg.effective_batch_size == 8

    def test_custom_values(self):
        """Custom values should be accepted."""
        cfg = TrainingConfig(
            batch_size=4,
            learning_rate=1e-4,
            max_steps=1000,
        )
        assert cfg.batch_size == 4
        assert cfg.learning_rate == 1e-4
        assert cfg.max_steps == 1000

    def test_invalid_batch_size(self):
        """Batch size must be positive."""
        with pytest.raises(ValueError, match="batch_size"):
            TrainingConfig(batch_size=0)

    def test_invalid_learning_rate(self):
        """Learning rate must be positive."""
        with pytest.raises(ValueError, match="learning_rate"):
            TrainingConfig(learning_rate=0)

    def test_invalid_max_steps(self):
        """Max steps must be positive."""
        with pytest.raises(ValueError, match="max_steps"):
            TrainingConfig(max_steps=0)

    def test_output_dir_default(self):
        """Default output dir."""
        cfg = TrainingConfig()
        assert cfg.output_dir == "./checkpoints"

    def test_seed_default(self):
        """Default seed for reproducibility."""
        cfg = TrainingConfig()
        assert cfg.seed == 42


class TestTrainingConfigIO:
    """Tests for training config save/load."""

    def test_save_and_load(self, tmp_path: Path):
        """Saved config should match loaded config."""
        cfg = TrainingConfig(
            batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=1e-5,
            max_steps=1000,
        )
        path = tmp_path / "training.yaml"

        save_training_config(cfg, path)
        loaded = load_training_config(path)

        assert loaded.batch_size == 4
        assert loaded.gradient_accumulation_steps == 8
        assert loaded.learning_rate == 1e-5
        assert loaded.max_steps == 1000

    def test_creates_parent_dirs(self, tmp_path: Path):
        """Save should create parent directories."""
        path = tmp_path / "nested" / "training.yaml"
        save_training_config(TrainingConfig(), path)
        assert path.exists()

    def test_load_partial(self, tmp_path: Path):
        """Partial YAML should use defaults."""
        import yaml

        path = tmp_path / "partial.yaml"
        with open(path, "w") as f:
            yaml.dump({"training": {"batch_size": 8}}, f)

        cfg = load_training_config(path)
        assert cfg.batch_size == 8
        assert cfg.learning_rate == 2e-4  # Default

    def test_load_without_training_key(self, tmp_path: Path):
        """YAML without 'training' key should still work."""
        import yaml

        path = tmp_path / "flat.yaml"
        with open(path, "w") as f:
            yaml.dump({"batch_size": 8, "learning_rate": 1e-4}, f)

        cfg = load_training_config(path)
        assert cfg.batch_size == 8
        assert cfg.learning_rate == 1e-4
