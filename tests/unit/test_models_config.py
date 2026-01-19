"""Tests for model and LoRA configuration."""

from pathlib import Path

import pytest

from gemma_finetune.models.config import (
    DEFAULT_LORA,
    DEFAULT_MODEL,
    LoRAConfig,
    ModelConfig,
    load_config,
    save_config,
)


class TestLoRAConfig:
    """Tests for LoRAConfig dataclass."""

    def test_default_values(self):
        """Default values should be sensible for Gemma 1B."""
        cfg = LoRAConfig()
        assert cfg.rank == 8
        assert cfg.alpha == 16
        assert cfg.dropout == 0.05
        assert "q_proj" in cfg.target_modules
        assert "v_proj" in cfg.target_modules

    def test_custom_values(self):
        """Custom values should be accepted."""
        cfg = LoRAConfig(rank=16, alpha=32, dropout=0.1)
        assert cfg.rank == 16
        assert cfg.alpha == 32
        assert cfg.dropout == 0.1

    def test_invalid_rank(self):
        """Rank must be positive."""
        with pytest.raises(ValueError, match="rank must be positive"):
            LoRAConfig(rank=0)
        with pytest.raises(ValueError, match="rank must be positive"):
            LoRAConfig(rank=-1)

    def test_invalid_alpha(self):
        """Alpha must be positive."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            LoRAConfig(alpha=0)

    def test_invalid_dropout(self):
        """Dropout must be in [0, 1)."""
        with pytest.raises(ValueError, match="Dropout must be"):
            LoRAConfig(dropout=-0.1)
        with pytest.raises(ValueError, match="Dropout must be"):
            LoRAConfig(dropout=1.0)

    def test_zero_dropout_valid(self):
        """Zero dropout should be valid."""
        cfg = LoRAConfig(dropout=0.0)
        assert cfg.dropout == 0.0


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_values(self):
        """Default values for Gemma model."""
        cfg = ModelConfig()
        assert cfg.model_id == "gemma_1b_en"
        assert cfg.dtype == "bfloat16"
        assert cfg.max_sequence_length == 512

    def test_custom_model_id(self):
        """Custom model ID should work."""
        cfg = ModelConfig(model_id="gemma_2b_en")
        assert cfg.model_id == "gemma_2b_en"

    def test_invalid_sequence_length(self):
        """Sequence length must be positive."""
        with pytest.raises(ValueError, match="max_sequence_length must be positive"):
            ModelConfig(max_sequence_length=0)

    def test_cache_dir(self):
        """Cache dir should be optional."""
        cfg1 = ModelConfig()
        assert cfg1.cache_dir is None
        cfg2 = ModelConfig(cache_dir="/tmp/cache")
        assert cfg2.cache_dir == "/tmp/cache"


class TestConfigIO:
    """Tests for config save/load functions."""

    def test_save_and_load(self, tmp_path: Path):
        """Saved config should match loaded config."""
        model_cfg = ModelConfig(
            model_id="gemma_2b_en",
            max_sequence_length=1024,
        )
        lora_cfg = LoRAConfig(rank=16, alpha=32, dropout=0.1)
        path = tmp_path / "config.yaml"

        save_config(model_cfg, lora_cfg, path)
        loaded_model, loaded_lora = load_config(path)

        assert loaded_model.model_id == "gemma_2b_en"
        assert loaded_model.max_sequence_length == 1024
        assert loaded_lora.rank == 16
        assert loaded_lora.alpha == 32
        assert loaded_lora.dropout == 0.1

    def test_creates_parent_dirs(self, tmp_path: Path):
        """Save should create parent directories."""
        path = tmp_path / "nested" / "dir" / "config.yaml"
        save_config(DEFAULT_MODEL, DEFAULT_LORA, path)
        assert path.exists()

    def test_load_partial_config(self, tmp_path: Path):
        """Partial YAML should use defaults for missing values."""
        import yaml

        path = tmp_path / "partial.yaml"
        partial = {"model": {"model_id": "custom_model"}}
        with open(path, "w") as f:
            yaml.dump(partial, f)

        model_cfg, lora_cfg = load_config(path)
        assert model_cfg.model_id == "custom_model"
        assert model_cfg.dtype == "bfloat16"  # Default
        assert lora_cfg.rank == 8  # Default

    def test_load_empty_config(self, tmp_path: Path):
        """Empty YAML should use all defaults."""
        import yaml

        path = tmp_path / "empty.yaml"
        with open(path, "w") as f:
            yaml.dump({}, f)

        model_cfg, lora_cfg = load_config(path)
        assert model_cfg.model_id == DEFAULT_MODEL.model_id
        assert lora_cfg.rank == DEFAULT_LORA.rank


class TestDefaultConfigs:
    """Tests for default configuration instances."""

    def test_default_lora_exists(self):
        """DEFAULT_LORA should be valid."""
        assert DEFAULT_LORA.rank > 0
        assert DEFAULT_LORA.alpha > 0

    def test_default_model_exists(self):
        """DEFAULT_MODEL should be valid."""
        assert DEFAULT_MODEL.model_id
        assert DEFAULT_MODEL.dtype == "bfloat16"
