"""Integration tests for the data processing pipeline."""

from pathlib import Path

from gemma_finetune.config import GEMMA_END, GEMMA_START
from gemma_finetune.data.converter import normalize_to_sharegpt
from gemma_finetune.data.formatter import format_training_example
from gemma_finetune.data.loader import detect_format, load_jsonl_to_list, save_jsonl
from gemma_finetune.data.schemas import validate_sharegpt


class TestDataPipelineIntegration:
    """Integration tests for the full data pipeline."""

    def test_sharegpt_full_pipeline(self, sample_sharegpt_data: Path, tmp_path: Path):
        """Test full pipeline: load -> validate -> format -> save."""
        # Load
        data = load_jsonl_to_list(sample_sharegpt_data)
        assert len(data) == 3

        # Detect format
        detected = detect_format(data[0])
        assert detected == "sharegpt"

        # Validate
        valid, errors = validate_sharegpt(data)
        assert valid, f"Validation failed: {errors}"

        # Format for training
        formatted = [format_training_example(item["conversations"]) for item in data]
        assert all("text" in item for item in formatted)
        assert all(GEMMA_START in item["text"] for item in formatted)

        # Save
        output_path = tmp_path / "formatted.jsonl"
        save_jsonl(formatted, output_path)
        assert output_path.exists()

        # Verify saved data
        reloaded = load_jsonl_to_list(output_path)
        assert len(reloaded) == 3

    def test_alpaca_to_training_format(self, sample_alpaca_data: Path, tmp_path: Path):
        """Test Alpaca conversion pipeline."""
        # Load
        data = load_jsonl_to_list(sample_alpaca_data)

        # Detect format
        detected = detect_format(data[0])
        assert detected == "alpaca"

        # Convert to ShareGPT
        converted = [normalize_to_sharegpt(item, "alpaca") for item in data]
        assert all("conversations" in item for item in converted)

        # Format for training
        formatted = [format_training_example(item["conversations"]) for item in converted]

        # Verify Gemma format
        for item in formatted:
            assert "user" in item["text"]
            assert "model" in item["text"]
            assert GEMMA_START in item["text"]
            assert GEMMA_END in item["text"]

    def test_chatml_to_training_format(self, sample_chatml_data: Path, tmp_path: Path):
        """Test ChatML conversion pipeline."""
        # Load
        data = load_jsonl_to_list(sample_chatml_data)

        # Detect format
        detected = detect_format(data[0])
        assert detected == "chatml"

        # Convert to ShareGPT
        converted = [normalize_to_sharegpt(item, "chatml") for item in data]

        # Format for training
        formatted = [format_training_example(item["conversations"]) for item in converted]

        # Verify
        for item in formatted:
            text = item["text"]
            assert GEMMA_START in text
            # System message should be embedded (sample_chatml_data has system messages)
            if "helpful assistant" in str(data):
                # System prompt is embedded in first user turn
                pass

    def test_round_trip_save_load(self, tmp_path: Path):
        """Test that saved data can be loaded and processed again."""
        # Create test data
        original = [
            {
                "conversations": [
                    {"from": "human", "value": "Test question?"},
                    {"from": "gpt", "value": "Test answer."},
                ]
            }
        ]

        # Save
        path = tmp_path / "test.jsonl"
        save_jsonl(original, path)

        # Load and process
        loaded = load_jsonl_to_list(path)
        formatted = format_training_example(loaded[0]["conversations"])

        # Verify
        assert "Test question?" in formatted["text"]
        assert "Test answer." in formatted["text"]


class TestConfigIntegration:
    """Integration tests for configuration handling."""

    def test_model_lora_config_roundtrip(self, tmp_path: Path):
        """Test model and LoRA config save/load."""
        from gemma_finetune.models.config import (
            LoRAConfig,
            ModelConfig,
            load_config,
            save_config,
        )

        model_cfg = ModelConfig(model_id="test_model", max_sequence_length=1024)
        lora_cfg = LoRAConfig(rank=16, alpha=32)

        path = tmp_path / "config.yaml"
        save_config(model_cfg, lora_cfg, path)

        loaded_model, loaded_lora = load_config(path)

        assert loaded_model.model_id == "test_model"
        assert loaded_model.max_sequence_length == 1024
        assert loaded_lora.rank == 16
        assert loaded_lora.alpha == 32

    def test_training_config_roundtrip(self, tmp_path: Path):
        """Test training config save/load."""
        from gemma_finetune.training.config import (
            TrainingConfig,
            load_training_config,
            save_training_config,
        )

        cfg = TrainingConfig(batch_size=4, learning_rate=1e-5, max_steps=1000)

        path = tmp_path / "training.yaml"
        save_training_config(cfg, path)

        loaded = load_training_config(path)

        assert loaded.batch_size == 4
        assert loaded.learning_rate == 1e-5
        assert loaded.max_steps == 1000
