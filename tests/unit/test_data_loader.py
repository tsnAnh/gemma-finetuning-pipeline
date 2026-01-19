"""Tests for data loading utilities."""

from pathlib import Path

import pytest

from gemma_finetune.data.loader import (
    detect_format,
    load_jsonl,
    load_jsonl_to_list,
    save_jsonl,
)


class TestJSONLLoading:
    """Tests for JSONL file loading."""

    def test_load_jsonl_iterator(self, sample_sharegpt_data: Path):
        """load_jsonl should return an iterator."""
        data = load_jsonl(sample_sharegpt_data)
        first = next(data)
        assert "conversations" in first

    def test_load_jsonl_to_list(self, sample_sharegpt_data: Path):
        """load_jsonl_to_list should return a list."""
        data = load_jsonl_to_list(sample_sharegpt_data)
        assert isinstance(data, list)
        assert len(data) == 3

    def test_load_empty_file(self, tmp_path: Path):
        """Empty file should return empty list."""
        empty_file = tmp_path / "empty.jsonl"
        empty_file.touch()
        data = load_jsonl_to_list(empty_file)
        assert data == []


class TestFormatDetection:
    """Tests for dataset format detection."""

    def test_detect_sharegpt(self):
        """Detect ShareGPT format."""
        sample = {"conversations": []}
        assert detect_format(sample) == "sharegpt"

    def test_detect_alpaca(self):
        """Detect Alpaca format."""
        sample = {"instruction": "test", "output": "result"}
        assert detect_format(sample) == "alpaca"

    def test_detect_chatml(self):
        """Detect ChatML format."""
        sample = {"messages": []}
        assert detect_format(sample) == "chatml"

    def test_detect_text(self):
        """Detect plain text format."""
        sample = {"text": "some text"}
        assert detect_format(sample) == "text"

    def test_detect_unknown(self):
        """Unknown format should raise ValueError."""
        sample = {"unknown_key": "value"}
        with pytest.raises(ValueError, match="Unknown dataset format"):
            detect_format(sample)


class TestSaveJSONL:
    """Tests for JSONL saving."""

    def test_save_and_reload(self, tmp_path: Path):
        """Saved data should match original."""
        data = [{"key": "value1"}, {"key": "value2"}]
        path = tmp_path / "output.jsonl"

        save_jsonl(data, path)
        reloaded = load_jsonl_to_list(path)

        assert reloaded == data

    def test_creates_parent_dirs(self, tmp_path: Path):
        """Should create parent directories."""
        path = tmp_path / "nested" / "dir" / "output.jsonl"
        data = [{"test": True}]

        save_jsonl(data, path)
        assert path.exists()
