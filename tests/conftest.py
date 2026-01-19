"""Pytest fixtures for Gemma finetuning tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# Set Keras backend before any imports
os.environ["KERAS_BACKEND"] = "jax"


@pytest.fixture
def sample_sharegpt_data(tmp_path: Path) -> Path:
    """Create sample ShareGPT format data."""
    data = [
        {
            "conversations": [
                {"from": "human", "value": "What is 2+2?"},
                {"from": "gpt", "value": "2+2 equals 4."},
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "Hello!"},
                {"from": "gpt", "value": "Hello! How can I help you today?"},
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "Explain gravity briefly."},
                {"from": "gpt", "value": "Gravity is a force that attracts objects with mass toward each other."},
            ]
        },
    ]

    path = tmp_path / "sharegpt.jsonl"
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    return path


@pytest.fixture
def sample_alpaca_data(tmp_path: Path) -> Path:
    """Create sample Alpaca format data."""
    data = [
        {"instruction": "Add these numbers", "input": "2+2", "output": "4"},
        {"instruction": "Greet the user", "input": "", "output": "Hello! How can I help?"},
        {"instruction": "Explain gravity", "input": "", "output": "Gravity is the force of attraction between masses."},
    ]

    path = tmp_path / "alpaca.jsonl"
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    return path


@pytest.fixture
def sample_chatml_data(tmp_path: Path) -> Path:
    """Create sample ChatML (OpenAI) format data."""
    data = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hello! How can I help?"},
            ]
        },
    ]

    path = tmp_path / "chatml.jsonl"
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    return path


@pytest.fixture
def prepared_data(tmp_path: Path) -> Path:
    """Create prepared (formatted for training) data."""
    data = [
        {"text": "<start_of_turn>user\nWhat is 2+2?<end_of_turn>\n<start_of_turn>model\n4<end_of_turn>"},
        {"text": "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\nHello!<end_of_turn>"},
    ]

    path = tmp_path / "prepared.jsonl"
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    return path


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Temporary directory for test outputs."""
    return tmp_path


@pytest.fixture(scope="session")
def has_gpu() -> bool:
    """Check if GPU is available."""
    try:
        import jax

        devices = jax.devices("gpu")
        return len(devices) > 0
    except Exception:
        return False


@pytest.fixture
def skip_if_no_gpu(has_gpu: bool) -> None:
    """Skip test if no GPU available."""
    if not has_gpu:
        pytest.skip("No GPU available")
