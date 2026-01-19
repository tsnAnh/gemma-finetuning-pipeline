# Code Standards

## Python Style

### General
- Python 3.10+ with type hints
- Line length: 100 chars (ruff enforced)
- Imports sorted with isort (via ruff)

### Naming
```python
# Modules: snake_case
data_pipeline.py
keras_inference.py

# Classes: PascalCase
class TrainingConfig:
class GemmaTrainer:

# Functions/variables: snake_case
def load_gemma_model():
batch_size = 2

# Constants: UPPER_SNAKE_CASE
GEMMA_START = "<start_of_turn>"
DEFAULT_LORA = LoRAConfig()
```

### Docstrings
```python
def format_gemma_chat(
    conversations: list[dict[str, str]],
    add_generation_prompt: bool = False,
) -> str:
    """Format conversations to Gemma chat template.

    Args:
        conversations: List of conversation turns with 'from' and 'value' keys.
        add_generation_prompt: If True, add model turn marker at end.

    Returns:
        Formatted string in Gemma chat template format.
    """
```

## Type Hints

```python
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

def load_config(path: str | Path) -> tuple[ModelConfig, LoRAConfig]:
    ...

def validate_sharegpt(data: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    ...
```

## Dataclasses

```python
@dataclass
class LoRAConfig:
    """LoRA hyperparameters."""

    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
```

## Error Handling

```python
# Validation errors
if not valid:
    raise ValueError(f"Unknown dataset format: {detected}")

# File operations
if not path.exists():
    raise FileNotFoundError(f"Config not found: {path}")

# Runtime checks
if not hasattr(model, "backbone"):
    raise RuntimeError("Model does not support LoRA")
```

## Testing

```python
class TestLoRAConfig:
    """Tests for LoRAConfig dataclass."""

    def test_default_values(self):
        """Default values should be sensible for Gemma 1B."""
        cfg = LoRAConfig()
        assert cfg.rank == 8

    def test_invalid_rank(self):
        """Rank must be positive."""
        with pytest.raises(ValueError, match="rank must be positive"):
            LoRAConfig(rank=0)
```

## File Organization

```
# Module init exports public API
# src/gemma_finetune/data/__init__.py
from .loader import load_jsonl, load_jsonl_to_list, save_jsonl
from .schemas import validate_sharegpt, validate_alpaca
from .converter import normalize_to_sharegpt
from .formatter import format_gemma_chat, format_training_example
```

## Configuration

```yaml
# configs/lora_default.yaml
model:
  model_id: "gemma_1b_en"
  dtype: "bfloat16"
  max_sequence_length: 512

lora:
  rank: 8
  alpha: 16
  dropout: 0.05
  target_modules:
    - "q_proj"
    - "v_proj"
```

## Git Commits

Use conventional commits:
```
feat: add LoRA weight saving
fix: handle empty conversations in formatter
docs: update training config documentation
test: add validation schema tests
refactor: extract optimizer into separate module
```
