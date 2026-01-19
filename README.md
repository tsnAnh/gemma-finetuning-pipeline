# Gemma Finetuning Pipeline

End-to-end Gemma 3 1B model finetuning pipeline with JAX/Keras, LoRA, and TFLite export for edge deployment.

## Features

- **JAX + Keras 3**: Google's recommended stack for Gemma finetuning
- **LoRA (Low-Rank Adaptation)**: Parameter-efficient finetuning for 12GB VRAM
- **Multi-format Dataset Support**: ShareGPT, Alpaca, ChatML
- **TFLite Export**: Edge deployment via ai-edge-torch
- **MediaPipe Integration**: `.task` bundle for mobile/edge inference

## Requirements

- Python 3.10+
- NVIDIA GPU with 12GB+ VRAM (RTX 3060 or better)
- CUDA 12.x

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd gemma-finetune

# Install with uv (recommended)
uv sync --extra cuda --extra dev

# Or with pip
pip install -e ".[cuda,dev]"
```

## Quick Start

### 1. Prepare Dataset

```bash
# Convert and validate your dataset
uv run python scripts/prepare_dataset.py \
    --input data/raw/my_dataset.jsonl \
    --output data/prepared/train.jsonl \
    --format sharegpt
```

### 2. Train Model

```bash
# Start finetuning
uv run python scripts/train.py \
    --data data/prepared/train.jsonl \
    --output checkpoints/my_model \
    --model-config configs/lora_default.yaml \
    --training-config configs/training_default.yaml
```

### 3. Export to TFLite

```bash
# Export for edge deployment
uv run python scripts/export_tflite.py \
    --checkpoint checkpoints/my_model \
    --output exports/model.tflite \
    --quantization int8
```

### 4. Chat with Model

```bash
# Interactive chat
uv run python scripts/chat.py \
    --model checkpoints/my_model
```

## Dataset Formats

### ShareGPT Format
```json
{
  "conversations": [
    {"from": "human", "value": "What is 2+2?"},
    {"from": "gpt", "value": "2+2 equals 4."}
  ]
}
```

### Alpaca Format
```json
{
  "instruction": "Add the numbers",
  "input": "2+2",
  "output": "4"
}
```

### ChatML Format
```json
{
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"}
  ]
}
```

## Configuration

### LoRA Config (`configs/lora_default.yaml`)
```yaml
model:
  model_id: "gemma_1b_en"
  dtype: "bfloat16"  # Required for Gemma 3
  max_sequence_length: 512

lora:
  rank: 8
  alpha: 16
  dropout: 0.05
  target_modules:
    - "q_proj"
    - "v_proj"
```

### Training Config (`configs/training_default.yaml`)
```yaml
training:
  learning_rate: 2e-4
  batch_size: 2
  gradient_accumulation_steps: 4
  max_steps: 500
  warmup_steps: 100
```

## Project Structure

```
gemma-finetune/
├── src/gemma_finetune/
│   ├── data/           # Dataset handling
│   ├── models/         # Model and LoRA config
│   ├── training/       # Training pipeline
│   ├── export/         # TFLite export
│   └── inference/      # Local inference
├── scripts/            # CLI tools
├── configs/            # Default configurations
├── tests/              # Test suite
└── docs/               # Documentation
```

## Important Notes

- **bfloat16 is mandatory** for Gemma 3 models (float16 causes overflow)
- LoRA weights must be **merged before quantization** for optimal quality
- Gemma uses **user/model roles only** (no system role)

## License

MIT
