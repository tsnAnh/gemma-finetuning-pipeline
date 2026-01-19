# Tech Stack: Gemma 3 1B Finetuning Pipeline

**Date:** 2026-01-19 | **Target:** RTX 3060 12GB VRAM

## Core Framework

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| ML Framework | JAX | 0.4.x | XLA compilation, autodiff |
| High-level API | Keras 3 | 3.x | Model loading, training API |
| Backend | JAX | - | Keras backend for XLA |

## Model & Training

| Component | Technology | Purpose |
|-----------|------------|---------|
| Base Model | `google/gemma-3-1b-it` | Instruction-tuned Gemma 3 1B |
| PEFT | Keras LoRA | Parameter-efficient finetuning |
| Quantization | bfloat16 | Memory efficiency + precision |
| Optimizer | AdamW | Standard LLM optimizer |

## Dataset Tools

| Component | Technology | Purpose |
|-----------|------------|---------|
| Data Loading | `datasets` (HuggingFace) | JSONL/CSV streaming |
| Validation | `pandera` | Schema validation |
| Processing | `jsonlines` | JSONL handling |
| Format | ShareGPT/ChatML | Multi-turn conversations |

## Edge Deployment

| Component | Technology | Purpose |
|-----------|------------|---------|
| Export Format | TFLite | Edge-optimized format |
| Converter | `ai-edge-torch` | JAX/Keras → TFLite |
| Runtime | MediaPipe GenAI | Mobile/edge inference |
| Quantization | INT8/INT4 | Size reduction for edge |

## Development Tools

| Component | Technology | Purpose |
|-----------|------------|---------|
| Python | 3.10+ | Runtime |
| Package Mgmt | `uv` | Fast dependency management |
| Testing | `pytest` | Unit/integration tests |
| Linting | `ruff` | Fast Python linter |
| Formatting | `ruff format` | Code formatting |
| Type Checking | `pyright` | Static type analysis |

## Project Structure

```
gemma-finetune/
├── src/
│   ├── data/           # Dataset processing
│   ├── models/         # Model loading, LoRA config
│   ├── training/       # Training loop, callbacks
│   ├── export/         # TFLite conversion
│   └── inference/      # Local inference utilities
├── scripts/            # CLI entry points
├── tests/              # Test suite
├── configs/            # Training configurations
├── data/               # Dataset storage (gitignored)
├── checkpoints/        # Model checkpoints (gitignored)
└── exports/            # Exported models (gitignored)
```

## Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | RTX 3060 12GB | RTX 4090 24GB |
| RAM | 16GB | 32GB |
| Storage | 50GB SSD | 100GB NVMe |
| CUDA | 12.x | 12.x |

## Key Hyperparameters (RTX 3060)

| Parameter | Value | Notes |
|-----------|-------|-------|
| LoRA Rank | 8-16 | Start with 8 |
| LoRA Alpha | 16-32 | 2x rank |
| Batch Size | 2-4 | Adjust based on sequence length |
| Gradient Accumulation | 4-8 | Effective batch 16-32 |
| Learning Rate | 1e-4 to 2e-4 | Conservative |
| Max Sequence Length | 512-1024 | Balance memory vs context |
| Precision | bfloat16 | Required for Gemma 3 |
