"""Global configuration and environment setup."""

import os

# Set Keras backend to JAX before any Keras imports
os.environ["KERAS_BACKEND"] = "jax"

# Hardware defaults optimized for RTX 3060 12GB
DEFAULT_DTYPE = "bfloat16"
MAX_SEQUENCE_LENGTH = 512
DEFAULT_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4

# LoRA defaults
DEFAULT_LORA_RANK = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05

# Training defaults
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_WARMUP_STEPS = 100
DEFAULT_MAX_STEPS = 500

# Gemma chat template markers
GEMMA_START = "<start_of_turn>"
GEMMA_END = "<end_of_turn>"
GEMMA_BOS = "<bos>"
