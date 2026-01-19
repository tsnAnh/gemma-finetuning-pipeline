# System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE                                │
├─────────────────────────────────────────────────────────────────────┤
│  Raw Data      →    Validation    →    Conversion    →    Formatted │
│  (ShareGPT/       (schemas.py)       (converter.py)      (formatter)│
│   Alpaca/ChatML)                                          (Gemma    │
│                                                            template)│
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        MODEL PIPELINE                                │
├─────────────────────────────────────────────────────────────────────┤
│  Base Model    →    LoRA Applied   →    Training    →    Checkpoint │
│  (keras_hub)       (lora.py)           (trainer.py)      (.npz)     │
│  Gemma 1B          rank=8, α=16        AdamW+cosine                 │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       EXPORT PIPELINE                                │
├─────────────────────────────────────────────────────────────────────┤
│  LoRA Merge    →    TFLite Convert  →    Quantize    →   .task     │
│  (merge.py)        (converter.py)       INT8/INT4      (bundle.py) │
│  CRITICAL!         ai-edge-torch                       MediaPipe    │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      INFERENCE PIPELINE                              │
├─────────────────────────────────────────────────────────────────────┤
│  Keras (GPU)   │   TFLite (CPU)   │   Chat Session                  │
│  Full model    │   Quantized      │   Multi-turn                    │
│  keras_inf.py  │   tflite_inf.py  │   chat.py                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
src/gemma_finetune/
├── __init__.py           # Package exports
├── config.py             # Global config (KERAS_BACKEND, tokens)
├── data/
│   ├── schemas.py        # Format validation (ShareGPT/Alpaca/ChatML)
│   ├── loader.py         # JSONL I/O, format detection
│   ├── converter.py      # Format conversion to ShareGPT
│   ├── formatter.py      # Gemma chat template formatting
│   └── dedup.py          # Deduplication utilities
├── models/
│   ├── config.py         # ModelConfig, LoRAConfig dataclasses
│   ├── loader.py         # load_gemma_model() via keras_hub
│   └── lora.py           # apply_lora, save/load weights
├── training/
│   ├── config.py         # TrainingConfig
│   ├── optimizer.py      # AdamW + cosine decay + warmup
│   ├── callbacks.py      # LoRASave, MetricsLogger, MemoryMonitor
│   ├── data_pipeline.py  # TF dataset creation for Keras
│   └── trainer.py        # GemmaTrainer orchestration
├── export/
│   ├── merge.py          # merge_lora_weights() - CRITICAL
│   ├── converter.py      # convert_to_tflite()
│   └── bundle.py         # create_mediapipe_task()
├── inference/
│   ├── keras_inference.py  # GPU inference
│   ├── tflite_inference.py # CPU/edge inference
│   └── chat.py             # ChatSession multi-turn
└── utils/
    └── gpu.py            # GPU memory utilities
```

## Data Flow

### 1. Dataset Preparation

```
Input (raw)          Validated           Converted           Formatted
┌──────────┐        ┌──────────┐        ┌──────────┐        ┌──────────┐
│ ShareGPT │───────→│ ShareGPT │───────→│ ShareGPT │───────→│ {"text": │
│ Alpaca   │validate│ Alpaca   │convert │          │format  │ "<start> │
│ ChatML   │        │ ChatML   │        │          │        │ user..."}│
└──────────┘        └──────────┘        └──────────┘        └──────────┘
```

### 2. Training Flow

```
Formatted Data → Tokenize → Batch → Forward → Loss → Backward → Update
                    ↓
              Only LoRA weights are updated (0.5% of params)
```

### 3. Export Flow

```
Checkpoint    Merge LoRA      TFLite         Quantize       Bundle
┌─────────┐   ┌─────────┐    ┌─────────┐    ┌─────────┐   ┌─────────┐
│ base +  │──→│ merged  │───→│ .tflite │───→│ INT8/4  │──→│ .task   │
│ lora.npz│   │ weights │    │         │    │ quant   │   │MediaPipe│
└─────────┘   └─────────┘    └─────────┘    └─────────┘   └─────────┘
              CRITICAL!
```

## Key Design Decisions

### 1. JAX + Keras 3 over PyTorch
- Google's official path for Gemma
- Native keras_hub support
- Better TPU compatibility

### 2. bfloat16 Mandatory
- Gemma 3 has numerical instability with float16
- bfloat16 required for stable training

### 3. LoRA Configuration
- rank=8 (smaller for 1B model)
- alpha=16 (2x rank scaling)
- Targets: q_proj, v_proj (attention layers)

### 4. Merge Before Quantize
- LoRA weights must merge before TFLite export
- Quantizing separate adapters loses precision
