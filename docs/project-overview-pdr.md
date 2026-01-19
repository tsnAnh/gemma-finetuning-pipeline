# Gemma Finetuning Pipeline - Product Development Requirements

## Executive Summary

End-to-end Gemma 3 1B model finetuning pipeline targeting RTX 3060 12GB VRAM, with TFLite export for Google AI Edge/MediaPipe deployment.

## Goals

1. **Memory-Efficient Training**: Finetune Gemma 3 1B within 12GB VRAM using LoRA
2. **Multi-Format Dataset Support**: Handle ShareGPT, Alpaca, ChatML formats
3. **Edge Deployment**: Export to TFLite with INT8/INT4 quantization
4. **MediaPipe Integration**: Create `.task` bundles for mobile inference

## Technical Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Framework | JAX + Keras 3 | Google's recommended path for Gemma |
| LoRA | Keras Hub built-in | Integrated support, optimized |
| Precision | bfloat16 | Mandatory for Gemma 3 (float16 overflows) |
| Export | ai-edge-torch | Official TFLite converter |
| Package Manager | uv | Fast, modern Python tooling |

## Target Hardware

- **GPU**: NVIDIA RTX 3060 12GB VRAM
- **Training Config**: batch_size=2, gradient_accumulation=4 (effective=8)
- **LoRA**: rank=8, alpha=16, targets q_proj/v_proj

## Constraints

1. **bfloat16 mandatory** - Gemma 3 overflows with float16
2. **No system role** - Gemma uses user/model only
3. **Merge before quantize** - LoRA weights must merge before TFLite export

## Success Criteria

- [ ] Train Gemma 3 1B on RTX 3060 without OOM
- [ ] Export functional TFLite model with INT8 quantization
- [ ] Create MediaPipe .task bundle
- [ ] Achieve <50ms inference latency on edge device

## Timeline

Phase 1-2 (Complete): Project setup, dataset tools
Phase 3-4 (Complete): Model loading, training pipeline
Phase 5-6 (Complete): TFLite export, inference utilities
Phase 7 (Complete): Testing & validation
