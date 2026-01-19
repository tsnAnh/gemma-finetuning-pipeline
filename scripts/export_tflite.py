#!/usr/bin/env python3
"""Export finetuned Gemma to TFLite."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["KERAS_BACKEND"] = "jax"

from gemma_finetune.export.bundle import create_mediapipe_task
from gemma_finetune.export.converter import convert_to_tflite, verify_tflite_model
from gemma_finetune.export.merge import merge_lora_weights, save_merged_model, verify_merge
from gemma_finetune.models.config import LoRAConfig, ModelConfig, load_config
from gemma_finetune.models.loader import load_gemma_model
from gemma_finetune.models.lora import apply_lora_to_model, load_lora_weights


def main() -> int:
    parser = argparse.ArgumentParser(description="Export finetuned Gemma to TFLite")
    parser.add_argument("--config", type=Path, help="Model/LoRA config YAML")
    parser.add_argument("--lora-weights", type=Path, required=True, help="LoRA weights .npz")
    parser.add_argument("--output-dir", type=Path, default=Path("./exports"), help="Output dir")
    parser.add_argument(
        "--quantization",
        choices=["none", "dynamic", "int8", "int4"],
        default="int8",
        help="Quantization type",
    )
    parser.add_argument("--create-task", action="store_true", help="Create MediaPipe .task bundle")
    parser.add_argument("--model-name", default="gemma-1b-finetuned", help="Model name for bundle")
    parser.add_argument("--skip-verify", action="store_true", help="Skip verification steps")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    if args.config:
        print(f"Loading config from {args.config}")
        model_cfg, lora_cfg = load_config(args.config)
    else:
        print("Using default config")
        model_cfg = ModelConfig()
        lora_cfg = LoRAConfig()

    # Load model with LoRA
    print("\n1. Loading base model...")
    try:
        model = load_gemma_model(model_cfg)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    print("\n2. Applying LoRA configuration...")
    model = apply_lora_to_model(model, lora_cfg)

    print(f"\n3. Loading LoRA weights from {args.lora_weights}...")
    try:
        load_lora_weights(model, args.lora_weights)
    except Exception as e:
        print(f"Error loading LoRA weights: {e}")
        return 1

    # Merge LoRA into base
    print("\n4. Merging LoRA weights into base model...")
    print("   (CRITICAL: Must merge BEFORE quantization)")
    model = merge_lora_weights(model)

    if not args.skip_verify and not verify_merge(model):
        print("Warning: LoRA merge verification failed")

    # Save merged model
    print("\n5. Saving merged model...")
    merged_path = output_dir / "merged"
    save_merged_model(model, merged_path)

    # Convert to TFLite
    print(f"\n6. Converting to TFLite ({args.quantization} quantization)...")
    tflite_filename = f"gemma_1b_{args.quantization}.tflite"
    tflite_path = output_dir / tflite_filename

    try:
        convert_to_tflite(merged_path, tflite_path, args.quantization)
    except Exception as e:
        print(f"Error during TFLite conversion: {e}")
        print("Try installing ai-edge-torch: pip install ai-edge-torch")
        return 1

    # Verify TFLite model
    if not args.skip_verify:
        print("\n7. Verifying TFLite model...")
        try:
            info = verify_tflite_model(tflite_path)
            print(f"   Size: {info['size_mb']:.1f} MB")
            print(f"   Inputs: {info['num_inputs']}")
            print(f"   Outputs: {info['num_outputs']}")
        except Exception as e:
            print(f"   Warning: Verification failed: {e}")

    # Create MediaPipe bundle
    if args.create_task:
        print("\n8. Creating MediaPipe task bundle...")
        task_output = output_dir / args.model_name
        tokenizer_path = merged_path / "preprocessor"

        try:
            create_mediapipe_task(
                tflite_path,
                tokenizer_path,
                task_output,
                model_name=args.model_name,
                max_sequence_length=model_cfg.max_sequence_length,
            )
        except Exception as e:
            print(f"   Warning: Could not create task bundle: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("Export Complete!")
    print("=" * 50)
    print(f"TFLite model: {tflite_path}")
    if args.create_task:
        task_path = output_dir / f"{args.model_name}.task"
        if task_path.exists():
            print(f"Task bundle: {task_path}")
    print(f"Merged model: {merged_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
