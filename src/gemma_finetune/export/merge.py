"""LoRA merge utilities for export."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ["KERAS_BACKEND"] = "jax"



def merge_lora_weights(model: Any) -> Any:
    """Merge LoRA adapters into base model weights.

    This produces a single model without LoRA layers.
    CRITICAL: Must be done BEFORE any quantization.

    Args:
        model: Model with LoRA adapters.

    Returns:
        Model with LoRA weights merged into base.
    """
    print("Merging LoRA weights into base model...")

    # Keras Hub models have merge_lora() method
    if hasattr(model, "backbone") and hasattr(model.backbone, "merge_lora"):
        model.backbone.merge_lora()
        print("LoRA weights merged successfully!")
        return model

    # If using PEFT/manual LoRA, attempt manual merge
    print("Warning: Built-in merge_lora() not found. Attempting manual merge...")

    merged_count = 0
    for layer in model.layers:
        if hasattr(layer, "lora_enabled") and layer.lora_enabled:
            _merge_layer_lora(layer)
            merged_count += 1

    if merged_count > 0:
        print(f"Manually merged {merged_count} LoRA layers")
    else:
        print("Warning: No LoRA layers found to merge")

    return model


def _merge_layer_lora(layer: Any) -> None:
    """Merge LoRA for a single layer.

    LoRA formula: W_merged = W_base + (lora_A @ lora_B) * scale

    Args:
        layer: Layer with LoRA adapters.
    """
    import numpy as np

    if not hasattr(layer, "kernel"):
        return

    base_weight = layer.kernel.numpy()

    # Check for LoRA matrices
    lora_a = getattr(layer, "lora_a", None)
    lora_b = getattr(layer, "lora_b", None)

    if lora_a is None or lora_b is None:
        return

    lora_a_val = lora_a.numpy()
    lora_b_val = lora_b.numpy()

    # Get scaling
    alpha = getattr(layer, "lora_alpha", 1.0)
    rank = getattr(layer, "lora_rank", 1)
    scale = alpha / rank if rank > 0 else 1.0

    # Compute merged weight
    lora_contribution = np.matmul(lora_a_val, lora_b_val) * scale
    merged_weight = base_weight + lora_contribution

    # Update weight
    layer.kernel.assign(merged_weight)

    # Disable LoRA
    if hasattr(layer, "lora_enabled"):
        layer.lora_enabled = False


def save_merged_model(model: Any, output_dir: str | Path) -> Path:
    """Save merged model for TFLite conversion.

    Args:
        model: Model with LoRA merged (or base model).
        output_dir: Output directory.

    Returns:
        Path to saved model directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "model.keras"
    preprocessor_path = output_path / "preprocessor"

    # Save model
    print(f"Saving model to {model_path}...")
    model.save(str(model_path))

    # Save preprocessor/tokenizer if available
    if hasattr(model, "preprocessor") and model.preprocessor is not None:
        print(f"Saving preprocessor to {preprocessor_path}...")
        try:
            model.preprocessor.save_to_preset(str(preprocessor_path))
        except Exception as e:
            print(f"Warning: Could not save preprocessor: {e}")

    print(f"Merged model saved to {output_path}")
    return output_path


def verify_merge(model: Any) -> bool:
    """Verify LoRA was properly merged.

    Args:
        model: Model that should have merged LoRA.

    Returns:
        True if no active LoRA layers found.
    """
    # Check for any remaining active LoRA
    for layer in model.layers:
        if hasattr(layer, "lora_enabled") and layer.lora_enabled:
            print(f"Warning: Layer {layer.name} still has LoRA enabled")
            return False

    # Check backbone if exists
    if hasattr(model, "backbone"):
        for layer in model.backbone.layers:
            if hasattr(layer, "lora_enabled") and layer.lora_enabled:
                print(f"Warning: Backbone layer {layer.name} still has LoRA enabled")
                return False

    return True
