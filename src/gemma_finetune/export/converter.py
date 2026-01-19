"""TFLite conversion utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

os.environ["KERAS_BACKEND"] = "jax"


def convert_to_tflite(
    model_path: str | Path,
    output_path: str | Path,
    quantization: Literal["none", "dynamic", "int8", "int4"] = "int8",
) -> Path:
    """Convert Keras model to TFLite format.

    Uses ai-edge-torch for optimized LLM conversion.

    Args:
        model_path: Path to saved Keras model directory.
        output_path: Output TFLite file path.
        quantization: Quantization type.

    Returns:
        Path to saved TFLite model.
    """
    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Converting to TFLite with {quantization} quantization...")

    try:
        # Try ai-edge-torch first (optimized for Gemma)
        return _convert_with_ai_edge_torch(model_path, output_path, quantization)
    except ImportError:
        print("ai-edge-torch not available, falling back to TF Lite converter")
        return convert_with_tf_lite_converter(model_path, output_path, quantization)


def _convert_with_ai_edge_torch(
    model_path: Path,
    output_path: Path,
    quantization: str,
) -> Path:
    """Convert using ai-edge-torch (Google's optimized converter)."""
    import keras
    from ai_edge_torch.generative import converter as edge_converter

    # Load merged model
    keras_model_path = model_path / "model.keras"
    model = keras.models.load_model(str(keras_model_path))

    # Configure quantization
    quant_config = None
    if quantization == "int8":
        quant_config = edge_converter.QuantConfig(
            bits=8,
            granularity="per_channel",
        )
    elif quantization == "int4":
        quant_config = edge_converter.QuantConfig(
            bits=4,
            granularity="per_channel",
        )
    elif quantization == "dynamic":
        quant_config = edge_converter.QuantConfig(
            bits=8,
            dynamic=True,
        )

    # Convert
    converter = edge_converter.Converter(
        model=model,
        quant_config=quant_config,
    )

    tflite_model = converter.convert()

    # Save
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved TFLite model: {output_path} ({size_mb:.1f} MB)")

    return output_path


def convert_with_tf_lite_converter(
    model_path: str | Path,
    output_path: str | Path,
    quantization: str = "dynamic",
) -> Path:
    """Convert using standard TensorFlow Lite converter.

    Fallback for when ai-edge-torch is not available.

    Args:
        model_path: Path to Keras model directory.
        output_path: Output TFLite file path.
        quantization: Quantization type.

    Returns:
        Path to saved TFLite model.
    """
    import keras
    import tensorflow as tf

    model_path = Path(model_path)
    output_path = Path(output_path)

    # Load Keras model
    keras_model_path = model_path / "model.keras"
    model = keras.models.load_model(str(keras_model_path))

    # Create converter from Keras model
    # First save as SavedModel format
    saved_model_path = model_path / "saved_model"
    model.export(str(saved_model_path), format="tf_saved_model")

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))

    # Quantization settings
    if quantization == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
    elif quantization == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    # Convert
    print("Converting with TensorFlow Lite converter...")
    tflite_model = converter.convert()

    # Save
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved TFLite model: {output_path} ({size_mb:.1f} MB)")

    return output_path


def verify_tflite_model(tflite_path: str | Path) -> dict:
    """Verify TFLite model can be loaded and get basic info.

    Args:
        tflite_path: Path to TFLite model.

    Returns:
        Model information dictionary.
    """
    import tensorflow as tf

    tflite_path = Path(tflite_path)

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return {
        "path": str(tflite_path),
        "size_mb": tflite_path.stat().st_size / (1024 * 1024),
        "num_inputs": len(input_details),
        "num_outputs": len(output_details),
        "input_shapes": [d["shape"].tolist() for d in input_details],
        "output_shapes": [d["shape"].tolist() for d in output_details],
        "input_dtypes": [d["dtype"].__name__ for d in input_details],
    }
