#!/usr/bin/env python3
"""Verify environment setup for Gemma finetuning."""

from __future__ import annotations

import sys


def main() -> int:
    """Check environment and print status."""
    print("=" * 50)
    print("Gemma Finetune Environment Check")
    print("=" * 50)

    errors = []

    # Check Python version
    py_version = sys.version_info
    if py_version >= (3, 10):
        print(f"[OK] Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        errors.append(f"Python 3.10+ required, found {py_version.major}.{py_version.minor}")
        print(f"[FAIL] Python version: {py_version.major}.{py_version.minor}")

    # Check JAX
    try:
        import jax

        print(f"[OK] JAX {jax.__version__}")

        # Check GPU
        devices = jax.devices("gpu")
        if devices:
            print(f"[OK] GPU devices: {len(devices)}")
            for i, d in enumerate(devices):
                print(f"     Device {i}: {d}")
        else:
            print("[WARN] No GPU devices found - will use CPU")

    except ImportError as e:
        errors.append(f"JAX not installed: {e}")
        print(f"[FAIL] JAX: {e}")

    # Check Keras
    try:
        import os

        os.environ["KERAS_BACKEND"] = "jax"
        import keras

        print(f"[OK] Keras {keras.__version__}")
        backend = keras.backend.backend()
        if backend == "jax":
            print(f"[OK] Keras backend: {backend}")
        else:
            print(f"[WARN] Keras backend: {backend} (expected 'jax')")

    except ImportError as e:
        errors.append(f"Keras not installed: {e}")
        print(f"[FAIL] Keras: {e}")

    # Check Keras Hub
    try:
        import keras_hub

        print(f"[OK] Keras Hub {keras_hub.__version__}")

    except ImportError as e:
        errors.append(f"Keras Hub not installed: {e}")
        print(f"[FAIL] Keras Hub: {e}")

    # Check bfloat16 support
    try:
        import jax.numpy as jnp

        x = jnp.array([1.0, 65504.0], dtype=jnp.bfloat16)
        if x.dtype == jnp.bfloat16:
            print("[OK] bfloat16 support")
        else:
            print("[WARN] bfloat16 may not be fully supported")

    except Exception as e:
        print(f"[WARN] bfloat16 check failed: {e}")

    # Check other dependencies
    deps = [
        ("datasets", "datasets"),
        ("pandera", "pandera"),
        ("jsonlines", "jsonlines"),
        ("yaml", "pyyaml"),
        ("tqdm", "tqdm"),
    ]

    for module, name in deps:
        try:
            __import__(module)
            print(f"[OK] {name}")
        except ImportError:
            print(f"[WARN] {name} not installed")

    # Check export dependencies (optional)
    print("\n--- Optional Export Dependencies ---")
    try:
        import tensorflow as tf

        print(f"[OK] TensorFlow {tf.__version__}")
    except ImportError:
        print("[INFO] TensorFlow not installed (needed for export)")

    try:
        import ai_edge_torch

        print(f"[OK] AI Edge Torch {ai_edge_torch.__version__}")
    except ImportError:
        print("[INFO] AI Edge Torch not installed (needed for export)")

    # Summary
    print("\n" + "=" * 50)
    if errors:
        print("Environment check FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1
    else:
        print("Environment ready for Gemma finetuning!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
