#!/usr/bin/env python3
"""Benchmark inference performance."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["KERAS_BACKEND"] = "jax"


def benchmark_keras(lora_path: str | None, num_runs: int = 10) -> dict:
    """Benchmark Keras inference.

    Args:
        lora_path: Optional path to LoRA weights.
        num_runs: Number of benchmark runs.

    Returns:
        Benchmark results.
    """
    from gemma_finetune.inference.keras_inference import KerasInference

    print("Loading Keras model...")
    engine = KerasInference.from_checkpoint(lora_path=lora_path)

    prompt = "Explain the theory of relativity in simple terms."

    # Warmup
    print("Warming up...")
    engine.generate(prompt, max_new_tokens=50)

    # Benchmark
    print(f"Running {num_runs} benchmarks...")
    times = []
    token_counts = []

    for i in range(num_runs):
        start = time.perf_counter()
        output = engine.generate(prompt, max_new_tokens=100)
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        token_counts.append(len(output.split()))

        print(f"  Run {i+1}: {elapsed:.2f}s, ~{token_counts[-1]} tokens")

    avg_time = sum(times) / len(times)
    avg_tokens = sum(token_counts) / len(token_counts)
    tokens_per_sec = avg_tokens / avg_time

    return {
        "engine": "keras",
        "runs": num_runs,
        "avg_time_sec": avg_time,
        "avg_tokens": avg_tokens,
        "tokens_per_sec": tokens_per_sec,
        "min_time_sec": min(times),
        "max_time_sec": max(times),
    }


def benchmark_tflite(model_path: str, tokenizer_path: str, num_runs: int = 10) -> dict:
    """Benchmark TFLite inference.

    Args:
        model_path: Path to TFLite model.
        tokenizer_path: Path to tokenizer.
        num_runs: Number of benchmark runs.

    Returns:
        Benchmark results.
    """
    from gemma_finetune.inference.tflite_inference import TFLiteInference

    print("Loading TFLite model...")
    engine = TFLiteInference(model_path, tokenizer_path)

    prompt = "Explain the theory of relativity in simple terms."

    # Warmup
    print("Warming up...")
    engine.generate(prompt, max_new_tokens=50)

    # Benchmark
    print(f"Running {num_runs} benchmarks...")
    times = []

    for i in range(num_runs):
        start = time.perf_counter()
        engine.generate(prompt, max_new_tokens=100)
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")

    avg_time = sum(times) / len(times)

    return {
        "engine": "tflite",
        "runs": num_runs,
        "avg_time_sec": avg_time,
        "tokens_per_sec": 100 / avg_time,  # Estimated
        "min_time_sec": min(times),
        "max_time_sec": max(times),
    }


def print_results(results: dict) -> None:
    """Print benchmark results."""
    print("\n" + "=" * 50)
    print(f"Benchmark Results ({results['engine'].upper()})")
    print("=" * 50)
    print(f"Runs:           {results['runs']}")
    print(f"Avg time:       {results['avg_time_sec']:.2f}s")
    print(f"Min time:       {results['min_time_sec']:.2f}s")
    print(f"Max time:       {results['max_time_sec']:.2f}s")
    print(f"Tokens/sec:     {results['tokens_per_sec']:.1f}")
    print("=" * 50)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark inference performance")

    # Mode selection
    parser.add_argument("--keras", action="store_true", help="Benchmark Keras inference")
    parser.add_argument("--tflite", action="store_true", help="Benchmark TFLite inference")

    # Model paths
    parser.add_argument("--lora", type=Path, help="LoRA weights path (for Keras)")
    parser.add_argument("--model", type=Path, help="Model path (TFLite file)")
    parser.add_argument("--tokenizer", type=Path, help="Tokenizer path (for TFLite)")

    # Benchmark options
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")

    args = parser.parse_args()

    if not args.keras and not args.tflite:
        print("Error: Specify --keras or --tflite")
        return 1

    if args.keras:
        try:
            results = benchmark_keras(str(args.lora) if args.lora else None, args.runs)
            print_results(results)
        except Exception as e:
            print(f"Keras benchmark failed: {e}")
            return 1

    if args.tflite:
        if not args.model or not args.tokenizer:
            print("Error: --model and --tokenizer required for TFLite benchmark")
            return 1

        try:
            results = benchmark_tflite(str(args.model), str(args.tokenizer), args.runs)
            print_results(results)
        except Exception as e:
            print(f"TFLite benchmark failed: {e}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
