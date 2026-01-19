#!/usr/bin/env python3
"""Prepare dataset for Gemma finetuning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gemma_finetune.data.converter import normalize_to_sharegpt
from gemma_finetune.data.dedup import (
    deduplicate_by_first_turn,
    deduplicate_exact,
    get_dedup_stats,
)
from gemma_finetune.data.formatter import format_training_example
from gemma_finetune.data.loader import detect_format, load_jsonl_to_list
from gemma_finetune.data.schemas import validate_sharegpt


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare dataset for Gemma finetuning")
    parser.add_argument("input", type=Path, help="Input JSONL file")
    parser.add_argument("output", type=Path, help="Output JSONL file")
    parser.add_argument(
        "--format",
        choices=["auto", "alpaca", "sharegpt", "chatml"],
        default="auto",
        help="Input format (default: auto-detect)",
    )
    parser.add_argument(
        "--dedupe",
        choices=["none", "exact", "first-turn"],
        default="none",
        help="Deduplication strategy",
    )
    parser.add_argument("--system", type=str, help="System prompt to embed in all examples")
    parser.add_argument("--validate", action="store_true", help="Validate dataset")
    parser.add_argument("--max-examples", type=int, help="Limit number of examples")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Load data
    print(f"Loading {args.input}...")
    try:
        data = load_jsonl_to_list(args.input)
    except Exception as e:
        print(f"Error loading file: {e}")
        return 1

    if not data:
        print("Error: Dataset is empty")
        return 1

    print(f"Loaded {len(data)} examples")

    # Detect format
    if args.format == "auto":
        try:
            fmt = detect_format(data[0])
            print(f"Detected format: {fmt}")
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    else:
        fmt = args.format
        print(f"Using format: {fmt}")

    # Convert to ShareGPT
    if fmt != "sharegpt":
        print("Converting to ShareGPT format...")
        try:
            data = [normalize_to_sharegpt(item, fmt) for item in data]
        except Exception as e:
            print(f"Error converting: {e}")
            return 1

    # Validate
    if args.validate:
        print("Validating...")
        valid, errors = validate_sharegpt(data)
        if not valid:
            print(f"Validation failed with {len(errors)} errors:")
            for err in errors[:10]:  # Show first 10
                print(f"  - {err}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")
            return 1
        print("Validation passed")

    # Deduplicate
    if args.dedupe != "none":
        print(f"Deduplicating ({args.dedupe})...")
        if args.dedupe == "exact":
            deduped = deduplicate_exact(data)
        elif args.dedupe == "first-turn":
            deduped = deduplicate_by_first_turn(data)
        else:
            deduped = data

        stats = get_dedup_stats(data, deduped)
        print(
            f"  Removed {stats['removed_count']} duplicates "
            f"({stats['reduction_percent']:.1f}%)"
        )
        data = deduped

    # Limit examples
    if args.max_examples and len(data) > args.max_examples:
        print(f"Limiting to {args.max_examples} examples")
        data = data[: args.max_examples]

    # Format for training
    print("Formatting for Gemma training...")
    formatted = []
    for item in data:
        try:
            example = format_training_example(
                item["conversations"],
                system_prompt=args.system,
            )
            formatted.append(example)
        except Exception as e:
            if args.verbose:
                print(f"  Warning: Skipping malformed example: {e}")

    # Save
    print(f"Saving {len(formatted)} examples to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for item in formatted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
