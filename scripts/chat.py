#!/usr/bin/env python3
"""Interactive chat with finetuned Gemma."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["KERAS_BACKEND"] = "jax"


def stream_print(token: str) -> None:
    """Print token without newline."""
    print(token, end="", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive chat with finetuned Gemma")

    # Model loading options (mutually exclusive groups)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--lora", type=Path, help="LoRA weights path (uses base model + LoRA)")
    model_group.add_argument("--merged", type=Path, help="Path to merged Keras model")
    model_group.add_argument("--tflite", type=Path, help="TFLite model path")

    # Additional options
    parser.add_argument("--tokenizer", type=Path, help="Tokenizer path (required for TFLite)")
    parser.add_argument("--system", type=str, help="System prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")

    args = parser.parse_args()

    # Load appropriate inference engine
    if args.tflite:
        print("Loading TFLite model...")
        if not args.tokenizer:
            print("Error: --tokenizer required for TFLite inference")
            return 1
        from gemma_finetune.inference.tflite_inference import TFLiteInference

        engine = TFLiteInference(args.tflite, args.tokenizer)

    elif args.merged:
        print("Loading merged Keras model...")
        from gemma_finetune.inference.keras_inference import KerasInference

        engine = KerasInference.from_merged_model(str(args.merged))

    else:  # args.lora
        print("Loading base model with LoRA...")
        from gemma_finetune.inference.keras_inference import KerasInference

        engine = KerasInference.from_checkpoint(lora_path=str(args.lora))

    # Create chat session
    from gemma_finetune.inference.chat import ChatSession

    chat = ChatSession(engine, system_prompt=args.system)

    # Interactive loop
    print("\n" + "=" * 50)
    print("Chat ready! Commands:")
    print("  'quit' or 'exit' - Exit chat")
    print("  'clear' - Clear conversation history")
    print("  'system <prompt>' - Set system prompt")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        lower_input = user_input.lower()
        if lower_input in ("quit", "exit"):
            print("Goodbye!")
            break

        if lower_input == "clear":
            chat.clear()
            print("Conversation cleared.\n")
            continue

        if lower_input.startswith("system "):
            new_system = user_input[7:].strip()
            chat.set_system_prompt(new_system)
            print(f"System prompt updated: {new_system[:50]}...\n")
            continue

        # Generate response
        print("Assistant: ", end="", flush=True)

        try:
            if args.no_stream:
                response = chat.chat(
                    user_input,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                print(response)
            else:
                response = chat.chat(
                    user_input,
                    stream_callback=stream_print,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                print()  # Newline after streaming
        except Exception as e:
            print(f"\nError: {e}")

        print()  # Extra newline between turns

    return 0


if __name__ == "__main__":
    sys.exit(main())
