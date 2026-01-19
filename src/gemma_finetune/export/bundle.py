"""MediaPipe task bundle creation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def create_mediapipe_task(
    tflite_path: str | Path,
    tokenizer_path: str | Path,
    output_path: str | Path,
    model_name: str = "gemma-1b-finetuned",
    model_version: str = "1.0.0",
    max_sequence_length: int = 512,
) -> Path:
    """Create MediaPipe .task bundle.

    Bundle contains:
    - TFLite model
    - Tokenizer files
    - Model metadata

    Args:
        tflite_path: Path to TFLite model.
        tokenizer_path: Path to tokenizer directory or file.
        output_path: Output path (without extension).
        model_name: Model name for metadata.
        model_version: Version string.
        max_sequence_length: Max sequence length.

    Returns:
        Path to created .task file.
    """
    tflite_path = Path(tflite_path)
    tokenizer_path = Path(tokenizer_path)
    output_path = Path(output_path)

    # Create temp directory for bundle contents
    bundle_dir = output_path.parent / f"{output_path.stem}_bundle_temp"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Copy TFLite model
        shutil.copy(tflite_path, bundle_dir / "model.tflite")
        print(f"Added TFLite model: {tflite_path.stat().st_size / (1024*1024):.1f} MB")

        # Copy tokenizer
        tokenizer_dest = bundle_dir / "tokenizer"
        tokenizer_dest.mkdir(parents=True, exist_ok=True)

        if tokenizer_path.is_dir():
            # Copy entire directory
            for file in tokenizer_path.iterdir():
                if file.is_file():
                    shutil.copy(file, tokenizer_dest / file.name)
            print(f"Added tokenizer directory: {tokenizer_path}")
        elif tokenizer_path.is_file():
            # Copy single file
            shutil.copy(tokenizer_path, tokenizer_dest / tokenizer_path.name)
            print(f"Added tokenizer file: {tokenizer_path}")
        else:
            print(f"Warning: Tokenizer path not found: {tokenizer_path}")

        # Create metadata
        metadata = {
            "name": model_name,
            "version": model_version,
            "model_type": "gemma_causal_lm",
            "task_type": "text_generation",
            "max_sequence_length": max_sequence_length,
            "created_by": "gemma-finetune",
        }

        with open(bundle_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print("Added metadata.json")

        # Create .task file (zip archive)
        task_path = output_path.with_suffix(".task")

        # Create zip and rename
        shutil.make_archive(
            str(output_path.with_suffix("")),
            "zip",
            bundle_dir,
        )

        zip_path = output_path.with_suffix(".zip")
        if zip_path.exists():
            zip_path.rename(task_path)

        task_size = task_path.stat().st_size / (1024 * 1024)
        print(f"Created MediaPipe task bundle: {task_path} ({task_size:.1f} MB)")

        return task_path

    finally:
        # Cleanup temp directory
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)


def extract_task_bundle(task_path: str | Path, output_dir: str | Path) -> Path:
    """Extract MediaPipe .task bundle.

    Args:
        task_path: Path to .task file.
        output_dir: Directory to extract to.

    Returns:
        Path to extracted directory.
    """
    task_path = Path(task_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.unpack_archive(task_path, output_dir, "zip")
    print(f"Extracted to {output_dir}")

    return output_dir


def get_task_metadata(task_path: str | Path) -> dict:
    """Get metadata from .task bundle without full extraction.

    Args:
        task_path: Path to .task file.

    Returns:
        Metadata dictionary.
    """
    import zipfile

    task_path = Path(task_path)

    with zipfile.ZipFile(task_path, "r") as zf:
        if "metadata.json" in zf.namelist():
            with zf.open("metadata.json") as f:
                return json.load(f)

    return {}
