"""Training callbacks for checkpointing and logging."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

os.environ["KERAS_BACKEND"] = "jax"

import keras


class LoRASaveCallback(keras.callbacks.Callback):
    """Save only LoRA weights periodically."""

    def __init__(self, output_dir: str, save_steps: int = 100):
        """Initialize callback.

        Args:
            output_dir: Directory to save checkpoints.
            save_steps: Save every N training steps.
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_steps = save_steps
        self.step = 0

    def on_train_batch_end(self, batch: int, logs: dict | None = None) -> None:
        self.step += 1
        if self.step % self.save_steps == 0:
            self._save_checkpoint()

    def on_train_end(self, logs: dict | None = None) -> None:
        # Save final checkpoint
        self._save_checkpoint(final=True)

    def _save_checkpoint(self, final: bool = False) -> None:
        from ..models.lora import save_lora_weights

        if final:
            path = self.output_dir / "lora_final.npz"
        else:
            path = self.output_dir / f"lora_step_{self.step}.npz"

        try:
            save_lora_weights(self.model, str(path))
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")


class MetricsLogger(keras.callbacks.Callback):
    """Log training metrics to JSON file."""

    def __init__(self, log_path: str):
        """Initialize callback.

        Args:
            log_path: Path to JSON log file.
        """
        super().__init__()
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics: list[dict] = []
        self.step = 0

    def on_train_batch_end(self, batch: int, logs: dict | None = None) -> None:
        self.step += 1
        if logs:
            entry = {
                "step": self.step,
                "batch": batch,
                "timestamp": datetime.now().isoformat(),
            }
            # Convert numpy/jax arrays to Python types
            for k, v in logs.items():
                try:
                    entry[k] = float(v)
                except (TypeError, ValueError):
                    entry[k] = str(v)
            self.metrics.append(entry)

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if logs:
            entry = {
                "epoch": epoch,
                "timestamp": datetime.now().isoformat(),
                "type": "epoch_end",
            }
            for k, v in logs.items():
                try:
                    entry[k] = float(v)
                except (TypeError, ValueError):
                    entry[k] = str(v)
            self.metrics.append(entry)

    def on_train_end(self, logs: dict | None = None) -> None:
        self._save_metrics()

    def _save_metrics(self) -> None:
        with open(self.log_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Saved metrics to {self.log_path}")


class MemoryMonitor(keras.callbacks.Callback):
    """Monitor GPU memory usage during training."""

    def __init__(self, log_every_n_steps: int = 50):
        """Initialize callback.

        Args:
            log_every_n_steps: Log memory every N steps.
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.step = 0

    def on_train_batch_end(self, batch: int, logs: dict | None = None) -> None:
        self.step += 1
        if self.step % self.log_every_n_steps == 0:
            self._log_memory()

    def _log_memory(self) -> None:
        try:
            import jax

            devices = jax.devices("gpu")
            if devices:
                device = devices[0]
                mem_stats = device.memory_stats()
                if mem_stats:
                    used_gb = mem_stats.get("bytes_in_use", 0) / (1024**3)
                    limit_gb = mem_stats.get("bytes_limit", 0) / (1024**3)
                    print(f"Step {self.step}: GPU Memory {used_gb:.2f}/{limit_gb:.2f} GB")
        except Exception:
            pass


class EarlyStoppingOnNaN(keras.callbacks.Callback):
    """Stop training if NaN loss is detected."""

    def on_train_batch_end(self, batch: int, logs: dict | None = None) -> None:
        import math

        if logs:
            loss = logs.get("loss", 0)
            if math.isnan(loss) or math.isinf(loss):
                print(f"NaN/Inf loss detected at step {batch}. Stopping training.")
                self.model.stop_training = True
