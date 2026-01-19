"""GPU detection and verification utilities."""

from __future__ import annotations


def get_gpu_info() -> dict:
    """Return GPU device info and memory stats.

    Returns:
        Dictionary with GPU information or availability status.
    """
    try:
        import jax

        devices = jax.devices("gpu")
        if not devices:
            return {"available": False, "reason": "No GPU devices found"}

        device = devices[0]
        info = {
            "available": True,
            "name": str(device),
            "count": len(devices),
            "platform": device.platform,
            "device_kind": getattr(device, "device_kind", "unknown"),
        }

        # Try to get memory stats
        try:
            mem_stats = device.memory_stats()
            if mem_stats:
                info["memory_bytes_in_use"] = mem_stats.get("bytes_in_use", 0)
                info["memory_bytes_limit"] = mem_stats.get("bytes_limit", 0)
                info["memory_gb_in_use"] = info["memory_bytes_in_use"] / (1024**3)
                info["memory_gb_limit"] = info["memory_bytes_limit"] / (1024**3)
        except Exception:
            pass

        return info

    except ImportError:
        return {"available": False, "reason": "JAX not installed"}
    except Exception as e:
        return {"available": False, "reason": str(e)}


def verify_bfloat16_support() -> bool:
    """Check if GPU supports bfloat16 operations.

    Gemma 3 requires bfloat16 precision to avoid overflow.

    Returns:
        True if bfloat16 is supported, False otherwise.
    """
    try:
        import jax.numpy as jnp

        x = jnp.array([1.0, 65504.0], dtype=jnp.bfloat16)
        result = x + 1.0
        return result.dtype == jnp.bfloat16

    except Exception:
        return False


def check_memory_available(required_gb: float = 4.0) -> tuple[bool, float]:
    """Check if sufficient GPU memory is available.

    Args:
        required_gb: Minimum required memory in GB.

    Returns:
        Tuple of (is_sufficient, available_gb).
    """
    info = get_gpu_info()

    if not info.get("available"):
        return False, 0.0

    limit_gb = info.get("memory_gb_limit", 0)
    used_gb = info.get("memory_gb_in_use", 0)
    available_gb = limit_gb - used_gb

    return available_gb >= required_gb, available_gb


def get_recommended_batch_size(available_memory_gb: float, model_size: str = "1b") -> int:
    """Get recommended batch size based on available memory.

    Args:
        available_memory_gb: Available GPU memory in GB.
        model_size: Model size identifier ("1b", "2b", etc.).

    Returns:
        Recommended batch size.
    """
    # Memory estimates per batch (1B model with LoRA, bfloat16)
    memory_per_batch = {
        "1b": 2.0,  # ~2GB per batch
        "2b": 3.5,  # ~3.5GB per batch
    }

    base_memory = 3.0  # Base model memory
    per_batch = memory_per_batch.get(model_size, 2.0)

    usable = available_memory_gb - base_memory
    if usable <= 0:
        return 1

    batch_size = max(1, int(usable / per_batch))
    return min(batch_size, 4)  # Cap at 4 for stability
