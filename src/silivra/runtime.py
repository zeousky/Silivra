"""Local hardware checks. Never collect device serial numbers or user content."""

import importlib.metadata
import platform
import resource
import subprocess
import sys

GIB = 1024**3


def hardware_info() -> dict:
    info = {
        "platform": sys.platform,
        "architecture": platform.machine(),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "metal_available": False,
    }
    if sys.platform == "darwin":
        info["unified_memory_bytes"] = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True
        ).strip())
    try:
        import mlx.core as mx
        info["mlx"] = importlib.metadata.version("mlx")
        info["metal_available"] = mx.metal.is_available()
        if info["metal_available"]:
            info["gpu"] = dict(mx.device_info())
    except ImportError:
        info["mlx"] = "not installed"
    return info


def require_apple_gpu(memory_gib: float = 6.0) -> dict:
    info = hardware_info()
    if info["platform"] != "darwin" or info["architecture"] != "arm64" or not info["metal_available"]:
        raise RuntimeError("Silivra requires an Apple silicon Mac with MLX Metal available.")
    if info.get("unified_memory_bytes", 0) < 14 * GIB:
        raise RuntimeError("This release targets Macs with at least 16 GB unified memory.")
    if not 0.25 <= memory_gib <= min(8.0, info["unified_memory_bytes"] / GIB / 2):
        raise ValueError("Memory budget must be 0.25–8 GiB and at most half of unified memory.")
    import mlx.core as mx
    mx.set_default_device(mx.gpu)
    # MLX treats this as a guideline. Training also checks measured peak usage.
    mx.set_memory_limit(int(memory_gib * GIB))
    mx.set_cache_limit(256 * 1024**2)
    return info


def memory_stats() -> dict:
    import mlx.core as mx
    return {
        "mlx_active_gib": mx.get_active_memory() / GIB,
        "mlx_peak_gib": mx.get_peak_memory() / GIB,
        "process_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GIB,
    }
