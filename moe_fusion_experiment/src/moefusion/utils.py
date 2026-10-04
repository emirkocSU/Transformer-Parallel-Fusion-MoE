"""Small shared utilities: seeding, hashing, profiler ranges, environment capture."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Named profiler ranges. Disabled by default so that timed training runs do not pay
# for them; scripts/profile_model.py enables them (identically for every architecture).
# ---------------------------------------------------------------------------
_RANGES_ENABLED = False


def set_profiling_ranges(enabled: bool) -> None:
    global _RANGES_ENABLED
    _RANGES_ENABLED = bool(enabled)


def prange(name: str):
    if _RANGES_ENABLED:
        return torch.profiler.record_function(name)
    return contextlib.nullcontext()


# ---------------------------------------------------------------------------
def stable_seed(*parts: Any) -> int:
    """Deterministic 63-bit seed from arbitrary parts (independent of PYTHONHASHSEED)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_json(path, obj, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=_json_default)
    os.replace(tmp, path)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def git_commit(root) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def code_fingerprint(root) -> str:
    """sha256 over all project source files (identifies the exact code even without git)."""
    root = Path(root)
    h = hashlib.sha256()
    for sub in ("src", "scripts", "configs", "tests"):
        for p in sorted((root / sub).rglob("*")):
            if p.is_file() and p.suffix in (".py", ".yaml", ".yml", ".json") and "__pycache__" not in p.parts:
                h.update(str(p.relative_to(root)).encode())
                h.update(p.read_bytes())
    return h.hexdigest()


def environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1e9, 2)
        info["ram_available_gb"] = round(vm.available / 1e9, 2)
    except Exception:
        pass
    try:
        du = shutil.disk_usage("/content" if os.path.isdir("/content") else "/")
        info["disk_free_gb"] = round(du.free / 1e9, 2)
    except Exception:
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu_name": p.name,
                "gpu_total_memory_gb": round(p.total_memory / 1e9, 2),
                "gpu_compute_capability": f"{p.major}.{p.minor}",
                "gpu_count": torch.cuda.device_count(),
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
                "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
            }
        )
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version,name,memory.total,clocks.max.sm", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            info["nvidia_smi_query"] = out.stdout.strip()
        except Exception:
            info["nvidia_smi_query"] = None
    try:
        import flash_attn  # type: ignore

        info["flash_attn_package"] = getattr(flash_attn, "__version__", "unknown")
    except Exception:
        info["flash_attn_package"] = None
    return info


def configure_torch_numerics() -> Dict[str, Any]:
    """Identical numeric settings for every run (recorded in manifests)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "allow_tf32_matmul": False,
        "allow_tf32_cudnn": False,
        "float32_matmul_precision": "highest",
        "autocast_dtype": "bfloat16",
        "note": "BF16 autocast for matmuls; FP32 master weights; FP32 router, norms, softmax/CE.",
    }


def fmt_num(n: float) -> str:
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.3f}{unit}"
    return f"{n:.0f}"
