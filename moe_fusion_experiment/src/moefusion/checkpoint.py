"""Resumable checkpoints: model, optimizer, step, tokens, wall-clock counters, all RNG states."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def rng_state() -> Dict[str, Any]:
    st = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["torch_cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: Dict[str, Any]) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in st:
        torch.cuda.set_rng_state_all(st["torch_cuda"])


def save_checkpoint(path, model, optimizer, step: int, state: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Atomic save (tmp file + rename). `state` holds counters such as tokens/train_seconds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": state,
        "rng": rng_state(),
        "config": config,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer, map_location="cpu", strict_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if strict_config is not None and payload.get("config") != strict_config:
        raise RuntimeError(f"CHECKPOINT MISMATCH: config stored in {path} differs from the current run config")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    set_rng_state(payload["rng"])
    return payload
