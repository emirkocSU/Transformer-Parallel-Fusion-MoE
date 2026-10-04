"""Resumable checkpoints: model, optimizer, step, tokens, wall-clock counters, all RNG states.

Disk safety: a checkpoint is written to `<path>.tmp` and atomically renamed, which transiently needs space for the
old AND the new file. If free space is insufficient for both, the old checkpoint is deleted first (logged); if it is
still insufficient the save is skipped (logged) instead of crashing the run with "No space left on device".
"""
from __future__ import annotations

import os
import random
import shutil
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
    # RNG states must be CPU ByteTensors, whatever device the checkpoint was mapped to.
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch_cpu"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in st:
        torch.cuda.set_rng_state_all([s.cpu() for s in st["torch_cuda"]])


def estimate_checkpoint_bytes(model, optimizer) -> int:
    """fp32 weights + optimizer state (AdamW: 2 fp32 moments; estimated as 2x params before the first step)."""
    n = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    opt = 0
    for st in optimizer.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                opt += v.numel() * v.element_size()
    if opt == 0:
        opt = 2 * sum(p.numel() * 4 for p in model.parameters())
    return int(n + opt)


def free_bytes(path) -> int:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def save_checkpoint(path, model, optimizer, step: int, state: Dict[str, Any], config: Dict[str, Any],
                    margin_gb: float = 2.0) -> Dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    need = estimate_checkpoint_bytes(model, optimizer)
    margin = int(margin_gb * 1e9)
    info: Dict[str, Any] = {"saved": False, "bytes_estimate": need, "deleted_old_first": False}
    if free_bytes(path.parent) < need + margin and path.exists():
        path.unlink()  # cannot hold old + new: give up atomicity rather than fail
        info["deleted_old_first"] = True
    free = free_bytes(path.parent)
    info["free_bytes_before"] = free
    if free < need + margin:
        info["reason"] = f"insufficient disk: {free / 1e9:.1f} GB free, {need / 1e9:.1f} GB + {margin_gb} GB margin needed"
        return info
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
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    except OSError as e:
        if tmp.exists():
            tmp.unlink()
        info["reason"] = f"write failed: {e}"
        return info
    info["saved"] = True
    return info


def load_checkpoint(path, model, optimizer, strict_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Always deserialised on the CPU: RNG states must be CPU tensors, and a 6-7 GB checkpoint must not create a
    second copy in GPU memory. model/optimizer.load_state_dict copy every tensor to its parameter's device
    (AdamW 'step' included, as required by the fused kernel)."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if strict_config is not None and payload.get("config") != strict_config:
        raise RuntimeError(f"CHECKPOINT MISMATCH: config stored in {path} differs from the current run config")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    set_rng_state(payload["rng"])
    return payload
