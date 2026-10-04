"""Local, deterministic token data.

Expected layout (produced by the user's Colab preparation, see data_reference/manifest.json):
    <data_dir>/train.npy        (n_train, seq_len + 1)  -- non-overlapping 1025-token chunks
    <data_dir>/validation.npy   (n_val,   seq_len + 1)
A (n, seq_len) layout is also accepted (last position then has no target).

The whole array is loaded into host RAM once (~0.6 GB), so timed training never touches disk
or network. The batch schedule is a pure function of (data_seed, step): every architecture sees
exactly the same sequences in exactly the same order, independently of the micro-batch size.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch


def load_token_array(path, seq_len: int, vocab_size: int, mmap: bool = False, check_range: bool = True) -> np.ndarray:
    arr = np.load(path, mmap_mode="r" if mmap else None)
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D array, got shape {arr.shape}")
    if arr.shape[1] not in (seq_len, seq_len + 1):
        raise ValueError(f"{path}: row length {arr.shape[1]} incompatible with seq_len {seq_len}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{path}: expected integer tokens, got {arr.dtype}")
    if check_range:
        mx, mn = int(arr.max()), int(arr.min())
        if mn < 0 or mx >= vocab_size:
            raise ValueError(f"{path}: token ids outside [0, {vocab_size}): min={mn} max={mx}")
    return arr


class BatchSchedule:
    """Deterministic epoch-wise permutation stream of row indices."""

    def __init__(self, n_rows: int, global_batch: int, seed: int):
        self.n_rows, self.global_batch, self.seed = int(n_rows), int(global_batch), int(seed)
        self._cache = {}

    def _perm(self, epoch: int) -> np.ndarray:
        if epoch not in self._cache:
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed, epoch, 0xDA7A])))
            self._cache = {epoch: rng.permutation(self.n_rows)}
        return self._cache[epoch]

    def rows_for_step(self, step: int) -> np.ndarray:
        start = step * self.global_batch
        out = np.empty(self.global_batch, dtype=np.int64)
        filled = 0
        while filled < self.global_batch:
            pos = start + filled
            epoch, off = divmod(pos, self.n_rows)
            take = min(self.global_batch - filled, self.n_rows - off)
            out[filled : filled + take] = self._perm(epoch)[off : off + take]
            filled += take
        return out

    def epoch_of_step(self, step: int) -> float:
        return (step * self.global_batch) / self.n_rows


def rows_to_tensor(arr: np.ndarray, rows: np.ndarray, device, pin: bool = True) -> torch.Tensor:
    batch = np.ascontiguousarray(arr[rows]).astype(np.int32, copy=False)
    t = torch.from_numpy(batch)
    if device is not None and torch.device(device).type == "cuda":
        if pin:
            t = t.pin_memory()
        t = t.to(device, non_blocking=True)
    return t.long()


def eval_rows(n_rows: int, n_eval: Optional[int]) -> np.ndarray:
    """Fixed, seed-independent validation subset: evenly spaced rows (identical for every run)."""
    if n_eval is None or n_eval >= n_rows:
        return np.arange(n_rows, dtype=np.int64)
    return np.unique(np.linspace(0, n_rows - 1, n_eval).round().astype(np.int64))


class TokenDataset:
    def __init__(self, data_dir, seq_len: int, vocab_size: int):
        data_dir = Path(data_dir)
        self.train = load_token_array(data_dir / "train.npy", seq_len, vocab_size)
        self.val = load_token_array(data_dir / "validation.npy", seq_len, vocab_size)
        self.seq_len = seq_len
        self.row_len = self.train.shape[1]
        if self.val.shape[1] != self.row_len:
            raise ValueError("train/validation row lengths differ")

    @property
    def targets_per_row(self) -> int:
        return self.seq_len if self.row_len == self.seq_len + 1 else self.seq_len - 1
