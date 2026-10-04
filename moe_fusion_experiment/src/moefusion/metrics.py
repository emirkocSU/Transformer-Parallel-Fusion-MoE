"""Machine-readable metric logging (JSONL) and router/expert utilisation statistics."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class JsonlLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")

    def log(self, record: Dict[str, Any]) -> None:
        self._f.write(json.dumps(record, default=_default) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def read_jsonl(path) -> List[Dict[str, Any]]:
    out = []
    p = Path(path)
    if not p.exists():
        return out
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def utilization_stats(counts: np.ndarray, top_k: int) -> Dict[str, Any]:
    """counts: (n_moe_layers, E) assignment counts accumulated over an interval."""
    counts = np.asarray(counts, dtype=np.float64)
    tot = counts.sum(axis=1, keepdims=True)
    frac = counts / np.maximum(tot, 1.0)  # per-layer fraction of assignments
    E = counts.shape[1]
    per_layer_cv = frac.std(axis=1) / np.maximum(frac.mean(axis=1), 1e-12)
    mins = frac.min(axis=1)
    maxs = frac.max(axis=1)
    ratio = np.where(mins > 0, maxs / np.maximum(mins, 1e-12), np.inf)
    agg = counts.sum(axis=0)
    agg_frac = agg / max(agg.sum(), 1.0)
    out: Dict[str, Any] = {f"expert_{e}_fraction": float(agg_frac[e]) for e in range(E)}
    out.update(
        {
            "util_cv_mean": float(per_layer_cv.mean()),
            "util_cv_max": float(per_layer_cv.max()),
            "util_maxmin_ratio_max": float(ratio.max()) if np.isfinite(ratio).all() else math.inf,
            "util_min_fraction": float(mins.min()),
            "util_max_fraction": float(maxs.max()),
            "per_layer_fraction": frac.round(5).tolist(),
        }
    )
    return out


def collapse_check(frac_per_layer: np.ndarray, top_k: int, min_fraction: float) -> List[int]:
    """Layers in which at most top_k experts receive >= min_fraction of assignments
    (i.e. the router effectively uses no more experts than it selects per token)."""
    frac = np.asarray(frac_per_layer)
    active = (frac >= min_fraction).sum(axis=1)
    return [int(i) for i in np.nonzero(active <= top_k)[0]]
