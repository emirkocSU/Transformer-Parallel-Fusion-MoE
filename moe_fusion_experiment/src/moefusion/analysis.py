"""Result aggregation and PRE-REGISTERED statistics (rules fixed in EXPERIMENT_SPEC.md before any run).

Pre-registered constants:
  EQUIV_MARGIN      = 0.02 nats   |dL| below this -> "approximately equal" (~2% perplexity)
  TARGET_OFFSETS    = (0.00, 0.05, 0.10, 0.25) nats above the WORST final periodic-subset loss
                      (every model reaches every target by construction; no target favours a model)
  SPEED_MARGIN      = 2%          step-time differences below this are "no speed difference"
  CI                = 95%, paired over validation sequences (same sequences for every model).
                      With ONE seed this interval captures evaluation-data noise only, NOT seed-to-seed
                      training variance; seed variance requires the 3-seed replication phase.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .metrics import read_jsonl
from .utils import read_json

EQUIV_MARGIN = 0.02
TARGET_OFFSETS = (0.0, 0.05, 0.10, 0.25)
SPEED_MARGIN = 0.02
Z95 = 1.959963984540054


def load_run(run_dir) -> Optional[Dict]:
    run_dir = Path(run_dir)
    if not (run_dir / "summary.json").exists():
        fail = run_dir / "failure.json"
        return {"name": run_dir.name, "failed": True, "failure": read_json(fail) if fail.exists() else None,
                "dir": str(run_dir)} if fail.exists() else None
    s = read_json(run_dir / "summary.json")
    ev = read_jsonl(run_dir / "eval_metrics.jsonl")
    periodic = sorted([e for e in ev if not e.get("final")], key=lambda e: e["step"])
    # keep the last record per step (a resumed run may re-evaluate a step)
    dedup = {}
    for e in periodic:
        dedup[e["step"]] = e
    periodic = [dedup[k] for k in sorted(dedup)]
    per_seq = None
    if (run_dir / "final_val_per_sequence_loss.npy").exists():
        per_seq = np.load(run_dir / "final_val_per_sequence_loss.npy").astype(np.float64)
    man = read_json(run_dir / "experiment_manifest.json") if (run_dir / "experiment_manifest.json").exists() else {}
    return {"name": s["config"]["model"]["name"], "failed": False, "summary": s, "periodic": periodic,
            "train": read_jsonl(run_dir / "train_metrics.jsonl"), "per_seq": per_seq, "manifest": man, "dir": str(run_dir)}


def paired_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Statistics of (b - a) over the SAME validation sequences."""
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    n = d.size
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = d[rng.integers(0, n, n)].mean()
    return {"mean": mean, "se": se, "ci95_normal": [mean - Z95 * se, mean + Z95 * se],
            "ci95_bootstrap": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n_sequences": int(n), "frac_sequences_b_better": float((d < 0).mean())}


def classify_diff(stat: Dict[str, float], margin: float = EQUIV_MARGIN) -> str:
    lo, hi = stat["ci95_normal"]
    m = stat["mean"]
    if abs(m) < margin and lo > -margin and hi < margin:
        return "approximately equal (within pre-registered margin)"
    if lo > 0:
        return "worse (CI excludes 0)" + ("" if m >= margin else ", but below practical margin")
    if hi < 0:
        return "better (CI excludes 0)" + ("" if -m >= margin else ", but below practical margin")
    return "inconclusive (CI includes 0)"


def interp_x_at_target(curve: List[Dict], target: float, xkey: str) -> Optional[float]:
    """First x at which the (piecewise-linear) validation-loss curve reaches <= target."""
    for i, e in enumerate(curve):
        if e["val_loss"] <= target:
            if i == 0:
                return float(e[xkey])
            p = curve[i - 1]
            l0, l1 = p["val_loss"], e["val_loss"]
            f = (l0 - target) / (l0 - l1) if l0 != l1 else 1.0
            return float(p[xkey] + f * (e[xkey] - p[xkey]))
    return None


def loss_at_x(curve: List[Dict], x: float, xkey: str) -> Optional[float]:
    xs = [e[xkey] for e in curve]
    if not xs or x < xs[0] or x > xs[-1]:
        return None
    return float(np.interp(x, xs, [e["val_loss"] for e in curve]))


def time_to_targets(runs: Dict[str, Dict]) -> Dict:
    finals = {k: r["periodic"][-1]["val_loss"] for k, r in runs.items() if r["periodic"]}
    worst = max(finals.values())
    out = {"rule": f"targets = worst final periodic-subset loss ({worst:.4f}) + {list(TARGET_OFFSETS)}", "targets": {}}
    for off in TARGET_OFFSETS:
        tgt = worst + off
        out["targets"][f"{tgt:.4f}"] = {
            k: {"train_seconds": interp_x_at_target(r["periodic"], tgt, "train_seconds"),
                "tokens": interp_x_at_target(r["periodic"], tgt, "tokens")} for k, r in runs.items()}
    return out


def loss_at_common_wallclock(runs: Dict[str, Dict]) -> Dict:
    t_star = min(r["periodic"][-1]["train_seconds"] for r in runs.values())
    return {"t_star_seconds": t_star,
            "definition": "validation loss (periodic subset) linearly interpolated at the training wall-clock at which "
                          "the FASTEST model finished its token budget",
            "loss": {k: loss_at_x(r["periodic"], t_star, "train_seconds") for k, r in runs.items()}}


def fairness_audit(runs: Dict[str, Dict], allowed=("name", "arch", "expert_hidden", "fusion_interval",
                                                     "matched_reference_hidden")) -> Dict:
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    rs = list(runs.values())
    s0 = rs[0]["summary"]
    for r in rs[1:]:
        s = r["summary"]
        add(f"train config identical ({r['name']} vs {rs[0]['name']})", s["config"]["train"] == s0["config"]["train"])
        diff = [k for k in s0["config"]["model"] if s0["config"]["model"][k] != s["config"]["model"][k] and k not in allowed]
        add(f"model config differs only in architecture fields ({r['name']})", not diff, f"unexpected: {diff}" if diff else "")
    add("identical optimizer steps", len({r["summary"]["steps"] for r in rs}) == 1, str({r["name"]: r["summary"]["steps"] for r in rs}))
    add("identical training tokens", len({r["summary"]["tokens_input"] for r in rs}) == 1)
    add("identical micro-batch / grad-accum", len({(r["summary"]["micro_batch_seqs"], r["summary"]["grad_accum"]) for r in rs}) == 1)
    add("identical evaluation steps", len({tuple(e["step"] for e in r["periodic"]) for r in rs}) == 1)
    add("identical final validation set", len({(r["summary"]["final_val_sequences"], r["summary"]["final_val_tokens"]) for r in rs}) == 1)
    gpus = {seg["gpu"] for r in rs for seg in r["summary"]["segments"]}
    add("all runs (incl. resumed segments) on the same GPU model", len(gpus) == 1, str(gpus))
    for key in ("attention_backend", "moe_backend", "parallel_backend"):
        add(f"identical {key}", len({r["summary"]["config"]["model"][key] for r in rs}) == 1)
    fps = {r["manifest"].get("code_fingerprint") for r in rs}
    add("identical code fingerprint", len(fps) == 1, str(fps))
    dh = {str(r["manifest"].get("dataset_hashes")) for r in rs}
    add("identical dataset hashes", len(dh) == 1)
    return {"passed": all(c["passed"] for c in checks), "checks": checks}
