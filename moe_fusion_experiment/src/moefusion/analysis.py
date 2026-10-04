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


# ----------------------------------------------------------------------------------------------------------
# MAIN verdict rule (EXPERIMENT_SPEC Amendment 2, fixed before the MAIN runs). Works for 1..n seeds.
#   d_s        paired mean difference X - Y on the full validation set in seed s (95% CI over sequences)
#   threshold  max(EQUIV_MARGIN, observed seed spread of the final loss of X and Y); with ONE seed the spread is
#              unknown, so the pre-registered practical margin (0.02 nats) is the noise floor
#   better     every seed: d_s < 0 with CI upper bound < 0, AND |mean d| >= threshold
#   worse      every seed: d_s > 0 with CI lower bound > 0, AND |mean d| >= threshold
#   equal      |mean d| < EQUIV_MARGIN and every |d_s| < EQUIV_MARGIN
#   otherwise  inconclusive
# ----------------------------------------------------------------------------------------------------------
# Amendment 4: with a SINGLE seed the decision threshold is the seed-to-seed variability of an architecture difference
# MEASURED in the diagnostic phase: B-A = -0.1719 (seed 42) vs -0.1189 (seed 43) -> |difference| = 0.053 nats.
SINGLE_SEED_THRESHOLD = 0.053

MULTI_SEED_KEYS = ["C_same_minus_B", "C_matched_minus_B", "C_matched_minus_C_same", "B_minus_A", "C_same_minus_A",
                   "C_matched_minus_A"]


def multi_seed_verdict(ds, cis, threshold):
    m = float(np.mean(ds))
    if all(d < 0 and c[1] < 0 for d, c in zip(ds, cis)) and abs(m) >= threshold:
        return "better (CI excludes 0 and abs(dL) >= %.3f nats)" % threshold
    if all(d > 0 and c[0] > 0 for d, c in zip(ds, cis)) and abs(m) >= threshold:
        return "worse (CI excludes 0 and abs(dL) >= %.3f nats)" % threshold
    if abs(m) < EQUIV_MARGIN and all(abs(d) < EQUIV_MARGIN for d in ds):
        return "approximately equal (within pre-registered margin in every seed)"
    same_sign = all(d < 0 for d in ds) or all(d > 0 for d in ds)
    return "inconclusive (" + ("consistent direction but abs(dL) below the %.3f-nat threshold" % threshold if same_sign
                               else "direction differs between seeds") + ")"


def multi_seed_summary(per_seed: Dict[int, Dict]) -> Dict:
    seeds = sorted(per_seed)
    models = [m for m in ("A", "B", "C_same", "C_matched") if all(m in per_seed[s].get("runs", {}) for s in seeds)]
    final = {}
    for m in models:
        v = {s: per_seed[s]["runs"][m]["final_val_loss"] for s in seeds}
        vals = list(v.values())
        final[m] = {"per_seed": v, "mean": float(np.mean(vals)), "spread": float(max(vals) - min(vals)),
                    "train_minutes_mean": float(np.mean([per_seed[s]["runs"][m]["train_seconds"] for s in seeds]) / 60)}
    comps = {}
    for key in MULTI_SEED_KEYS:
        x, y = key.split("_minus_")
        if x not in final or y not in final or not all(key in per_seed[s].get("paired", {}) for s in seeds):
            continue
        ds = [per_seed[s]["paired"][key]["mean"] for s in seeds]
        cis = [per_seed[s]["paired"][key]["ci95_normal"] for s in seeds]
        spread = max(final[x]["spread"], final[y]["spread"])
        threshold = max(EQUIV_MARGIN, spread) if len(seeds) > 1 else max(EQUIV_MARGIN, SINGLE_SEED_THRESHOLD)
        comps[key] = {"per_seed_d": dict(zip(seeds, ds)), "per_seed_ci": dict(zip(seeds, cis)), "mean_d": float(np.mean(ds)),
                      "seed_spread": spread if len(seeds) > 1 else None, "threshold": threshold,
                      "verdict": multi_seed_verdict(ds, cis, threshold)}
    wall = {}
    for x in ("C_same", "C_matched"):
        if x in final and "B" in final:
            per = {}
            for s in seeds:
                cw = per_seed[s].get("common_wallclock") or {}
                lx, lb = (cw.get("loss") or {}).get(x), (cw.get("loss") or {}).get("B")
                per[s] = None if lx is None or lb is None else lx - lb
            vals = [v for v in per.values() if v is not None]
            wall[f"{x}_minus_B_at_common_wallclock"] = {
                "per_seed": per,
                "verdict": ("better per wall-clock in every seed" if vals and len(vals) == len(seeds) and all(v < 0 for v in vals)
                            else "worse per wall-clock in every seed" if vals and len(vals) == len(seeds) and all(v > 0 for v in vals)
                            else "mixed / not available")}
    fusion = None
    if "C_matched_minus_B" in comps and "C_same_minus_B" in comps:
        vm, vs = comps["C_matched_minus_B"]["verdict"], comps["C_same_minus_B"]["verdict"]
        if vm.startswith("better"):
            fusion = "USEFUL: with the SAME parameter/compute budget, parallel+fusion beats the pure parallel model."
        elif vs.startswith("better"):
            fusion = ("ADDS QUALITY AT EXTRA COST: fusion improves quality only when it adds parameters/compute; "
                      "see the wall-clock comparison for whether it pays for its cost.")
        elif vs.startswith("approximately") or vs.startswith("worse"):
            fusion = "NOT USEFUL in this setting: adding fusion layers does not improve on the pure parallel model."
        else:
            fusion = ("INCONCLUSIVE: the fusion effect lies between the equivalence margin and the decision threshold; "
                      "replicate B and the C variants with seed 43 before concluding.")
        if len(seeds) == 1:
            fusion += " (single seed: not replicated)"
    return {"seeds": seeds, "models": models, "final_loss": final, "comparisons": comps, "wallclock": wall,
            "fusion_verdict": fusion,
            "rule": "d_s = paired dL per seed; better/worse require the same sign with a CI excluding 0 in EVERY seed AND "
                    "abs(mean d) >= threshold = max(%.2f, observed seed spread) [single seed: max(%.2f, %.3f measured in the "
                    "diagnostic phase)]; equal requires abs(d) < %.2f nats in every seed; anything else is inconclusive."
                    % (EQUIV_MARGIN, EQUIV_MARGIN, SINGLE_SEED_THRESHOLD, EQUIV_MARGIN)}
