"""Aggregate run logs -> results.json, then plots and FINAL_REPORT.md. Never hardcodes numbers."""
import argparse
from pathlib import Path

import _common  # noqa: F401
from _common import banner

import numpy as np

from moefusion.analysis import (EQUIV_MARGIN, SPEED_MARGIN, classify_diff, fairness_audit, load_run,
                                loss_at_common_wallclock, paired_diff, time_to_targets)
from moefusion.utils import read_json, write_json

SHORT = {"A_serial": "A", "B_parallel": "B", "C_parallel_fusion_samewidth": "C_same", "C_parallel_fusion_matched": "C_matched"}
PAIRS = [("B", "A", "Q1: B vs A (cost of removing same-layer Attention->MoE dependency)"),
         ("C_same", "B", "Q2: C_same vs B (effect of periodic fusion, extra compute)"),
         ("C_same", "A", "C_same vs A (NOT a compute-fair comparison)"),
         ("C_matched", "A", "Q3: C_matched vs A (compute/parameter matched)"),
         ("C_matched", "B", "C_matched vs B"),
         ("C_matched", "C_same", "C_matched vs C_same (width vs. fusion budget)")]


def _maybe(path):
    p = Path(path)
    return read_json(p) if p.exists() else None


def aggregate(root: Path) -> dict:
    runs, failed = {}, []
    for d in sorted((root / "runs").glob("*")):
        r = load_run(d)
        if r is None:
            continue
        if r["failed"]:
            failed.append(r)
            continue
        runs[SHORT.get(r["name"], r["name"])] = r
    order = ["A", "B", "C_same", "C_matched"]
    runs = {k: runs[k] for k in order if k in runs} | {k: v for k, v in runs.items() if k not in order}
    res = {"root": str(root), "n_completed": len(runs), "failed_runs": [{"dir": f["dir"], "failure": f["failure"]} for f in failed],
           "pre_registered": {"equivalence_margin_nats": EQUIV_MARGIN, "speed_margin": SPEED_MARGIN}}
    res["runs"] = {}
    for k, r in runs.items():
        s = r["summary"]
        last_router = next((t for t in reversed(r["train"]) if "per_layer_fraction" in t), {})
        res["runs"][k] = {
            "name": r["name"], "train_config": s["config"]["train"], "micro_batch_seqs": s["micro_batch_seqs"],
            "grad_accum": s["grad_accum"], "final_val_loss": s["final_val_loss"], "final_val_ppl": s["final_val_ppl"],
            "final_val_tokens": s["final_val_tokens"], "steps": s["steps"], "tokens": s["tokens_input"],
            "train_seconds": s["train_seconds"], "tokens_per_sec": s["tokens_per_sec_overall"],
            "step_time_median": s["step_time_after10"].get("median"), "step_time_p95": s["step_time_after10"].get("p95"),
            "step_time_mean": s["step_time_after10"].get("mean"), "step_time_std": s["step_time_after10"].get("std"),
            "data_wait_median": s["data_wait"].get("median"),
            "peak_mem_alloc_gb": s["peak_mem_alloc_gb"], "peak_mem_reserved_gb": s["peak_mem_reserved_gb"],
            "params_total": s["budget_analytic"]["params_total"],
            "params_active": s["budget_analytic"]["params_active_per_token"],
            "flops_train_per_token": s["budget_analytic"]["flops_train_per_token"],
            "n_moe": s["budget_analytic"]["n_moe_total"], "expert_hidden": s["budget_analytic"]["expert_hidden"],
            "router_final": {k2: last_router.get(k2) for k2 in ("util_cv_mean", "util_cv_max", "util_maxmin_ratio_max",
                                                                 "util_min_fraction", "router_entropy", "router_entropy_max",
                                                                 "per_layer_fraction")},
            "events": [e for e in _events(Path(r["dir"]))],
        }
    res["paired"] = {}
    for b, a, label in PAIRS:
        if a in runs and b in runs and runs[a]["per_seq"] is not None and runs[b]["per_seq"] is not None:
            st = paired_diff(runs[a]["per_seq"], runs[b]["per_seq"])
            st["label"] = label
            st["classification"] = classify_diff(st)
            res["paired"][f"{b}_minus_{a}"] = st
    if all(k in runs for k in ("A", "B", "C_same")):
        la, lb, lc = (res["runs"][k]["final_val_loss"] for k in ("A", "B", "C_same"))
        gap = lb - la
        bvsa = res["paired"].get("B_minus_A", {})
        meaningful = bvsa.get("ci95_normal", [0, 0])[0] > 0
        res["recovery"] = {"gap_B_minus_A": gap, "gap_significant": meaningful,
                           "recovered_by_C_same": (lb - lc) / gap if meaningful and gap != 0 else None,
                           "note": "fraction of the B-A loss gap removed by C_same; undefined unless B is significantly worse than A"}
        if "C_matched" in runs:
            lm = res["runs"]["C_matched"]["final_val_loss"]
            res["recovery"]["recovered_by_C_matched"] = (lb - lm) / gap if meaningful and gap != 0 else None
    if len(runs) >= 2:
        res["time_to_target"] = time_to_targets(runs)
        res["common_wallclock"] = loss_at_common_wallclock(runs)
        res["fairness_audit"] = fairness_audit(runs)
    res["benchmark"] = _maybe(root / "benchmark" / "benchmark.json")
    res["overlap"] = _maybe(root / "profiles" / "overlap_analysis.json")
    res["dataset"] = _maybe(root / "dataset_verification.json")
    res["environment"] = _maybe(root / "env" / "environment.json")
    res["attention_backend"] = _maybe(root / "env" / "attention_backend.json")
    res["model_budget"] = _maybe(root / "inspect" / "model_budget.json")
    res["probe"] = _maybe(root / "probe.json")
    res["smoke"] = _maybe(root / "smoke" / "smoke_report.json")
    res["tests"] = _maybe(root / "tests" / "pytest_summary.json")
    return res


def _events(d: Path):
    from moefusion.metrics import read_jsonl

    return read_jsonl(d / "events.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g. /content/moe_fusion_runs/pilot")
    ap.add_argument("--mode", default="pilot")
    args = ap.parse_args()
    root = Path(args.root)
    banner(f"AGGREGATE RESULTS ({root})")
    res = aggregate(root)
    res["mode"] = args.mode
    write_json(root / "results.json", res)
    from make_plots import make_all_plots
    from make_report import write_report

    plots = make_all_plots(root, res)
    write_report(root, res, plots)
    print(f"  results.json, {len(plots)} plots and FINAL_REPORT.md written to {root}")
    for k, v in res.get("paired", {}).items():
        print(f"  {k:24s} dL={v['mean']:+.4f}  95% CI [{v['ci95_normal'][0]:+.4f}, {v['ci95_normal'][1]:+.4f}]  -> {v['classification']}")
    fa = res.get("fairness_audit")
    if fa:
        print(f"  fairness audit: {'PASSED' if fa['passed'] else 'FAILED'}")
        for c in fa["checks"]:
            if not c["passed"]:
                print("    FAILED:", c)


if __name__ == "__main__":
    main()
