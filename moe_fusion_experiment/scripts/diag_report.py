"""Apply the pre-registered diagnostic rules R1-R4 (EXPERIMENT_SPEC Amendment 3) and write DIAG_REPORT.md."""
import argparse
import shutil
from pathlib import Path

import _common  # noqa: F401
from _common import banner

import numpy as np

from moefusion.metrics import read_jsonl
from moefusion.utils import now_iso, read_json, write_json

GROUPS = {"seed42": "diag_seed42", "dense": "diag_dense", "warmup10": "diag_warmup10", "seed43": "diag_seed43"}
MARGIN = 0.02


def gap(out: Path, tag: str):
    """paired B - A on the full validation set of one diagnostic group."""
    p = out / tag / "results.json"
    if not p.exists():
        return None
    st = (read_json(p).get("paired") or {}).get("B_minus_A")
    return None if st is None else {"d": st["mean"], "ci": st["ci95_normal"]}


def churn(out: Path, tag: str, model_dir: str, early_frac: float = 0.25):
    """mean top-1 routing churn over the first quarter of training and over the whole run."""
    runs = sorted((out / tag / "runs").glob(f"{model_dir}_seed*"))
    if not runs:
        return None
    ev = [e for e in read_jsonl(runs[0] / "eval_metrics.jsonl") if not e.get("final") and "routing_churn_top1_mean" in e]
    if not ev:
        return None
    last = max(e["step"] for e in ev)
    early = [e["routing_churn_top1_mean"] for e in ev if e["step"] <= early_frac * last] or [ev[0]["routing_churn_top1_mean"]]
    return {"early": float(np.mean(early)), "all": float(np.mean([e["routing_churn_top1_mean"] for e in ev])),
            "per_eval": [(e["step"], round(e["routing_churn_top1_mean"], 4)) for e in ev]}


def dead_intervals(out: Path, tag: str, model_dir: str):
    runs = sorted((out / tag / "runs").glob(f"{model_dir}_seed*"))
    if not runs:
        return None
    return [e["step"] for e in read_jsonl(runs[0] / "events.jsonl") if e["event"] == "dead_expert_alert"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/moe_fusion_runs")
    ap.add_argument("--drive-dir", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    banner("DIAGNOSTIC REPORT (pre-registered rules R1-R4)")
    g = {k: gap(out, t) for k, t in GROUPS.items()}
    c = {"A": churn(out, GROUPS["seed42"], "A_serial"), "B": churn(out, GROUPS["seed42"], "B_parallel"),
         "A_s43": churn(out, GROUPS["seed43"], "A_serial"), "B_s43": churn(out, GROUPS["seed43"], "B_parallel")}
    dead = {k: dead_intervals(out, GROUPS["seed42"], m) for k, m in (("A", "A_serial"), ("B", "B_parallel"))}
    rules = {}
    base = g["seed42"]
    if base:
        d0 = base["d"]
        # R1 seed robustness
        s43 = g["seed43"]
        if s43:
            robust = np.sign(s43["d"]) == np.sign(d0) and abs(s43["d"]) >= MARGIN and abs(d0) >= MARGIN
            rules["R1_seed"] = {"seed42": d0, "seed43": s43["d"], "result": "ROBUST across seeds" if robust else "NOT robust"}
        # R2 MoE-specific?
        dn = g["dense"]
        if dn:
            specific = (np.sign(dn["d"]) != np.sign(d0)) or abs(dn["d"]) < 0.5 * abs(d0)
            rules["R2_dense"] = {"moe": d0, "dense": dn["d"],
                                 "result": "MoE-SPECIFIC (dense gap < half of the MoE gap or opposite sign)" if specific
                                 else "NOT MoE-specific (dense shows at least half of the MoE gap)"}
        # R3 warm-up
        w = g["warmup10"]
        if w:
            explained = abs(w["d"]) <= 0.5 * abs(d0)
            rules["R3_warmup"] = {"warmup2pct": d0, "warmup10pct": w["d"],
                                  "result": "LARGELY EXPLAINED by early training / warm-up (gap at least halves)" if explained
                                  else "NOT explained by warm-up"}
    if c["A"] and c["B"]:
        ratio = c["A"]["early"] / max(c["B"]["early"], 1e-9)
        rules["R4_routing"] = {"A_early_churn": c["A"]["early"], "B_early_churn": c["B"]["early"], "ratio": ratio,
                               "result": "SUPPORTS routing fluctuation (A >= 1.5x B)" if ratio >= 1.5 else
                               ("does NOT support routing fluctuation (A <= 1.1x B)" if ratio <= 1.1 else "weak / inconclusive")}
    # overall reading (pre-registered mapping)
    r = {k: v["result"] for k, v in rules.items()}
    if not r.get("R1_seed", "").startswith("ROBUST"):
        overall = ("The B>A gap is NOT robust: it does not reach the 0.02-nat margin with the same sign in both seeds. "
                   "Treat the pilot gap as not established (seed noise or a small effect); do not build on it.")
    elif r.get("R3_warmup", "").startswith("LARGELY"):
        overall = ("The B>A gap is largely a warm-up / early-training artefact of the shared recipe: compare architectures "
                   "with a longer warm-up (or per-architecture tuned warm-up) before drawing conclusions.")
    elif r.get("R2_dense", "").startswith("MoE-SPECIFIC") and r.get("R4_routing", "").startswith("SUPPORTS"):
        overall = ("REAL, MoE-SPECIFIC effect consistent with routing stability: in this regime the parallel block keeps the "
                   "router more stable than the serial block (StableMoE-type routing fluctuation in A).")
    elif r.get("R2_dense", "").startswith("MoE-SPECIFIC"):
        overall = "Real and MoE-specific, but the routing-churn measurement does not confirm the routing-stability mechanism."
    else:
        overall = ("Robust but NOT MoE-specific: the parallel advantage also appears with a dense FFN, i.e. it is a property of "
                   "this training regime (small, heavily under-trained, one shared recipe) rather than of MoE routing.")
    summary = {"time": now_iso(), "gaps_B_minus_A": g, "routing_churn": c, "dead_expert_alert_steps": dead,
               "rules": rules, "overall": overall, "margin": MARGIN}
    write_json(out / "DIAG_SUMMARY.json", summary)
    L = ["# DIAGNOSTIC REPORT - why did B (parallel) beat A (serial)?\n",
         "_Pre-registered rules: EXPERIMENT_SPEC.md Amendment 3. Pilot budget (25M tokens) per run._\n",
         "## Measured B - A (paired, full validation set; negative = B better)\n",
         "| group | B - A (nats) | 95% CI |\n|---|---|---|"]
    for k, v in g.items():
        L.append(f"| {k} | " + ("n/a | n/a |" if v is None else f"{v['d']:+.4f} | [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] |"))
    L.append("\n## Routing stability (top-1 expert churn of a fixed validation probe between evaluations)\n")
    for k, v in c.items():
        if v:
            L.append(f"* {k}: first quarter {v['early']:.4f}, whole run {v['all']:.4f}")
    L.append(f"* dead-expert alerts (seed 42): A at steps {dead.get('A')}, B at steps {dead.get('B')}\n")
    L.append("## Rules\n")
    for k, v in rules.items():
        L.append(f"* **{k}**: {v['result']}  `{ {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != 'result'} }`")
    L.append(f"\n## Overall reading\n\n**{overall}**\n")
    L.append("Single diagnostic runs per condition: these rules decide what to test next; they are not a replacement for "
             "replication.\n")
    (out / "DIAG_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    if args.drive_dir:
        try:
            dst = Path(args.drive_dir) / "runs"
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out / "DIAG_REPORT.md", dst / "DIAG_REPORT.md")
            shutil.copy2(out / "DIAG_SUMMARY.json", dst / "DIAG_SUMMARY.json")
        except Exception as e:  # noqa: BLE001
            print("  [drive] could not copy the diagnostic report:", e)


if __name__ == "__main__":
    main()
