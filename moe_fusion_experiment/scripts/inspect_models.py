"""Parameter / FLOP budget table for all configurations, cross-checked against instantiated modules.
Fails if the compute-matched C deviates by more than 2% in expert parameters or active MoE FLOPs."""
import argparse
import sys
from pathlib import Path

import _common  # noqa: F401
from _common import CORE_MODELS, EXIT_GATE_FAILED, banner

import torch

from moefusion.config import assert_fair, load_experiment
from moefusion.flop_counter import analytic_budget, matched_mismatch, training_flops
from moefusion.model import MoEFusionLM, count_parameters
from moefusion.utils import fmt_num, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot")
    ap.add_argument("--models", default=",".join(CORE_MODELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tolerance", type=float, default=0.02)
    args = ap.parse_args()
    banner("MODEL INSPECTION - PARAMETER / FLOP BUDGET")
    names = args.models.split(",")
    cfgs = {n: load_experiment(n, args.run) for n in names}
    notes = assert_fair(list(cfgs.values()))
    rows, errors = {}, []
    for n, (m, t) in cfgs.items():
        a = analytic_budget(m, t.seq_len)
        with torch.device("meta"):
            measured = count_parameters(MoEFusionLM(m))
        for key, akey in (("total", "params_total"), ("attention", "params_attention"), ("router", "params_router"),
                          ("experts", "params_experts"), ("embedding", "params_embedding")):
            if measured[key] != a[akey]:
                errors.append(f"{n}: measured {key}={measured[key]} != analytic {a[akey]}")
        a["train_flops_total"] = training_flops(m, t.total_tokens, t.seq_len)
        a["tokens"] = t.total_tokens
        rows[n] = {"analytic": a, "measured": measured}
    cols = ["Model", "Layers", "Attn", "MoE", "Fusion", "d", "heads", "E", "k", "hidden", "total", "experts",
            "active/tok", "FLOPs/tok(train)", "bf16 GB"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for n, r in rows.items():
        a = r["analytic"]
        m = cfgs[n][0]
        lines.append("| " + " | ".join(str(x) for x in [
            n, m.n_layers, a["n_attention_layers"], a["n_moe_backbone"], a["n_moe_fusion"], m.d_model, m.n_heads,
            m.n_experts, m.top_k, m.expert_hidden, fmt_num(a["params_total"]), fmt_num(a["params_experts"]),
            fmt_num(a["params_active_per_token"]), fmt_num(a["flops_train_per_token"]), f"{a['bf16_param_memory_gb']:.2f}"]) + " |")
    table = "\n".join(lines)
    print(table)
    mm = {}
    for n, (m, t) in cfgs.items():
        if m.matched_reference_hidden:
            base = next((cfgs[b][0] for b in names if cfgs[b][0].arch == "serial"
                         and cfgs[b][0].expert_hidden == m.matched_reference_hidden), None)
            if base is None:
                continue
            mm[n] = matched_mismatch(base, m)
            print(f"\n  compute matching {n} vs {base.name}:")
            for k, v in mm[n].items():
                print(f"     {k:28s} {100 * v:+.4f}%")
            for k in ("expert_params_rel_diff", "active_moe_flops_rel_diff"):
                if abs(mm[n][k]) > args.tolerance:
                    errors.append(f"WARNING-FAIL: {n} {k} = {100 * mm[n][k]:.2f}% exceeds {100 * args.tolerance:.0f}%")
    print("\n  allowed (architecture-defining) differences:")
    for x in notes:
        print("    -", x)
    out = Path(args.out) / "inspect"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "model_budget.json", {"rows": rows, "matched": mm, "fairness_notes": notes, "errors": errors})
    (out / "model_budget.md").write_text(table + "\n")
    if errors:
        for e in errors:
            print("  ERROR:", e)
        sys.exit(EXIT_GATE_FAILED)
    print("  analytic counts == instantiated module counts for every model: OK")


if __name__ == "__main__":
    main()
