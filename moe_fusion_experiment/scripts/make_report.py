"""FINAL_REPORT.md generator. Every number is read from results.json; interpretation rules are the
pre-registered ones in EXPERIMENT_SPEC.md. MEASURED / INTERPRETATION / SPECULATION are kept separate."""
import argparse
from pathlib import Path

import _common  # noqa: F401

from moefusion.analysis import EQUIV_MARGIN, SPEED_MARGIN
from moefusion.utils import fmt_num, now_iso, read_json


def _f(x, nd=4, sign=False):
    if x is None:
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _ci(st):
    lo, hi = st["ci95_normal"]
    return f"{st['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def _speed(R, a, b):
    """relative step-time difference of b vs a (median, steady state)."""
    if a not in R or b not in R or not R[a]["step_time_median"]:
        return None
    return R[b]["step_time_median"] / R[a]["step_time_median"] - 1.0


def write_report(root: Path, res: dict, plots: dict) -> Path:
    R = res.get("runs", {})
    P = res.get("paired", {})
    mode = res.get("mode", "pilot")
    L = []
    w = L.append
    w(f"# FINAL REPORT - Serial vs Parallel vs Parallel+Fusion MoE ({mode.upper()})\n")
    w(f"_Generated automatically {now_iso()} from experiment logs in `{root}`. No number in this file is hand-written._\n")
    if mode != "main":
        w("> **STATUS: PILOT.** One paired seed, ~25M tokens per model. The pre-registered purpose of the pilot is to detect "
          "catastrophic bugs and obtain a first, NON-CONFIRMATORY comparison. Confidence intervals below capture "
          "validation-data noise only, not seed-to-seed training variance. No claim in this report is confirmatory.\n")
    if res.get("failed_runs"):
        w("> **FAILED RUNS:** " + "; ".join(f"`{f['dir']}`: {(f['failure'] or {}).get('reason')}" for f in res["failed_runs"]) + "\n")

    w("## A. Experiment question\n")
    w("Can the strict Attention -> MoE dependency be removed from most Transformer layers (Attention and MoE as parallel "
      "branches) and the lost interaction recovered by periodic Fusion MoE layers, giving a better quality-versus-wall-clock "
      "trade-off?\n")
    w("## B. Architecture definitions\n")
    w("* **A (serial):** `a = x + Attn(RMSNorm(x)); y = a + MoE(RMSNorm(a))`\n"
      "* **B (parallel):** `y = x + Attn(RMSNorm_a(x)) + MoE(RMSNorm_m(x))` - identical parameter set to A (bit-identical init).\n"
      "* **C (parallel + fusion):** B blocks, plus `x = x + FusionMoE(RMSNorm_f(x))` after blocks 4, 8 and 12 "
      "(12 attention + 15 MoE applications). `C_same`: expert hidden 1920 (extra compute, mechanism test). "
      "`C_matched`: expert hidden 1536 so that 15 x 1536 = 12 x 1920 (compute/parameter matched).\n"
      "* Shared: 12 layers, d_model 768, 12 heads x 64, RoPE, RMSNorm, 8 SwiGLU experts top-2 (no token dropping), "
      "tied embeddings, vocab 32,000, context 1024, BF16 autocast.\n")

    ds = res.get("dataset") or {}
    w("## C. Dataset\n")
    if ds:
        arr = ds.get("arrays", {})
        w(f"* Source: FineWeb-Edu sample-10BT (pinned revision, first 300k documents, document-level split).\n"
          f"* train.npy {arr.get('train', {}).get('shape')} / validation.npy {arr.get('validation', {}).get('shape')} "
          f"({arr.get('train', {}).get('row_layout')}).\n"
          f"* Leakage check: {ds.get('separation')}\n"
          f"* Re-tokenisation provenance check: " +
          ", ".join(f"{k}: {v.get('status')} ({v.get('packing_mode')})" for k, v in (ds.get('tokenization_spot_check') or {}).items()) + "\n")
        if ds.get("rebuild"):
            rb = ds["rebuild"]
            w(f"* Data was rebuilt from the pinned source by `prepare_fineweb.py`: byte-identical to the original "
              f"preparation = **{rb.get('identical_to_original')}**; inferred conventions {rb.get('conventions')}.\n")
        if ds.get("warnings"):
            w("* Dataset warnings: " + "; ".join(ds["warnings"]) + "\n")
    w("## D. Tokenizer\n")
    if ds.get("tokenizer"):
        t = ds["tokenizer"]
        w(f"One ByteLevel-BPE artifact for every model: vocab {t['vocab_size']}, special tokens {t['special_tokens']}, "
          f"sha256 `{t['sha256']}`.\n")
    w("## E. Training setup (identical for every architecture)\n")
    if R:
        r0 = next(iter(res["runs"].values()))
        tc = r0.get("train_config") or {}
        w(f"* Steps {r0['steps']}, tokens {r0['tokens']:,}, global batch {tc.get('global_batch_seqs')} x {tc.get('seq_len')} tokens, "
          f"micro-batch {r0.get('micro_batch_seqs')} x grad-accum {r0.get('grad_accum')} (common to all, from the memory probe), "
          f"AdamW({tc.get('beta1')}, {tc.get('beta2')}, wd {tc.get('weight_decay')}), peak LR {tc.get('lr')}, "
          f"{100 * (tc.get('warmup_frac') or 0):.0f}% linear warm-up, cosine to {tc.get('min_lr_ratio')} x peak, clip {tc.get('grad_clip')}, "
          f"balance-loss coefficient {tc.get('balance_coef')} (mean over MoE applications), z-loss coefficient {tc.get('zloss_coef')} "
          "(always logged), BF16 autocast with FP32 master weights / router / norms / loss.\n"
          "* Same data order (BatchSchedule seeded by the paired seed), same evaluation points, same validation rows; "
          "evaluation and checkpoint time excluded from the wall-clock axis.\n")
    fa = res.get("fairness_audit")
    if fa:
        w(f"* **Automated fairness audit: {'PASSED' if fa['passed'] else 'FAILED'}**\n")
        w("| check | result | detail |\n|---|---|---|")
        for c in fa["checks"]:
            w(f"| {c['check']} | {'PASS' if c['passed'] else '**FAIL**'} | {c['detail']} |")
        w("")
    env = res.get("environment") or {}
    w("## F. Hardware / software environment\n")
    w(f"GPU {env.get('gpu_name')} ({env.get('gpu_total_memory_gb')} GB, cc {env.get('gpu_compute_capability')}), "
      f"driver/nvidia-smi `{env.get('nvidia_smi_query')}`, PyTorch {env.get('torch')}, CUDA {env.get('torch_cuda_version')}, "
      f"Python {env.get('python')}. Attention backend: **{(res.get('attention_backend') or {}).get('selected')}** "
      f"(fallback used: {(res.get('attention_backend') or {}).get('fallback_used')}). MoE backend: reference. "
      "Primary comparison parallel execution: reference (sequential dispatch) for every architecture.\n")
    if env.get("warnings"):
        w("Environment warnings: " + "; ".join(env["warnings"]) + "\n")

    mb = res.get("model_budget") or {}
    w("## G/H. Parameter and FLOP budget\n")
    if mb.get("rows"):
        w("| model | MoE apps | expert hidden | total params | expert params | active params/token | train FLOPs/token |\n|---|---|---|---|---|---|---|")
        for n, r in mb["rows"].items():
            a = r["analytic"]
            w(f"| {n} | {a['n_moe_total']} | {a['expert_hidden']} | {fmt_num(a['params_total'])} | {fmt_num(a['params_experts'])} | "
              f"{fmt_num(a['params_active_per_token'])} | {fmt_num(a['flops_train_per_token'])} |")
        for n, m in (mb.get("matched") or {}).items():
            w(f"\nCompute matching `{n}` vs A: expert params {100 * m['expert_params_rel_diff']:+.3f}%, active MoE FLOPs "
              f"{100 * m['active_moe_flops_rel_diff']:+.3f}%, total params {100 * m['total_params_rel_diff']:+.3f}%, "
              f"train FLOPs/token {100 * m['train_flops_rel_diff']:+.3f}% (FLOP convention: see flop_counter.py).")
        w("")

    w("## I. Validation results (MEASURED)\n")
    if R:
        w("| model | final val LM loss (full val set) | perplexity | val tokens |\n|---|---|---|---|")
        for k, r in R.items():
            w(f"| {k} | {r['final_val_loss']:.4f} | {r['final_val_ppl']:.2f} | {r['final_val_tokens']:,} |")
        w("\nPaired differences (row model minus column model, same validation sequences; 95% CI):\n")
        w("| comparison | dL (nats) [95% CI] | bootstrap CI | seqs where first is better | classification |\n|---|---|---|---|---|")
        for k, st in P.items():
            w(f"| {st['label']} | {_ci(st)} | [{st['ci95_bootstrap'][0]:+.4f}, {st['ci95_bootstrap'][1]:+.4f}] | "
              f"{100 * st['frac_sequences_b_better']:.1f}% | {st['classification']} |")
        w("")
    w("## J/K. Throughput and wall-clock (MEASURED)\n")
    if R:
        w("| model | training wall-clock (min) | tokens/s | median step (ms) | p95 step (ms) | median data wait (ms) |\n|---|---|---|---|---|---|")
        for k, r in R.items():
            w(f"| {k} | {r['train_seconds'] / 60:.2f} | {r['tokens_per_sec']:,.0f} | {1e3 * r['step_time_median']:.1f} | "
              f"{1e3 * r['step_time_p95']:.1f} | {1e3 * (r['data_wait_median'] or 0):.2f} |")
        w("")
    tt = res.get("time_to_target")
    if tt:
        w(f"**Time-to-target** ({tt['rule']}; periodic-subset curve, linear interpolation):\n")
        ks = list(R)
        w("| target loss | " + " | ".join(f"{k} (min / M tokens)" for k in ks) + " |\n|---|" + "---|" * len(ks))
        for tgt, d in tt["targets"].items():
            cells = []
            for k in ks:
                v = d.get(k, {})
                cells.append("not reached" if v.get("train_seconds") is None else
                             f"{v['train_seconds'] / 60:.2f} / {v['tokens'] / 1e6:.1f}")
            w(f"| {tgt} | " + " | ".join(cells) + " |")
        w("")
    cw = res.get("common_wallclock")
    if cw:
        w(f"**Loss at common wall-clock** T* = {cw['t_star_seconds'] / 60:.2f} min ({cw['definition']}): " +
          ", ".join(f"{k} {_f(v)}" for k, v in cw["loss"].items()) + "\n")
    b = res.get("benchmark")
    if b:
        w(f"**Systems benchmark** (separate from training; one micro-batch of {b['micro_batch']} sequences, untrained weights, "
          f"{b['rounds']} interleaved rotated rounds):\n")
        w("| variant | forward ms | fwd+bwd ms | optimizer step ms | tokens/s (step) | peak GB |\n|---|---|---|---|---|---|")
        for k, v in b["full_model"].items():
            w(f"| {k} | {v['forward']['median_of_round_medians_ms']:.1f} | {v['forward_backward']['median_of_round_medians_ms']:.1f} | "
              f"{v['optimizer_step']['median_of_round_medians_ms']:.1f} | {v['optimizer_step']['tokens_per_sec']:,.0f} | {v['peak_mem_alloc_gb']:.1f} |")
        w("")
    w("## L. VRAM (MEASURED)\n")
    if R:
        w(", ".join(f"{k}: {_f(r['peak_mem_alloc_gb'], 1)} GB allocated / {_f(r['peak_mem_reserved_gb'], 1)} GB reserved" for k, r in R.items()) + "\n")
    w("## M. Router / expert behaviour (MEASURED)\n")
    if R:
        w("| model | utilisation CV (mean over layers) | worst max/min ratio | min expert fraction | router entropy (max ln8=2.079) | alerts |\n|---|---|---|---|---|---|")
        for k, r in R.items():
            rf = r["router_final"]
            alerts = [e["event"] for e in r["events"] if e["event"] != "checkpoint_saved"]
            w(f"| {k} | {_f(rf.get('util_cv_mean'), 3)} | {_f(rf.get('util_maxmin_ratio_max'), 2)} | {_f(rf.get('util_min_fraction'), 4)} | "
              f"{_f(rf.get('router_entropy'), 3)} | {', '.join(sorted(set(alerts))) or 'none'} |")
        w("")
    w("## N. CUDA concurrency / profiling evidence (MEASURED)\n")
    if b and b.get("branches"):
        for name, br in b["branches"].items():
            for m_ in ("forward", "forward_backward"):
                a = br[m_]
                an = a["analysis"]
                w(f"* `{name}` {m_}: attention {a['attention_branch']['median_ms']:.2f} ms, MoE {a['moe_branch']['median_ms']:.2f} ms, "
                  f"sum {an['sum_of_branches_ms']:.2f} ms, reference {a['parallel_reference']['median_ms']:.2f} ms, concurrent "
                  f"{a['parallel_concurrent']['median_ms']:.2f} ms -> speed-up {an['concurrent_vs_reference_speedup']:.3f}x, "
                  f"saved {an['time_saved_vs_sum_ms']:.2f} ms = {100 * an['overlap_fraction_of_shorter_branch']:.1f}% of the shorter branch. "
                  f"**{an['interpretation']}**")
        eq = b.get("concurrent_equivalence_full_size", {})
        w(f"* Full-size reference vs concurrent equivalence: forward max |diff| {eq.get('forward_max_abs_diff')}, worst gradient "
          f"relative L2 diff {eq.get('grad_worst_relative_l2_diff')} -> {'PASS' if eq.get('passed') else 'FAIL'}.")
    ov = res.get("overlap")
    if ov:
        for k, a in ov.items():
            if "error" in a:
                w(f"* trace `{k}`: {a['error']}")
                continue
            fr, sb = a["forward_range_based"], a["stream_based_fwd_bwd"]
            w(f"* trace `{k}`: {a['n_kernels']} kernels; forward attention/MoE kernel overlap {fr['overlap_us']:.0f} us "
              f"({100 * fr['overlap_fraction_of_shorter']:.1f}% of the shorter branch); whole-step stream overlap: " +
              (f"{sb['overlap_us']:.0f} us ({100 * sb['overlap_fraction_of_shorter']:.1f}%)" if sb.get("applicable") else f"n/a ({sb['reason']})"))
    w("\nTraces: `profiles/*.json` (open in https://ui.perfetto.dev). Method and caveats: `src/moefusion/trace_analysis.py`.\n")

    w("## O. Failures or anomalies\n")
    anomalies = [f"{k}: {e}" for k, r in R.items() for e in r["events"] if e["event"] != "checkpoint_saved"]
    anomalies += [f"failed run {f['dir']}" for f in res.get("failed_runs", [])]
    anomalies += (env.get("warnings") or []) + (ds.get("warnings") or [])
    w("\n".join(f"* {a}" for a in anomalies) if anomalies else "None recorded.")
    w("\n## P. Statistical uncertainty\n")
    w("Single paired seed. Paired CIs over validation sequences quantify evaluation noise only. Seed-to-seed variance "
      "of small LMs is typically of the same order as the effects of interest, so differences below the pre-registered "
      f"margin ({EQUIV_MARGIN} nats) - and any difference before the 3-seed replication - must be treated as provisional.\n")

    # ---------------------------------------------------------------- answers
    w("## Q. Answers to the pre-registered questions\n")
    w("_MEASURED values are followed by the INTERPRETATION that the pre-registered rules allow._\n")
    def pst(key):
        return P.get(key)
    q1 = pst("B_minus_A")
    w(f"**Q1 - quality lost from A to B:** " + (f"MEASURED dL(B-A) = {_ci(q1)} nats. INTERPRETATION: {q1['classification']}." if q1 else "n/a"))
    rec = res.get("recovery") or {}
    q2 = pst("C_same_minus_B")
    w(f"\n**Q2 - recovered by C:** " + (f"MEASURED dL(C_same-B) = {_ci(q2)}; recovered fraction of the B-A gap: "
      f"{_f(rec.get('recovered_by_C_same'), 2)} (C_same), {_f(rec.get('recovered_by_C_matched'), 2)} (C_matched). "
      f"INTERPRETATION: fusion effect {q2['classification']}." + ("" if rec.get("gap_significant") else
      " The recovered fraction is undefined because B is not significantly worse than A.") if q2 else "n/a"))
    q3 = pst("C_matched_minus_A")
    w(f"\n**Q3 - does C help when compute is matched:** " + (f"MEASURED dL(C_matched-A) = {_ci(q3)}, dL(C_matched-B) = "
      f"{_ci(pst('C_matched_minus_B')) if pst('C_matched_minus_B') else 'n/a'}. INTERPRETATION vs A: {q3['classification']}." if q3 else "n/a"))
    sB = _speed(R, "A", "B")
    w(f"\n**Q4 - A100 wall-clock speed-up of the parallel structure:** MEASURED training step time B vs A: "
      f"{_f(None if sB is None else 100 * sB, 2, True)}% (both use the same sequential dispatch)." +
      (f" Benchmark: B[concurrent] vs B[reference] optimizer step ratio "
       f"{b['full_model']['B_parallel[concurrent]']['optimizer_step']['median_of_round_medians_ms'] / b['full_model']['B_parallel[reference]']['optimizer_step']['median_of_round_medians_ms']:.3f}."
       if b else "") +
      f" INTERPRETATION: differences below {100 * SPEED_MARGIN:.0f}% are treated as no speed difference. A mathematically parallel "
      "block yields hardware speed-up only through kernel concurrency or fused projections (see Q5/Q6); with sequential "
      "dispatch it performs the same work as A.")
    if b and b.get("branches"):
        k0 = next(iter(b["branches"]))
        an = b["branches"][k0]["forward_backward"]["analysis"]
        w(f"\n**Q5 - do attention and MoE kernels overlap:** MEASURED (block {k0}, fwd+bwd): {an['interpretation']} "
          "Trace-level overlap numbers are in section N.")
        w(f"\n**Q6 - fraction of theoretical parallelism realised:** MEASURED time saved / shorter-branch time = "
          f"{100 * an['overlap_fraction_of_shorter_branch']:.1f}% (forward+backward, one block; 100% = perfect overlap, "
          "<=0% = none).")
    else:
        w("\n**Q5/Q6:** benchmark not available.")
    w("\n**Q7 - time-to-target:** see the table in section J/K.")
    if R:
        best_tok = min(R, key=lambda k: R[k]["final_val_loss"])
        w(f"\n**Q8 - best validation loss per token (equal tokens for all):** MEASURED lowest final loss: **{best_tok}** "
          f"({R[best_tok]['final_val_loss']:.4f}). Whether it is distinguishable from the others: see the paired table.")
    if cw:
        valid = {k: v for k, v in cw["loss"].items() if v is not None}
        if valid:
            bw = min(valid, key=valid.get)
            w(f"\n**Q9 - best validation loss per wall-clock second:** MEASURED lowest loss at common wall-clock T*: **{bw}** "
              f"({valid[bw]:.4f}). Note that C_same also spends more compute per token.")
    q10 = pst("C_same_minus_B")
    if q10:
        promising = q10["ci95_normal"][1] < 0
        w(f"\n**Q10 - is fusion_interval=4 promising enough for 2/8/final-only ablations?** Pre-registered rule: yes if dL(C_same-B) "
          f"has a 95% CI entirely below 0. MEASURED: {_ci(q10)} -> **{'YES (provisional)' if promising else 'NOT by the pre-registered rule'}**"
          + (" - pilot only; confirm in MAIN before spending ablation compute." if mode != "main" else ""))
    w("\n## R. Limitations\n")
    w("* Small scale (12 layers, d 768, ~160M active / ~480-590M total params, 25M-100M tokens): evidence is small-scale "
      "architectural evidence, not a claim about billion/trillion-parameter LLMs.\n"
      "* One seed in the pilot; CI excludes training-seed variance.\n"
      "* Hyperparameters were NOT tuned per architecture (by design); a recipe tuned for A could favour A.\n"
      "* Wall-clock depends on this implementation (reference MoE with one host sync per MoE layer, SDPA attention) and "
      "on Colab conditions; runs are sequential in one session, the systems benchmark interleaves models to control drift.\n"
      "* The periodic validation curve uses a fixed 1,024-sequence subset; final numbers use the full validation set.\n")
    w("## S. Recommended next experiment\n")
    w("* If the fairness audit passed and no run failed: run MAIN (100M tokens) for the same four models, then the 3-seed "
      "replication (seeds 42/43/44) before any strong conclusion.\n"
      "* Run the fusion-interval ablation (2 / 8 / final-only, compute-matched widths from `matched_hidden`) only if Q10 "
      "is positive in MAIN.\n"
      "* Systems track (separate label): fused Attention/MoE input projections and concurrent execution in training.\n")
    w("## Plots\n")
    for k, p in plots.items():
        w(f"![{k}](plots/{Path(p).name})")
    w("\n---\n**SPECULATION** is intentionally absent from the measured sections above; any hypothesis about *why* a pattern "
      "occurs must be tested in a follow-up experiment.\n")
    path = Path(root) / "FINAL_REPORT.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    res = read_json(Path(a.root) / "results.json")
    print(write_report(Path(a.root), res, {}))
