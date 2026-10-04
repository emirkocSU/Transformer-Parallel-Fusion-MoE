"""Publication-quality plots read from experiment logs (never hardcoded).

Colour = model identity, fixed order (validated categorical palette, light mode):
A blue, B orange, C_same aqua, C_matched yellow; line styles are a secondary encoding so identity
never relies on colour alone. Thin 2px lines, >=8px markers, recessive grid, legend + direct labels.
"""
import argparse
from pathlib import Path

import _common  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from moefusion.analysis import load_run  # noqa: E402
from moefusion.utils import read_json  # noqa: E402

ORDER = ["A", "B", "C_same", "C_matched"]
COLOR = {"A": "#2a78d6", "B": "#eb6834", "C_same": "#1baf7a", "C_matched": "#eda100"}
STYLE = {"A": "-", "B": "--", "C_same": "-.", "C_matched": ":"}
MARK = {"A": "o", "B": "s", "C_same": "^", "C_matched": "D"}
LABEL = {"A": "A serial", "B": "B parallel", "C_same": "C parallel+fusion (same width)",
         "C_matched": "C parallel+fusion (matched)"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SHORT = {"A_serial": "A", "B_parallel": "B", "C_parallel_fusion_samewidth": "C_same", "C_parallel_fusion_matched": "C_matched"}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5, "axes.titlesize": 12,
        "axes.titleweight": "bold", "legend.frameon": False, "figure.dpi": 150,
    })


def _runs(root: Path):
    """model -> run; with several seeds the curves are the seed MEAN, and '_band' holds the per-step min/max."""
    groups = {}
    for d in sorted((root / "runs").glob("*")):
        r = load_run(d)
        if r and not r["failed"]:
            groups.setdefault(SHORT.get(r["name"], r["name"]), []).append(r)
    out = {}
    for k, rs in groups.items():
        if len(rs) == 1:
            out[k] = rs[0]
            continue
        n = min(len(r["periodic"]) for r in rs)
        per = []
        for i in range(n):
            es = [r["periodic"][i] for r in rs]
            per.append({"step": es[0]["step"], "tokens": es[0]["tokens"],
                        "train_seconds": float(np.mean([e["train_seconds"] for e in es])),
                        "val_loss": float(np.mean([e["val_loss"] for e in es])),
                        "_lo": min(e["val_loss"] for e in es), "_hi": max(e["val_loss"] for e in es)})
        m = min(len(r["train"]) for r in rs)
        train = [{"tokens": rs[0]["train"][i]["tokens"], "lm_loss": float(np.mean([r["train"][i]["lm_loss"] for r in rs]))}
                 for i in range(m)]
        out[k] = {"name": rs[0]["name"], "periodic": per, "train": train, "n_seeds": len(rs)}
    return {k: out[k] for k in ORDER if k in out} | {k: v for k, v in out.items() if k not in ORDER}


def _curve_plot(runs, xkey, ykey, xlabel, ylabel, title, path, xscale=1.0, transform=None, subtitle=None):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ends = []
    for k, r in runs.items():
        xs = np.array([e[xkey] for e in r["periodic"]]) / xscale
        ys = np.array([e["val_loss"] for e in r["periodic"]])
        if transform:
            ys = transform(ys)
        # skip the step-0 point for readability of the converged region (it is ~ln V for all models)
        sl = slice(1, None)
        if "_lo" in r["periodic"][-1]:  # several seeds: min-max band around the seed mean
            lo = np.array([e["_lo"] for e in r["periodic"]])
            hi = np.array([e["_hi"] for e in r["periodic"]])
            if transform:
                lo, hi = transform(lo), transform(hi)
            ax.fill_between(xs[sl], lo[sl], hi[sl], color=COLOR.get(k, INK2), alpha=0.18, linewidth=0)
        ax.plot(xs[sl], ys[sl], STYLE.get(k, "-"), color=COLOR.get(k, INK2), lw=2, marker=MARK.get(k, "o"),
                ms=4.5, markeredgecolor=SURFACE, markeredgewidth=1.0, label=LABEL.get(k, k))
        ends.append((xs[-1], ys[-1], k))
    # direct labels at line ends; only labels whose end points are close in x are nudged apart vertically
    ends.sort(key=lambda t: t[1])
    y0, y1 = ax.get_ylim()
    x0, x1 = ax.get_xlim()
    gap, placed = 0.045 * (y1 - y0), []
    for x, y, k in ends:
        yy = y
        for px, py in placed:
            if abs(px - x) < 0.08 * (x1 - x0) and abs(py - yy) < gap:
                yy = py + gap
        placed.append((x, yy))
        ax.annotate(k, xy=(x, y), xytext=(x + 0.012 * (x1 - x0), yy), textcoords="data", va="center",
                    fontsize=9, color=INK2, annotation_clip=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=18 if subtitle else 6)
    if subtitle:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _bars(values: dict, ylabel, title, path, err=None, fmt="{:.0f}", note=None):
    values = {k: v for k, v in values.items() if v is not None}
    if not values:
        return None
    keys = list(values)
    fig, ax = plt.subplots(figsize=(max(5.5, 1.35 * len(keys) + 2), 4.2))
    xs = np.arange(len(keys))
    cols = [COLOR.get(k.split("[")[0], "#8a8984") for k in keys]
    ax.bar(xs, [values[k] for k in keys], color=cols, width=0.62, edgecolor=SURFACE, linewidth=2, zorder=3)
    if err:
        for i, k in enumerate(keys):
            if err.get(k) is not None:
                ax.plot([i, i], [values[k], err[k]], color=INK2, lw=1.2, zorder=4)
                ax.plot([i - 0.08, i + 0.08], [err[k], err[k]], color=INK2, lw=1.2, zorder=4)
    for i, k in enumerate(keys):
        top = max(values[k], (err or {}).get(k) or values[k])
        ax.annotate(fmt.format(values[k]), (i, top), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(xs, [k.replace("[", "\n[") for k in keys], fontsize=8.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="x", visible=False)
    if note:
        ax.text(0, -0.28 if any("[" in k for k in keys) else -0.2, note, transform=ax.transAxes, fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def make_all_plots(root: Path, res: dict):
    _style()
    root = Path(root)
    pdir = root / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    runs = _runs(root)
    plots = {}
    if not runs:
        return plots
    sub = "validation LM loss on the fixed periodic subset; identical sequences for every model"
    plots["loss_vs_tokens"] = _curve_plot(runs, "tokens", "val_loss", "training tokens (millions)", "validation LM loss (nats)",
                                          "Validation loss vs training tokens", pdir / "loss_vs_tokens.png", 1e6, subtitle=sub)
    plots["loss_vs_wallclock"] = _curve_plot(runs, "train_seconds", "val_loss", "training wall-clock (minutes, eval excluded)",
                                             "validation LM loss (nats)", "Validation loss vs wall-clock (primary)",
                                             pdir / "loss_vs_wallclock.png", 60.0, subtitle=sub)
    plots["ppl_vs_tokens"] = _curve_plot(runs, "tokens", "val_loss", "training tokens (millions)", "validation perplexity",
                                         "Perplexity vs training tokens", pdir / "ppl_vs_tokens.png", 1e6, np.exp, subtitle=sub)
    plots["ppl_vs_wallclock"] = _curve_plot(runs, "train_seconds", "val_loss", "training wall-clock (minutes, eval excluded)",
                                            "validation perplexity", "Perplexity vs wall-clock", pdir / "ppl_vs_wallclock.png",
                                            60.0, np.exp, subtitle=sub)
    R = res.get("runs", {})
    plots["throughput_comparison"] = _bars({k: R[k]["tokens_per_sec"] for k in R}, "tokens / second",
                                           "End-to-end training throughput", pdir / "throughput_comparison.png",
                                           fmt="{:,.0f}", note="total training tokens / training wall-clock (sync-to-sync steps)")
    plots["step_time_comparison"] = _bars({k: 1e3 * R[k]["step_time_median"] for k in R}, "ms per optimizer step",
                                          "Training step time (median; whisker = p95)", pdir / "step_time_comparison.png",
                                          err={k: 1e3 * R[k]["step_time_p95"] for k in R}, fmt="{:.0f}",
                                          note="steps after the first 10; global batch 65,536 tokens")
    plots["vram_comparison"] = _bars({k: R[k]["peak_mem_alloc_gb"] for k in R}, "GB", "Peak GPU memory allocated (training)",
                                     pdir / "vram_comparison.png", fmt="{:.1f}")
    # expert utilisation heat maps (sequential single hue)
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blues", ["#f1f6fd", "#9cc3ef", "#2a78d6", "#0f3f7a"])
    ks = [k for k in R if R[k]["router_final"].get("per_layer_fraction")]
    if ks:
        fig, axes = plt.subplots(1, len(ks), figsize=(3.0 * len(ks) + 1.5, 4.8), squeeze=False, layout="constrained")
        vmax = max(np.max(R[k]["router_final"]["per_layer_fraction"]) for k in ks)
        for i, (ax, k) in enumerate(zip(axes[0], ks)):
            f = np.asarray(R[k]["router_final"]["per_layer_fraction"])
            im = ax.imshow(f, aspect="auto", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_title(f"{k}  (CV {R[k]['router_final']['util_cv_mean']:.3f})", fontsize=10)
            ax.set_xlabel("expert")
            ax.set_xticks(range(f.shape[1]))
            ax.set_yticks(range(f.shape[0]))
            if i == 0:
                ax.set_ylabel("MoE application (execution order)")
            ax.grid(False)
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85, label="fraction of assignments (uniform = 1/E)")
        fig.suptitle("Expert utilisation, last logging interval", x=0.01, ha="left", fontsize=12, fontweight="bold")
        fig.savefig(pdir / "expert_utilization.png")
        plt.close(fig)
        plots["expert_utilization"] = str(pdir / "expert_utilization.png")
    # training loss (EMA) vs tokens
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for k, r in runs.items():
        t = np.array([e["tokens"] for e in r["train"]]) / 1e6
        l = np.array([e["lm_loss"] for e in r["train"]])
        ema = np.empty_like(l)
        acc = l[0]
        for i, v in enumerate(l):
            acc = 0.9 * acc + 0.1 * v
            ema[i] = acc
        ax.plot(t, ema, STYLE.get(k, "-"), color=COLOR.get(k, INK2), lw=2, label=LABEL.get(k, k))
    ax.set_ylim(top=min(ax.get_ylim()[1], 8.0))
    ax.set_xlabel("training tokens (millions)")
    ax.set_ylabel("train LM loss (EMA 0.9)")
    ax.set_title("Training LM loss (auxiliary router loss excluded)", loc="left")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(pdir / "train_loss_vs_tokens.png")
    plt.close(fig)
    plots["train_loss_vs_tokens"] = str(pdir / "train_loss_vs_tokens.png")
    # systems benchmark
    b = res.get("benchmark")
    if b and b.get("full_model"):
        vals = {k: v["optimizer_step"]["median_of_round_medians_ms"] for k, v in b["full_model"].items() if "error" not in v}
        vals = {k.replace("C_samewidth", "C_same").replace("A_serial", "A").replace("B_parallel", "B"): v for k, v in vals.items()}
        plots["benchmark_step_time"] = _bars(vals, "ms per micro-batch step", "Systems benchmark: full optimizer step "
                                             f"(micro-batch {b['micro_batch']})", pdir / "benchmark_step_time.png",
                                             fmt="{:.0f}", note="median of interleaved-round medians; [reference] = sequential "
                                             "dispatch, [concurrent] = two CUDA streams")
    return {k: v for k, v in plots.items() if v}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    res = read_json(Path(a.root) / "results.json")
    print(make_all_plots(Path(a.root), res))
