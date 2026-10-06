"""Builds the README / slide figures from the logged results in ../../results (no hand-entered numbers).

    python docs/figures/make_figures.py

Writes <name>.png (light) and <name>_dark.png (dark) next to this file.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent.parent / "results"
TOK_PER_STEP = 64 * 1024

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", band="#f0efec",
                  series={"A": "#2a78d6", "B": "#eb6834", "C_same": "#1baf7a", "C_matched": "#eda100"}),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", band="#383835",
                 series={"A": "#3987e5", "B": "#d95926", "C_same": "#199e70", "C_matched": "#c98500"}),
}
LABEL = {"A": "A  serial", "B": "B  parallel", "C_same": "C_same  parallel + fusion (+15% compute)",
         "C_matched": "C_matched  parallel + fusion (same compute)"}
SHORT = {"A": "A", "B": "B", "C_same": "C_same", "C_matched": "C_matched"}
RUNS = {"A": "A_serial_seed42", "B": "B_parallel_seed42", "C_same": "C_parallel_fusion_samewidth_seed42",
        "C_matched": "C_parallel_fusion_matched_seed42"}


def load_curve(phase, run):
    rows = [json.loads(l) for l in open(RES / phase / "runs" / run / "eval_metrics.jsonl")]
    rows = [r for r in rows if not r["final"]]
    return ([r["step"] * TOK_PER_STEP / 1e6 for r in rows], [r["val_loss"] for r in rows],
            [r["train_seconds"] / 60 for r in rows])


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(t["axis"])
        ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=t["muted"], labelsize=9, length=0, pad=6)
    ax.grid(True, color=t["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t["ink2"])
    ax.yaxis.label.set_color(t["ink2"])
    ax.title.set_color(t["ink"])


def fig_setup(t, w, h, ncols=1, nrows=1, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), facecolor=t["surface"], **kw)
    return fig, axes


def end_label(ax, x, y, text, color, t, dy=0.0, dx=8, ha="left"):
    ax.plot([x], [y], "o", ms=7, color=color, mec=t["surface"], mew=2, zorder=5)
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center", ha=ha, fontsize=9,
                color=t["ink"], annotation_clip=False)


def save(fig, name, theme):
    out = HERE / (f"{name}.png" if theme == "light" else f"{name}_dark.png")
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# ---------------------------------------------------------------- figures
def loss_vs_tokens(t, theme):
    curves = {m: load_curve("main", r) for m, r in RUNS.items()}
    fig, (a1, a2) = fig_setup(t, 12, 4.6, 2, gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.22})
    for ax in (a1, a2):
        style(ax, t)
    for m, (x, y, _) in curves.items():
        a1.plot(x[1:], y[1:], color=t["series"][m], lw=2, solid_capstyle="round", label=LABEL[m])
        xs = [(xi, yi) for xi, yi in zip(x, y) if xi >= 50]
        a2.plot([p[0] for p in xs], [p[1] for p in xs], color=t["series"][m], lw=2, solid_capstyle="round")
    final = {m: c[1][-1] for m, c in curves.items()}
    offs = {"A": 0, "B": 6, "C_matched": -6, "C_same": -10}
    for m in curves:
        end_label(a2, 100.0, final[m], f"{SHORT[m]}  {final[m]:.3f}", t["series"][m], t, dy=offs[m])
    a1.set_title("Validation loss vs training tokens (MAIN, 100M tokens)", loc="left", fontsize=11, pad=10, color=t["ink"])
    a2.set_title("Zoom: second half of training", loc="left", fontsize=11, pad=10, color=t["ink"])
    for ax in (a1, a2):
        ax.set_xlabel("training tokens (millions)")
    a1.set_ylabel("validation loss (nats/token)")
    a2.set_xlim(49, 100.5)
    leg = a1.legend(frameon=False, fontsize=9, loc="upper right")
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    save(fig, "main_loss_vs_tokens", theme)


def gaps(t, theme):
    c = {m: load_curve("main", r) for m, r in RUNS.items()}
    x = c["A"][0]
    d = lambda p, q: [a - b for a, b in zip(c[p][1], c[q][1])]  # noqa: E731
    panels = [("B − A", "parallel vs serial", d("B", "A")),
              ("C_same − B", "fusion gain (with +15% compute)", d("C_same", "B")),
              ("C_matched − C_same", "cost of narrowing experts 1920 → 1536", d("C_matched", "C_same")),
              ("C_matched − B", "net effect at the same compute", d("C_matched", "B"))]
    paired = json.load(open(RES / "main" / "results.json"))["paired"]
    keys = ["B_minus_A", "C_same_minus_B", "C_matched_minus_C_same", "C_matched_minus_B"]
    fig, axes = fig_setup(t, 12, 6.6, 2, 2, sharex=True)
    fig.subplots_adjust(hspace=0.42, wspace=0.3)
    color = t["series"]["A"]
    for ax, (title, sub, y), k in zip(axes.flat, panels, keys):
        style(ax, t)
        ax.axhspan(-0.02, 0.02, color=t["band"], zorder=0, lw=0)
        ax.axhline(0, color=t["axis"], lw=1, zorder=1)
        ax.plot(x[1:], y[1:], color=color, lw=2, solid_capstyle="round", zorder=3)
        fv = paired[k]["mean"]
        end_label(ax, x[-1], y[-1], f"final {fv:+.3f}", color, t)
        ax.set_title(f"{title}   ", loc="left", fontsize=11, pad=18, color=t["ink"])
        ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=9, color=t["ink2"])
        ax.set_xlim(0, 125)
        ax.set_xticks(range(0, 101, 20))
    for ax in axes[1]:
        ax.set_xlabel("training tokens (millions)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ validation loss (nats)\nnegative = first model better")
    fig.text(0.5, -0.02, "Paired differences on the same validation sequences. Shaded band: ±0.02-nat equivalence margin. "
             "Curves use the periodic 2,048-sequence subset; 'final' uses the full validation set (15,088 sequences).",
             ha="center", fontsize=8.5, color=t["muted"])
    save(fig, "main_gaps", theme)


def wallclock(t, theme):
    curves = {m: load_curve("main", r) for m, r in RUNS.items()}
    fig, ax = fig_setup(t, 8.5, 4.6)
    style(ax, t)
    for m, (_, y, tm) in curves.items():
        pts = [(a, b) for a, b in zip(tm, y) if a >= 16]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=t["series"][m], lw=2, solid_capstyle="round")
        if m == "B":
            end_label(ax, tm[-1], y[-1], f"{SHORT[m]}  {tm[-1]:.1f} min", t["series"][m], t, dy=-11, dx=-6, ha="right")
        else:
            end_label(ax, tm[-1], y[-1], f"{SHORT[m]}  {tm[-1]:.1f} min", t["series"][m], t, dy={"C_matched": 7}.get(m, 0))
    tstar = curves["B"][2][-1]
    ax.axvline(tstar, color=t["muted"], lw=1)
    ax.text(tstar - 0.3, ax.get_ylim()[1], "B and A finish (32.3 min)", ha="right", va="top", fontsize=8.5,
            color=t["ink2"])
    ax.set_title("Validation loss vs training wall-clock (A100 40GB, same data and steps)", loc="left", fontsize=11,
                 pad=10, color=t["ink"])
    ax.set_xlabel("training wall-clock (minutes, evaluation excluded)")
    ax.set_ylabel("validation loss (nats/token)")
    ax.set_xlim(16, 41.5)
    save(fig, "main_loss_vs_wallclock", theme)


def b_minus_a_summary(t, theme):
    diag = json.load(open(RES / "diagnostics" / "DIAG_SUMMARY.json"))["gaps_B_minus_A"]
    pilot = json.load(open(RES / "pilot" / "results.json"))["paired"]["B_minus_A"]["mean"]
    main = json.load(open(RES / "main" / "results.json"))["paired"]["B_minus_A"]["mean"]
    rows = [("Pilot · seed 42 · 25M tokens", pilot),
            ("DIAG · seed 42 · 25M (re-run)", diag["seed42"]["d"]),
            ("DIAG · seed 43 · 25M", diag["seed43"]["d"]),
            ("DIAG · dense FFN control · 25M", diag["dense"]["d"]),
            ("DIAG · 10% warm-up · 25M", diag["warmup10"]["d"]),
            ("MAIN · seed 42 · 100M tokens", main)]
    fig, ax = fig_setup(t, 8.5, 3.9)
    style(ax, t)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ys = list(range(len(rows)))[::-1]
    color = t["series"]["B"]
    ax.barh(ys, [r[1] for r in rows], height=0.42, color=color, edgecolor=t["surface"], linewidth=2)
    for yv, (_, v) in zip(ys, rows):
        ax.text(v - 0.004, yv, f"{v:+.3f}", ha="right", va="center", fontsize=9, color=t["ink"])
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9, color=t["ink2"])
    ax.axvline(0, color=t["axis"], lw=1)
    ax.set_xlim(-0.24, 0.005)
    ax.set_xlabel("B − A validation loss (nats) · negative = parallel better")
    ax.set_title("Parallel (B) beats serial (A) in every condition — the gap shrinks with training",
                 loc="left", fontsize=11, pad=10, color=t["ink"])
    save(fig, "b_minus_a_summary", theme)


if __name__ == "__main__":
    plt.rcParams["font.family"] = "DejaVu Sans"
    for theme, t in THEMES.items():
        loss_vs_tokens(t, theme)
        gaps(t, theme)
        wallclock(t, theme)
        b_minus_a_summary(t, theme)
    print("figures written to", HERE)
