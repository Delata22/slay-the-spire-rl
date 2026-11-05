#!/usr/bin/env python3
"""Learning curves per variant (algo + preset + encoding), aggregated over seeds.

  python plot.py [--scenario 4] [--out evolution.png]

A variant = several runs that differ only by training seed (_s0, _s1, _s2).
Line = mean over seeds; band = min-max range across seeds (= reproducibility).
Baselines (random / heuristic / MCTS oracle) are horizontal lines.
"""
import argparse, json, re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = Path(__file__).resolve().parents[2]

# reference palette (dataviz skill), dark variant
SURFACE, INK, INK2, GRID = "#1a1a19", "#ffffff", "#c3c2b7", "#3a3a37"
SLOTS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
REF_STYLE = {"random": ":", "heuristic": "--", "optimal (pioche connue)": "-"}
# Display labels for result keys that are stored in French in the shipped JSON files.
LABELS = {"optimal (pioche connue)": "optimal (known draw order)"}


def variant_of(run):
    return re.sub(r"_s\d+$", "", run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, default=4)
    ap.add_argument("--out", default="evolution.png")
    ap.add_argument("--extra", type=int, default=0, help="variable-deck scenario")
    ap.add_argument("--pool", default="", help="train / eval (variable-deck scenario)")
    ap.add_argument("--deck", default="", help="archetype (per-archetype experiments)")
    ap.add_argument("--encounter", default="", help="fixed encounter")
    a = ap.parse_args()
    if a.deck or a.encounter:
        base = ROOT / "results" / "first_combat" / f"{a.deck or 'random'}_{a.encounter}_x{a.extra}"
    else:
        base = ROOT / "results" / "first_combat" / (f"seed{a.scenario}" + (f"_x{a.extra}" if a.extra else ""))
    ev = json.load(open(base / (f"eval_{a.pool}.json" if a.pool else "eval.json")))

    # group runs by variant: {variant: {step: [value per seed]}}
    curves = defaultdict(lambda: defaultdict(list))
    for run, rows in ev["runs"].items():
        for r in rows:
            curves[variant_of(run)][r["step"]].append(r["hp_lost"])
    mean_at = lambda v, s: sum(curves[v][s]) / len(curves[v][s])
    order = sorted(curves, key=lambda v: min(mean_at(v, s) for s in curves[v]))
    colors = {v: SLOTS[i % len(SLOTS)] for i, v in enumerate(order)}
    xmax = max(s for c in curves.values() for s in c)
    floor_ = min(v["hp_lost"] for v in ev["baselines"].values())
    zoom_hi = 1.12 * max(mean_at(v, s) for v in curves for s in curves[v] if s >= xmax * 0.3)

    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [2, 3]})
    for ax, title, ylim in ((axes[0], "HP lost per combat (↓ better) — full view", None),
                            (axes[1], "Zoom on the end of training", (0.85 * floor_, zoom_hi))):
        lo, hi = ylim if ylim else (0, 1.08 * max(mean_at(v, 0) for v in curves))
        opt = ev["baselines"].get("optimal (pioche connue)")
        if opt and ylim:
            ax.axhspan(lo, opt["hp_lost"], color="#d55181", alpha=0.07, lw=0, zorder=0)
        gap = (hi - lo) * 0.042; placed = []
        for y, name in sorted(((v["hp_lost"], k) for k, v in ev["baselines"].items()), reverse=True):
            if not (lo <= y <= hi):
                continue
            ax.axhline(y, color=INK2, lw=1, ls=REF_STYLE.get(name, "-."), zorder=1)
            ty = y
            for p in placed:
                if abs(ty - p) < gap: ty = p - gap
            placed.append(ty)
            ax.annotate(f"{LABELS.get(name, name)}  {y:.1f}", (xmax, y), xytext=(xmax * 1.02, ty),
                        color=INK2, va="bottom", fontsize=8.5)
        for v in order:
            steps = sorted(curves[v])
            mean = [mean_at(v, s) for s in steps]
            ax.fill_between(steps, [min(curves[v][s]) for s in steps], [max(curves[v][s]) for s in steps],
                            color=colors[v], alpha=0.18, lw=0, zorder=2)
            ax.plot(steps, mean, color=colors[v], lw=2, marker="o", ms=3.5, label=v, zorder=3)
            if ylim and lo <= mean[-1] <= hi:
                ax.annotate(f" {mean[-1]:.1f}", (steps[-1], mean[-1]), color=colors[v],
                            fontsize=9, va="center", fontweight="bold")
        ax.set_title(title, color=INK, loc="left", fontsize=11)
        ax.set_xlabel("environment steps seen in training")
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.set_xlim(-xmax * 0.02, xmax * 1.3); ax.set_ylim(lo, hi)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda t, _: f"{t / 1000:.0f}k"))
    axes[0].legend(frameon=False, loc="upper left", labelcolor=INK, fontsize=9, bbox_to_anchor=(0.10, 0.97))
    n_seeds = max(len(c[max(c)]) for c in curves.values())
    title = f"deck {a.deck} vs {a.encounter}" if a.deck else f"Scenario seed {ev['scenario']}"
    if a.extra:
        title += (f" + {a.extra} random cards — cards "
                  + ("NEVER SEEN in training" if a.pool == "eval" else "seen in training"))
    fig.suptitle(f"{title} | {ev['n_eval']} paired seeds, {n_seeds} training seeds "
                 f"(line = mean, band = range)", color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    out = base / a.out; fig.savefig(out, dpi=130); print(out)


if __name__ == "__main__":
    main()
