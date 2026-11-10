#!/usr/bin/env python3
"""Builds the README figures from the shipped evaluation files (results/first_combat/*).

  python experiments/make_figures.py            # -> docs/figures/*.png

Metric everywhere: regret = HP lost minus HP lost by the best known play on the same combat (best of
an exhaustive search that knows the draw order and of every evaluated policy; see
experiments/first_combat/recompute_optimal.py). Lower is better; regret is >= 0 by construction. Learned agents: final checkpoint, training seeds pooled.
Error bars: 95 % bootstrap over combats.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "first_combat"
OUT = ROOT / "docs" / "figures"
OPT = "optimal (pioche connue)"          # key stored in the evaluation files
MCTS = "mcts_oracle@2000"                  # reference search, with its UCT bug
MCTS_FIX = "mcts_fixed@2000"               # same search, bug fixed (patches/scumsearch_fixed_uct.patch)
MCTS_NOTE = "Hatched: MCTS references, which clone the RNG and therefore see the draw order."

SEEN, UNSEEN = "#2a78d6", "#eb6834"      # categorical slots 1-2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def boot(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


def regret(exp, pool, policy, step=None):
    """Per-combat regret of a baseline (`policy` = key) or of a run family (`policy` = prefix,
    all training seeds pooled, final checkpoint unless `step` is given)."""
    suffix = f"_{pool}" if pool else ""
    z = np.load(RES / exp / f"eval_episodes{suffix}.npz")
    opt = z[OPT][:, 1]
    if policy.startswith("mcts_fixed"):                 # written by eval_mcts_fixed.py
        return np.load(RES / exp / f"eval_mcts_fixed{suffix}.npz")[policy][:, 1] - opt
    if policy in z.files:
        return z[policy][:, 1] - opt
    keys = [k for k in z.files if k.startswith(policy + "_s") and "__" in k and "smoke" not in k]
    runs = sorted({k.split("__")[0] for k in keys})
    out = []
    for r in runs:
        steps = sorted(int(k.split("__")[1]) for k in keys if k.split("__")[0] == r)
        s = step if step is not None else steps[-1]
        out.append(z[f"{r}__{s}"][:, 1] - opt)
    if not out:
        raise KeyError(f"{exp}/{pool}: no run matches {policy}")
    return np.concatenate(out)


def seen_unseen_bars(ax, exp, groups, title):
    """Grouped bars: regret on cards seen in training vs held-out cards."""
    x = np.arange(len(groups)); w = 0.36
    top = [max(max(boot(regret(exp, p, k))[1] for p in ("train", "eval"))
               for _, k in groups)]
    for j, (pool, color, label) in enumerate((("train", SEEN, "cards seen in training"),
                                              ("eval", UNSEEN, "held-out cards (never seen)"))):
        means, lo, hi = [], [], []
        for _, key in groups:
            r = regret(exp, pool, key)
            m = r.mean(); c = boot(r)
            means.append(m); lo.append(m - c[0]); hi.append(c[1] - m)
        xs = x + (j - 0.5) * w
        bars = ax.bar(xs, means, w - 0.03, color=color, label=label, yerr=[lo, hi],
                      error_kw=dict(ecolor=INK, lw=0.9, capsize=2.5))
        for b, (_, key) in zip(bars, groups):
            if key.startswith("mcts"):
                b.set_hatch("///"); b.set_edgecolor("white"); b.set_linewidth(0)
        for xi, m, h in zip(xs, means, hi):
            ax.text(xi, m + h + 0.04 * top[0], f"{m:.2f}", ha="center", va="bottom",
                    fontsize=8, color=MUTED)
    ax.set_xticks(x, [g for g, _ in groups], fontsize=9)
    ax.set_ylim(0, top[0] * 1.3)
    ax.set_ylabel("regret (HP lost above best known play)")
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9, borderaxespad=0.2)
    if any(k.startswith("mcts") for _, k in groups):
        ax.text(0, -0.2, MCTS_NOTE, transform=ax.transAxes, fontsize=8, color=MUTED, va="top")


def fig_ladder():
    fig, ax = plt.subplots(figsize=(9.8, 4.3))
    seen_unseen_bars(ax, "strength_cultist_x2", [
        ("V0  one-hot ids\nfixed action slots", "dqn_tuned_onehot_pool"),
        ("V1  measured effects\nfixed action slots", "dqn_tuned_effects"),
        ("V2  measured effects\naction scorer", "dqn_scorer_effects"),
        ("MCTS @2000\noriginal (bugged)", MCTS),
        ("MCTS @2000\nfixed", MCTS_FIX),
    ], "Representation ladder: Strength deck vs Cultist, 2 random cards per combat")
    fig.savefig(OUT / "ladder.png"); plt.close(fig)


def fig_card_description():
    fig, ax = plt.subplots(figsize=(9.8, 4.3))
    seen_unseen_bars(ax, "strength_cultist_x5", [
        ("nominal\n(printed values)", "dqn_scorer_nominal_effects"),
        ("measured\n(in current context)", "dqn_scorer_effects"),
        ("measured + nominal", "dqn_scorer_both_effects"),
        ("MCTS @2000\noriginal (bugged)", MCTS),
        ("MCTS @2000\nfixed", MCTS_FIX),
    ], "How to describe a card: scorer agent, 5 random cards per combat")
    fig.savefig(OUT / "card_description.png"); plt.close(fig)


def fig_random_pool():
    fig, ax = plt.subplots(figsize=(9.8, 4.3))
    seen_unseen_bars(ax, "seed4_x3", [
        ("DQN\none-hot ids", "dqn_tuned_onehot"),
        ("DQN\nmeasured effects", "dqn_tuned_effects"),
        ("PPO\none-hot ids", "ppo_tuned_onehot"),
        ("PPO\nmeasured effects", "ppo_tuned_effects"),
        ("MCTS @2000\noriginal (bugged)", MCTS),
        ("MCTS @2000\nfixed", MCTS_FIX),
    ], "Random card pool (53 train / 18 held-out cards) vs Jaw Worm, 3 random cards")
    fig.savefig(OUT / "random_pool.png"); plt.close(fig)


def fig_learning_curves():
    """Regret on held-out cards vs training steps, V0 / V1 / V2 (mean over training seeds)."""
    exp, pool = "strength_cultist_x2", "eval"
    z = np.load(RES / exp / "eval_episodes_eval.npz")
    opt = z[OPT][:, 1]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for label, fam, color in (("V0  one-hot ids, fixed slots", "dqn_tuned_onehot_pool", "#eb6834"),
                              ("V1  measured effects, fixed slots", "dqn_tuned_effects", "#1baf7a"),
                              ("V2  measured effects, scorer", "dqn_scorer_effects", "#2a78d6")):
        keys = [k for k in z.files if k.startswith(fam + "_s") and "__" in k]
        steps = sorted({int(k.split("__")[1]) for k in keys})
        runs = sorted({k.split("__")[0] for k in keys})
        curve = [np.mean([(z[f"{r}__{st}"][:, 1] - opt).mean() for r in runs]) for st in steps]
        ax.plot(np.array(steps) / 1000, curve, "-o", color=color, lw=2, ms=4, label=label)
    for key, ls, text in ((MCTS, "--", "MCTS @2000, original (bugged)"), (MCTS_FIX, ":", "MCTS @2000, fixed")):
        m = regret(exp, pool, key).mean()
        ax.axhline(m, ls=ls, color=MUTED, lw=1.2)
        ax.text(0, m, text + " - sees draw order", ha="left", va="bottom", fontsize=8, color=MUTED)
    ax.set_yscale("log")
    ticks = [0.2, 0.5, 1, 2, 5, 10, 20, 50]
    ax.set_yticks(ticks, [str(t) for t in ticks]); ax.minorticks_off()
    ax.set_xlabel("training steps (thousands)")
    ax.set_ylabel("regret on held-out cards (log)")
    ax.set_title("Learning curves on unseen cards (Strength deck vs Cultist)", loc="left",
                 fontsize=11, color=INK)
    ax.legend(loc="upper right", fontsize=9)
    fig.savefig(OUT / "learning_curves.png"); plt.close(fig)


def fig_pool_size(exp_prefix="random_cultist_x3_p", sizes=(6, 12, 24, 54)):
    """Training-pool diversity sweep (experiments/first_combat/run_pool_size.sh).
    Left: regret on the 18 held-out cards (dots = training seeds, line = mean).
    Right: degradation = regret on held-out cards - regret on the run's own training cards."""
    if not all((RES / f"{exp_prefix}{n}" / "eval_episodes_eval.npz").exists() for n in sizes):
        print("pool-size sweep: results missing, figure skipped")
        return
    agents = (("action scorer, measured + nominal effects", "dqn_scorer_both_mask_effects", "#2a78d6"),
              ("fixed slots, one-hot card ids", "dqn_tuned_onehot_pool", "#eb6834"))

    def per_seed(n, pool, fam):
        z = np.load(RES / f"{exp_prefix}{n}" / f"eval_episodes_{pool}.npz")
        opt = z[OPT][:, 1]
        runs = sorted({k.split("__")[0] for k in z.files if k.startswith(fam + "_s")})
        out = []
        for r in runs:
            last = max(int(k.split("__")[1]) for k in z.files if k.startswith(r + "__"))
            out.append((z[f"{r}__{last}"][:, 1] - opt).mean())
        return np.array(out)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9))
    for k, (label, fam, color) in enumerate(agents):
        dx = (k - 0.5) * 0.06
        unseen = [per_seed(n, "eval", fam) for n in sizes]
        deg = [per_seed(n, "eval", fam) - per_seed(n, "train", fam) for n in sizes]
        for ax, vals in ((a1, unseen), (a2, deg)):
            for n, v in zip(sizes, vals):
                ax.scatter(np.full(len(v), n) * 2 ** dx, v, s=18, color=color, alpha=0.45, lw=0)
            ax.plot(np.array(sizes) * 2 ** dx, [v.mean() for v in vals], "-o", color=color, lw=2,
                    ms=6, label=label)
    for key, ls, text, va in ((MCTS, "--", "MCTS @2000, original", "bottom"),
                              (MCTS_FIX, ":", "MCTS @2000, fixed", "top")):
        m = regret(f"{exp_prefix}{sizes[0]}", "eval", key).mean()
        a1.axhline(m, ls=ls, color=MUTED, lw=1.2)
        a1.text(sizes[-1] * 1.05, m, text, va="bottom", ha="right", fontsize=8, color=MUTED)
    a1.set_ylim(bottom=0)
    a2.axhline(0, color=MUTED, lw=0.8)
    for ax in (a1, a2):
        ax.set_xscale("log", base=2); ax.set_xticks(sizes, [str(n) for n in sizes]); ax.minorticks_off()
        ax.set_xlabel("distinct cards in the training pool")
    a1.set_ylabel("regret on held-out cards")
    a2.set_ylabel("held-out minus seen regret")
    a1.set_title("Regret on the same 18 held-out cards", loc="left", fontsize=10, color=INK)
    a2.set_title("Degradation from seen to held-out cards", loc="left", fontsize=10, color=INK)
    fig.suptitle("Training-card diversity vs generalisation (Cultist, 3 random cards, 150k steps, 3 seeds)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=2, fontsize=9)
    fig.savefig(OUT / "pool_size.png"); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_ladder(); fig_card_description(); fig_random_pool(); fig_learning_curves(); fig_pool_size()
    print(f"figures -> {OUT}")
