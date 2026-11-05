#!/usr/bin/env python3
"""Generalisation: gap between cards SEEN in training and UNSEEN (held-out) cards.

Assumes `evaluate.py --extra-cards N --pool train` and `--pool eval` have been run.

  python compare_pools.py [--extra 3] [--scenario 4]

For each variant (algo + preset + encoding, averaged over training seeds):
  - HP lost on the training pool (same distribution as training);
  - HP lost on the eval pool (never-seen cards);
  - the gap between the two = cost of novelty; THIS is the quantity that separates encodings;
  - the gap to the brute-force optimum on each pool (the scenario's difficulty differs between pools).
"""
import argparse, json, re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load(base, pool):
    ev = json.load(open(base / f"eval_{pool}.json"))
    arr = dict(np.load(base / f"eval_episodes_{pool}.npz"))
    return ev, arr


def variant(run):
    return re.sub(r"_s\d+$", "", run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, default=4)
    ap.add_argument("--extra", type=int, default=3)
    a = ap.parse_args()
    base = ROOT / "results" / "first_combat" / f"seed{a.scenario}_x{a.extra}"

    data = {}
    for pool in ("train", "eval"):
        ev, arr = load(base, pool)
        finals = defaultdict(list)
        for run, rows in ev["runs"].items():
            finals[variant(run)].append(rows[-1]["hp_lost"])
        data[pool] = (ev, arr, finals)

    rng = np.random.default_rng(0)
    print(f"scenario seed {a.scenario} + {a.extra} random cards per episode")
    for pool in ("train", "eval"):
        ev = data[pool][0]
        refs = " | ".join(f"{k} {v['hp_lost']:.2f}" for k, v in ev["baselines"].items()
                          if k in ("optimal (pioche connue)", "heuristic", "mcts_oracle@2000", "random"))
        print(f"  pool {pool:5}: {refs}")

    print(f"\n{'variant':26} {'seen cards':>12} {'unseen cards':>16} {'gap':>18}")
    for v in sorted(data["eval"][2], key=lambda k: np.mean(data["eval"][2][k])):
        tr = np.mean(data["train"][2][v]); evl = np.mean(data["eval"][2][v])
        # paired gap, training seed by training seed
        keys = [k for k in data["eval"][1] if k.startswith(v + "_s") and k.endswith("__400000")]
        diffs = []
        for k in keys:
            if k in data["train"][1]:
                diffs.append(data["eval"][1][k][:, 1] - data["train"][1][k][:, 1])
        if diffs:
            x = np.concatenate(diffs)
            m = x[rng.integers(0, len(x), (4000, len(x)))].mean(1)
            gap = f"{x.mean():+.2f} [{np.percentile(m,2.5):+.2f}, {np.percentile(m,97.5):+.2f}]"
        else:
            gap = f"{evl - tr:+.2f}"
        print(f"{v:26} {tr:>12.2f} {evl:>16.2f} {gap:>18}")

    opt_tr = data["train"][0]["baselines"]["optimal (pioche connue)"]["hp_lost"]
    opt_ev = data["eval"][0]["baselines"]["optimal (pioche connue)"]["hp_lost"]
    print(f"\nNote: the optimum is {opt_tr:.2f} on the training pool and {opt_ev:.2f} on the eval pool "
          f"({opt_ev - opt_tr:+.2f}); part of the gap above comes from difficulty, not novelty.")


if __name__ == "__main__":
    main()
