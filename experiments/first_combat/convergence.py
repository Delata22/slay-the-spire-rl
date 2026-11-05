#!/usr/bin/env python3
"""Convergence speed: in steps, in combats, and in wall-clock time.

Criterion: first checkpoint reaching 95% of the total gain between checkpoint 0 and the last one.
Throughput comes from the training logs; it includes the environment, the encoding, the forward
pass and the network update.

WARNING: runs were executed 4 in parallel, so absolute times are pessimistic by a factor of
2 to 3. Ratios between encodings remain valid (same contention).

  python convergence.py [--dir strength_cultist_x3] [--pool train]
"""
import argparse, glob, json, re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "first_combat"


def table(d, pool="train"):
    f = d / (f"eval_{pool}.json" if (d / f"eval_{pool}.json").exists() else "eval.json")
    if not f.exists():
        return []
    ev = json.load(open(f))
    curves = defaultdict(lambda: defaultdict(list))
    for run, rows in ev["runs"].items():
        v = re.sub(r"_s\d+$", "", run)
        for r in rows:
            curves[v][r["step"]].append(r["hp_lost"])
    out = []
    for v, steps in sorted(curves.items()):
        xs = sorted(steps)
        y = [float(np.mean(steps[s])) for s in xs]
        target = y[-1] + 0.05 * (y[0] - y[-1])
        n_steps = xs[next((i for i, val in enumerate(y) if val <= target), len(y) - 1)]
        eps, rates = [], []
        for jl in glob.glob(str(d / f"{v}_s*" / "train_log.jsonl")):
            eps.append(sum(1 for l in open(jl) if json.loads(l)["t"] <= n_steps))
        for lg in glob.glob(str(d / "logs" / f"train_{v}_s*.log")):
            rates += [int(x) for x in re.findall(r"\| (\d+) (?:pas|steps)/s", open(lg).read())]
        r = float(np.median(rates)) if rates else float("nan")
        out.append({"variant": v, "steps": n_steps, "combats": float(np.mean(eps)) if eps else float("nan"),
                    "steps_per_s": r, "minutes": n_steps / r / 60 if r == r else float("nan"),
                    "ms_per_step": 1000 / r if r == r else float("nan"), "final_hp_lost": y[-1]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="")
    ap.add_argument("--pool", default="train")
    a = ap.parse_args()
    dirs = [RES / a.dir] if a.dir else [d for d in sorted(RES.iterdir()) if d.is_dir()]
    for d in dirs:
        rows = table(d, a.pool)
        if not rows:
            continue
        print(f"=== {d.name}")
        print(f"{'variant':28} {'steps':>9} {'combats':>9} {'steps/s':>7} {'time':>8} {'ms/step':>8} {'final HP':>9}")
        for r in rows:
            print(f"{r['variant']:28} {r['steps']:>9,} {r['combats']:>9.0f} {r['steps_per_s']:>7.0f} "
                  f"{r['minutes']:>7.1f}m {r['ms_per_step']:>8.2f} {r['final_hp_lost']:>9.2f}".replace(",", " "))
        print()


if __name__ == "__main__":
    main()
