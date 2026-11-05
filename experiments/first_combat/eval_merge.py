#!/usr/bin/env python3
"""Evaluate the checkpoints of a few runs and merge into eval.json / eval_episodes.npz.

  python eval_merge.py <run1> <run2> ...

Faster than evaluate.py when only a few runs are new: baselines
(random / heuristic / MCTS) and other runs are not recomputed.
"""
import sys, json, glob, os, time, re
from pathlib import Path
import numpy as np
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as E  # noqa: E402

BASE = Path(__file__).resolve().parents[2] / "results/first_combat/seed4"


def main():
    runs = sys.argv[1:]
    specs = []
    for r in runs:
        cfg = json.load(open(BASE / r / "config.json"))
        for f in sorted(glob.glob(str(BASE / r / "ckpt_*"))):
            step = int(os.path.basename(f).split("_")[1].split(".")[0])
            specs.append({"kind": "sb3" if f.endswith(".zip") else "torch", "path": f, "algo": cfg["algo"],
                          "encoding": cfg.get("encoding", "onehot"), "run": r, "step": step,
                          "key": f"{r}/{step}"})
    seeds = list(range(500)); chunk = 25
    jobs = [(sp, 4, seeds[i:i + chunk]) for sp in specs for i in range(0, 500, chunk)]
    t0 = time.time(); res = {}
    with mp.get_context("spawn").Pool(20) as pool:
        for key, out in pool.imap_unordered(E.worker, jobs):
            for s, *v in out:
                res.setdefault(key, {})[s] = v
    print(f"eval of {len(runs)} runs ({len(specs)} checkpoints): {time.time() - t0:.0f} s", flush=True)

    summary = json.load(open(BASE / "eval.json"))
    arrays = dict(np.load(BASE / "eval_episodes.npz"))
    for sp in specs:
        r = np.array([res[sp["key"]][s] for s in seeds], float)
        row = {"step": sp["step"], "win": float(r[:, 0].mean()), "win_ci": E.boot_ci(r[:, 0]),
               "hp_lost": float(r[:, 1].mean()), "hp_lost_ci": E.boot_ci(r[:, 1]), "turns": float(r[:, 2].mean())}
        rows = [x for x in summary["runs"].get(sp["run"], []) if x["step"] != sp["step"]]
        summary["runs"][sp["run"]] = rows + [row]
        arrays[sp["key"].replace("/", "__")] = r
    for v in summary["runs"].values():
        v.sort(key=lambda x: x["step"])
    json.dump(summary, open(BASE / "eval.json", "w"), indent=1)
    np.savez_compressed(BASE / "eval_episodes.npz", **arrays)

    fin = {}
    for run, rows in summary["runs"].items():
        fin.setdefault(re.sub(r"_s\d+$", "", run), []).append(rows[-1]["hp_lost"])
    print("\nvariants (HP lost at the last checkpoint):")
    for v, a in sorted(fin.items(), key=lambda kv: np.mean(kv[1])):
        print(f"  {v:24} {np.mean(a):.2f}   ({' / '.join(f'{x:.2f}' for x in a)})")


if __name__ == "__main__":
    main()
