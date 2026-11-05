#!/usr/bin/env python3
"""Compare two policies on the SAME eval seeds and transcribe the combats where they diverge.

  python compare_replays.py <key_A> <key_B> [--n 3] [--worse]

Keys are those of eval_episodes.npz, e.g.:
  dqn_tuned_onehot_s0__400000   mcts_oracle@2000
By default, shows the combats where A loses the FEWEST HP relative to B (--worse for the opposite).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as E

BASE = Path(__file__).resolve().parents[2] / "results/first_combat/seed4"


def spec_for(key):
    if key.startswith("mcts_oracle@"):
        return {"kind": "mcts", "sims": int(key.split("@")[1]), "key": key}
    if key in ("random", "heuristic"):
        return {"kind": key, "key": key}
    run, step = key.split("__")
    cfg = json.load(open(BASE / run / "config.json"))
    path = BASE / run / f"ckpt_{int(step):07d}.pt"
    if not path.exists():
        path = BASE / run / f"ckpt_{int(step):07d}.zip"
    return {"kind": "sb3" if path.suffix == ".zip" else "torch", "path": str(path),
            "algo": cfg["algo"], "encoding": cfg.get("encoding", "onehot"), "run": run, "key": key}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--worse", action="store_true")
    args = ap.parse_args()

    d = np.load(BASE / "eval_episodes.npz")
    A, B = d[args.a], d[args.b]
    diff = A[:, 1] - B[:, 1]          # HP lost by A - HP lost by B
    print(f"{args.a}  vs  {args.b}")
    print(f"  HP lost: {A[:,1].mean():.2f} vs {B[:,1].mean():.2f} | "
          f"A better in {np.mean(diff<0):.0%} of combats, tied {np.mean(diff==0):.0%}, "
          f"worse in {np.mean(diff>0):.0%}")
    print(f"  mean gap {diff.mean():+.2f} HP | A worst case {diff.max():+.0f} | A best case {diff.min():+.0f}\n")

    order = np.argsort(-diff if args.worse else diff)[:args.n]
    specs = [spec_for(args.a), spec_for(args.b)]
    envs = {sp["key"]: E.get_env(4, sp.get("encoding", "onehot")) for sp in specs}
    pols = {sp["key"]: E.load_policy(sp, envs[sp["key"]]) for sp in specs}
    for seed in order:
        print("=" * 78)
        print(f"EVAL SEED {seed}   (gap {diff[seed]:+.0f} HP)")
        for sp in specs:
            k = sp["key"]
            info, txt = E.play(pols[k], envs[k], int(seed), trace=True)
            print(f"\n--- {k}: {info['hp_lost']} HP lost, {info['turns']} turns ---")
            print(txt)
    print("=" * 78)


if __name__ == "__main__":
    main()
