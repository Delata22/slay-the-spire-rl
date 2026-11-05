#!/usr/bin/env python3
"""Test every card of the Ironclad pool in the env: crashes, infinite loops, encoding.

Writes the list of cards to exclude to src/stsenv/ironclad_pool_unsafe.json.
A card is flagged if it raises an exception, if combats get truncated (MAX_STEPS reached)
or if the effects encoding produces non-finite values.

  python validate_pool.py [--episodes 12] [--workers 16]
"""
import argparse, json, sys, traceback
from pathlib import Path
import multiprocessing as mp

import numpy as np

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))


def check(name, episodes=12):
    from stsenv.combat_env import CombatEnv, MAX_STEPS
    try:
        env = CombatEnv(4, encoding="effects", extra_cards=2)
        env.pool = [name]                      # force this card alone
        rng = np.random.default_rng(0)
        trunc = 0
        for ep in range(episodes):
            o, _ = env.reset(options={"episode_seed": 10_000 + ep})
            if not np.all(np.isfinite(o)):
                return name, "non-finite observation"
            done = tr = False
            while not (done or tr):
                m = env.action_masks()
                if not m.any():
                    return name, "no legal action"
                o, r, done, tr, info = env.step(int(rng.choice(np.flatnonzero(m))))
                if not np.all(np.isfinite(o)):
                    return name, "non-finite observation"
            trunc += info["steps"] >= MAX_STEPS
        if trunc > episodes * 0.25:
            return name, f"truncated combats ({trunc}/{episodes})"
        return name, None
    except Exception as e:
        return name, f"{type(e).__name__}: {e}".split("\n")[0][:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    pool = json.load(open(SRC / "stsenv/ironclad_pool.json"))
    with mp.get_context("spawn").Pool(a.workers) as p:
        res = p.starmap(check, [(c, a.episodes) for c in pool])
    bad = {c: why for c, why in res if why}
    print(f"{len(pool) - len(bad)}/{len(pool)} usable cards")
    for c, why in sorted(bad.items()):
        print(f"  excluded: {c:22} {why}")
    out = SRC / "stsenv/ironclad_pool_unsafe.json"
    out.write_text(json.dumps(sorted(bad), indent=0))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
