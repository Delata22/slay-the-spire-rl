#!/usr/bin/env python3
"""Evaluates the corrected MCTS (patches/scumsearch_fixed_uct.patch) on the same paired combat seeds
as every other policy, for each shipped experiment. Writes, next to each eval_episodes*.npz,
eval_mcts_fixed[_<pool>].npz with keys "mcts_fixed@<sims>" (rows: win, hp_lost, turns).

  python eval_mcts_fixed.py [--sims 200,2000] [--workers 12] [--check]

--check first replays the ORIGINAL MCTS on a few seeds and asserts it matches the shipped results,
i.e. that the patched build leaves the simulator and the original search unchanged.
"""
import argparse, multiprocessing as mp, os, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
RES = ROOT / "results" / "first_combat"

# folder -> (env spec as in evaluate.py, pools, number of eval seeds[, cards excluded before the split])
EXPERIMENTS = {
    "seed4": ((4, 0, "eval", None), [None], 500),
    "seed4_x3": ((4, 3, "{pool}", None), ["train", "eval"], 500, ["BARRICADE"]),
    "strength_cultist_x2": ((("strength", "cultist"), 2, "{pool}", None), ["train", "eval"], 500),
    "strength_cultist_x5": ((("strength", "cultist"), 5, "{pool}", None), ["train", "eval"], 400),
    **{f"random_cultist_x3_p{n}": ((("random", "cultist"), 3, "{pool}", n), ["train", "eval"], 500)
       for n in (6, 12, 24, 54)},
}


def job(args):
    spec, fixed, sims, seeds = args
    import evaluate as E
    from stsenv.policies import MCTSOraclePolicy
    scenario, extra, pool, tps = spec
    env = E.get_env(scenario, "onehot", extra, pool, tps)
    pol = MCTSOraclePolicy(sims, fixed=fixed)
    out = []
    for s in seeds:
        info, _ = E.play(pol, env, s)
        out.append((s, info["win"], info["hp_lost"], info["turns"]))
    return out


def make_pool(workers, exclude=()):
    """Fresh worker pool; excluded cards reach the spawned workers through STS_EXCLUDE_CARDS,
    read when stsenv.combat_env is imported."""
    os.environ["STS_EXCLUDE_CARDS"] = ",".join(exclude)
    return mp.get_context("spawn").Pool(workers)


def run(pool_, spec, fixed, sims, seeds, workers):
    chunk = max(5, len(seeds) // (workers * 2))
    jobs = [(spec, fixed, sims, seeds[i:i + chunk]) for i in range(0, len(seeds), chunk)]
    rows = {}
    for out in pool_.imap_unordered(job, jobs):
        for s, *v in out:
            rows[s] = v
    return np.array([rows[s] for s in seeds], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", default="200,2000")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only", default="", help="comma-separated experiment folders")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    names = [n for n in EXPERIMENTS if not a.only or n in a.only.split(",")]
    cache = {}
    if a.check:
        with make_pool(a.workers) as P:
            r = run(P, (4, 0, "eval", None), False, 2000, list(range(24)), a.workers)
            ref = np.load(RES / "seed4" / "eval_episodes.npz")["mcts_oracle@2000"][:24]
            assert np.array_equal(r, ref), "original MCTS does not reproduce the shipped seed4 results"
        with make_pool(a.workers, ["BARRICADE"]) as P:
            r = run(P, (4, 3, "eval", None), False, 200, list(range(24)), a.workers)
            ref = np.load(RES / "seed4_x3" / "eval_episodes_eval.npz")["mcts_oracle@200"][:24]
            assert np.array_equal(r, ref), "seed4_x3 pairing broken (card pool split changed?)"
        print("check: original MCTS reproduces the shipped seed4 and seed4_x3 results exactly")
    for name in names:
        spec_t, pools, n, *excl = EXPERIMENTS[name]
        exclude = excl[0] if excl else []
        with make_pool(a.workers, exclude) as P:
            for pool in pools:
                spec = tuple(p.format(pool=pool) if isinstance(p, str) else p for p in spec_t)
                # the held-out pool does not depend on the training-pool size: compute it once
                key = (spec[0], spec[1], spec[2], spec[3] if spec[2] == "train" else None, n, tuple(exclude))
                arrays = {}
                for sims in (int(x) for x in a.sims.split(",")):
                    t0 = time.time()
                    if (key, sims) not in cache:
                        cache[(key, sims)] = run(P, spec, True, sims, list(range(n)), a.workers)
                    r = cache[(key, sims)]
                    arrays[f"mcts_fixed@{sims}"] = r
                    print(f"{name:26} {pool or '':5} mcts_fixed@{sims:<5} HP lost {r[:, 1].mean():5.2f}"
                          f"  win {r[:, 0].mean():.1%}  ({time.time() - t0:.0f} s)", flush=True)
                np.savez_compressed(RES / name / f"eval_mcts_fixed{'_' + pool if pool else ''}.npz", **arrays)

if __name__ == "__main__":
    main()
