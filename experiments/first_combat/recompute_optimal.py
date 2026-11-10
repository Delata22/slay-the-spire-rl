#!/usr/bin/env python3
"""Recomputes the reference used for regret ("best known play") for the shipped experiments and
updates it in place in optimal_hp_lost*.npy, eval_episodes*.npz and eval*.json.

Best known play, per combat = min over
  - the exhaustive search of stsenv/optimal.py (draw order known; copies of a card merged), and
  - every evaluated policy, in practice the fixed MCTS @20000 (it also sees the draw order).
The search alone is slightly pessimistic: merging card copies ignores that WHICH copy is discarded
changes the next reshuffle. Earlier versions also never explored potions or card selections.
Paired differences between policies do not depend on the reference.

  python recompute_optimal.py [--workers 12] [--only seed4,seed4_x3]

Afterwards, every evaluated policy must have regret >= 0 on every combat: the script checks it.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from eval_mcts_fixed import EXPERIMENTS  # noqa: E402

RES = ROOT / "results" / "first_combat"
OPT = "optimal (pioche connue)"          # key stored in the evaluation files


def boot_ci(x, n=2000, rng=np.random.default_rng(0)):
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    from stsenv import combat_env, optimal
    names = [n for n in EXPERIMENTS if not a.only or n in a.only.split(",")]
    # experiments that exclude cards from the pool go last: the exclusion is not undone
    names.sort(key=lambda n: len(EXPERIMENTS[n]) > 3)
    for name in names:
        spec_t, pools, n, *excl = EXPERIMENTS[name]
        if excl:
            combat_env.exclude_cards(excl[0]); optimal._ENV.clear()
        for pool in pools:
            spec = tuple(p.format(pool=pool) if isinstance(p, str) else p for p in spec_t)
            sfx = f"_{pool}" if pool else ""
            search = optimal.optimal_hp_lost(spec, list(range(n)), workers=a.workers)
            npz_path = RES / name / f"eval_episodes{sfx}.npz"
            z = dict(np.load(npz_path))
            old = z[OPT][:, 1].copy()
            assert len(old) == len(search)
            policies = {k: v[:, 1] for k, v in z.items() if k != OPT}
            fixed_path = RES / name / f"eval_mcts_fixed{sfx}.npz"
            if fixed_path.exists():
                policies.update({k: v[:, 1] for k, v in np.load(fixed_path).items()})
            best_policy = np.min(np.stack(list(policies.values())), axis=0)
            new = np.minimum(search, best_policy)
            from_policy = {k: int(((v < search) & (v == new)).sum()) for k, v in policies.items()}
            from_policy = {k: c for k, c in from_policy.items() if c}
            z[OPT] = np.stack([np.ones_like(new), new, np.zeros_like(new)], axis=1)
            worst = min(float((v - new).min()) for v in policies.values())
            np.savez_compressed(npz_path, **z)
            np.save(RES / name / f"optimal_search_hp_lost{sfx}.npy", search)
            for npy in (RES / name / f"optimal_hp_lost{sfx}.npy", RES / name / "optimal_hp_lost.npy"):
                if npy.exists() or npy.name == f"optimal_hp_lost{sfx}.npy":
                    np.save(npy, new)
            js = RES / name / f"eval{sfx}.json"
            d = json.load(open(js))
            d["baselines"][OPT].update(hp_lost=float(new.mean()), hp_lost_ci=boot_ci(new))
            json.dump(d, open(js, "w"), indent=1)
            print(f"{name:26} {pool or '':5} reference HP lost {old.mean():.3f} -> {new.mean():.3f}"
                  f" | search alone {search.mean():.3f}, improved {int((search < old).sum())} combats"
                  f" | beaten by a policy on {int((best_policy < search).sum())}/{n}: {from_policy}", flush=True)
            if (search > old).any():
                print(f"   warning: search worse than the previous reference on {int((search > old).sum())} combats")
            assert worst >= 0


if __name__ == "__main__":
    main()
