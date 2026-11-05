#!/usr/bin/env python3
"""How often does MCTS miss an available lethal move?

Plays combats with MCTS; at each decision, brute-forces whether some card sequence kills this
turn, and checks whether the chosen action starts such a sequence.

  python lethal_check.py [--n 40] [--sims 2000]
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from stsenv.combat_env import CombatEnv, sts, AT, legal_map  # noqa: E402
from stsenv.policies import MCTSOraclePolicy  # noqa: E402


def lethal_first_actions(bc, depth=4):
    """Action indices (in amap) that start a lethal sequence this turn."""
    good = set()

    def rec(state, first, d):
        if state.outcome == sts.Outcome.PLAYER_VICTORY:
            good.add(first); return True
        if d >= depth or state.outcome != sts.Outcome.UNDECIDED:
            return False
        _, amap, _ = legal_map(state)
        for i, a in amap.items():
            if a.action_type != AT.CARD:
                continue
            c = state.clone(); a.execute(c)
            rec(c, first if first is not None else i, d + 1)
        return False

    _, amap, _ = legal_map(bc)
    for i, a in amap.items():
        if a.action_type != AT.CARD:
            continue
        c = bc.clone(); a.execute(c)
        rec(c, i, 1)
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--dir", default="", help="results folder: probe a trained agent instead of MCTS")
    ap.add_argument("--run", default="")
    ap.add_argument("--pool", default="eval")
    a = ap.parse_args()
    if a.dir:                                   # probe a trained agent
        sys.path.insert(0, str(Path(__file__).parent))
        import evaluate as E, json as J
        base = E.ROOT / "results" / "first_combat" / a.dir
        cfg = J.load(open(base / a.run / "config.json"))
        spec = {"kind": "scorer" if cfg.get("head") == "scorer" else "torch",
                "path": str(sorted((base / a.run).glob("ckpt_*"))[-1]), "algo": cfg["algo"],
                "encoding": cfg.get("encoding", "onehot"), "run": a.run, "key": a.run}
        env = E.get_env((cfg.get("deck", "random"), cfg.get("encounter")), spec["encoding"],
                        cfg.get("extra_cards", 0), a.pool)
        pol = E.load_policy(spec, env)
        name = a.run
    else:
        env = CombatEnv(4); pol = MCTSOraclePolicy(a.sims); name = f"MCTS @{a.sims}"
    occasions = missed = 0
    for ep in range(a.n):
        env.reset(options={"episode_seed": ep})
        done = tr = False
        while not (done or tr):
            good = lethal_first_actions(env.bc)
            act = pol.act(env)
            if good:
                occasions += 1
                missed += act not in good
            _, _, done, tr, _ = env.step(act)
    print(f"{name} over {a.n} combats: {occasions} situations with a lethal move available, "
          f"{missed} missed ({missed / max(1, occasions):.0%})")


if __name__ == "__main__":
    main()
