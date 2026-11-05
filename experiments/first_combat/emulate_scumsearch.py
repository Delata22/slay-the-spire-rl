#!/usr/bin/env python3
"""Re-implements the ScumSearch procedure in Python to understand its choices.

ScumSearch = random playouts to the end of the combat, MAX backup, and it returns the first
action of the best sequence found (`bestActionSequence.front()`), with
  eval_win  = 100 * (35 + HP + 4*potions - 0.01*turn)
  eval_loss = heuristic (damage dealt, monsters alive, turns...)

  python emulate_scumsearch.py [--seed 306] [--turn 3] [--playouts 3000]
"""
import argparse, random, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from stsenv.combat_env import CombatEnv, sts, legal_map  # noqa: E402
from stsenv.policies import MCTSOraclePolicy  # noqa: E402


def evaluate_end(bc):
    if bc.outcome == sts.Outcome.PLAYER_VICTORY:
        return 100 * (35 + bc.player.cur_hp - bc.turn * 0.01)
    return -1e6 + bc.player.cur_hp          # loss: very bad


def playout(bc, rng):
    """Random playout to the end; returns (evaluation, description of the first action)."""
    st = bc.clone(); first = None
    for _ in range(200):
        if st.outcome != sts.Outcome.UNDECIDED:
            break
        _, amap, _ = legal_map(st)
        if not amap:
            break
        i = rng.choice(sorted(amap))
        if first is None:
            first = amap[i].desc(st).strip()
        amap[i].execute(st)
    return evaluate_end(st), first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=306)
    ap.add_argument("--turn", type=int, default=3)
    ap.add_argument("--playouts", type=int, default=3000)
    a = ap.parse_args()

    env = CombatEnv(4)
    pol = MCTSOraclePolicy(2000)
    env.reset(options={"episode_seed": a.seed})
    while env.bc.turn + 1 < a.turn and env.bc.outcome == sts.Outcome.UNDECIDED:
        env.step(pol.act(env))
    bc = env.bc
    m = bc.monsters[0]
    print(f"seed {a.seed}, turn {bc.turn + 1} | player {bc.player.cur_hp} HP, {bc.player.energy} energy | "
          f"{m.name} {m.cur_hp} HP blk{m.block} vuln{m.vulnerable} -> {m.intent}")

    rng = random.Random(0)
    best = (-1e18, None); by_first = defaultdict(lambda: [0, -1e18])
    for _ in range(a.playouts):
        ev, first = playout(bc, rng)
        if first is None:
            continue
        by_first[first][0] += 1
        by_first[first][1] = max(by_first[first][1], ev)
        if ev > best[0]:
            best = (ev, first)

    print(f"\n{a.playouts} random playouts; best evaluation found: {best[0]:.0f} "
          f"via a sequence starting with '{best[1]}'\n")
    print(f"{'first action':52} {'playouts':>9} {'best eval':>15}")
    for k, (n, mx) in sorted(by_first.items(), key=lambda kv: -kv[1][1]):
        print(f"{k:52} {n:>9} {mx:>15.0f}")
    print(f"\nfor comparison: mcts_recommend @2000 picks '{sts.mcts_recommend(bc, 2000).desc(bc).strip()}'")


if __name__ == "__main__":
    main()
