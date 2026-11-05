#!/usr/bin/env python3
"""Why does MCTS miss a lethal move? Diagnosis on one specific state.

  python diagnose_mcts.py [--seed 306] [--turn 3]

Replays the MCTS policy up to the start of the requested turn, then:
  - brute-forces every playable card sequence for that turn (via clones) and reports which ones kill;
  - queries mcts_recommend with increasing budgets.
"""
import argparse, itertools, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from stsenv.combat_env import CombatEnv, sts, AT, legal_map  # noqa: E402
from stsenv.policies import MCTSOraclePolicy  # noqa: E402


def state_line(bc):
    m = bc.monsters[0]
    return (f"HP {bc.player.cur_hp}/{bc.player.max_hp} block {bc.player.block} energy {bc.player.energy} | "
            f"{m.name} {m.cur_hp}/{m.max_hp} blk{m.block} vuln{m.vulnerable} str{m.strength} -> {m.intent} | "
            f"hand: {', '.join(c.name + ('+' if c.upgraded else '') for c in bc.hand)}")


def best_sequences(bc, depth=4):
    """Brute force: every action sequence for this turn (excluding end turn), via clones."""
    out = []

    def rec(state, seq):
        if len(seq) >= depth or state.outcome != sts.Outcome.UNDECIDED:
            out.append((seq, state)); return
        mask, amap, _ = legal_map(state)
        plays = [(i, a) for i, a in amap.items() if a.action_type == AT.CARD]
        if not plays:
            out.append((seq, state)); return
        out.append((seq, state))
        for i, a in plays:
            c = state.clone()
            name = state.hand[a.source_idx].name + ("+" if state.hand[a.source_idx].upgraded else "")
            a.execute(c)
            rec(c, seq + [name])

    rec(bc.clone(), [])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=306)
    ap.add_argument("--turn", type=int, default=3)
    ap.add_argument("--sims", default="200,2000,20000,100000")
    a = ap.parse_args()

    env = CombatEnv(4)
    pol = MCTSOraclePolicy(2000)
    env.reset(options={"episode_seed": a.seed})
    while env.bc.turn + 1 < a.turn and env.bc.outcome == sts.Outcome.UNDECIDED:
        env.step(pol.act(env))
    bc = env.bc
    print(f"seed {a.seed}, start of turn {bc.turn + 1}")
    print(state_line(bc), "\n")

    seqs = best_sequences(bc)
    kills = [(s, st) for s, st in seqs if st.outcome == sts.Outcome.PLAYER_VICTORY]
    print(f"brute force: {len(seqs)} sequences explored this turn, {len(kills)} kill the monster")
    for s, st in sorted(kills, key=len)[:4]:
        print(f"   LETHAL: {' + '.join(s)}")
    if not kills:
        best = min(seqs, key=lambda x: x[1].monsters[0].cur_hp)
        print(f"   no lethal move; best leaves {best[1].monsters[0].cur_hp} HP via {' + '.join(best[0])}")
    print()

    for n in (int(x) for x in a.sims.split(",")):
        picks = {}
        for _ in range(3):
            act = sts.mcts_recommend(bc, n)
            picks[act.desc(bc).strip()] = picks.get(act.desc(bc).strip(), 0) + 1
        print(f"mcts_recommend @{n:>6} : " + " | ".join(f"{k} x{v}" for k, v in picks.items()))


if __name__ == "__main__":
    main()
