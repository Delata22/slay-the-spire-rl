"""BLIND optimum: the best policy that does not see the draw order.

`optimal.py` solves each combat knowing the future (an unreachable bound).
Here we impose the POMDP constraint: the policy must play the SAME action in all worlds it
cannot tell apart. This is the true bound for a blind agent.

Building the worlds — the tricky part. Two arbitrary combat seeds give different monster HP
and opening hands, which the agent SEES: these are not indistinguishable worlds. So we draw
many seeds and group them by identical initial observation; within a group, worlds differ
only by the hidden order of the rest of the draw pile. That is exactly the uncertainty the
agent faces.

Algorithm: expectimax over beliefs, with
  - shortcut: a belief reduced to a single world is worth the clairvoyant optimum (solved fast);
  - branch and bound: cut any branch whose HP already lost reaches the best known result
    (HP lost can only increase).
"""
from __future__ import annotations

import sys
from collections import defaultdict

from .combat_env import sts, legal_map, AT, CombatEnv


def obs_key(bc):
    """What the agent sees. Never the pile order, only their size (the contents are inferred)."""
    if bc.outcome != sts.Outcome.UNDECIDED:
        return ("end", int(bc.outcome == sts.Outcome.PLAYER_VICTORY), bc.player.cur_hp)
    p = bc.player
    mons = tuple((m.cur_hp, m.block, m.vulnerable, m.weak, m.strength, m.intent)
                 for m in bc.monsters if m.alive)
    hand = tuple(sorted(c.name + ("+" if c.upgraded else "") for c in bc.hand))
    return (p.cur_hp, p.block, p.energy, p.strength, p.dexterity, bc.turn,
            mons, hand, len(bc.draw_pile), len(bc.discard_pile), len(bc.exhaust_pile))


def action_key(bc, a):
    """An action as the agent sees it: the card and the target, not the hand index."""
    if a.action_type == AT.CARD:
        c = bc.hand[a.source_idx]
        return ("card", c.name + ("+" if c.upgraded else ""), a.target_idx)
    return ("end_turn",)


def clairvoyant(bc, hp0, bound=10 ** 9):
    """HP lost under perfect play in THIS world (the future is known). Branch and bound."""
    best = [bound]

    def rec(st):
        if st.outcome != sts.Outcome.UNDECIDED:
            lost = hp0 - st.player.cur_hp if st.outcome == sts.Outcome.PLAYER_VICTORY else hp0
            best[0] = min(best[0], lost)
            return
        if hp0 - st.player.cur_hp >= best[0]:
            return
        _, amap, _ = legal_map(st)
        seen = set()
        for i, a in sorted(amap.items()):
            k = action_key(st, a)
            if k in seen:
                continue
            seen.add(k)
            c = st.clone(); a.execute(c); rec(c)

    rec(bc)
    return best[0]


class BlindSolver:
    def __init__(self, node_cap=400_000):
        self.nodes = 0
        self.cap = node_cap
        self.capped = False

    def value(self, worlds, hp0, cutoff=float("inf")):
        """Mean HP lost under the best policy shared by all `worlds`.

        `cutoff`: branch-and-bound limit. If no action does better, return +inf
        ("no better than the bound"), never the bound itself, or the parent would underestimate.
        """
        self.nodes += 1
        if self.nodes > self.cap:
            self.capped = True
            return float("inf")

        first = worlds[0]
        if first.outcome != sts.Outcome.UNDECIDED:
            return sum((hp0 - w.player.cur_hp) if w.outcome == sts.Outcome.PLAYER_VICTORY else hp0
                       for w in worlds) / len(worlds)
        if len(worlds) == 1:                                   # nothing hidden left to average over
            return clairvoyant(first, hp0)

        wmaps = [legal_map(w)[1] for w in worlds]
        actions = list(dict.fromkeys(action_key(first, a) for a in wmaps[0].values()))

        best = float("inf")
        feasible = False
        for ak in actions:
            groups = defaultdict(list)
            ok = True
            for w, wmap in zip(worlds, wmaps):
                wi = next((j for j, b in wmap.items() if action_key(w, b) == ak), None)
                if wi is None:                                 # illegal action here: inconsistent belief
                    ok = False; break
                c = w.clone(); wmap[wi].execute(c)
                groups[obs_key(c)].append(c)
            if not ok:
                continue
            feasible = True
            lim = min(best, cutoff)
            total = sum(len(g) for g in groups.values())
            acc = 0.0
            for g in sorted(groups.values(), key=len, reverse=True):
                w_g = len(g) / total
                # the most this group can cost before exceeding `lim`
                child = self.value(g, hp0, cutoff=(lim - acc) / w_g)
                acc += w_g * child
                if acc >= lim:                                 # already worse than what we can do
                    acc = float("inf"); break
            best = min(best, acc)
        if not feasible:                                       # no playable action
            return sum(hp0 - w.player.cur_hp for w in worlds) / len(worlds)
        return best


def build_beliefs(scenario, n_seeds, min_worlds=2):
    """Group seeds by initial observation: worlds within a group are indistinguishable."""
    env = CombatEnv(scenario)
    groups = defaultdict(list)
    for s in range(n_seeds):
        env.reset(options={"episode_seed": s})
        groups[obs_key(env.bc)].append(env.bc.clone())
    return [g for g in groups.values() if len(g) >= min_worlds], env


def blind_optimal(scenario=4, n_seeds=400, max_worlds=8, node_cap=400_000, verbose=True):
    """-> (mean HP lost under the optimal blind policy, clairvoyant HP lost, details)."""
    beliefs, env = build_beliefs(scenario, n_seeds)
    beliefs.sort(key=len, reverse=True)
    rows = []
    for g in beliefs:
        worlds = g[:max_worlds]
        hp0 = worlds[0].player.cur_hp
        solver = BlindSolver(node_cap)
        v = solver.value([w.clone() for w in worlds], hp0)
        clair = sum(clairvoyant(w.clone(), hp0) for w in worlds) / len(worlds)
        rows.append((len(worlds), v, clair, solver.nodes, solver.capped))
        if verbose:
            print(f"  belief of {len(worlds)} worlds: blind {v:5.2f} | clairvoyant {clair:5.2f} | "
                  f"{solver.nodes:>7} nodes" + ("  (CAPPED)" if solver.capped else ""), flush=True)
    n = sum(r[0] for r in rows)
    blind = sum(r[0] * r[1] for r in rows) / n
    clair = sum(r[0] * r[2] for r in rows) / n
    return blind, clair, rows


if __name__ == "__main__":
    import time
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    max_worlds = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    t0 = time.time()
    blind, clair, rows = blind_optimal(4, n_seeds, max_worlds)
    print(f"\n{len(rows)} beliefs | BLIND optimum {blind:.2f} HP | clairvoyant optimum "
          f"{clair:.2f} HP on the same worlds | price of ignorance {blind - clair:+.2f} HP "
          f"| {time.time() - t0:.0f} s")
