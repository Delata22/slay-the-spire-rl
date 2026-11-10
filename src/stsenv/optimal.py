"""Brute-force optimal play with the draw order KNOWN.

This is the scenario's absolute upper bound: a blind agent cannot reach it, but it shows
how much there really is left to gain (the "oracle" MCTS is far from it).

Depth-first search over action sequences until the end of combat, with two prunings:
identical cards in hand are merged, and any branch that has already lost as much HP as
the best known result is cut. ~3,000 nodes and 0.1 s per combat on this scenario.
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np

from .combat_env import CombatEnv, sts, legal_map, AT

_ENV = {}


def _env(spec):
    """spec = scenario number, or (scenario, extra_cards, pool)."""
    if spec not in _ENV:
        s, extra, pool, *rest = spec if isinstance(spec, tuple) else (spec, 0, "eval")
        tps = rest[0] if rest else None
        if isinstance(s, tuple):                      # (deck archetype, encounter)
            deck, enc = s
            _ENV[spec] = CombatEnv(4, extra_cards=extra, pool=pool, deck=deck, encounter=enc,
                                   train_pool_size=tps)
        else:
            _ENV[spec] = CombatEnv(s, extra_cards=extra, pool=pool)
    return _ENV[spec]


def solve(args):
    """-> (seed, HP lost under perfect play). A loss counts as losing all HP."""
    scenario, seed = args
    env = _env(scenario)
    env.reset(options={"episode_seed": seed})
    best = [10 ** 9]

    def rec(st, hp0):
        if st.outcome != sts.Outcome.UNDECIDED:
            lost = hp0 - st.player.cur_hp if st.outcome == sts.Outcome.PLAYER_VICTORY else hp0
            best[0] = min(best[0], lost)
            return
        # Branch-and-bound on HP lost so far. Only admissible if HP never goes back up: with a healing
        # card (Reaper) a line that loses HP first and heals later is cut. Without the cut the search
        # does not finish (> 10 min per combat), so the reported reference is the best of this search
        # and of every evaluated policy (experiments/first_combat/recompute_optimal.py).
        if hp0 - st.player.cur_hp >= best[0]:
            return
        _, amap, _ = legal_map(st)
        seen = set()
        for i, a in sorted(amap.items()):
            # Copies of the same card are merged (otherwise the search is 20-100x slower). This is an
            # approximation: for a player who knows the draw order, WHICH copy goes to the discard pile
            # changes the next reshuffle, so the result is "best play up to card-copy symmetry".
            # Every non-card action (end turn, each potion, each card-selection option) is distinct;
            # earlier versions keyed them all as "end", so potions and selections were never explored.
            if a.action_type == AT.CARD:
                c = st.hand[a.source_idx]
                key = ("card", int(c.id), c.upgraded, c.cost_for_turn, a.target_idx)
            else:
                key = ("other", i)
            if key in seen:
                continue
            seen.add(key)
            c = st.clone(); a.execute(c); rec(c, hp0)

    rec(env.bc.clone(), env.bc.player.cur_hp)
    return seed, best[0]


def optimal_hp_lost(scenario, seeds, workers=16, cache: Path | None = None):
    """Array of HP lost under perfect play, aligned with `seeds`. Cached if `cache` is given."""
    if cache is not None and Path(cache).exists():
        arr = np.load(cache)
        if len(arr) >= len(seeds):
            return arr[:len(seeds)]
    with mp.get_context("fork").Pool(workers) as pool:
        res = dict(pool.map(solve, [(scenario, s) for s in seeds]))
    arr = np.array([res[s] for s in seeds], dtype=float)
    if cache is not None:
        np.save(cache, arr)
    return arr
