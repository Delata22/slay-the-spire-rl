"""NOMINAL effects of a card: what is printed on it, out of context.

The `combat_env.probe` probe measures what the card does *now*: Strike shows 9 damage if the
target is Vulnerable, 6 otherwise. Here we want the opposite — the reference value, "Strike = 6".

Method: measure the card in a NEUTRAL combat built for the purpose — a deck containing only
this card, a player with no status, a single enemy with no block or debuff. The result thus
depends on no context, and it is cached on disk (nominal_fx.json).

This is the card's description, obtained without a hand-written table and without touching the C++.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .combat_env import (sts, legal_map, probe, AT, D_EFF_PROBE, make_scenario_gc, _card,
                         N_CARD_TYPES)

_CACHE_PATH = os.path.join(os.path.dirname(__file__), "nominal_fx.json")


def _load_cache():
    try:
        with open(_CACHE_PATH) as f:
            return {k: np.array(v, dtype=np.float32) for k, v in json.load(f).items()}
    except (OSError, ValueError):      # missing, or being rewritten by another process
        return {}


_CACHE = _load_cache()
_REF_GC = None

D_NOMINAL = D_EFF_PROBE


def _resolve(name):
    """'BASH+' -> (BASH, upgraded)."""
    return (name[:-1], True) if name.endswith("+") else (name, False)


def _reference_gc():
    """A GameContext at the first combat, whose deck is emptied to hold the card under study."""
    global _REF_GC
    if _REF_GC is None:
        _REF_GC = make_scenario_gc(4)          # Jaw Worm, starting deck, no status
    return _REF_GC


def nominal(name: str, copies: int = 6, tries: int = 12) -> np.ndarray:
    """Effects vector of card `name` in a neutral context (cached)."""
    if name in _CACHE:
        return _CACHE[name]
    base, upgraded = _resolve(name)
    gc = _reference_gc()
    size = len(gc.deck)
    for _ in range(copies):
        c = _card(base)
        if upgraded:
            c.upgrade()
        gc.obtain_card(c)
    for idx in range(size - 1, -1, -1):        # remove the starting deck: only the card remains
        gc.remove_card(idx)
    vec = np.zeros(D_NOMINAL, dtype=np.float32)
    try:
        for t in range(tries):
            gc.seed = 7_000_000 + t
            bc = sts.BattleContext(); bc.init(gc)
            _, amap, _ = legal_map(bc)
            act = next((a for a in amap.values()
                        if a.action_type == AT.CARD and bc.hand[a.source_idx].name == bc.hand[0].name),
                       None)
            if act is not None:
                vec = np.asarray(probe(bc, act), dtype=np.float32)
                break
    finally:                                   # restore the starting deck
        for idx in range(len(gc.deck) - 1, -1, -1):
            gc.remove_card(idx)
        for c in ("STRIKE_RED",) * 5 + ("DEFEND_RED",) * 4 + ("BASH",):
            gc.obtain_card(_card(c))
    _CACHE[name] = vec
    # atomic write: parallel workers share this cache file
    merged = {**_load_cache(), **_CACHE}
    tmp = f"{_CACHE_PATH}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump({k: np.asarray(v).tolist() for k, v in merged.items()}, f)
    os.replace(tmp, _CACHE_PATH)
    return vec


def nominal_for_card(card) -> np.ndarray:
    """Same, from an in-game card (handles upgrades)."""
    name = str(card.id).split(".")[-1]
    return nominal(name + ("+" if card.upgraded else ""))


