"""V2 — "scorer" head: f(state, action descriptor) -> score, instead of a masked Discrete(N).

Key difference from V0/V1: the action space is no longer a 116-slot array where each slot has
a fixed meaning ("card in hand slot 3 on target 0"). Each legal action is described by a
VECTOR, and the network scores the candidates one by one. Three consequences:

- the number of actions can vary freely (bigger hand, more enemies, potions);
- a never-seen action is scored by what it does, not by its position in an array;
- the hand slot stops carrying information: two identical Strikes get the same score.

Action descriptor (52 dimensions, no card identity):
  action type (4)         card / end turn / potion / select
  static (9)              cost, targeted, upgraded, card type
  measured effects (31)   damage, block, HP, energy, statuses... (combat_env probe)
  target (8)              target's HP, block, Vulnerable, Weak, intent

END TURN is probed like the others: its descriptor contains what the enemy turn costs.
"""
from __future__ import annotations

import numpy as np

from .combat_env import (CombatEnv, sts, AT, legal_map, probe, status_value, monsters_hp,
                         _pile_summary, D_PLAYER, D_PILE_SUM, D_MON, D_EFF_PROBE,
                         N_MON, N_CARD_TYPES, PSTATUS, MSTATUS)
from .nominal import nominal_for_card, D_NOMINAL

# Three ways to describe a card's effects:
#   "measured"  what it does NOW        (Strike = 9 on a Vulnerable target) — contextual
#   "nominal"   what is printed on it   (Strike = 6, always)                — context-free
#   "both"      both concatenated: the reference value AND the current value
FX_MODES = {"measured": D_EFF_PROBE, "nominal": D_NOMINAL, "both": D_EFF_PROBE + D_NOMINAL}


# Cards whose effect depends on the combat RNG. The "measured" probe turns them into ORACLES: it runs
# the card in a clone that shares the RNG state, so it predicts the exact roll (verified:
# Sword Boomerang reveals the actual split of its 3 hits; Havoc reveals the top card of the
# draw pile). With `leak="mask"`, their measurement is replaced by the nominal value and a
# "random effect" flag is raised: the agent knows there is uncertainty, without knowing the outcome.
RANDOM_EFFECT_CARDS = {
    "SWORD_BOOMERANG",   # 3 hits on random targets
    "HAVOC",             # plays the top card of the draw pile -> leaks draw order
    "TRUE_GRIT",         # exhausts a random card from hand
    "WILD_STRIKE",       # shuffles a Wound into a random position
    "RECKLESS_CHARGE",   # same with a Dazed
    "DROPKICK",          # conditional draw
}


def is_random_card(bc, a):
    if a.action_type != AT.CARD:
        return False
    return str(bc.hand[a.source_idx].id).split(".")[-1] in RANDOM_EFFECT_CARDS


def fx_vector(bc, a, mode, leak="allow"):
    """Effects part of an action descriptor, according to the chosen mode."""
    is_random = leak == "mask" and is_random_card(bc, a)
    measure = lambda: (nominal_for_card(bc.hand[a.source_idx]) if is_random
                       else np.asarray(probe(bc, a), dtype=np.float32))
    if a.action_type != AT.CARD:          # end turn, potion, select: no nominal value
        m = measure()
        return m if mode == "measured" else (np.zeros(D_NOMINAL, np.float32) if mode == "nominal"
                                             else np.concatenate([m, np.zeros(D_NOMINAL, np.float32)]))
    if mode == "measured":
        return measure()
    nom = nominal_for_card(bc.hand[a.source_idx])
    return nom if mode == "nominal" else np.concatenate([measure(), nom])

# --- V3: the draw pile as a multiset ------------------------------------------------------
# A card in the draw pile only exposes its static part (cost, type, upgrade, targeting): it cannot
# be probed without being played. It is therefore paired with REMEMBERED effects, recorded when the
# agent played it earlier (cache held by the environment). This is what a human player knows: they
# know Strike because they have played it before. A flag says whether the effects are known.
D_CARD_STATIC = 3 + N_CARD_TYPES
D_CARD = D_CARD_STATIC + D_EFF_PROBE
D_PILE_SET = 2 + 2 * D_CARD          # size, known fraction, mean and max of descriptors

D_KIND = 5                              # + a "random effect" flag
D_STATIC = 3 + N_CARD_TYPES
D_TARGET = 8
D_ACT = D_KIND + D_STATIC + D_EFF_PROBE + D_TARGET
D_STATE_BASE = D_PLAYER + 3 * D_PILE_SUM + N_MON * D_MON
D_STATE = D_STATE_BASE + D_ACT              # + candidate mean: context of the alternatives
D_STATE_BASE_SET = D_PLAYER + 3 * D_PILE_SET + N_MON * D_MON
D_STATE_SET = D_STATE_BASE_SET + D_ACT      # V3: piles described card by card, then aggregated


def state_vector(bc):
    """The state WITHOUT the hand: what is playable is described by the candidates."""
    x = np.zeros(D_STATE_BASE, dtype=np.float32)
    p = bc.player
    x[0:8] = (p.cur_hp / max(1, p.max_hp), p.max_hp / 100, p.block / 50, p.energy / 5,
              p.strength / 10, p.dexterity / 10, bc.turn / 20, len(bc.hand) / 10)
    for k, s in enumerate(PSTATUS):
        x[8 + k] = min(status_value(p, s), 20) / 10
    o = D_PLAYER
    for pile in (bc.draw_pile, bc.discard_pile, bc.exhaust_pile):
        _pile_summary(x, o, pile); o += D_PILE_SUM
    for j, m in enumerate(bc.monsters[:N_MON]):
        b = o + j * D_MON
        if not m.alive:
            x[b] = 1; continue
        di = m.intent_damage(bc)
        x[b:b + 12] = (1, 1, m.targetable, m.cur_hp / max(1, m.max_hp), m.max_hp / 100, m.block / 30,
                       m.strength / 10, m.vulnerable / 5, m.weak / 5, m.is_attacking,
                       di.damage / 30 if m.is_attacking else 0,
                       di.attack_count / 5 if m.is_attacking else 0)
        for k, s in enumerate(MSTATUS):
            x[b + 12 + k] = min(status_value(m, s), 20) / 10
    return x


def card_key(c):
    return (int(c.id), bool(c.upgraded))


def card_static(c):
    d = np.zeros(D_CARD_STATIC, dtype=np.float32)
    d[0] = c.cost / 3; d[1] = c.requires_target; d[2] = c.upgraded
    d[3 + int(c.type)] = 1
    return d


def pile_multiset(pile, fx_cache, row_cache=None):
    """Aggregate a multiset of cards: size, fraction with known effects, mean and max.

    ORDER never enters the representation — it is precisely the hidden information."""
    out = np.zeros(D_PILE_SET, dtype=np.float32)
    n = len(pile)
    out[0] = n / 10
    if n == 0:
        return out
    rows, known = [], 0
    for c in pile:
        k = card_key(c)
        row = row_cache.get(k) if row_cache is not None else None
        if row is None:
            fx = fx_cache.get(k)
            row = np.concatenate([card_static(c),
                                  fx if fx is not None else np.zeros(D_EFF_PROBE, dtype=np.float32)])
            if row_cache is not None:
                row_cache[k] = row
        known += fx_cache.get(k) is not None
        rows.append(row)
    m = np.stack(rows)
    out[1] = known / n
    out[2:2 + D_CARD] = m.mean(0)
    out[2 + D_CARD:] = m.max(0)
    return out


def state_vector_multiset(bc, fx_cache, row_cache=None):
    """V3 state: like state_vector, but the three piles are described card by card."""
    x = np.zeros(D_STATE_BASE_SET, dtype=np.float32)
    p = bc.player
    x[0:8] = (p.cur_hp / max(1, p.max_hp), p.max_hp / 100, p.block / 50, p.energy / 5,
              p.strength / 10, p.dexterity / 10, bc.turn / 20, len(bc.hand) / 10)
    for k, s_ in enumerate(PSTATUS):
        x[8 + k] = min(status_value(p, s_), 20) / 10
    o = D_PLAYER
    for pile in (bc.draw_pile, bc.discard_pile, bc.exhaust_pile):
        x[o:o + D_PILE_SET] = pile_multiset(pile, fx_cache, row_cache); o += D_PILE_SET
    for j, m in enumerate(bc.monsters[:N_MON]):
        b = o + j * D_MON
        if not m.alive:
            x[b] = 1; continue
        di = m.intent_damage(bc)
        x[b:b + 12] = (1, 1, m.targetable, m.cur_hp / max(1, m.max_hp), m.max_hp / 100, m.block / 30,
                       m.strength / 10, m.vulnerable / 5, m.weak / 5, m.is_attacking,
                       di.damage / 30 if m.is_attacking else 0,
                       di.attack_count / 5 if m.is_attacking else 0)
        for k, s_ in enumerate(MSTATUS):
            x[b + 12 + k] = min(status_value(m, s_), 20) / 10
    return x


def action_descriptor(bc, a, fx_mode="measured", leak="allow"):
    d = np.zeros(D_KIND + D_STATIC + FX_MODES[fx_mode] + D_TARGET, dtype=np.float32)
    kind = {AT.CARD: 0, AT.END_TURN: 1, AT.POTION: 2}.get(a.action_type, 3)
    d[kind] = 1
    d[4] = is_random_card(bc, a)          # effect depends on the RNG: the shown value is uncertain
    o = D_KIND
    if a.action_type == AT.CARD:
        c = bc.hand[a.source_idx]
        d[o] = c.cost_for_turn / 3; d[o + 1] = c.requires_target; d[o + 2] = c.upgraded
        d[o + 3 + int(c.type)] = 1
    o += D_STATIC
    fx = fx_vector(bc, a, fx_mode, leak)
    d[o:o + len(fx)] = fx
    o += len(fx)
    mons = list(bc.monsters)
    if a.action_type in (AT.CARD, AT.POTION) and a.target_idx < len(mons):
        m = mons[a.target_idx]
        if m.alive:
            di = m.intent_damage(bc)
            d[o:o + 8] = (1, m.cur_hp / max(1, m.max_hp), m.max_hp / 100, m.block / 30,
                          m.vulnerable / 5, m.weak / 5, m.is_attacking,
                          di.damage / 30 if m.is_attacking else 0)
    return d


class ScorerCombatEnv(CombatEnv):
    """Same dynamics as CombatEnv, but exposes (state, candidate descriptors)."""

    def __init__(self, *args, piles="summary", fx="measured", leak="allow", **kwargs):
        # "scorer" is not a state encoding: the state is produced by state_vector()
        if kwargs.get("encoding") is None or str(kwargs.get("encoding")).startswith("scorer"):
            kwargs["encoding"] = "effects"      # the state comes from state_vector(), not ENCODERS
        super().__init__(*args, **kwargs)
        self.piles = piles                      # "summary" (V2) or "multiset" (V3)
        self.fx_mode = fx                       # "measured" | "nominal" | "both"
        self.leak = leak                        # "allow" (legacy) | "mask" (no RNG oracle)
        self.act_dim_ = D_KIND + D_STATIC + FX_MODES[fx] + D_TARGET
        self.fx_cache = {}                      # remembered effects per already-played card
        self._row_cache = {}                    # assembled descriptor, invalidated when effects arrive
        base = D_STATE_BASE_SET if piles == "multiset" else D_STATE_BASE
        self.state_dim = base + self.act_dim_
        self.act_dim = self.act_dim_

    def _prefill_piles(self):
        """In nominal mode, the effects of pile cards are known without having played them."""
        if self.fx_mode == "measured":
            return
        for pile in (self.bc.draw_pile, self.bc.discard_pile, self.bc.hand):
            for c in pile:
                k = card_key(c)
                if k not in self.fx_cache:
                    self.fx_cache[k] = nominal_for_card(c); self._row_cache.pop(k, None)

    def candidates(self):
        """-> (state [D_STATE], descriptors [K, D_ACT], matching action indices).

        Two identical cards in hand give the same descriptor: probed only once."""
        bc = self.bc
        idxs = sorted(self.amap)
        cache, rows = {}, []
        for i in idxs:
            a = self.amap[i]
            if a.action_type == AT.CARD:
                c = bc.hand[a.source_idx]
                key = (0, int(c.id), c.upgraded, c.cost_for_turn, a.target_idx)
            else:
                key = (int(a.action_type), a.source_idx, a.target_idx, a.select_idx, 0)
            if key not in cache:
                cache[key] = action_descriptor(bc, a, self.fx_mode, self.leak)
                if a.action_type == AT.CARD:    # remember what this card does, for the piles
                    c = bc.hand[a.source_idx]
                    ck = card_key(c)
                    if ck not in self.fx_cache:
                        # for the piles, always keep an effects vector of size D_EFF_PROBE
                        self.fx_cache[ck] = (nominal_for_card(c) if self.fx_mode == "nominal"
                                             else cache[key][D_KIND + D_STATIC:
                                                             D_KIND + D_STATIC + D_EFF_PROBE])
                        self._row_cache.pop(ck, None)
            rows.append(cache[key])
        descs = np.stack(rows) if rows else np.zeros((0, self.act_dim_), dtype=np.float32)
        if self.piles == "multiset":
            self._prefill_piles()
        base = (state_vector_multiset(bc, self.fx_cache, self._row_cache)
                if self.piles == "multiset" else state_vector(bc))
        mean = descs.mean(0) if len(descs) else np.zeros(self.act_dim_, dtype=np.float32)
        return np.concatenate([base, mean]).astype(np.float32), descs, idxs

    def obs(self):                                   # compatibility: state only
        return self.candidates()[0]
