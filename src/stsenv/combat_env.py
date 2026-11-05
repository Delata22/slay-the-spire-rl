"""Single-combat environment on top of sts_lightspeed.

Scenario = (scenario seed) -> the run is played by the built-in heuristic up to the
first combat: deck, relics, HP and encounter are then FROZEN.
Episode  = that combat replayed with a different combat RNG seed (gc.seed is overwritten
before BattleContext.init, which derives aiRng/shuffleRng/monsterHpRng from gc.seed + floor):
draw order and monster AI choices vary, nothing else does.
=> the policy cannot memorise a draw order (a POMDP, as in the real game).

Action space: masked Discrete(N_ACTIONS), indexed by slot:
  [0, 50)    CARD      hand h (0..9) x target t (0..4); untargeted card -> t = 0 only
  50         END_TURN
  [51, 76)   POTION    slot p (0..4) x target t (0..4); potion discard ignored
  [76, 116)  SINGLE_CARD_SELECT select_idx (0..39)
Unrepresentable actions (MULTI_CARD_SELECT, select_idx >= 40) are played
automatically (first legal one) and do not count as agent decisions.
"""
from __future__ import annotations

import json, os, sys
import numpy as np

sys.path.insert(0, os.environ.get(
    "STS_LIGHTSPEED_BUILD",
    os.path.join(os.path.dirname(__file__), "..", "..", "external", "sts_lightspeed", "build")))
import slaythespire as sts  # noqa: E402

import gymnasium as gym  # noqa: E402

AT = sts.SearchActionType
N_HAND, N_MON, N_POT, N_SEL = 10, 5, 5, 40
A_CARD, A_END, A_POT, A_SEL = 0, 50, 51, 76
N_ACTIONS = A_SEL + N_SEL
N_CARD_IDS = len(sts.CardId.__members__)
N_CARD_TYPES = len(sts.CardType.__members__)
PSTATUS = list(sts.PlayerStatus.__members__.values())
MSTATUS = list(sts.MonsterStatus.__members__.values())
MAX_STEPS = 300

# --- observation sizes ---
D_PLAYER = 8 + len(PSTATUS)
D_SLOT = 4 + N_CARD_TYPES + N_CARD_IDS              # present, cost, targeted?, upgrade, type, id
D_PILES = 3 * N_CARD_IDS                             # draw / discard / exhaust contents (not the order)
D_MON = 12 + len(MSTATUS)
OBS_DIM = D_PLAYER + N_HAND * D_SLOT + D_PILES + N_MON * D_MON


# ---------------------------------------------------------------- scenario
def make_scenario_gc(scenario_seed: int, ascension: int = 0):
    """GameContext stopped at the seed's first combat (non-combat played by the heuristic)."""
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, scenario_seed, ascension)
    ag = sts.Agent(); ag.simulation_count_base = 1; ag.pause_on_battle = True
    for _ in range(50):
        ag.playout(gc)
        if gc.screen_state == sts.ScreenState.BATTLE:
            return gc
    raise RuntimeError(f"seed {scenario_seed}: no combat reached")


def describe_scenario(gc) -> str:
    deck = [str(c.id).split(".")[-1] + ("+" if c.upgraded else "") for c in gc.deck]
    rel = [str(r.id).split(".")[-1] for r in gc.relics]
    return (f"seed={gc.seed} floor={gc.floor_num} encounter={str(gc.encounter).split('.')[-1]} "
            f"HP={gc.cur_hp}/{gc.max_hp}\n  deck ({len(deck)}): {', '.join(sorted(deck))}\n"
            f"  relics: {', '.join(rel)}")


def new_battle(template_gc, episode_seed: int, encounter=None):
    template_gc.seed = episode_seed
    bc = sts.BattleContext()
    if encounter is None:
        bc.init(template_gc)
    else:
        bc.init_encounter(template_gc, encounter)      # forced encounter, deck/HP unchanged
    return bc




# ---------------------------------------------------------------- variable deck
# The simulator's Ironclad pool (extracted from CardPools.h). Cards drawn each episode are
# added to the scenario deck and then removed, so the template is never modified.
_POOL_PATH = os.path.join(os.path.dirname(__file__), "ironclad_pool.json")
IRONCLAD_POOL = json.load(open(_POOL_PATH)) if os.path.exists(_POOL_PATH) else []
from .decks import ARCHETYPES, ENCOUNTERS, archetype_pools  # noqa: E402
_BAD_PATH = os.path.join(os.path.dirname(__file__), "ironclad_pool_unsafe.json")
UNSAFE_CARDS = set(json.load(open(_BAD_PATH))) if os.path.exists(_BAD_PATH) else set()


def card_pools(split=0.75, seed=0, exclude_unsafe=True, subset=None):
    """Split a pool into disjoint (training cards, eval cards).

    `subset` restricts to one archetype's pool. An 8-card archetype gives 6 seen / 2 unseen:
    generalisation is then tested on thematically related cards, not arbitrary ones."""
    source = IRONCLAD_POOL if subset is None else subset
    cards = [c for c in source if not (exclude_unsafe and c in UNSAFE_CARDS)]
    order = sorted(cards)
    np.random.default_rng(seed).shuffle(order)
    if not order:
        return [], []
    cut = max(1, int(round(len(order) * split))) if len(order) > 1 else 1
    return order[:cut], order[cut:] or order[:cut]


def _card(name):
    return sts.Card(sts.CardId.__members__[name])


class _ExtraCards:
    """Add cards to the GameContext while the combat is initialised, then remove them."""

    def __init__(self, gc, names):
        self.gc = gc; self.names = names; self.n = len(names)

    def __enter__(self):
        self.size_before = len(self.gc.deck)
        for name in self.names:
            self.gc.obtain_card(_card(name))
        return self.gc

    def __exit__(self, *exc):
        for idx in range(len(self.gc.deck) - 1, self.size_before - 1, -1):
            self.gc.remove_card(idx)
        assert len(self.gc.deck) == self.size_before


# ---------------------------------------------------------------- actions
def action_index(a, bc) -> int:
    t = a.action_type
    if t == AT.CARD:
        h, tgt = a.source_idx, a.target_idx
        if h >= N_HAND: return -1
        if not bc.hand[h].requires_target:
            tgt = 0 if tgt == 0 else -2          # -2: duplicate of an untargeted card
        return -2 if tgt == -2 else A_CARD + h * N_MON + tgt
    if t == AT.END_TURN:
        return A_END
    if t == AT.POTION:
        p, tgt = a.source_idx, a.target_idx
        return A_POT + p * N_MON + tgt if (p < N_POT and tgt < N_MON) else -2
    if t == AT.SINGLE_CARD_SELECT:
        return A_SEL + a.select_idx if a.select_idx < N_SEL else -1
    return -1


def legal_map(bc):
    """-> (mask[N_ACTIONS] bool, {index: SearchAction}, unrepresentable actions)."""
    mask = np.zeros(N_ACTIONS, dtype=bool); amap = {}; unmapped = []
    for a in sts.get_legal_actions(bc):
        i = action_index(a, bc)
        if i >= 0:
            if i not in amap:
                amap[i] = a; mask[i] = True
        elif i == -1:
            unmapped.append(a)
    return mask, amap, unmapped



# ---------------------------------------------------------------- observation
def encode(bc) -> np.ndarray:
    x = np.zeros(OBS_DIM, dtype=np.float32)
    p = bc.player
    x[0:8] = (p.cur_hp / max(1, p.max_hp), p.max_hp / 100, p.block / 50, p.energy / 5,
              p.strength / 10, p.dexterity / 10, bc.turn / 20, len(bc.hand) / 10)
    for k, s in enumerate(PSTATUS):
        x[8 + k] = min(status_value(p, s), 20) / 10
    o = D_PLAYER
    for h, c in enumerate(bc.hand[:N_HAND]):
        b = o + h * D_SLOT
        x[b] = 1; x[b + 1] = c.cost_for_turn / 3; x[b + 2] = c.requires_target; x[b + 3] = c.upgraded
        x[b + 4 + int(c.type)] = 1
        x[b + 4 + N_CARD_TYPES + int(c.id)] = 1
    o += N_HAND * D_SLOT
    for k, pile in enumerate((bc.draw_pile, bc.discard_pile, bc.exhaust_pile)):
        b = o + k * N_CARD_IDS
        for c in pile:
            x[b + int(c.id)] += 0.2
    o += D_PILES
    for j, m in enumerate(bc.monsters[:N_MON]):
        b = o + j * D_MON
        if not m.alive:
            x[b] = 1; continue
        di = m.intent_damage(bc)
        x[b:b + 12] = (1, 1, m.targetable, m.cur_hp / max(1, m.max_hp), m.max_hp / 100, m.block / 30,
                       m.strength / 10, m.vulnerable / 5, m.weak / 5, m.is_attacking,
                       di.damage / 30 if m.is_attacking else 0, di.attack_count / 5 if m.is_attacking else 0)
        for k, s in enumerate(MSTATUS):
            x[b + 12 + k] = min(status_value(m, s), 20) / 10
    return x


def monsters_hp(bc):
    return sum(max(0, m.cur_hp) for m in bc.monsters if m.alive)


def status_value(entity, status):
    """Value of a status, robust to binary statuses.

    Upstream bug (`Player::getStatusRuntime`, Player.cpp:61): presence is tested with a bit mask,
    but the value is read with `statusMap.at(s)`. For a status without a counter (Barricade),
    the bit is set with no map entry -> std::out_of_range. We then fall back to 0/1.
    """
    try:
        return entity.get_status(status)
    except Exception:
        try:
            return 1 if entity.has_status(status) else 0
        except Exception:
            return 0


# ------------------------------------------------- "measured effects" encoding
# Instead of a one-hot card identity, each card in hand is described by WHAT IT DOES:
# clone the combat (2 µs), play the card, record the deltas.
# -> no card identity in the observation, so it transfers to unseen cards.

D_EFF_STATIC = 5 + N_CARD_TYPES        # present, cost, playable, targeted, upgraded + type
D_EFF_PROBE = 14 + len(PSTATUS)      # 14 measurements + 1 delta per exposed player status
D_SLOT_EFF = D_EFF_STATIC + D_EFF_PROBE
D_PILE_SUM = 2 + N_CARD_TYPES          # size, mean cost, type distribution
OBS_DIM_EFF = D_PLAYER + N_HAND * D_SLOT_EFF + 3 * D_PILE_SUM + N_MON * D_MON


def _snap(bc):
    """Summary state used to measure a card's effect. The 17 exposed statuses are included:
    without them, a Power like Demon Form (which does set a status of value 2) produced an
    empty effect vector, although its effect IS observable."""
    p = bc.player
    return (p.cur_hp, p.block, p.energy, p.strength, p.dexterity,
            sum(m.cur_hp for m in bc.monsters if m.alive),
            sum(m.vulnerable for m in bc.monsters if m.alive),
            sum(m.weak for m in bc.monsters if m.alive),
            sum(1 for m in bc.monsters if m.alive),
            len(bc.hand), len(bc.discard_pile), len(bc.exhaust_pile),
            *(status_value(p, st) for st in PSTATUS))


def probe(bc, action):
    """Play `action` in a clone and return the measured-effects vector."""
    before = _snap(bc)
    c = bc.clone()
    try:
        action.execute(c)
    except Exception:
        return [0.0] * D_EFF_PROBE
    a = _snap(c)
    dmg = before[5] - a[5]
    return [dmg / 30, (before[8] - a[8]) / 2,                       # damage, monsters killed
            (a[1] - before[1]) / 20, (a[0] - before[0]) / 20,       # block gained, player HP
            (a[2] - before[2]) / 3,                                 # remaining energy (delta)
            (a[3] - before[3]) / 5, (a[4] - before[4]) / 5,         # strength, dexterity
            (a[6] - before[6]) / 5, (a[7] - before[7]) / 5,         # Vulnerable, Weak applied
            (a[9] - before[9] + 1) / 5,                             # cards drawn (net)
            (a[10] - before[10]) / 5, (a[11] - before[11]) / 2,     # discard, exhaust
            1.0 if dmg > 0 else 0.0, 1.0 if a[1] > before[1] else 0.0,
            # statuses gained or lost: this is where Powers become visible
            *[(a[12 + k] - before[12 + k]) / 5 for k in range(len(PSTATUS))]]


def _pile_summary(x, o, pile):
    n = len(pile)
    x[o] = n / 10
    if n:
        x[o + 1] = sum(c.cost_for_turn for c in pile) / (3 * n)
        for c in pile:
            x[o + 2 + int(c.type)] += 1 / n


def encode_effects(bc, amap):
    x = np.zeros(OBS_DIM_EFF, dtype=np.float32)
    p = bc.player
    x[0:8] = (p.cur_hp / max(1, p.max_hp), p.max_hp / 100, p.block / 50, p.energy / 5,
              p.strength / 10, p.dexterity / 10, bc.turn / 20, len(bc.hand) / 10)
    for k, s in enumerate(PSTATUS):
        x[8 + k] = min(status_value(p, s), 20) / 10
    o = D_PLAYER
    for h, c in enumerate(bc.hand[:N_HAND]):
        b = o + h * D_SLOT_EFF
        act = next((amap[i] for i in range(h * N_MON, (h + 1) * N_MON) if i in amap), None)
        x[b] = 1; x[b + 1] = c.cost_for_turn / 3; x[b + 2] = act is not None
        x[b + 3] = c.requires_target; x[b + 4] = c.upgraded
        x[b + 5 + int(c.type)] = 1
        if act is not None:
            x[b + D_EFF_STATIC:b + D_SLOT_EFF] = probe(bc, act)
    o += N_HAND * D_SLOT_EFF
    for pile in (bc.draw_pile, bc.discard_pile, bc.exhaust_pile):
        _pile_summary(x, o, pile); o += D_PILE_SUM
    for j, m in enumerate(bc.monsters[:N_MON]):
        b = o + j * D_MON
        if not m.alive:
            x[b] = 1; continue
        di = m.intent_damage(bc)
        x[b:b + 12] = (1, 1, m.targetable, m.cur_hp / max(1, m.max_hp), m.max_hp / 100, m.block / 30,
                       m.strength / 10, m.vulnerable / 5, m.weak / 5, m.is_attacking,
                       di.damage / 30 if m.is_attacking else 0, di.attack_count / 5 if m.is_attacking else 0)
        for k, s in enumerate(MSTATUS):
            x[b + 12 + k] = min(status_value(m, s), 20) / 10
    return x


# ------------------------------------------- COMPACT one-hot (per-experiment vocabulary)
# The full one-hot wastes 371 dimensions, ~360 of which are always zero. Here the vocabulary
# is limited to cards that can actually appear: starting deck + training pool
# + held-out eval cards (otherwise out of vocabulary) + common status/curse cards.
# One "unknown" slot covers everything else.
PARASITES = ["WOUND", "BURN", "DAZED", "SLIMED", "VOID", "BURN_PLUS"]


def build_vocab(template_gc, names):
    ids = {int(c.id) for c in template_gc.deck}
    for n in list(names) + PARASITES:
        m = sts.CardId.__members__.get(n)
        if m is not None:
            ids.add(int(m))
    order = sorted(ids)
    return {cid: i for i, cid in enumerate(order)}, len(order) + 1   # +1: "unknown" slot


def encode_onehot_vocab(bc, amap, vocab, vsize):
    d_slot = 4 + N_CARD_TYPES + vsize
    dim = D_PLAYER + N_HAND * d_slot + 3 * vsize + N_MON * D_MON
    x = np.zeros(dim, dtype=np.float32)
    p = bc.player
    x[0:8] = (p.cur_hp / max(1, p.max_hp), p.max_hp / 100, p.block / 50, p.energy / 5,
              p.strength / 10, p.dexterity / 10, bc.turn / 20, len(bc.hand) / 10)
    for k, st in enumerate(PSTATUS):
        x[8 + k] = min(status_value(p, st), 20) / 10
    o = D_PLAYER
    for h, c in enumerate(bc.hand[:N_HAND]):
        b = o + h * d_slot
        x[b] = 1; x[b + 1] = c.cost_for_turn / 3; x[b + 2] = c.requires_target; x[b + 3] = c.upgraded
        x[b + 4 + int(c.type)] = 1
        x[b + 4 + N_CARD_TYPES + vocab.get(int(c.id), vsize - 1)] = 1
    o += N_HAND * d_slot
    for k, pile in enumerate((bc.draw_pile, bc.discard_pile, bc.exhaust_pile)):
        b = o + k * vsize
        for c in pile:
            x[b + vocab.get(int(c.id), vsize - 1)] += 0.2
    o += 3 * vsize
    for j, m in enumerate(bc.monsters[:N_MON]):
        b = o + j * D_MON
        if not m.alive:
            x[b] = 1; continue
        di = m.intent_damage(bc)
        x[b:b + 12] = (1, 1, m.targetable, m.cur_hp / max(1, m.max_hp), m.max_hp / 100, m.block / 30,
                       m.strength / 10, m.vulnerable / 5, m.weak / 5, m.is_attacking,
                       di.damage / 30 if m.is_attacking else 0, di.attack_count / 5 if m.is_attacking else 0)
        for k, st in enumerate(MSTATUS):
            x[b + 12 + k] = min(status_value(m, st), 20) / 10
    return x


ENCODERS = {"onehot": (lambda bc, amap: encode(bc), OBS_DIM),
            "effects": (encode_effects, OBS_DIM_EFF)}


# ---------------------------------------------------------------- env
class CombatEnv(gym.Env):
    """Reward: +1 win / -1 loss (or exceeding MAX_STEPS)
    - HP lost / max HP at each step   (the cost that matters after the combat)
    + 0.5 x fraction of monster HP removed at each step  (dense shaping)."""
    metadata = {"render_modes": []}

    def __init__(self, scenario_seed=4, seed_range=(1_000_000, 2_000_000), shaping=0.5,
                 encoding="onehot", extra_cards=0, pool="train", pool_split=0.75, pool_seed=0,
                 deck="random", encounter=None, train_pool_size=None):
        self.template = make_scenario_gc(scenario_seed)
        self.seed_range = seed_range; self.shaping = shaping
        # encounter: the seed's by default, or a forced one (decks.ENCOUNTERS)
        self.encounter_name = encounter
        self.encounter = None if encounter is None else sts.MonsterEncounter.__members__[
            ENCOUNTERS.get(encounter, encounter)]
        # deck: coherent archetype (decks.ARCHETYPES), split into seen / unseen cards
        self.deck_name = deck
        train_pool, eval_pool = archetype_pools(deck, IRONCLAD_POOL, pool_split, pool_seed)
        train_pool = [c for c in train_pool if c not in UNSAFE_CARDS]
        eval_pool = [c for c in eval_pool if c not in UNSAFE_CARDS] or train_pool
        full_train_pool = train_pool
        if train_pool_size:
            # nested subsets: the first N cards of the (already shuffled) training pool
            train_pool = train_pool[:train_pool_size]
        self.train_pool_size = train_pool_size
        self.pool_name = pool
        self.pool = {"train": train_pool, "eval": eval_pool, "all": train_pool + eval_pool}[pool]
        self.encoding = encoding
        if encoding == "onehot_pool":
            # per-experiment vocabulary: starting deck + seen cards + eval cards
            vocab, vsize = build_vocab(self.template, sorted(set(full_train_pool) | set(eval_pool)))
            self.vocab_size = vsize
            self._encode = lambda bc, amap: encode_onehot_vocab(bc, amap, vocab, vsize)
            obs_dim = D_PLAYER + N_HAND * (4 + N_CARD_TYPES + vsize) + 3 * vsize + N_MON * D_MON
        else:
            self._encode, obs_dim = ENCODERS[encoding]
        self.extra_cards = extra_cards
        self.extra = []
        self.observation_space = gym.spaces.Box(-5, 5, (obs_dim,), np.float32)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.rng = np.random.default_rng()
        self.bc = None

    def _settle(self):
        """Auto-play unrepresentable decisions."""
        while True:
            self.mask, self.amap, unmapped = legal_map(self.bc)
            if self.mask.any() or not unmapped or self.bc.outcome != sts.Outcome.UNDECIDED:
                return
            unmapped[0].execute(self.bc)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        ep_seed = (options or {}).get("episode_seed")
        if ep_seed is None:
            ep_seed = int(self.rng.integers(*self.seed_range))
        if self.extra_cards:
            # reproducible draw: depends only on the episode seed
            rng = np.random.default_rng(1_000_003 + ep_seed)
            self.extra = [self.pool[i] for i in rng.choice(len(self.pool), self.extra_cards, replace=True)]
            with _ExtraCards(self.template, self.extra):
                self.bc = new_battle(self.template, ep_seed, self.encounter)
        else:
            self.bc = new_battle(self.template, ep_seed, self.encounter)
        self.ep_seed = ep_seed; self.steps = 0
        self.hp0 = self.bc.player.cur_hp; self.mhp0 = max(1, monsters_hp(self.bc))
        self._settle()
        return self.obs(), {}

    def obs(self):
        return self._encode(self.bc, self.amap)

    def action_masks(self):
        return self.mask.copy()

    def step(self, action):
        bc = self.bc
        hp_b, mhp_b = bc.player.cur_hp, monsters_hp(bc)
        self.amap[int(action)].execute(bc)
        self.steps += 1
        self._settle()
        r = -(hp_b - bc.player.cur_hp) / bc.player.max_hp + self.shaping * (mhp_b - monsters_hp(bc)) / self.mhp0
        done = bc.outcome != sts.Outcome.UNDECIDED
        trunc = not done and (self.steps >= MAX_STEPS or not self.mask.any())
        if done:
            r += 1.0 if bc.outcome == sts.Outcome.PLAYER_VICTORY else -1.0
        elif trunc:
            r -= 1.0
        info = {}
        if done or trunc:
            info = episode_stats(bc, self.hp0, self.steps)
        return self.obs(), float(r), done, trunc, info


def episode_stats(bc, hp0, steps):
    win = bc.outcome == sts.Outcome.PLAYER_VICTORY
    return {"win": bool(win), "hp_lost": hp0 - (bc.player.cur_hp if win else 0),
            "turns": bc.turn + 1, "steps": steps}
