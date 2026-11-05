"""Deck archetypes and encounters — each designed to test one specific skill.

An archetype = a pool of mutually coherent cards. Each episode, `extra_cards` cards are drawn
from it and added to the starting deck. The deck thus stays on-theme while varying.

`question` states what the archetype is meant to reveal; `probe` names the associated
behaviour metric, computed by `experiments/first_combat/behavior.py`.
"""

ARCHETYPES = {
    # ------------------------------------------------------------------ setup
    "strength": {
        "cards": ["INFLAME", "SPOT_WEAKNESS", "FLEX", "HEAVY_BLADE", "TWIN_STRIKE", "PUMMEL"],
        "holdout": ["LIMIT_BREAK", "SWORD_BOOMERANG"],   # unseen: doubles Strength / hits 3x
        "question": "does it invest in Strength BEFORE attacking, even though it pays nothing that turn?",
        "probe": "strength_first",
    },
    "debuff": {
        "cards": ["UPPERCUT", "CLOTHESLINE", "THUNDERCLAP", "DISARM", "HEAVY_BLADE", "IRON_WAVE"],
        "holdout": ["SHOCKWAVE", "INTIMIDATE"],          # unseen: other debuff appliers
        "question": "does it apply Vulnerable/Weak before its big hits rather than after?",
        "probe": "debuff_before_big_hit",
    },
    "block": {
        "cards": ["SHRUG_IT_OFF", "TRUE_GRIT", "IRON_WAVE", "BODY_SLAM", "GHOSTLY_ARMOR", "FLAME_BARRIER"],
        "holdout": ["ENTRENCH", "SECOND_WIND"],          # unseen: doubles block / converts
        "question": "does it stack block before playing Body Slam, which deals damage equal to block?",
        "probe": "body_slam_block",
    },
    "resource": {
        "cards": ["OFFERING", "BLOODLETTING", "POMMEL_STRIKE", "BATTLE_TRANCE", "ARMAMENTS", "HEADBUTT"],
        "holdout": ["SEEING_RED", "DUAL_WIELD"],         # unseen: energy / duplication
        "question": "does it spend HP for energy and card draw when it pays off?",
        "probe": "hp_for_tempo",
    },
    "aoe": {
        "cards": ["CLEAVE", "WHIRLWIND", "THUNDERCLAP", "SWORD_BOOMERANG", "DROPKICK", "TWIN_STRIKE"],
        "holdout": ["IMMOLATE", "SHOCKWAVE"],            # unseen: high-damage AoE / AoE debuff
        "question": "against several enemies, does it prefer AoE attacks over single-target?",
        "probe": "aoe_usage",
    },
    "powers": {
        "cards": ["INFLAME", "SPOT_WEAKNESS", "SHRUG_IT_OFF", "HEAVY_BLADE", "TWIN_STRIKE", "IRON_WAVE"],
        "holdout": ["DEMON_FORM", "METALLICIZE"],        # unseen: Powers, delayed effect but
        "question": "does it generalise to never-seen Powers, whose probe measures the status applied "
                    "(Demon Form 2, Metallicize 3) but not the future value?",
        "probe": "strength_first",
    },
    # ------------------------------------------------------------------ controls
    "random": {
        "cards": None,          # whole Ironclad pool: control with no thematic coherence
        "question": "control: same budgets, arbitrary cards",
        "probe": None,
    },
    "starter": {
        "cards": [],            # starting deck only
        "question": "control: fixed deck, no cards added",
        "probe": None,
    },
}

# Act 1 encounters, easiest to hardest. One per training run.
ENCOUNTERS = {
    "cultist": "CULTIST",                    # 1 enemy, gains Strength: a race
    "jaw_worm": "JAW_WORM",                  # 1 enemy, alternates attack and defence
    "louses": "TWO_LOUSE",                   # 2 weak enemies: target choice
    "small_slimes": "SMALL_SLIMES",          # 2 enemies, applies Weak
    "blue_slaver": "BLUE_SLAVER",            # applies Entangled: tempo loss
    "red_slaver": "RED_SLAVER",              # harder, applies Weak
    "three_louses": "THREE_LOUSE",           # 3 enemies: tests AoE attacks
    "fungi": "TWO_FUNGI_BEASTS",             # Vulnerable on death: punishes slowness
    "gremlin_nob": "GREMLIN_NOB",            # ELITE: punishes skills (Enrage)
    "lagavulin": "LAGAVULIN",                # ELITE: sleeps, then applies Weak
    "sentries": "THREE_SENTRIES",            # ELITE: 3 targets, adds Dazed
}


def archetype_pools(name, full_pool, split=0.75, seed=0):
    """-> (cards seen in training, held-out eval cards).

    For an archetype, both lists are hand-written: held-out cards stay thematically close
    (same mechanics, different effects), otherwise we would be testing novelty rather than
    generalisation. For 'random', the full pool is split at random."""
    import numpy as np
    spec = ARCHETYPES[name]
    if spec["cards"] is None:                       # 'random' control: whole Ironclad pool
        order = sorted(full_pool)
        np.random.default_rng(seed).shuffle(order)
        cut = int(round(len(order) * split))
        return order[:cut], order[cut:]
    return list(spec["cards"]), list(spec.get("holdout") or spec["cards"])
