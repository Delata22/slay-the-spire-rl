"""Invariants of the combat environments. Needs the compiled simulator (scripts/setup_simulator.sh)."""
import numpy as np
import pytest

sts = pytest.importorskip("slaythespire", reason="simulator not built (scripts/setup_simulator.sh)")

from stsenv.combat_env import CombatEnv, card_pools  # noqa: E402
from stsenv.decks import archetype_pools  # noqa: E402
from stsenv.optimal import solve  # noqa: E402
from stsenv.policies import HeuristicPolicy, RandomPolicy  # noqa: E402
from stsenv.scorer_env import ScorerCombatEnv, D_KIND, D_STATIC, D_EFF_PROBE  # noqa: E402


def deck_names(gc):
    return sorted(str(c.id) + ("+" if c.upgraded else "") for c in gc.deck)


def play(env, policy, seed):
    env.reset(options={"episode_seed": seed})
    done = tr = False
    info = {}
    while not (done or tr):
        _, _, done, tr, info = env.step(policy.act(env))
    return info


@pytest.fixture(scope="module")
def env():
    return CombatEnv(4, extra_cards=3, pool="train", deck="random", encounter="cultist")


def test_extra_cards_never_modify_the_template(env):
    before = deck_names(env.template)
    for s in range(20):
        env.reset(options={"episode_seed": s})
        assert len(env.bc.draw_pile) + len(env.bc.hand) >= len(before) + 3
    assert deck_names(env.template) == before


def test_episodes_are_reproducible(env):
    a = play(env, RandomPolicy(0), 123)
    b = play(env, RandomPolicy(0), 123)
    assert a == b


def test_train_and_eval_pools_are_disjoint():
    from stsenv.combat_env import IRONCLAD_POOL
    train, held_out = archetype_pools("random", IRONCLAD_POOL)
    assert train and held_out and not set(train) & set(held_out)
    tr, ev = card_pools()
    assert not set(tr) & set(ev)


def test_nested_training_pools():
    pools = [CombatEnv(4, extra_cards=3, pool="train", deck="random", encounter="cultist",
                       train_pool_size=n).pool for n in (6, 12, 24)]
    assert [len(p) for p in pools] == [6, 12, 24]
    assert pools[1][:6] == pools[0] and pools[2][:12] == pools[1]


def test_scorer_descriptors_ignore_hand_position():
    env = ScorerCombatEnv(4, extra_cards=0, deck="random", encounter="cultist", fx="both")
    env.reset(options={"episode_seed": 0})
    _, descs, idxs = env.candidates()
    names = {}
    for d, i in zip(descs, idxs):
        a = env.amap[i]
        if a.action_type == sts.SearchActionType.CARD:
            c = env.bc.hand[a.source_idx]
            names.setdefault((str(c.id), c.upgraded, a.target_idx), []).append(d)
    dup = [v for v in names.values() if len(v) > 1]
    for v in dup:                                   # identical cards -> identical descriptors
        assert all(np.array_equal(v[0], x) for x in v[1:])


def test_leak_mask_hides_rng_outcome():
    """With leak='mask', a random-effect card is described by its printed values, not by
    the outcome the probe would read from the cloned RNG."""
    allow = ScorerCombatEnv(4, extra_cards=0, deck="random", encounter="cultist", fx="measured")
    mask = ScorerCombatEnv(4, extra_cards=0, deck="random", encounter="cultist", fx="measured",
                           leak="mask")
    for e in (allow, mask):
        e.template.obtain_card(sts.Card(sts.CardId.SWORD_BOOMERANG))
    found = False
    for seed in range(40):
        allow.reset(options={"episode_seed": seed}); mask.reset(options={"episode_seed": seed})
        for e in (allow, mask):
            e.candidates()
        for i, a in allow.amap.items():
            if a.action_type == sts.SearchActionType.CARD and \
                    str(allow.bc.hand[a.source_idx].id).endswith("SWORD_BOOMERANG"):
                from stsenv.scorer_env import action_descriptor
                d_mask = action_descriptor(mask.bc, mask.amap[i], "measured", "mask")
                assert d_mask[4] == 1.0                     # "random effect" flag raised
                fx = d_mask[D_KIND + D_STATIC:D_KIND + D_STATIC + D_EFF_PROBE]
                assert np.isclose(fx[0], 9 / 30)            # nominal 3 x 3 damage, not a sampled split
                found = True
                break
        if found:
            break
    assert found, "Sword Boomerang never drawn in the opening hand"


def test_optimal_is_a_lower_bound(env):
    pol = HeuristicPolicy()
    spec = (("random", "cultist"), 3, "train")
    for seed in range(5):
        _, best = solve((spec, seed))
        info = play(env, pol, seed)
        assert best <= info["hp_lost"]


def test_fixed_mcts_finds_the_lethal_line():
    """Seed 306, turn 3 of the fixed-deck scenario: three Strikes kill the Jaw Worm. The original
    search spends every simulation on Bash+; the fixed one (patches/scumsearch_fixed_uct.patch) kills."""
    from stsenv.policies import MCTSOraclePolicy
    env = CombatEnv(4)
    env.reset(options={"episode_seed": 306})
    pol = MCTSOraclePolicy(2000)
    while env.bc.turn + 1 < 3:
        env.step(pol.act(env))
    try:
        fixed = sts.mcts_recommend(env.bc, 200, fixed=True)
    except TypeError:
        pytest.skip("simulator built without patches/scumsearch_fixed_uct.patch")
    original = sts.mcts_recommend(env.bc, 200)
    assert "Strike" in fixed.desc(env.bc)
    assert "Bash" in original.desc(env.bc)
