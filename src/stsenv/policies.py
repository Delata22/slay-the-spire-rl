"""Reference (non-learned) policies and the network shared by the learned algorithms.

All policies expose  act(env) -> action index  (env: CombatEnv after reset/step).
"""
from __future__ import annotations

import numpy as np
import torch, torch.nn as nn

from .combat_env import sts, AT, A_END, action_index


class RandomPolicy:
    name = "random"
    def __init__(self, seed=0): self.rng = np.random.default_rng(seed)
    def act(self, env): return int(self.rng.choice(np.flatnonzero(env.action_masks())))


class HeuristicPolicy:
    """Generic rules (knows no card by name):
    - if incoming damage exceeds block: play a SKILL (Defend…);
    - otherwise play an ATTACK/POWER, most expensive first (Bash before Strike);
    - target: the targetable monster with the lowest HP;
    - nothing playable -> end turn."""
    name = "heuristic"

    def act(self, env):
        bc = env.bc
        incoming = 0
        for m in bc.monsters:
            if m.alive and m.is_attacking:
                d = m.intent_damage(bc); incoming += (d.damage + m.strength) * max(1, d.attack_count)
        need_block = incoming > bc.player.block
        mons = list(bc.monsters)
        # card-select screen (Armaments, Dual Wield...): "end turn" is not legal there
        best, best_key = (A_END if A_END in env.amap else min(env.amap)), None
        for i, a in env.amap.items():
            if a.action_type != AT.CARD:
                continue
            c = bc.hand[a.source_idx]
            is_skill = c.type == sts.CardType.SKILL
            if c.type in (sts.CardType.STATUS, sts.CardType.CURSE):
                continue
            tgt_hp = mons[a.target_idx].cur_hp if c.requires_target else 0
            key = (is_skill == need_block, c.cost_for_turn, -tgt_hp)
            if best_key is None or key > best_key:
                best, best_key = i, key
        return best


class MCTSOraclePolicy:
    """ScumSearch (mcts_recommend): its simulations copy the RNG -> it KNOWS the future draw
    order. A cheating upper bound, not an imitable target."""
    def __init__(self, sims=2000): self.sims = sims; self.name = f"mcts_oracle@{sims}"

    def act(self, env):
        a = sts.mcts_recommend(env.bc, self.sims)
        i = action_index(a, env.bc)
        if i in env.amap:
            return i
        if i == -2 and a.action_type == AT.CARD:           # untargeted card, target != 0
            return a.source_idx * 5
        return int(np.flatnonzero(env.action_masks())[0])


# ---------------------------------------------------------------- shared network
class MaskedMLP(nn.Module):
    """obs -> logits (or Q-values) over the N actions; + optional value head."""
    def __init__(self, obs_dim, n_actions, hidden=(256, 256), value_head=False):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]; d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_actions)
        self.v = nn.Linear(d, 1) if value_head else None

    def forward(self, obs, mask=None):
        z = self.body(obs)
        out = self.head(z)
        if mask is not None:
            out = out.masked_fill(~mask, -1e9)
        return out, (self.v(z).squeeze(-1) if self.v is not None else None)


class TorchGreedyPolicy:
    """Argmax of a MaskedMLP (REINFORCE / DQN)."""
    def __init__(self, net, name): self.net = net.eval(); self.name = name

    @torch.no_grad()
    def act(self, env):
        o = torch.as_tensor(env.obs()).unsqueeze(0)
        m = torch.as_tensor(env.action_masks()).unsqueeze(0)
        return int(self.net(o, m)[0].argmax(-1))


class SB3GreedyPolicy:
    def __init__(self, model, name): self.model = model; self.name = name

    def act(self, env):
        a, _ = self.model.predict(env.obs(), action_masks=env.action_masks(), deterministic=True)
        return int(a)


# ---------------------------------------------------------------- V2: scorer head
class Scorer(nn.Module):
    """f(state, action descriptor) -> score. The same network scores every candidate, so the
    number of actions can vary and an unseen action is scored by its effects."""

    def __init__(self, state_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        layers, d = [], state_dim + act_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]; d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)

    def forward(self, state, descs, mask=None):
        """state [B, S]; descs [B, K, A] -> scores [B, K]."""
        B, K, _ = descs.shape
        x = torch.cat([state.unsqueeze(1).expand(B, K, -1), descs], dim=-1)
        out = self.head(self.body(x)).squeeze(-1)
        return out if mask is None else out.masked_fill(~mask, -1e9)

    def score_one(self, state, descs):
        return self.forward(state.unsqueeze(0), descs.unsqueeze(0))[0]


class ScorerPolicy:
    """Scorer argmax over the current state's candidates."""
    def __init__(self, net, name): self.net = net.eval(); self.name = name

    @torch.no_grad()
    def act(self, env):
        s, descs, idxs = env.candidates()
        sc = self.net.score_one(torch.as_tensor(s), torch.as_tensor(descs))
        return idxs[int(sc.argmax())]
