#!/usr/bin/env python3
"""V2 — training with a scorer head: Q(s, a) = f(state ⊕ descriptor(a)).

Difference from `train.py` (V0/V1): no fixed-size action array. The network scores each
candidate, and the max/softmax runs over a variable number of candidates. An unseen action is
scored by what it does.

  python train_scorer.py --algo dqn --deck strength --encounter cultist --extra-cards 2 --steps 300000
  python train_scorer.py --algo reinforce ...

Outputs: same conventions as train.py (checkpoints, train_log.jsonl, config.json).
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from stsenv.scorer_env import ScorerCombatEnv, D_STATE, D_STATE_SET, D_ACT  # noqa: E402
from stsenv.policies import Scorer  # noqa: E402
from train import ckpt_schedule, Logger, scenario_dir  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
K_MAX = 12                      # candidates stored per transition (measured here: max 6; grows with the number of enemies)


def pad(descs, k_max=K_MAX, act_dim=None):
    d = np.zeros((k_max, act_dim or D_ACT), dtype=np.float16)
    m = np.zeros(k_max, dtype=bool)
    n = min(len(descs), k_max)
    d[:n] = descs[:n]; m[:n] = True
    return d, m


# ------------------------------------------------------------------ DQN with scorer head
def train_dqn(a, env, out, log):
    dev = torch.device(a.device)
    S, A = env.state_dim, env.act_dim
    q = Scorer(S, A).to(dev); qt = Scorer(S, A).to(dev); qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=a.lr, weight_decay=0)
    N = a.buffer
    bS = np.zeros((N, S), np.float16); bA = np.zeros((N, A), np.float16)
    bS2 = np.zeros((N, S), np.float16); bD2 = np.zeros((N, K_MAX, A), np.float16)
    bM2 = np.zeros((N, K_MAX), bool); bR = np.zeros(N, np.float32); bDone = np.zeros(N, np.float32)
    ptr = size = 0
    rng = np.random.default_rng(a.seed)
    sched = ckpt_schedule(a.steps); ci = 0
    env.reset(seed=a.seed)
    s, descs, idxs = env.candidates()
    eps_end = max(1, int(a.steps * a.eps_frac))

    for step in range(a.steps + 1):
        while ci < len(sched) and step >= sched[ci]:
            torch.save({k: v.cpu() for k, v in q.state_dict().items()}, out / f"ckpt_{sched[ci]:07d}.pt")
            ci += 1
        if step == a.steps:
            break
        eps = max(0.05, 1 - 0.95 * step / eps_end)
        if rng.random() < eps or len(idxs) == 1:
            k = int(rng.integers(len(idxs)))
        else:
            with torch.no_grad():
                sc = q.score_one(torch.as_tensor(s, device=dev), torch.as_tensor(descs, device=dev))
                k = int(sc.argmax())
        _, r, done, tr, info = env.step(idxs[k])
        s2, d2, i2 = env.candidates()

        pd2, pm2 = pad(d2, act_dim=A)
        bS[ptr], bA[ptr], bR[ptr], bS2[ptr], bD2[ptr], bM2[ptr], bDone[ptr] = \
            s, descs[k], r, s2, pd2, pm2, float(done or tr)
        ptr = (ptr + 1) % N; size = min(size + 1, N)

        if done or tr:
            log.episode(step + 1, info)
            env.reset(); s, descs, idxs = env.candidates()
        else:
            s, descs, idxs = s2, d2, i2

        if size >= a.learning_starts and step % a.train_freq == 0:
            j = rng.integers(0, size, a.batch)
            T = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=dev)
            st, act, rew, st2, d2b, m2b, dn = (T(bS[j]), T(bA[j]), T(bR[j]), T(bS2[j]),
                                               T(bD2[j]), torch.as_tensor(bM2[j], device=dev), T(bDone[j]))
            with torch.no_grad():
                m2f = m2b.clone(); m2f[~m2b.any(1)] = True        # terminal transition: neutral mask
                nxt = qt(st2, d2b, m2f).max(1).values
                y = rew + a.gamma * (1 - dn) * nxt
            qsa = q(st, act.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(qsa, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0); opt.step()
        if step % a.target_every == 0:
            qt.load_state_dict(q.state_dict())


# ------------------------------------------------------------------ REINFORCE with scorer head
def train_reinforce(a, env, out, log):
    net = Scorer(env.state_dim, env.act_dim)
    value = nn.Sequential(nn.Linear(env.state_dim, 128), nn.ReLU(), nn.Linear(128, 1))
    opt = torch.optim.Adam(list(net.parameters()) + list(value.parameters()), lr=a.lr)
    sched = ckpt_schedule(a.steps); ci = 0; steps = 0
    env.reset(seed=a.seed)
    while True:
        while ci < len(sched) and steps >= sched[ci]:
            torch.save(net.state_dict(), out / f"ckpt_{sched[ci]:07d}.pt"); ci += 1
        if steps >= a.steps:
            break
        traj, rets = [], []
        for _ in range(a.batch_episodes):
            env.reset(); rs = []; done = tr = False
            while not (done or tr):
                s, descs, idxs = env.candidates()
                with torch.no_grad():
                    sc = net.score_one(torch.as_tensor(s), torch.as_tensor(descs))
                k = int(torch.distributions.Categorical(logits=sc).sample())
                traj.append((s, descs, k))
                _, r, done, tr, info = env.step(idxs[k]); rs.append(r); steps += 1
            g, gs = 0.0, []
            for r in reversed(rs):
                g = r + a.gamma * g; gs.append(g)
            rets += gs[::-1]; log.episode(steps, info)
        loss = 0.0
        G = torch.tensor(rets, dtype=torch.float32)
        V = value(torch.as_tensor(np.stack([t[0] for t in traj]))).squeeze(-1)
        adv = (G - V).detach()
        for (s, descs, k), ad in zip(traj, adv):
            sc = net.score_one(torch.as_tensor(s), torch.as_tensor(descs))
            loss = loss - torch.log_softmax(sc, 0)[k] * ad
        loss = loss / len(traj) + 0.5 * F.mse_loss(V, G)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="dqn", choices=["dqn", "reinforce"])
    ap.add_argument("--scenario", type=int, default=4)
    ap.add_argument("--deck", default="random")
    ap.add_argument("--encounter", default=None)
    ap.add_argument("--extra-cards", type=int, default=2)
    ap.add_argument("--pool", default="train")
    ap.add_argument("--fx", default="measured", choices=["measured", "nominal", "both"],
                    help="card description: effects measured in context, nominal values, or both")
    ap.add_argument("--piles", default="summary", choices=["summary", "multiset"],
                    help="V3: 'multiset' describes the draw pile card by card (contents known, order hidden)")
    ap.add_argument("--leak", default="allow", choices=["allow", "mask"],
                    help="'mask': random-effect cards get their nominal values, so the probe cannot leak RNG")
    ap.add_argument("--train-pool-size", type=int, default=None,
                    help="keep only the first N cards of the training pool (nested subsets)")
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--buffer", type=int, default=60_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--learning-starts", type=int, default=5_000)
    ap.add_argument("--train-freq", type=int, default=2)
    ap.add_argument("--target-every", type=int, default=1_000)
    ap.add_argument("--eps-frac", type=float, default=0.25)
    ap.add_argument("--batch-episodes", type=int, default=32)
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    suffix = (("_multiset" if a.piles == "multiset" else "") + ("" if a.fx == "measured" else f"_{a.fx}")
               + ("_mask" if a.leak == "mask" else ""))
    name = f"{a.algo}_scorer{suffix}_effects_s{a.seed}" + (f"_{a.tag}" if a.tag else "")
    out = ROOT / "results" / "first_combat" / scenario_dir(a.scenario, a.extra_cards, a.deck, a.encounter, a.train_pool_size) / name
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("ckpt_*"):
        f.unlink()
    json.dump({**vars(a), "head": "scorer",
               "encoding": f"scorer:{a.piles}:{a.fx}:{a.leak}"},
              open(out / "config.json", "w"), indent=1)

    env = ScorerCombatEnv(a.scenario, extra_cards=a.extra_cards, pool=a.pool,
                          deck=a.deck, encounter=a.encounter, piles=a.piles, fx=a.fx,
                          leak=a.leak, train_pool_size=a.train_pool_size)
    S_DIM = env.state_dim
    print(f"{name} | state {S_DIM} + action {env.act_dim} | {a.steps} steps -> {out}", flush=True)
    t0 = time.time()
    (train_dqn if a.algo == "dqn" else train_reinforce)(a, env, out, Logger(out))
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
