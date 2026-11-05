#!/usr/bin/env python3
"""Train an agent on the first combat of a fixed seed, with checkpoints.

  python train.py --algo ppo|reinforce|dqn [--scenario 4] [--steps 200000] [--tag name]

Outputs: results/first_combat/seed<S>/<algo>[_tag]/
  ckpt_<steps>.pt|.zip checkpoints (steps = environment steps seen)
  train_log.jsonl      training-episode stats (sampled, not greedy)
  config.json
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from stsenv.combat_env import CombatEnv, N_ACTIONS, describe_scenario  # noqa: E402
from stsenv.policies import MaskedMLP  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))


def scenario_dir(scenario, extra_cards, deck="random", encounter=None, train_pool_size=None):
    """One results folder per experiment: deck archetype, encounter, number of cards."""
    if train_pool_size:
        return f"{deck}_{encounter or 'seed' + str(scenario)}_x{extra_cards}_p{train_pool_size}"
    if encounter or deck not in ("random", None):
        return f"{deck}_{encounter or 'seed' + str(scenario)}_x{extra_cards}"
    return f"seed{scenario}" + (f"_x{extra_cards}" if extra_cards else "")


def ckpt_schedule(budget):
    """0, 5k, 10k (early learning), then every 10% of the budget."""
    return sorted({int(budget * k / 10) for k in range(11)} | {b for b in (5_000, 10_000) if b < budget})


# Presets: "default" = usual settings; "tuned" = adapted to short episodes (~15 decisions)
PRESETS = {
    "ppo": {"default": {},
            "tuned": dict(lr=5e-4, gamma=1.0, ent_coef=0.003, n_steps=512, n_epochs=10, batch=512),
            "fast": dict(lr=1e-3, gamma=1.0, ent_coef=0.01, n_steps=256, n_epochs=20, batch=256)},
    "dqn": {"default": {},
            "tuned": dict(lr=3e-4, gamma=1.0, batch=256, train_freq=2, target_every=1000,
                          buffer=100_000, eps_frac=0.25)},
}


class Logger:
    def __init__(self, out):
        self.f = open(out / "train_log.jsonl", "w"); self.buf = []; self.t0 = time.time()

    def episode(self, steps, info):
        self.f.write(json.dumps({"t": steps, **info}) + "\n")
        self.buf.append(info)
        if len(self.buf) >= 200:
            w = np.mean([b["win"] for b in self.buf]); h = np.mean([b["hp_lost"] for b in self.buf])
            print(f"  [{steps:>8}] train (200 ep.) win {w:.2f}  HP lost {h:5.1f}  "
                  f"| {steps / (time.time() - self.t0):.0f} steps/s", flush=True)
            self.buf = []; self.f.flush()


# ------------------------------------------------------------------ REINFORCE
def train_reinforce(a, out, log):
    env = CombatEnv(a.scenario, encoding=a.encoding, extra_cards=a.extra_cards, pool=a.pool,
                      deck=a.deck, encounter=a.encounter, train_pool_size=a.train_pool_size); env.reset(seed=a.seed)
    net = MaskedMLP(env.observation_space.shape[0], N_ACTIONS, value_head=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr or 3e-4)
    sched = ckpt_schedule(a.steps); ci = 0; steps = 0
    while True:
        while ci < len(sched) and steps >= sched[ci]:
            torch.save(net.state_dict(), out / f"ckpt_{sched[ci]:07d}.pt"); ci += 1
        if steps >= a.steps: break
        O, M, A, G = [], [], [], []
        for _ in range(a.batch_episodes):
            o, _ = env.reset(); rs = []; done = tr = False
            while not (done or tr):
                m = env.action_masks()
                with torch.no_grad():
                    logits, _ = net(torch.as_tensor(o)[None], torch.as_tensor(m)[None])
                act = int(torch.distributions.Categorical(logits=logits[0]).sample())
                O.append(o); M.append(m); A.append(act)
                o, r, done, tr, info = env.step(act); rs.append(r); steps += 1
            g, gs = 0.0, []
            for r in reversed(rs):
                g = r + a.gamma * g; gs.append(g)
            G += gs[::-1]; log.episode(steps, info)
        O = torch.as_tensor(np.array(O)); M = torch.as_tensor(np.array(M))
        A = torch.as_tensor(A); G = torch.as_tensor(G, dtype=torch.float32)
        logits, v = net(O, M)
        dist = torch.distributions.Categorical(logits=logits)
        adv = (G - v).detach()
        loss = -(dist.log_prob(A) * adv).mean() + 0.5 * F.mse_loss(v, G) - a.ent_coef * dist.entropy().mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()


# ------------------------------------------------------------------ DQN (double, masked)
def train_dqn(a, out, log):
    env = CombatEnv(a.scenario, encoding=a.encoding, extra_cards=a.extra_cards, pool=a.pool,
                      deck=a.deck, encounter=a.encounter, train_pool_size=a.train_pool_size); env.reset(seed=a.seed)
    OBS = env.observation_space.shape[0]
    dev = torch.device(a.device)
    q = MaskedMLP(OBS, N_ACTIONS).to(dev); qt = MaskedMLP(OBS, N_ACTIONS).to(dev); qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=a.lr or 1e-4)
    N = a.buffer
    bO = np.zeros((N, OBS), np.float16); bO2 = np.zeros((N, OBS), np.float16)
    bM2 = np.zeros((N, N_ACTIONS), bool); bA = np.zeros(N, np.int64); bR = np.zeros(N, np.float32); bD = np.zeros(N, np.float32)
    ptr = size = 0; rng = np.random.default_rng(a.seed)
    sched = ckpt_schedule(a.steps); ci = 0
    o, _ = env.reset(); eps_end_at = max(1, int(a.steps * a.eps_frac))
    for steps in range(a.steps + 1):
        while ci < len(sched) and steps >= sched[ci]:
            torch.save({k: v.cpu() for k, v in q.state_dict().items()}, out / f"ckpt_{sched[ci]:07d}.pt"); ci += 1
        if steps == a.steps: break
        eps = max(0.05, 1 - 0.95 * steps / eps_end_at)
        m = env.action_masks()
        if rng.random() < eps:
            act = int(rng.choice(np.flatnonzero(m)))
        else:
            with torch.no_grad():
                act = int(q(torch.as_tensor(o, device=dev)[None], torch.as_tensor(m, device=dev)[None])[0].argmax())
        o2, r, done, tr, info = env.step(act)
        bO[ptr], bA[ptr], bR[ptr], bO2[ptr], bM2[ptr], bD[ptr] = o, act, r, o2, env.action_masks(), float(done or tr)
        ptr = (ptr + 1) % N; size = min(size + 1, N)
        o = o2
        if done or tr:
            log.episode(steps + 1, info); o, _ = env.reset()
        if size >= a.learning_starts and steps % a.train_freq == 0:
            idx = rng.integers(0, size, a.batch)
            T = lambda x, dt=None: torch.as_tensor(x, dtype=dt, device=dev)
            O = T(bO[idx], torch.float32); O2 = T(bO2[idx], torch.float32)
            M2 = T(bM2[idx]); A = T(bA[idx]); R = T(bR[idx]); D = T(bD[idx])
            with torch.no_grad():
                M2 = M2 | (D[:, None] > 0)                         # terminal: empty mask -> allow all (ignored)
                a2 = q(O2, M2)[0].argmax(-1)
                y = R + a.gamma * (1 - D) * qt(O2)[0].gather(1, a2[:, None]).squeeze(1)
            qsa = q(O)[0].gather(1, A[:, None]).squeeze(1)
            loss = F.smooth_l1_loss(qsa, y)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0); opt.step()
        if steps % a.target_every == 0:
            qt.load_state_dict(q.state_dict())


# ------------------------------------------------------------------ PPO (sb3-contrib)
def train_ppo(a, out, log):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback

    def make(i):
        def f():
            e = CombatEnv(a.scenario, encoding=a.encoding, extra_cards=a.extra_cards, pool=a.pool,
                      deck=a.deck, encounter=a.encounter, train_pool_size=a.train_pool_size); e.reset(seed=a.seed * 1000 + i); return e
        return f

    venv = SubprocVecEnv([make(i) for i in range(a.n_envs)])
    model = MaskablePPO("MlpPolicy", venv, n_steps=a.n_steps, batch_size=a.batch, n_epochs=a.n_epochs,
                        learning_rate=a.lr or 3e-4,
                        gamma=a.gamma, ent_coef=a.ent_coef, seed=a.seed, device="cpu", verbose=0,
                        policy_kwargs=dict(net_arch=[256, 256]))
    sched = ckpt_schedule(a.steps)

    class CB(BaseCallback):
        def __init__(self): super().__init__(); self.ci = 0

        def _save(self):
            while self.ci < len(sched) and self.num_timesteps >= sched[self.ci]:
                self.model.save(out / f"ckpt_{sched[self.ci]:07d}.zip"); self.ci += 1

        def _on_training_start(self): self._save()

        def _on_step(self):
            for info in self.locals["infos"]:
                if "win" in info:
                    log.episode(self.num_timesteps, {k: info[k] for k in ("win", "hp_lost", "turns", "steps")})
            self._save(); return True

    model.learn(a.steps, callback=CB())
    model.save(out / f"ckpt_{a.steps:07d}.zip")
    venv.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["ppo", "reinforce", "dqn"])
    ap.add_argument("--scenario", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")  # dqn
    ap.add_argument("--n-envs", type=int, default=8)            # ppo
    ap.add_argument("--config", default="default")              # hyperparameter preset
    ap.add_argument("--encoding", default="onehot", choices=["onehot", "onehot_pool", "effects"])
    ap.add_argument("--extra-cards", type=int, default=0, help="cards drawn at random each episode")
    ap.add_argument("--pool", default="train", choices=["train", "eval", "all"])
    ap.add_argument("--deck", default="random", help="deck archetype (stsenv/decks.py)")
    ap.add_argument("--encounter", default=None, help="fixed encounter (cultist, gremlin_nob...)")
    ap.add_argument("--train-pool-size", type=int, default=None,
                    help="keep only the first N cards of the training pool (nested subsets)")
    ap.add_argument("--n-steps", type=int, default=256)         # ppo
    ap.add_argument("--n-epochs", type=int, default=4)          # ppo
    ap.add_argument("--eps-frac", type=float, default=0.5)      # dqn: fraction of the budget for eps 1 -> 0.05
    ap.add_argument("--batch-episodes", type=int, default=32)   # reinforce
    ap.add_argument("--buffer", type=int, default=50_000)       # dqn
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--learning-starts", type=int, default=5_000)
    ap.add_argument("--train-freq", type=int, default=4)
    ap.add_argument("--target-every", type=int, default=2_000)
    a = ap.parse_args()
    for k, v in PRESETS.get(a.algo, {}).get(a.config, {}).items():
        setattr(a, k, v)
    torch.manual_seed(a.seed)

    name = f"{a.algo}_{a.config}_{a.encoding}_s{a.seed}" + (f"_{a.tag}" if a.tag else "")
    out = ROOT / "results" / "first_combat" / scenario_dir(a.scenario, a.extra_cards, a.deck, a.encounter, a.train_pool_size) / name
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("ckpt_*"): f.unlink()
    json.dump(vars(a), open(out / "config.json", "w"), indent=1)
    print(f"{name} | {a.steps} steps | -> {out}\n{describe_scenario(CombatEnv(a.scenario).template)}", flush=True)
    t0 = time.time()
    {"ppo": train_ppo, "reinforce": train_reinforce, "dqn": train_dqn}[a.algo](a, out, Logger(out))
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
