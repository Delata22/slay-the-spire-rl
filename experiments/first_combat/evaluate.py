#!/usr/bin/env python3
"""Evaluate all checkpoints + baseline policies on the SAME eval seeds.

  python evaluate.py [--scenario 4] [--n-eval 500] [--workers 12] [--runs ppo,dqn,...] [--mcts 200,2000]

Eval seeds = 0 .. n-1 (training seeds are drawn from [1e6, 2e6): disjoint).
All policies play exactly the same combats -> paired comparisons.
Learned policies: argmax (greedy).

Outputs in results/first_combat/seed<S>/:
  eval.json                     aggregated metrics + 95% bootstrap CI
  eval_episodes.npz             per-episode results (for paired tests)
  replays/<run>/<ckpt>.txt      replayed combats, turn by turn (eval seeds 0, 1, 2)
"""
import argparse, json, os, sys, time
from pathlib import Path
import multiprocessing as mp

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

_ENV = {}
# Display labels for result keys that are stored in French in the shipped JSON files.
LABELS = {"optimal (pioche connue)": "optimal (known draw order)"}


def load_policy(spec, env):
    import torch
    from stsenv.combat_env import N_ACTIONS
    from stsenv import policies as P
    kind = spec["kind"]
    if kind == "random": return P.RandomPolicy(0)
    if kind == "heuristic": return P.HeuristicPolicy()
    if kind == "mcts": return P.MCTSOraclePolicy(spec["sims"])
    if kind == "scorer":
        net = P.Scorer(env.state_dim, env.act_dim)
        net.load_state_dict(torch.load(spec["path"], weights_only=True))
        return P.ScorerPolicy(net, spec["run"])
    if kind == "torch":
        net = P.MaskedMLP(env.observation_space.shape[0], N_ACTIONS, value_head=spec["algo"] == "reinforce")
        net.load_state_dict(torch.load(spec["path"], weights_only=True))
        return P.TorchGreedyPolicy(net, spec["run"])
    if kind == "sb3":
        from sb3_contrib import MaskablePPO
        return P.SB3GreedyPolicy(MaskablePPO.load(spec["path"], device="cpu"), spec["run"])
    raise ValueError(kind)


def get_env(scenario, encoding="onehot", extra_cards=0, pool="eval", train_pool_size=None):
    """`scenario`: seed number, or (deck, encounter) for the per-archetype experiments."""
    key = (scenario, encoding, extra_cards, pool, train_pool_size)
    if key not in _ENV:
        from stsenv.combat_env import CombatEnv
        if str(encoding).startswith("scorer"):
            from stsenv.scorer_env import ScorerCombatEnv
            parts = str(encoding).split(":")           # "scorer:<piles>:<fx>" (old format: "scorer")
            piles = parts[1] if len(parts) > 2 else ("multiset" if encoding.endswith("multiset") else "summary")
            fx = parts[2] if len(parts) > 2 else "measured"
            leak = parts[3] if len(parts) > 3 else "allow"
            CombatEnv = lambda *ar, **kw: ScorerCombatEnv(*ar, piles=piles, fx=fx, leak=leak, **kw)
        if isinstance(scenario, tuple):
            deck, enc = scenario
            _ENV[key] = CombatEnv(4, encoding=encoding, extra_cards=extra_cards, pool=pool,
                                  deck=deck, encounter=enc, train_pool_size=train_pool_size)
        else:
            _ENV[key] = CombatEnv(scenario, encoding=encoding, extra_cards=extra_cards, pool=pool)
    return _ENV[key]


def fmt_state(bc):
    p = bc.player
    mons = " | ".join(
        f"{m.name} {m.cur_hp}/{m.max_hp}" + (f" blk{m.block}" if m.block else "") +
        (f" vuln{m.vulnerable}" if m.vulnerable else "") + (f" str{m.strength}" if m.strength else "") +
        f" → {m.intent}" + (f" ({m.intent_damage(bc).damage}x{m.intent_damage(bc).attack_count})" if m.is_attacking else "")
        for m in bc.monsters if m.alive)
    hand = ", ".join(c.name + ("+" if c.upgraded else "") for c in bc.hand)
    return (f"── Turn {bc.turn + 1} │ HP {p.cur_hp}/{p.max_hp} block {p.block} energy {p.energy}\n"
            f"   enemies: {mons}\n   hand   : {hand}")


def play(pol, env, ep_seed, trace=False):
    env.reset(options={"episode_seed": ep_seed})
    lines, last_turn, done, tr, info = [], -1, False, False, {}
    while not (done or tr):
        if trace and env.bc.turn != last_turn:
            last_turn = env.bc.turn; lines.append(fmt_state(env.bc))
        a = pol.act(env)
        if trace:
            lines.append(f"     ▸ {env.amap[a].desc(env.bc).strip()}")
        _, _, done, tr, info = env.step(a)
    if trace:
        lines.append(f"══ {'WIN' if info['win'] else 'LOSS'} — HP lost {info['hp_lost']}, {info['turns']} turns")
    return info, "\n".join(lines)


def worker(job):
    spec, scenario, seeds = job
    import torch; torch.set_num_threads(1)
    scenario, extra, pool, tps = scenario if isinstance(scenario, tuple) else (scenario, 0, "eval", None)
    env = get_env(scenario, spec.get("encoding", "onehot"), extra, pool, tps); pol = load_policy(spec, env)
    out = []
    for s in seeds:
        info, _ = play(pol, env, s)
        out.append((s, info["win"], info["hp_lost"], info["turns"]))
    return spec["key"], out


def replay_job(job):
    spec, scenario, seeds = job
    import torch; torch.set_num_threads(1)
    scenario, extra, pool, tps = scenario if isinstance(scenario, tuple) else (scenario, 0, "eval", None)
    env = get_env(scenario, spec.get("encoding", "onehot"), extra, pool, tps); pol = load_policy(spec, env)
    return spec, [play(pol, env, s, trace=True)[1] for s in seeds]


def boot_ci(x, n=2000, rng=np.random.default_rng(0)):
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, default=4)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--runs", default="", help="subfolders to evaluate (default: all except *_smoke)")
    ap.add_argument("--mcts", default="200,2000")
    ap.add_argument("--replay-seeds", default="0,1,2")
    ap.add_argument("--extra-cards", type=int, default=0)
    ap.add_argument("--deck", default="", help="deck archetype (per-archetype experiments)")
    ap.add_argument("--encounter", default="", help="fixed encounter")
    ap.add_argument("--ckpt-steps", default="",
                    help="comma-separated checkpoint steps to evaluate (default: all checkpoints)")
    ap.add_argument("--train-pool-size", type=int, default=None,
                    help="evaluate runs trained on the first N cards of the training pool")
    ap.add_argument("--pool", default="eval", choices=["train", "eval", "all"],
                    help="card pool at eval: 'eval' = cards NEVER seen in training")
    a = ap.parse_args()

    if a.deck or a.encounter:
        deck = a.deck or "random"
        base = ROOT / "results" / "first_combat" / (f"{deck}_{a.encounter or 'seed' + str(a.scenario)}_x{a.extra_cards}"
                                                    + (f"_p{a.train_pool_size}" if a.train_pool_size else ""))
        env_spec = ((deck, a.encounter or None), a.extra_cards, a.pool, a.train_pool_size)
    else:
        base = ROOT / "results" / "first_combat" / (f"seed{a.scenario}" + (f"_x{a.extra_cards}" if a.extra_cards else ""))
        env_spec = (a.scenario, a.extra_cards, a.pool, None)
    runs = [d for d in sorted(base.iterdir()) if d.is_dir() and (d / "config.json").exists()]
    runs = [d for d in runs if (d.name in a.runs.split(",")) if a.runs] or [d for d in runs if not d.name.endswith("_smoke")]

    specs = [{"kind": "random", "key": "random"}, {"kind": "heuristic", "key": "heuristic"}]
    specs += [{"kind": "mcts", "sims": int(s), "key": f"mcts_oracle@{s}"} for s in a.mcts.split(",") if s]
    for d in runs:
        cfg = json.load(open(d / "config.json"))
        keep = {int(x) for x in a.ckpt_steps.split(",") if x}
        for f in sorted(d.glob("ckpt_*")):
            step = int(f.stem.split("_")[1])
            if keep and step not in keep:
                continue
            kind = "scorer" if cfg.get("head") == "scorer" else ("sb3" if f.suffix == ".zip" else "torch")
            specs.append({"kind": kind, "path": str(f),
                          "algo": cfg["algo"], "encoding": cfg.get("encoding", "onehot"),
                          "run": d.name, "step": step, "key": f"{d.name}/{step}"})

    seeds = list(range(a.n_eval)); chunk = max(10, a.n_eval // a.workers)
    jobs = [(sp, env_spec, seeds[i:i + chunk]) for sp in specs for i in range(0, len(seeds), chunk)]
    print(f"{len(specs)} policies × {a.n_eval} combats ({len(runs)} runs: {', '.join(d.name for d in runs)})", flush=True)
    t0 = time.time(); res = {}
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for key, out in pool.imap_unordered(worker, jobs):
            for s, *vals in out:
                res.setdefault(key, {})[s] = vals
        print(f"eval: {time.time() - t0:.0f} s", flush=True)
        rseeds = [int(s) for s in a.replay_seeds.split(",")]
        replays = pool.map(replay_job, [(sp, env_spec, rseeds) for sp in specs])

    summary = {"scenario": a.scenario, "n_eval": a.n_eval, "baselines": {}, "runs": {}}
    arrays = {}

    # upper bound: perfect play with the draw order known (brute force, ~0.1 s/combat)
    from stsenv.optimal import optimal_hp_lost
    opt = optimal_hp_lost(env_spec, seeds, workers=a.workers,
                          cache=base / f"optimal_hp_lost_{a.pool}.npy")
    summary["baselines"]["optimal (pioche connue)"] = {
        "win": 1.0, "win_ci": [1.0, 1.0], "hp_lost": float(opt.mean()),
        "hp_lost_ci": boot_ci(opt), "turns": float("nan")}
    arrays["optimal (pioche connue)"] = np.stack([np.ones_like(opt), opt, np.zeros_like(opt)], axis=1)
    for sp in specs:
        r = np.array([res[sp["key"]][s] for s in seeds], float)      # row i = eval seed i (pairing)
        row = {"win": float(r[:, 0].mean()), "win_ci": boot_ci(r[:, 0]),
               "hp_lost": float(r[:, 1].mean()), "hp_lost_ci": boot_ci(r[:, 1]), "turns": float(r[:, 2].mean())}
        arrays[sp["key"].replace("/", "__")] = r
        if "run" in sp:
            summary["runs"].setdefault(sp["run"], []).append({"step": sp["step"], **row})
        else:
            summary["baselines"][sp["key"]] = row
    for v in summary["runs"].values():
        v.sort(key=lambda x: x["step"])
    summary["pool"] = a.pool; summary["extra_cards"] = a.extra_cards
    suffix = "" if not a.extra_cards else f"_{a.pool}"
    json.dump(summary, open(base / f"eval{suffix}.json", "w"), indent=1)
    np.savez_compressed(base / f"eval_episodes{suffix}.npz", **arrays)

    for sp, texts in replays:
        name = sp.get("run", "baselines"); stem = f"{sp['step']:07d}" if "step" in sp else sp["key"]
        d = base / "replays" / name; d.mkdir(parents=True, exist_ok=True)
        body = "\n\n".join(f"### eval seed {s}\n{t}" for s, t in zip(rseeds, texts))
        (d / f"{stem}.txt").write_text(f"{sp['key']}\n\n{body}\n")

    print(f"\n{'policy':28} {'wins':>10} {'HP lost [CI95]':>22} {'turns':>6}")
    for k, v in summary["baselines"].items():
        print(f"{LABELS.get(k, k):28} {v['win']:>10.1%} {v['hp_lost']:>8.1f} [{v['hp_lost_ci'][0]:.1f}, {v['hp_lost_ci'][1]:.1f}] {v['turns']:>6.1f}")
    for run, rows in summary["runs"].items():
        for v in (rows[0], rows[-1]):
            k = f"{run} @{v['step']}"
            print(f"{k:28} {v['win']:>10.1%} {v['hp_lost']:>8.1f} [{v['hp_lost_ci'][0]:.1f}, {v['hp_lost_ci'][1]:.1f}] {v['turns']:>6.1f}")
    print(f"\n-> {base / f'eval{suffix}.json'}\n-> {base / 'replays'}/")


if __name__ == "__main__":
    main()
