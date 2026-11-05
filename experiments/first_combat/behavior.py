#!/usr/bin/env python3
"""Behaviour probes: has the agent understood the MECHANICS, not just the HP outcome?

HP lost tells who wins, not why. These probes look at what is played, and when.

  python behavior.py --deck strength --encounter cultist [--episodes 300]

Probes:
  strength_first        when a Strength card is in hand on turn 1, is it played BEFORE any
                        attack? (it pays nothing on the turn itself: it is an investment)
  debuff_before_big_hit fraction of big attacks (>= 10 damage) that land on an already Vulnerable target
  body_slam_block       mean block when Body Slam is played (it deals damage equal to block)
  hp_for_tempo          number of HP-cost cards played per combat (Offering, Bloodletting)
  aoe_usage             fraction of area attacks when at least 2 enemies are alive

Each probe is measured for the agent, the heuristic and the MCTS oracle: without a reference,
a number like "62 %" means nothing.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import evaluate as E  # noqa: E402
from stsenv.combat_env import sts, AT  # noqa: E402

STRENGTH_CARDS = {"Inflame", "Spot Weakness", "Flex", "Limit Break"}
HP_COST_CARDS = {"Offering", "Bloodletting", "Hemokinesis", "Seeing Red", "Rupture"}
AOE_CARDS = {"Cleave", "Whirlwind", "Thunderclap", "Immolate", "Shockwave", "Sword Boomerang"}


def rollout(pol, env, seed):
    """Replay one combat and return the list of cards played, with their context."""
    env.reset(options={"episode_seed": seed})
    log, done, tr, info = [], False, False, {}
    while not (done or tr):
        bc = env.bc
        a = pol.act(env)
        act = env.amap[a]
        if act.action_type == AT.CARD:
            card = bc.hand[act.source_idx]
            tgt = bc.monsters[act.target_idx] if act.target_idx < len(list(bc.monsters)) else None
            before = sum(m.cur_hp for m in bc.monsters if m.alive)
            log.append({"turn": bc.turn, "card": card.name, "block": bc.player.block,
                        "strength": bc.player.strength, "alive": sum(1 for m in bc.monsters if m.alive),
                        "target_vuln": tgt.vulnerable if tgt else 0,
                        "hand": [c.name for c in bc.hand]})
            _, _, done, tr, info = env.step(a)
            log[-1]["damage"] = before - sum(m.cur_hp for m in env.bc.monsters if m.alive)
        else:
            _, _, done, tr, info = env.step(a)
    return log, info


def probes(logs):
    out = {}
    # --- Strength investment ---
    occasions = invested = 0
    for log in logs:
        t1 = [e for e in log if e["turn"] == 0]
        if not t1 or not (STRENGTH_CARDS & set(t1[0]["hand"])):
            continue
        occasions += 1
        played = [e["card"] for e in t1]
        first_atk = next((i for i, c in enumerate(played) if c not in STRENGTH_CARDS
                          and any(e["damage"] > 0 for e in t1 if e["card"] == c)), len(played))
        invested += any(c in STRENGTH_CARDS for c in played[:first_atk])
    out["strength_first"] = (invested / occasions, occasions) if occasions else (float("nan"), 0)

    # --- debuff before big hits ---
    big = [e for log in logs for e in log if e["damage"] >= 10]
    out["debuff_before_big_hit"] = ((sum(e["target_vuln"] > 0 for e in big) / len(big), len(big))
                                    if big else (float("nan"), 0))
    # --- Body Slam played with block ---
    bs = [e for log in logs for e in log if e["card"] == "Body Slam"]
    out["body_slam_block"] = ((float(np.mean([e["block"] for e in bs])), len(bs))
                              if bs else (float("nan"), 0))
    # --- HP-cost cards ---
    n = sum(1 for log in logs for e in log if e["card"] in HP_COST_CARDS)
    out["hp_for_tempo"] = (n / len(logs), n)
    # --- area attacks with several enemies ---
    multi = [e for log in logs for e in log if e["alive"] >= 2 and e["damage"] > 0]
    out["aoe_usage"] = ((sum(e["card"] in AOE_CARDS for e in multi) / len(multi), len(multi))
                        if multi else (float("nan"), 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="strength")
    ap.add_argument("--encounter", default="cultist")
    ap.add_argument("--extra", type=int, default=3)
    ap.add_argument("--pool", default="train")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--runs", default="", help="comma-separated run subfolders to probe (default: all)")
    a = ap.parse_args()

    base = E.ROOT / "results" / "first_combat" / f"{a.deck}_{a.encounter}_x{a.extra}"
    specs = [{"kind": "heuristic", "key": "heuristic"}, {"kind": "mcts", "sims": 2000, "key": "mcts@2000"}]
    wanted = a.runs.split(",") if a.runs else None
    for d in sorted(p for p in base.iterdir() if p.is_dir() and (p / "config.json").exists()):
        if wanted and d.name not in wanted:
            continue
        cfg = json.load(open(d / "config.json"))
        ck = sorted(d.glob("ckpt_*"))[-1]
        specs.append({"kind": "sb3" if ck.suffix == ".zip" else "torch", "path": str(ck),
                      "algo": cfg["algo"], "encoding": cfg.get("encoding", "onehot"),
                      "run": d.name, "key": d.name})

    rows = {}
    for sp in specs:
        env = E.get_env((a.deck, a.encounter), sp.get("encoding", "onehot"), a.extra, a.pool)
        pol = E.load_policy(sp, env)
        logs, infos = [], []
        for s in range(a.episodes):
            log, info = rollout(pol, env, s)
            logs.append(log); infos.append(info)
        rows[sp["key"]] = (probes(logs), float(np.mean([i["hp_lost"] for i in infos])))
        print(".", end="", flush=True)
    print()

    names = list(next(iter(rows.values()))[0])
    print(f"\ndeck '{a.deck}' vs {a.encounter}, {a.pool} cards, {a.episodes} combats")
    print(f"{'policy':26} {'HP lost':>10} " + " ".join(f"{n:>22}" for n in names))
    for k, (pr, hp) in rows.items():
        cells = []
        for n in names:
            v, cnt = pr[n]
            cells.append(f"{'—':>22}" if cnt == 0 or v != v else f"{v:>18.2f}({cnt:>3})")
        print(f"{k:26} {hp:>10.2f} " + " ".join(cells))
    out = base / f"behavior_{a.pool}.json"
    json.dump({k: {"hp_lost": hp, **{n: pr[n] for n in names}} for k, (pr, hp) in rows.items()},
              open(out, "w"), indent=1)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
