#!/usr/bin/env python3
"""Where does the gap to optimal come from? Regret distribution, then transcripts of the worst combats.

  python gap_analysis.py --run dqn_scorer_both_effects_s0 --dir strength_cultist_x5 --pool eval

1. distribution: fraction of combats played perfectly, share of total regret carried by the worst ones;
2. for the worst combats: the agent's line AND the optimal line, side by side.

The optimal line knows the draw order: it is not a reachable target, but a reference for naming
the mistake (missed lethal move, wasted block, setup too slow...).
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import evaluate as E  # noqa: E402
from stsenv.combat_env import sts, legal_map, AT  # noqa: E402


def optimal_line(bc, hp0, cap=200_000, borne_initiale=10 ** 9):
    """Best action sequence with knowledge of the future -> (HP lost, descriptions, cap reached).

    `cap` bounds the number of nodes: beyond it, the best line found so far is returned, which may
    not be optimal. The computed regret is then UNDER-estimated (the true optimum would be better).

    `borne_initiale` (initial bound) seeds the search with the agent's result."""
    best = [borne_initiale, None]        # seeded with the agent's result: only search for better
    nodes = [0]

    def rec(st, seq):
        nodes[0] += 1
        if nodes[0] > cap:
            return
        if st.outcome != sts.Outcome.UNDECIDED:
            lost = hp0 - st.player.cur_hp if st.outcome == sts.Outcome.PLAYER_VICTORY else hp0
            if lost < best[0]:
                best[0], best[1] = lost, list(seq)
            return
        if hp0 - st.player.cur_hp >= best[0]:
            return
        _, amap, _ = legal_map(st)
        seen, order = set(), []
        for i, a in sorted(amap.items()):
            key = (st.hand[a.source_idx].name if a.action_type == AT.CARD else "end", a.target_idx)
            if key in seen:
                continue
            seen.add(key)
            # try damaging moves first: good lines are found early,
            # so pruning kicks in quickly even with a node cap
            c0 = st.clone(); before = sum(m.cur_hp for m in st.monsters if m.alive)
            try:
                a.execute(c0)
                gain = before - sum(m.cur_hp for m in c0.monsters if m.alive)
            except Exception:
                gain = -1
            order.append((-gain, i, a))
        for _, i, a in sorted(order, key=lambda x: x[0]):
            c = st.clone()
            label = (f"{st.hand[a.source_idx].name}" + (f" -> m{a.target_idx}" if a.target_idx < 5
                       and st.hand[a.source_idx].requires_target else "")) \
                if a.action_type == AT.CARD else "end turn"
            a.execute(c)
            rec(c, seq + [(st.turn + 1, label)])

    rec(bc.clone(), [])
    return best + [nodes[0] > cap]


def agent_line(pol, env, seed):
    env.reset(options={"episode_seed": seed})
    seq, done, tr, info = [], False, False, {}
    while not (done or tr):
        bc = env.bc
        a = pol.act(env)
        act = env.amap[a]
        lib = (bc.hand[act.source_idx].name + (f" -> m{act.target_idx}" if act.target_idx < 5
               and bc.hand[act.source_idx].requires_target else "")) \
            if act.action_type == AT.CARD else "end turn"
        seq.append((bc.turn + 1, lib))
        _, _, done, tr, info = env.step(a)
    return info, seq


def fmt(seq):
    out, cur = [], None
    for t, lib in seq:
        if t != cur:
            out.append(f"  T{t}:"); cur = t
        out[-1] += f" {lib} |"
    return "\n".join(x.rstrip(" |") for x in out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="strength_cultist_x5")
    ap.add_argument("--run", default="dqn_scorer_both_effects_s0")
    ap.add_argument("--pool", default="eval")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--worst", type=int, default=3)
    ap.add_argument("--cap", type=int, default=200_000, help="node cap for the optimal line search")
    a = ap.parse_args()

    base = E.ROOT / "results" / "first_combat" / a.dir
    cfg = json.load(open(base / a.run / "config.json"))
    deck, enc = cfg.get("deck", "random"), cfg.get("encounter")
    spec = {"kind": "scorer" if cfg.get("head") == "scorer" else "torch",
            "path": str(sorted((base / a.run).glob("ckpt_*"))[-1]),
            "algo": cfg["algo"], "encoding": cfg.get("encoding", "onehot"), "run": a.run, "key": a.run}
    env = E.get_env((deck, enc), spec["encoding"], cfg.get("extra_cards", 0), a.pool)
    pol = E.load_policy(spec, env)

    rows = []
    for s in range(a.n):
        info, seq = agent_line(pol, env, s)
        env.reset(options={"episode_seed": s})
        opt, oseq, capped = optimal_line(env.bc, env.bc.player.cur_hp, a.cap,
                                        borne_initiale=info["hp_lost"] + 1)
        if oseq is None:                     # nothing better than the agent within the explored budget
            opt, oseq = info["hp_lost"], seq
        rows.append({"seed": s, "agent": info["hp_lost"], "opt": opt, "regret": info["hp_lost"] - opt,
                     "aseq": seq, "oseq": oseq, "turns": info["turns"], "capped": capped})

    reg = np.array([r["regret"] for r in rows], float)
    print(f"{a.run}: {a.n} combats, pool {a.pool}")
    print(f"  HP lost {np.mean([r['agent'] for r in rows]):.2f} | optimal {np.mean([r['opt'] for r in rows]):.2f} "
          f"| mean regret {reg.mean():.2f}")
    print(f"  combats played optimally: {np.mean(reg <= 0):.0%} | regret <= 1 HP: {np.mean(reg <= 1):.0%}")
    top = np.argsort(-reg)
    part = reg[top[:max(1, a.n // 20)]].sum() / max(1e-9, reg.sum())
    print(f"  the worst 5% of combats carry {part:.0%} of total regret")
    print(f"  regret: median {np.median(reg):.1f}, p90 {np.percentile(reg, 90):.1f}, max {reg.max():.0f}")
    nb = sum(r["capped"] for r in rows)
    if nb:
        print(f"  WARNING: {nb}/{len(rows)} combats hit the node cap; "
              f"the optimum there is approximate, so the reported regret is UNDER-estimated")

    # --- aggregates: where does the gap come from? ---
    ta = np.array([max(t for t, _ in r["aseq"]) for r in rows], float)
    to = np.array([max(t for t, _ in r["oseq"]) for r in rows], float)
    print(f"\n  turns: agent {ta.mean():.2f} | optimal {to.mean():.2f} | "
          f"the agent takes more turns in {np.mean(ta > to):.0%} of combats")
    for d in (0, 1, 2):
        m = (ta - to) == d
        if m.sum():
            print(f"    +{d} turn(s) ({m.sum():3d} combats) -> mean regret {reg[m].mean():5.2f}")
    m = (ta - to) >= 3
    if m.sum():
        print(f"    +3 turns or more ({m.sum():3d} combats) -> mean regret {reg[m].mean():5.2f}")

    from collections import Counter
    ca, co = Counter(), Counter()
    for r in rows:
        for _, lib in r["aseq"]:
            ca[lib.split(" ->")[0]] += 1
        for _, lib in r["oseq"]:
            co[lib.split(" ->")[0]] += 1
    print(f"\n  cards played per combat:  {'card':18} {'agent':>7} {'optimal':>8}")
    for c in sorted(set(ca) | set(co), key=lambda k: -(co[k] + ca[k])):
        print(f"{'':28} {c:18} {ca[c]/len(rows):>7.2f} {co[c]/len(rows):>8.2f}")

    first_a = Counter(next((lib.split(" ->")[0] for t, lib in r["aseq"] if t == 1), "-") for r in rows)
    first_o = Counter(next((lib.split(" ->")[0] for t, lib in r["oseq"] if t == 1), "-") for r in rows)
    print(f"\n  first card of the combat: {'card':18} {'agent':>7} {'optimal':>8}")
    for c in sorted(set(first_a) | set(first_o), key=lambda k: -(first_o[k] + first_a[k]))[:6]:
        print(f"{'':28} {c:18} {first_a[c]/len(rows):>7.0%} {first_o[c]/len(rows):>8.0%}")

    for i in top[:a.worst]:
        r = rows[i]
        print("\n" + "=" * 70)
        print(f"seed {r['seed']}: agent {r['agent']} HP lost, optimal {r['opt']} (regret {r['regret']:.0f})")
        print("--- agent"); print(fmt(r["aseq"]))
        print("--- optimal (knows the draw pile)"); print(fmt(r["oseq"]))


if __name__ == "__main__":
    main()
