# Reproducing every table and figure

All commands run from `experiments/first_combat/` with `OMP_NUM_THREADS=1` (without it each multiprocessing
worker opens one BLAS thread per core and training is ~3.7× slower). Times were measured on an
i7-14650HX (16 cores / 24 threads) + RTX 5070 Ti laptop, with 4–8 runs in parallel; a run alone is
2–3× faster. Each grid script writes training logs to `results/first_combat/<experiment>/logs/`.

Evaluation always uses combat seeds 0..499 (training seeds are drawn from [1e6, 2e6)), so every policy plays
the same 500 combats. `evaluate.py` evaluates every checkpoint of every run in the experiment folder plus the
references (random, heuristic, MCTS@200/2000, exact optimal) and writes `eval*.json` and
`eval_episodes*.npz`. The optimal-play bound is cached in `optimal_hp_lost_<pool>.npy`.

| README item | experiment folder | commands | time |
|---|---|---|---|
| Result 1 (fixed deck, algorithms × presets × encodings) | `seed4` | `bash run_matrix.sh` then `python evaluate.py --scenario 4` | ~2 h training (21 runs × 400k steps), ~15 min eval |
| Result 2 (random pool, held-out cards) | `seed4_x3` | `bash run_deck.sh`, then `python evaluate.py --extra-cards 3 --pool eval` and `--pool train` | ~1.5 h + 2 × 10 min |
| Result 3 (V0 / V1 / V2 ladder), top figure | `strength_cultist_x2` | `bash run_v2.sh`, then `python evaluate.py --deck strength --encounter cultist --extra-cards 2 --pool eval` and `--pool train` | ~45 min + 2 × 10 min |
| Result 4 (card description) | `strength_cultist_x5` | `bash run_fx.sh` (nominal, both, V3) and `bash run_v3.sh` (measured, V3 measured), then `evaluate.py --deck strength --encounter cultist --extra-cards 5 --n-eval 400 --pool eval` / `--pool train` | ~40 min + 2 × 10 min |
| Result 5 (training-pool size sweep) | `random_cultist_x3_p{6,12,24,54}` | `bash run_pool_size.sh` (trains and evaluates) | ~40 min training (24 runs, 8 in parallel) + ~35 min eval |
| Behaviour probes | `strength_<encounter>_x3` | `bash run_archetypes.sh`, `bash finish_archetypes.sh` (evaluate + `behavior.py` + registry) | ~1 h per encounter |
| Gap analysis | any folder with scorer runs | `python gap_analysis.py --dir strength_cultist_x5 --run dqn_scorer_both_effects_s0 --pool eval` | ~1 min |
| MCTS diagnosis | `seed4` | `python diagnose_mcts.py`, `python emulate_scumsearch.py`, `python lethal_check.py` | ~5 min |
| Figures | — | `python ../make_figures.py` (reads only the shipped `results/`) | ~20 s |

Other utilities:

- `registry.py` regenerates `EXPERIMENTS.md`, a per-experiment summary of every evaluation file.
- `convergence.py` measures steps / combats / wall-clock time to reach 95 % of the final gain.
- `plot.py` draws learning curves (HP lost vs training steps) for one experiment folder.
- `validate_pool.py` plays every Ironclad card in the simulator and lists the ones that crash it.

Pinned software versions used for the reported runs: Python 3.12.3, torch 2.14.0 (CUDA), numpy 2.5.3,
gymnasium 1.3.0, sb3-contrib 2.9.0, `sts_lightspeed` @7476a81 + `sts-rl-agent` @54d7402 patch, pybind11 v2.13.6.
