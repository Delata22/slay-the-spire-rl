#!/usr/bin/env bash
# V0 / V1 / V2 ladder on the same scenario (strength deck vs cultist, k=2, held-out cards):
#   V0  masked Discrete(116) + compact one-hot      (card identity, fixed slot)
#   V1  masked Discrete(116) + effect vectors       (effects, fixed slot)
#   V2  scorer f(s, phi(a))                         (effects, variable number of actions)
# Everything else is identical: DQN, same hyperparameters, same seeds, same budget.
set -u
cd "$(dirname "$0")"
DECK=${DECK:-strength}
ENC=${ENC:-cultist}
EXTRA=${EXTRA:-2}
STEPS=${STEPS:-300000}
SEEDS=${SEEDS:-"0 1 2"}
JOBS=${JOBS:-4}
D=../../results/first_combat/${DECK}_${ENC}_x${EXTRA}/logs; mkdir -p "$D"
{
  for s in $SEEDS; do
    echo "v0 $s"; echo "v1 $s"; echo "v2 $s"
  done
} | xargs -P "$JOBS" -L1 bash -c '
  case $0 in
    v0) nice -n 10 python train.py --algo dqn --config tuned --encoding onehot_pool --seed $1 \
          --deck '"$DECK"' --encounter '"$ENC"' --extra-cards '"$EXTRA"' --pool train \
          --steps '"$STEPS"' > '"$D"'/train_dqn_tuned_onehot_pool_s$1.log 2>&1 ;;
    v1) nice -n 10 python train.py --algo dqn --config tuned --encoding effects --seed $1 \
          --deck '"$DECK"' --encounter '"$ENC"' --extra-cards '"$EXTRA"' --pool train \
          --steps '"$STEPS"' > '"$D"'/train_dqn_tuned_effects_s$1.log 2>&1 ;;
    v2) nice -n 10 python train_scorer.py --algo dqn --seed $1 \
          --deck '"$DECK"' --encounter '"$ENC"' --extra-cards '"$EXTRA"' --pool train \
          --steps '"$STEPS"' > '"$D"'/train_dqn_scorer_effects_s$1.log 2>&1 ;;
  esac
  echo "done: $0 s$1"'
