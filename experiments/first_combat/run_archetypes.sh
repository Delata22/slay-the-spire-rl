#!/usr/bin/env bash
# One experiment = one deck archetype x one encounter. One monster per training run.
# Usage: DECK=strength ENCOUNTERS="cultist red_slaver gremlin_nob" bash run_archetypes.sh
set -u
cd "$(dirname "$0")"
DECK=${DECK:-strength}
ENCOUNTERS=${ENCOUNTERS:-"cultist"}
STEPS=${STEPS:-300000}
EXTRA=${EXTRA:-3}
SEEDS=${SEEDS:-"0 1"}
ALGO=${ALGO:-dqn}
JOBS=${JOBS:-4}
for enc in $ENCOUNTERS; do
  for s in $SEEDS; do
    for code in onehot effects; do echo "$enc $s $code"; done
  done
done | xargs -P "$JOBS" -L1 bash -c '\
    D=../../results/first_combat/'"$DECK"'_$0_x'"$EXTRA"'/logs; mkdir -p $D; \
    nice -n 10 python train.py --algo '"$ALGO"' --config tuned --encoding $2 --seed $1 \
      --deck '"$DECK"' --encounter $0 --extra-cards '"$EXTRA"' --pool train \
      --steps '"$STEPS"' --n-envs 4 > $D/train_'"$ALGO"'_tuned_$2_s$1.log 2>&1; \
    echo "done: '"$DECK"' $0 $2 s$1"'
