#!/usr/bin/env bash
# Matrix: 2 algos x hyperparameter presets x encodings x 3 training seeds.
# 400k steps each (the plateau was reached by 400k in the 1M run).
set -u
cd "$(dirname "$0")"
STEPS=${STEPS:-400000}
JOBS=${JOBS:-4}
LOGS=../../results/first_combat/seed4/logs; mkdir -p $LOGS
{
  for s in 0 1 2; do
    for c in default tuned fast; do echo "ppo $c onehot  $s"; done
    for c in default tuned;      do echo "dqn $c onehot  $s"; done
    echo "ppo tuned   effects $s"
    echo "dqn default effects $s"
  done
} | xargs -P $JOBS -L1 bash -c 'set -- $0 $1 $2 $3; \
    nice -n 10 python train.py --algo $1 --config $2 --encoding $3 --seed $4 \
      --steps '"$STEPS"' --n-envs 4 > '"$LOGS"'/train_$1_$2_$3_s$4.log 2>&1; \
    echo "done: $1 $2 $3 s$4"'
