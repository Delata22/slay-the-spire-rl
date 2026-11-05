#!/usr/bin/env bash
# "Variable deck" scenario: 3 cards drawn at random from the Ironclad pool each episode.
# Train on the TRAIN pool (54 cards), evaluate on the EVAL pool (17 never-seen cards).
# This is the head-on test of the effect encoding: an unknown id tells the one-hot nothing,
# whereas an effect vector stays readable.
set -u
cd "$(dirname "$0")"
STEPS=${STEPS:-400000}
EXTRA=${EXTRA:-3}
JOBS=${JOBS:-4}
LOGS=../../results/first_combat/seed4_x${EXTRA}/logs; mkdir -p "$LOGS"
{
  for s in 0 1 2; do
    for enc in onehot effects; do
      echo "dqn tuned $enc $s"
      echo "ppo tuned $enc $s"
    done
  done
} | xargs -P "$JOBS" -L1 bash -c 'set -- $0 $1 $2 $3; \
    nice -n 10 python train.py --algo $1 --config $2 --encoding $3 --seed $4 \
      --extra-cards '"$EXTRA"' --pool train --steps '"$STEPS"' --n-envs 4 \
      > '"$LOGS"'/train_$1_$2_$3_s$4.log 2>&1; \
    echo "done: $1 $2 $3 s$4"'
