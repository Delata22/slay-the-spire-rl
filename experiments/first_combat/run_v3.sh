#!/usr/bin/env bash
# V2 vs V3 at k=5: does describing the draw pile card by card change anything?
#   V2  scorer + summarized piles (size, mean cost, breakdown by type)
#   V3  scorer + piles as a MULTISET (each card: static + memorized effects), order never given
# Everything else identical: DQN, same hyperparameters, same seeds, same budget.
set -u
cd "$(dirname "$0")"
DECK=${DECK:-strength}
ENC=${ENC:-cultist}
EXTRA=${EXTRA:-5}
STEPS=${STEPS:-150000}
SEEDS=${SEEDS:-"0 1 2"}
JOBS=${JOBS:-4}
D=../../results/first_combat/${DECK}_${ENC}_x${EXTRA}/logs; mkdir -p "$D"
{
  for s in $SEEDS; do echo "summary $s"; echo "multiset $s"; done
} | xargs -P "$JOBS" -L1 bash -c '
  nice -n 10 python train_scorer.py --algo dqn --piles $0 --seed $1 \
    --deck '"$DECK"' --encounter '"$ENC"' --extra-cards '"$EXTRA"' --pool train \
    --steps '"$STEPS"' > '"$D"'/train_dqn_scorer_$0_s$1.log 2>&1
  echo "done: $0 s$1"'
