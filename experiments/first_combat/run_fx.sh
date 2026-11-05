#!/usr/bin/env bash
# How should a card be described? Three answers, same budget and seeds (scorer, k=5):
#   measured  what it does NOW        (Strike = 9 on a Vulnerable target) — contextual
#   nominal   what is printed on it   (Strike = 6, always)                — context-free
#   both      the two concatenated
# + a rerun of V3 (draw pile as a multiset) with NOMINAL values: this time draw-pile cards
#   are described from the first turn, without having to be played first.
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
  for s in $SEEDS; do
    echo "summary nominal $s"; echo "summary both $s"; echo "multiset nominal $s"
  done
} | xargs -P "$JOBS" -L1 bash -c '
  nice -n 10 python train_scorer.py --algo dqn --piles $0 --fx $1 --seed $2 \
    --deck '"$DECK"' --encounter '"$ENC"' --extra-cards '"$EXTRA"' --pool train \
    --steps '"$STEPS"' > '"$D"'/train_dqn_scorer_$0_$1_s$2.log 2>&1
  echo "done: $0 $1 s$2"'
