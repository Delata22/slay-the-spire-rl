#!/usr/bin/env bash
# Training-pool diversity sweep: does generalisation to unseen cards improve with the number of
# distinct training cards? Nested training pools of 6, 12, 24 and 54 cards; the 18 held-out cards
# are the same for every size. Two agents, same DQN budget and seeds:
#   scorer   f(state, phi(action)), card described by measured + nominal effects (no card identity)
#   onehot   Discrete(116) masked head, compact one-hot card identity
# Random-effect cards are masked in the probe (--leak mask) so the descriptor cannot leak the RNG.
set -u
cd "$(dirname "$0")"
DECK=${DECK:-random}
ENC=${ENC:-cultist}
EXTRA=${EXTRA:-3}
STEPS=${STEPS:-150000}
SIZES=${SIZES:-"6 12 24 54"}
SEEDS=${SEEDS:-"0 1 2"}
JOBS=${JOBS:-8}
LOGS=../../results/first_combat/pool_size_logs; mkdir -p "$LOGS"
{
  for n in $SIZES; do for s in $SEEDS; do
    echo "scorer $n $s"; echo "onehot $n $s"
  done; done
} | xargs -P "$JOBS" -L1 bash -c '
  case $0 in
    scorer) nice -n 10 python train_scorer.py --algo dqn --fx both --leak mask --seed $2 \
              --deck '"$DECK"' --encounter '"$ENC"' --extra-cards '"$EXTRA"' --pool train \
              --train-pool-size $1 --steps '"$STEPS"' > '"$LOGS"'/scorer_p$1_s$2.log 2>&1 ;;
    onehot) nice -n 10 python train.py --algo dqn --config tuned --encoding onehot_pool --seed $2 \
              --deck '"$DECK"' --encounter '"$ENC"' --extra-cards '"$EXTRA"' --pool train \
              --train-pool-size $1 --steps '"$STEPS"' > '"$LOGS"'/onehot_p$1_s$2.log 2>&1 ;;
  esac
  echo "done: $0 p$1 s$2 ($(date +%H:%M))"'

# Evaluation: final and mid-training checkpoints, 500 paired combat seeds, on the run's own
# training pool ("seen") and on the 18 held-out cards ("unseen").
for n in $SIZES; do
  for pool in train eval; do
    nice -n 10 python evaluate.py --deck "$DECK" --encounter "$ENC" --extra-cards "$EXTRA" \
      --train-pool-size "$n" --pool "$pool" --n-eval 500 --workers 12 --mcts 2000 \
      --ckpt-steps "0,$((STEPS / 2)),$STEPS" > "$LOGS/eval_p${n}_${pool}.log" 2>&1
    echo "eval done: p$n $pool ($(date +%H:%M))"
  done
done
echo "ALL DONE ($(date +%H:%M))"
