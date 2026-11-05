#!/usr/bin/env bash
# Wait for an archetype's training runs to finish, then evaluate, probe behavior,
# and regenerate the registry. Usage: DECK=strength ENCOUNTERS="cultist ..." bash finish_archetypes.sh
set -u
cd "$(dirname "$0")"
DECK=${DECK:-strength}
ENCOUNTERS=${ENCOUNTERS:-"cultist"}
EXTRA=${EXTRA:-3}
NRUNS=${NRUNS:-4}
EPISODES=${EPISODES:-150}
for enc in $ENCOUNTERS; do
  D=../../results/first_combat/${DECK}_${enc}_x${EXTRA}
  until [ "$(grep -lE 'terminé|done in' $D/logs/train_*.log 2>/dev/null | wc -l)" -ge "$NRUNS" ]; do sleep 30; done
  for pool in eval train; do
    python evaluate.py --deck "$DECK" --encounter "$enc" --extra-cards "$EXTRA" --pool $pool \
      --n-eval 500 --workers 14 > "$D/logs/eval_$pool.log" 2>&1
  done
  python behavior.py --deck "$DECK" --encounter "$enc" --extra "$EXTRA" --pool train \
    --episodes "$EPISODES" > "$D/logs/behavior.log" 2>&1
  echo "evaluated: $DECK $enc"
done
python registry.py
