#!/usr/bin/env bash
# B2 CEM horizon sweep — E1 L2 only, matched seed.
# PushT offset n=50 seed 0; Reacher live_reset n=20 seed 0.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"

echo "== PushT offset L2 horizon sweep seed=${SEED} =="
for H in 2 3 5 8; do
  python eval_live.py --env pusht --protocol online_offset --pair-mode offset \
    --episodes 50 --seed "$SEED" --collector kinematic --collect-episodes 200 \
    --plan-cost l2_z --horizon "$H" --no-video --device "$DEVICE" \
    --log-dir "eval_results/pusht/phase_b_horizon/offset_seed${SEED}_h${H}"
done

echo "== Reacher live_reset L2 horizon sweep seed=${SEED} =="
for H in 2 3 5 8; do
  python eval_live.py --env reacher --protocol live_reset \
    --episodes 20 --seed "$SEED" --plan-cost l2_z --horizon "$H" \
    --no-video --device "$DEVICE" \
    --log-dir "eval_results/reacher/phase_b_horizon/live_seed${SEED}_h${H}"
done

echo "done. compare success_rate in each log-dir metrics.json"
