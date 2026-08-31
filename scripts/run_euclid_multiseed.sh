#!/usr/bin/env bash
# Multi-seed Euclidean φ replicate: short_horizon n=20 + offset n=50.
# Weights: lewm_phi_v2 (frozen). Seeds 0,1,2 × E1/E2/E4.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export STABLEWM_HOME="${STABLEWM_HOME:-$ROOT/../stablewm}"
CKPT="$STABLEWM_HOME/checkpoints/pusht/lewm_phi_v2/reach.pt"
EVAL="$ROOT/eval_results/pusht/lewm_phi_euclid_multiseed"
PY="${PY:-$ROOT/.venv/bin/python}"

if [[ ! -f "$CKPT" ]]; then
  echo "missing weights: $CKPT" >&2
  exit 1
fi
mkdir -p "$EVAL"

run_one() {
  local proto="$1" seed="$2" tag="$3" extra="$4"
  local logdir="$EVAL/$proto/seed${seed}/$tag"
  mkdir -p "$logdir"
  echo "======== $proto seed=$seed $tag ========"
  # shellcheck disable=SC2086
  "$PY" eval_live.py $extra --seed "$seed" --log-dir "$logdir"
}

{
  echo "START $(date -Is) CKPT=$CKPT"
  for seed in 0 1 2; do
    SH="--env pusht --protocol online_offset --pair-mode short_horizon --episodes 20 --device cuda --no-video --collector kinematic --collect-episodes 320"
    run_one short "$seed" E1_l2_c1 "$SH --plan-cost l2_z"
    run_one short "$seed" E2_phi_v2 "$SH --plan-cost phi_d --phi-weights $CKPT"
    run_one short "$seed" E4_random "$SH --plan-cost phi_d"

    OFF="--env pusht --protocol online_offset --pair-mode offset --episodes 50 --device cuda --no-video --collector kinematic --collect-episodes 200"
    run_one offset "$seed" E1_l2_c1 "$OFF --plan-cost l2_z"
    run_one offset "$seed" E2_phi_v2 "$OFF --plan-cost phi_d --phi-weights $CKPT"
    run_one offset "$seed" E4_random "$OFF --plan-cost phi_d"
  done
  echo "ALL_EUCLID_MULTISEED_DONE $(date -Is)"
} 2>&1 | tee "$EVAL/campaign_console.log"

"$PY" "$ROOT/scripts/aggregate_euclid_multiseed.py" --root "$EVAL"
echo "wrote aggregate under $EVAL"
