#!/usr/bin/env bash
# Multi-seed eval for imagined-φ (H1) vs v2 / L2 / random.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export STABLEWM_HOME="${STABLEWM_HOME:-$ROOT/../stablewm}"
V2="$STABLEWM_HOME/checkpoints/pusht/lewm_phi_v2/reach.pt"
IMG="$STABLEWM_HOME/checkpoints/pusht/lewm_phi_imagined_v1/reach.pt"
EVAL="$ROOT/eval_results/pusht/lewm_phi_imagined_v1_eval"
PY="${PY:-$ROOT/.venv/bin/python}"
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
  echo "START $(date -Is)"
  for seed in 0 1 2; do
    SH="--env pusht --protocol online_offset --pair-mode short_horizon --episodes 20 --device cuda --no-video --collector kinematic --collect-episodes 320"
    run_one short "$seed" E1_l2_c1 "$SH --plan-cost l2_z"
    run_one short "$seed" E2_phi_v2 "$SH --plan-cost phi_d --phi-weights $V2"
    run_one short "$seed" E2_phi_imagined "$SH --plan-cost phi_d --phi-weights $IMG"
    run_one short "$seed" E4_random "$SH --plan-cost phi_d"

    OFF="--env pusht --protocol online_offset --pair-mode offset --episodes 50 --device cuda --no-video --collector kinematic --collect-episodes 200"
    run_one offset "$seed" E1_l2_c1 "$OFF --plan-cost l2_z"
    run_one offset "$seed" E2_phi_v2 "$OFF --plan-cost phi_d --phi-weights $V2"
    run_one offset "$seed" E2_phi_imagined "$OFF --plan-cost phi_d --phi-weights $IMG"
    run_one offset "$seed" E4_random "$OFF --plan-cost phi_d"
  done
  echo "ALL_IMAGINED_EVAL_DONE $(date -Is)"
} 2>&1 | tee "$EVAL/campaign_console.log"

"$PY" "$ROOT/scripts/aggregate_imagined_multiseed.py" --root "$EVAL"
