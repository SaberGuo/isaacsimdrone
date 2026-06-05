#!/usr/bin/env bash
# Run baseline + apf(att=0.3,rep=-0.5) + apf(att=0.5,rep=-0.5) sequentially in background.
# Usage: nohup ./run_3way_compare.sh > logs/apf_ablation_logs/3way_main.log 2>&1 &

set -euo pipefail

export CONDA_PREFIX="$HOME/anaconda3/envs/env_isaaclab"

REPO_ROOT="/home/descfly/code/isaac"
ISAACLAB_SH="$REPO_ROOT/IsaacLab/isaaclab.sh"
TRAIN_SCRIPT="$REPO_ROOT/omniperception_isaacdrone-v3/scripts/test_train_skrl.py"
LOG_DIR="$REPO_ROOT/omniperception_isaacdrone-v3/logs/apf_ablation_logs"

TIMESTEPS=2000000
NUM_ENVS=50

export PYTHONPATH="$REPO_ROOT/omniperception_isaacdrone-v3/source/omniperception_isaacdrone"

mkdir -p "$LOG_DIR"

run_exp() {
  local name="$1"; shift
  local log="$LOG_DIR/${name}.log"
  echo "======================================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START: $name"
  echo "[CMD] $ISAACLAB_SH -p $TRAIN_SCRIPT $*"
  echo "======================================================================"
  cd "$REPO_ROOT/IsaacLab"
  "$ISAACLAB_SH" -p "$TRAIN_SCRIPT" "$@" > "$log" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE: $name"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: $name (exit $rc)"
  fi
  echo ""
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 3-way comparison started"
echo "Experiments: baseline | apf(att=0.3,rep=-0.5) | apf(att=0.5,rep=-0.5)"
echo "Timesteps: $TIMESTEPS  NumEnvs: $NUM_ENVS"
echo ""

#run_exp "baseline_no_apf" \
#  --timesteps $TIMESTEPS --num_envs $NUM_ENVS --headless

#run_exp "apf_att0.3_rep-0.5" \
#  --enable_apf --apf_attractive_weight 0.3 --apf_repulsive_weight -0.5 \
#  --timesteps $TIMESTEPS --num_envs $NUM_ENVS --headless

run_exp "apf_att0.5_rep-0.5" \
  --enable_apf --apf_attractive_weight 0.5 --apf_repulsive_weight -0.5 \
  --timesteps $TIMESTEPS --num_envs $NUM_ENVS --headless

echo "======================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All 3 experiments finished."
echo "Run compare_apf_vs_baseline.py to analyse results."
echo "======================================================================"
