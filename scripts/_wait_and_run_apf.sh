#!/usr/bin/env bash
# Wait for baseline PID to finish, then launch APF run
BASELINE_PID=1854914
REPO_ROOT="/home/descfly/code/isaac"
ISAACLAB_SH="$REPO_ROOT/IsaacLab/isaaclab.sh"
TRAIN_SCRIPT="$REPO_ROOT/omniperception_isaacdrone-v3/scripts/test_train_skrl.py"
LOG="$REPO_ROOT/omniperception_isaacdrone-v3/logs/apf_ablation_logs/apf_att0.5_rep-0.5.log"

echo "[INFO] Waiting for baseline PID $BASELINE_PID to finish..."
while kill -0 $BASELINE_PID 2>/dev/null; do
  sleep 30
done
echo "[INFO] Baseline done. Launching APF run (att=0.5, rep=-0.5)..."

cd "$REPO_ROOT/IsaacLab"
export PYTHONPATH="$REPO_ROOT/omniperception_isaacdrone-v3/source/omniperception_isaacdrone"
"$ISAACLAB_SH" -p "$TRAIN_SCRIPT" \
  --enable_apf \
  --apf_attractive_weight 0.5 \
  --apf_repulsive_weight -0.5 \
  --timesteps 2000000 \
  --num_envs 50 \
  --headless \
  > "$LOG" 2>&1
echo "[INFO] APF run finished. Exit code: $?"
