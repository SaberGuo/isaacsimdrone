#!/usr/bin/bash
# --------------------------------------------------
# IsaacSimDrone Test6 - SKRL PPO Training
# Usage:
#   ./run_test6_train.sh
#   ./run_test6_train.sh --num_envs 16 --timesteps 100000 --headless
#   ./run_test6_train.sh --num_envs 32 --timesteps 2000000 --headless
#   ./run_test6_train.sh --resume --num_envs 32 --timesteps 2000000 --headless
#
# Common flags:
#   --resume                     Auto-resume from newest checkpoint under logs/
#                                (alias for --resume_latest passed to the python script)
#   --num_envs <N>               Number of parallel envs (default: 32)
#   --timesteps <N>              Total training timesteps (default: 2_000_000)
#   --headless                   Run without GUI (recommended for training)
#   --seed <N>                   Random seed (default: 42)
#   --num_obstacles <N>          Number of obstacles (default: 100)
#   --learning_rate <float>      LR (default: 1e-4)
#   --rollouts <N>               Rollout steps per update (default: 256)
#   --learning_epochs <N>        PPO epochs per update (default: 8)
#   --checkpoint_interval <N>    Save checkpoint every N steps (default: 50000)
# --------------------------------------------------

export CONDA_PREFIX="$HOME/anaconda3/envs/env_isaaclab"
export PYTHONPATH="$HOME/code/isaac/omniperception_isaacdrone-v3/source/omniperception_isaacdrone"

# Resolve paths relative to this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ISAACLAB_SH="$REPO_ROOT/IsaacLab/isaaclab.sh"
TRAIN_SCRIPT="$REPO_ROOT/omniperception_isaacdrone-v3/scripts/test_train_skrl.py"

# Translate --resume -> --resume_latest before forwarding to python script
PASS_ARGS=()
for a in "$@"; do
    case "$a" in
        --resume) PASS_ARGS+=("--resume_latest") ;;
        *) PASS_ARGS+=("$a") ;;
    esac
done

cd "$REPO_ROOT/IsaacLab" || exit 1

echo "[INFO] Launching SKRL training ..."
echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Conda env: $CONDA_PREFIX"
echo "[INFO] Train script: $TRAIN_SCRIPT"
echo "[INFO] Forwarded args: ${PASS_ARGS[*]}"

if [ ! -f "$ISAACLAB_SH" ]; then
    echo "[ERROR] IsaacLab launcher not found: $ISAACLAB_SH"
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "[ERROR] Training script not found: $TRAIN_SCRIPT"
    exit 1
fi

"$ISAACLAB_SH" -p "$TRAIN_SCRIPT" --headless "${PASS_ARGS[@]}" &
PID=$!
wait $PID
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Training exited with code $EXIT_CODE"
else
    echo "[INFO] Training finished successfully."
fi

exit $EXIT_CODE
