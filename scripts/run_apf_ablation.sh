#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# APF Ablation Study Launcher for v3
# ------------------------------------------------------------------------------
# Usage:
#   ./run_apf_ablation.sh
#   ./run_apf_ablation.sh --scale large
#   ./run_apf_ablation.sh --dry-run
#   ./run_apf_ablation.sh --apf-only
#
# For full background execution (survives terminal close):
#   nohup ./run_apf_ablation.sh --scale default > ablation.log 2>&1 &
# ------------------------------------------------------------------------------

# --- CLI defaults -----------------------------------------------------------
SCALE="default"
DRY_RUN=false
FORCE=false
APF_ONLY=false
BASELINE_ONLY=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scale)
      SCALE="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=true; shift ;;
    --force)
      FORCE=true; shift ;;
    --apf-only)
      APF_ONLY=true; shift ;;
    --baseline-only)
      BASELINE_ONLY=true; shift ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --scale {default|large}   Training scale (default: default)"
      echo "  --dry-run                 Print experiment plan without running"
      echo "  --force                   Ignore existing runs and re-run all"
      echo "  --apf-only                Skip baseline, run only APF variants"
      echo "  --baseline-only           Run only the baseline (no APF)"
      echo "  -h, --help                Show this help"
      echo ""
      echo "Examples:"
      echo "  $0 --dry-run"
      echo "  $0 --scale large --apf-only"
      echo "  nohup $0 > ablation.log 2>&1 &"
      exit 0 ;;
    *)
      echo "[ERROR] Unknown option: $1"; exit 1 ;;
  esac
done

# --- Scale presets ----------------------------------------------------------
if [[ "$SCALE" == "large" ]]; then
  TIMESTEPS=10000000
  NUM_ENVS=128
else
  TIMESTEPS=2000000
  NUM_ENVS=50
fi

# --- Experiment matrix ------------------------------------------------------
declare -a EXP_NAMES
declare -a EXP_ARGS

if [[ "$APF_ONLY" != "true" ]]; then
  EXP_NAMES+=("baseline_no_apf")
  EXP_ARGS+=("--timesteps $TIMESTEPS --num_envs $NUM_ENVS --headless")
fi

if [[ "$BASELINE_ONLY" != "true" ]]; then
  for att in 0.3 0.5 1.0; do
    for rep in -0.3 -0.5 -1.0; do
      name="apf_att${att}_rep${rep}"
      EXP_NAMES+=("$name")
      EXP_ARGS+=("--enable_apf --apf_attractive_weight $att --apf_repulsive_weight $rep --timesteps $TIMESTEPS --num_envs $NUM_ENVS --headless")
    done
  done
fi

# --- Path resolution --------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ISAACLAB_SH="$REPO_ROOT/IsaacLab/isaaclab.sh"
TRAIN_SCRIPT="$REPO_ROOT/omniperception_isaacdrone-v3/scripts/test_train_skrl.py"
LOGS_DIR="$REPO_ROOT/omniperception_isaacdrone-v3/logs"
ABLATION_LOG_DIR="$LOGS_DIR/apf_ablation_logs"

export PYTHONPATH="$REPO_ROOT/omniperception_isaacdrone-v3/source/omniperception_isaacdrone"

CHECK_DEDUP_SCRIPT="$SCRIPT_DIR/_check_dedup.py"
if [[ ! -f "$CHECK_DEDUP_SCRIPT" ]]; then
  echo "[ERROR] Dedup helper not found: $CHECK_DEDUP_SCRIPT"
  exit 1
fi

if [[ ! -f "$ISAACLAB_SH" ]]; then
  echo "[ERROR] IsaacLab launcher not found: $ISAACLAB_SH"
  exit 1
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "[ERROR] Training script not found: $TRAIN_SCRIPT"
  exit 1
fi

# --- Python cmd detection ---------------------------------------------------
PYTHON_CMD=""
for cmd in python python3; do
  if command -v "$cmd" > /dev/null 2>&1; then
    if "$cmd" -c "import sys; sys.exit(0)" > /dev/null 2>&1; then
      PYTHON_CMD="$cmd"
      break
    fi
  fi
done
if [[ -z "$PYTHON_CMD" ]]; then
  echo "[ERROR] Working Python not found in PATH"
  exit 1
fi

# --- Deduplication helper ---------------------------------------------------
check_existing() {
  local enable_apf="$1"
  local att="$2"
  local rep="$3"
  local ts="$4"
  local ne="$5"

  "$PYTHON_CMD" "$CHECK_DEDUP_SCRIPT" \
    --logs-dir "$LOGS_DIR" \
    --enable-apf "$enable_apf" \
    --apf-attractive-weight "$att" \
    --apf-repulsive-weight "$rep" \
    --timesteps "$ts" \
    --num-envs "$ne" 2>/dev/null
}

# --- Header -----------------------------------------------------------------
echo "=============================================================================="
echo "  APF Ablation Study Launcher"
echo "=============================================================================="
echo "[INFO] Scale        : $SCALE"
echo "[INFO] Timesteps    : $TIMESTEPS"
echo "[INFO] Num envs     : $NUM_ENVS"
echo "[INFO] Experiments  : ${#EXP_NAMES[@]}"
echo "[INFO] Logs dir     : $LOGS_DIR"
echo "[INFO] Ablation logs: $ABLATION_LOG_DIR"
echo "[INFO] Dry-run      : $DRY_RUN"
echo "[INFO] Force        : $FORCE"
echo ""

mkdir -p "$ABLATION_LOG_DIR"

skipped=0
to_run=0

# --- Main loop --------------------------------------------------------------
for i in "${!EXP_NAMES[@]}"; do
  name="${EXP_NAMES[$i]}"
  args="${EXP_ARGS[$i]}"
  log_file="$ABLATION_LOG_DIR/${name}.log"

  # Parse APF params from name for dedup
  if [[ "$name" == "baseline_no_apf" ]]; then
    enable_apf=false
    att="0.5"
    rep="-0.5"
  else
    enable_apf=true
    att="${name#apf_att}"
    att="${att%%_rep*}"
    rep="${name#*_rep}"
  fi

  status="RUN"
  if [[ "$FORCE" != "true" ]]; then
    result=$(check_existing "$enable_apf" "$att" "$rep" "$TIMESTEPS" "$NUM_ENVS" || true)
    if [[ "$result" == "FOUND" ]]; then
      status="SKIP"
      ((skipped++)) || true
    else
      ((to_run++)) || true
    fi
  else
    ((to_run++)) || true
  fi

  echo "[$status] $name"
  if [[ "$status" == "RUN" && "$DRY_RUN" != "true" ]]; then
    echo "[INFO] Launching: $name"
    echo "[INFO] Command  : $ISAACLAB_SH -p $TRAIN_SCRIPT $args"
    echo "[INFO] Log file : $log_file"
    cd "$REPO_ROOT/IsaacLab"
    nohup "$ISAACLAB_SH" -p "$TRAIN_SCRIPT" $args > "$log_file" 2>&1 &
    PID=$!
    wait $PID
    exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
      echo "[WARN] Experiment '$name' exited with code $exit_code"
    else
      echo "[INFO] Completed: $name"
    fi
  fi
  echo ""
done

echo "=============================================================================="
echo "  Summary: $to_run run, $skipped skipped"
echo "  Dry-run: $DRY_RUN, Force: $FORCE"
echo "=============================================================================="

if [[ "$DRY_RUN" == "true" ]]; then
  echo ""
  echo "[TIP] This was a dry-run. To execute, run without --dry-run:"
  echo "      ./run_apf_ablation.sh --scale $SCALE"
fi
