#!/usr/bin/env bash
# ==============================================================================
# Vast.ai Training Runner Script
# ==============================================================================
# Launches training with W&B logging, checkpoint syncing.
#
# Usage:
#   bash scripts/train_vast.sh [OPTIONS] [-- <train.py args>]
#
# Examples:
#   bash scripts/train_vast.sh
#   bash scripts/train_vast.sh --run-id run_01 --num-epochs 20 --batch-size 128
#   bash scripts/train_vast.sh --auto-stop --run-id exp_prod --num-epochs 50
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Ensure uv is in PATH
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Load environment variables
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

AUTO_STOP=false
RUN_IN_TMUX=false
PASSTHROUGH_ARGS=()

# Parse custom wrapper flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --auto-stop)
            AUTO_STOP=true
            shift
            ;;
        --tmux)
            RUN_IN_TMUX=true
            shift
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "=========================================================="
echo " Training on Vast.ai"
echo "=========================================================="

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu --format=csv,noheader || true
fi

# WandB check
if [ -n "${WANDB_API_KEY}" ]; then
    echo "  [W&B] Metrics will be logged to project: '${WANDB_PROJECT:-wm}'"
else
    echo "  [WARN] WANDB_API_KEY is not set. Metrics will not be synced to wandb."
fi

# Hugging Face check
if [ -n "${HF_TOKEN}" ] && [ -n "${HF_REPO_ID}" ]; then
    echo "  [HF] Checkpoints will be automatically synced to Hugging Face: '${HF_REPO_ID}'"
else
    echo "  [INFO] Hugging Face credentials not configured. Checkpoints will be stored locally."
fi


if [ "$AUTO_STOP" = true ]; then
    echo "  [AUTO-STOP] Instance will be stopped after training completes to save costs."
fi
echo "=========================================================="

# If user requested TMUX and we're not inside tmux, launch inside a detached session
if [ "$RUN_IN_TMUX" = true ] && [ -z "$TMUX" ]; then
    SESSION_NAME="wm_train"
    echo "Starting training in detached tmux session '${SESSION_NAME}'..."
    tmux new-session -d -s "${SESSION_NAME}" "bash scripts/train_vast.sh ${PASSTHROUGH_ARGS[*]}"
    echo "Training started in background!"
    echo "  To attach and view logs: tmux attach -t ${SESSION_NAME}"
    echo "  To detach from session : Press Ctrl+B then D"
    exit 0
fi

# Helper function to stop Vast.ai instance
stop_vast_instance() {
    if [ "$AUTO_STOP" = true ]; then
        echo "[AUTO-STOP] Preparing to stop Vast.ai instance..."
        INSTANCE_ID="${VAST_CONTAINERLABEL:-$CONTAINER_ID}"
        if [ -n "${INSTANCE_ID}" ]; then
            echo "[AUTO-STOP] Stopping instance ${INSTANCE_ID} using VastAI Python SDK..."
            python -c "
import os
from vastai import VastAI
api_key = os.getenv('VAST_API_KEY') or None
vast = VastAI(api_key=api_key) if api_key else VastAI()
vast.stop_instance(id=int('${INSTANCE_ID}'))
print('[AUTO-STOP] Instance stopped successfully.')
" || echo "[AUTO-STOP] Failed to stop instance automatically."
        else
            echo "[AUTO-STOP] Notice: No instance ID found in environment (VAST_CONTAINERLABEL or CONTAINER_ID)."
        fi
    fi
}


trap stop_vast_instance EXIT

# Run training
echo "Executing: python src/train.py ${PASSTHROUGH_ARGS[*]}"
python src/train.py "${PASSTHROUGH_ARGS[@]}"

echo "=========================================================="
echo " Training finished successfully!"
echo " Check W&B dashboard and Hugging Face repository for saved checkpoints."
echo "=========================================================="

