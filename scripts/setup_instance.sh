#!/usr/bin/env bash
# ==============================================================================
# Vast.ai Instance Setup & Bootstrap Script
# ==============================================================================
# Sets up the environment, GPU packages, uv, dependencies, W&B, and dataset.
# Run this once on the remote Vast.ai instance after launching.
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=========================================================="
echo " Starting Vast.ai Environment Setup in ${PROJECT_ROOT}"
echo "=========================================================="

# 1. System packages for rendering, Mujoco, and utilities
echo "[1/6] Installing necessary system packages..."
if command -v tmux &> /dev/null && command -v git &> /dev/null && command -v rsync &> /dev/null && ldconfig -p | grep -q "libGL.so" 2>/dev/null; then
    echo "  System packages already installed, skipping apt-get."
elif command -v apt-get &> /dev/null; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y -q || true
    apt-get install -y -q --no-install-recommends \
        git curl wget tmux htop rsync \
        libgl1-mesa-glx libosmesa6-dev libglew-dev patchelf \
        build-essential || true
fi


# 2. Install uv (fast Python package installer & resolver)
echo "[2/6] Ensuring 'uv' is installed..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
echo "Using uv version: $(uv --version)"

# 3. Load .env if present
echo "[3/6] Loading environment variables..."
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
    echo "  Loaded environment variables from .env"
else
    echo "  [WARN] No .env file found! Please copy your .env file to the instance."
fi

# 4. Install Python dependencies
echo "[4/6] Installing project dependencies..."
pip install --no-cache-dir --upgrade -e .

# 5. Verify GPU, PyTorch, and WandB
echo "[5/6] Verifying GPU and credentials..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
fi

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'bfloat16 supported: {torch.cuda.is_bf16_supported()}')
"

# Login to W&B if key provided
if [ -n "${WANDB_API_KEY}" ]; then
    echo "Authenticating Weights & Biases..."
    wandb login "${WANDB_API_KEY}" || true
else
    echo "  [INFO] WANDB_API_KEY not set. Set it in .env to enable W&B metric syncing."
fi

# Test Hugging Face connectivity if configured
if [ -n "${HF_TOKEN}" ]; then
    echo "Testing Hugging Face connectivity..."
    python scripts/sync_hf.py test || echo "  [WARN] Hugging Face connection test failed. Check credentials in .env"
else
    echo "  [INFO] HF_TOKEN not set in .env. Checkpoints will be saved locally only."
fi


# 6. Pre-download dataset (default: tworooms)
DATASET_NAME="${1:-tworooms}"
DATA_DIR="data/${DATASET_NAME}"
echo "[6/6] Checking training dataset (${DATASET_NAME})..."
if [ -d "${DATA_DIR}" ] && [ -n "$(ls -A "${DATA_DIR}" 2>/dev/null)" ]; then
    echo "  Dataset '${DATASET_NAME}' already present at ${DATA_DIR}."
else
    echo "  Downloading '${DATASET_NAME}' dataset..."
    python src/dataset/download.py "${DATASET_NAME}" --dest_dir "${DATA_DIR}"
fi

echo "=========================================================="
echo " Setup complete! You are ready to start training."
echo " Run: bash scripts/train_vast.sh"
echo "=========================================================="
