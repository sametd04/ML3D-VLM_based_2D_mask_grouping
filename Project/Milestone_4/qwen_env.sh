#!/usr/bin/env bash
# Sets up a standalone venv ("qwen_env") for running Qwen3-VL via transformers.
# Kept separate from the MaskClustering env (start_env.sh) since that one pins
# older torch/numpy versions for Open3D/PyTorch3D compatibility.
#
# Usage:
#   chmod +x qwen_env.sh
#   ./qwen_env.sh [path-to-create-env-in]
#
# If no path is given, the env is created at ./qwen_env (relative to where you run this).

set -euo pipefail

ENV_NAME="qwen_env"
ENV_DIR="${1:-$(pwd)/${ENV_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=================================================="
echo " Setting up '${ENV_NAME}' at: ${ENV_DIR}"
echo "=================================================="

if ! command -v "${PYTHON_BIN}" &>/dev/null; then
    echo "ERROR: ${PYTHON_BIN} not found. Set PYTHON_BIN env var to your python3 path." >&2
    exit 1
fi

echo "==> [1/6] Creating virtual environment"
if [ -d "${ENV_DIR}" ]; then
    echo "    Removing existing env at ${ENV_DIR}"
    rm -rf "${ENV_DIR}"
fi
"${PYTHON_BIN}" -m venv "${ENV_DIR}"

# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"

echo "==> [2/6] Upgrading pip"
pip install --upgrade pip --quiet

echo "==> [3/6] Installing torch (cu124) then transformers stack"
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate "qwen-vl-utils==0.0.14"
pip install pyyaml opencv-python pillow numpy tqdm scikit-learn

echo "==> [4/6] Installing Jupyter kernel support"
pip install ipykernel

echo "==> [5/6] Registering '${ENV_NAME}' as a Jupyter kernel"
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})"

echo "==> [6/6] Verifying installation"
python - <<'PYEOF'
import torch
import transformers
from transformers import Qwen3VLForConditionalGeneration  # noqa: F401

print(f"transformers version: {transformers.__version__}")
print(f"torch version:       {torch.__version__}")
print(f"CUDA available:      {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:                 {torch.cuda.get_device_name(0)}")
PYEOF

deactivate

echo ""
echo "=================================================="
echo " Done."
echo " - Env location:  ${ENV_DIR}"
echo " - In Jupyter, switch kernel to: 'Python (${ENV_NAME})'"
echo ""
echo " To run inference:"
echo "   source ${ENV_DIR}/bin/activate"
echo "   python inference.py ...   # or open run_inference.ipynb"
echo "=================================================="