#!/bin/bash
# Setup SAM 3.1 environment on the ML3D cluster
# Run this once to create the conda environment and install dependencies.
#
# Usage:
#   bash setup_sam3_env.sh
#
# Prerequisites:
#   - Request access to SAM 3.1 checkpoints on HuggingFace: https://huggingface.co/facebook/sam3.1
#   - Generate a HuggingFace access token

set -e

echo "=== SAM 3.1 Environment Setup ==="

# Check CUDA version
echo "Checking CUDA..."
nvidia-smi | grep "CUDA Version" || echo "WARNING: nvidia-smi not available (run on a GPU node)"

# Create conda environment
echo "Creating conda env 'sam3' with Python 3.12..."
conda create -n sam3 python=3.12 -y
eval "$(conda shell.bash hook)"
conda activate sam3

# Install PyTorch with CUDA support
echo "Installing PyTorch 2.10.0 with CUDA 12.8..."
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# Clone and install SAM 3
SAM3_DIR="${SAM3_DIR:-$(cd "$(dirname "$0")/../.." && pwd)/third_party/sam3}"
if [ ! -d "$SAM3_DIR" ]; then
    echo "Cloning SAM 3 repository..."
    git clone https://github.com/facebookresearch/sam3.git "$SAM3_DIR"
fi

echo "Installing SAM 3..."
cd "$SAM3_DIR"
pip install -e ".[notebooks]"

# Install optional fast inference dependencies
echo "Installing flash-attn and other optimizations..."
pip install einops ninja
pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128 || echo "WARNING: flash-attn-3 failed (non-critical)"

# Additional dependencies
pip install opencv-python matplotlib scikit-learn tqdm pillow

# HuggingFace authentication
echo ""
echo "=== HuggingFace Authentication ==="
echo "You need to authenticate to download SAM 3.1 checkpoints."
echo "1. Go to https://huggingface.co/facebook/sam3.1 and request access"
echo "2. Generate a token at https://huggingface.co/settings/tokens"
echo "3. Run: huggingface-cli login"
echo ""

# Verify installation
echo "=== Verifying installation ==="
python -c "import sam3; print(f'SAM 3 installed at: {sam3.__file__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "=== Setup complete ==="
echo "To use: conda activate sam3"
echo "To test: python run_sam3_masks.py --test"
