#!/bin/bash
# One-shot environment setup for the MaskClustering + CropFormer pipeline on the ML3D cluster.
#
# Run this ONCE, in an interactive GPU session, from the repo root:
#
#   ssh <you>@ml3d.vc.in.tum.de
#   salloc --gpus=1
#   cd /path/to/ML3D-VLM_based_2D_mask_grouping
#   bash setup_env.sh
#
# It creates the conda env "maskclustering", installs PyTorch/pytorch3d/detectron2/CropFormer
# (cloning and building them under MaskClustering/third_party/, since that folder is
# gitignored and not shipped in the repo), and pip installs everything else from
# requirements.txt. Safe to re-run — it skips steps that are already done.
#
# Afterwards, just: conda activate maskclustering && jupyter notebook ...
# and run "Project/Milestone 1/maskclustering_pipeline.ipynb" top to bottom.

set -e

ENV_NAME="maskclustering"
PYTHON_VERSION="3.10"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MC_DIR="$REPO_ROOT/MaskClustering"
THIRD_PARTY="$MC_DIR/third_party"

echo "=== MaskClustering environment setup ==="

if ! command -v conda &> /dev/null; then
    echo "conda not found on PATH. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
    echo "Conda env '$ENV_NAME' already exists, reusing it."
else
    echo "Creating conda env '$ENV_NAME' (python $PYTHON_VERSION)..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi
conda activate "$ENV_NAME"

echo ""
echo "--- PyTorch (CUDA 11.8) ---"
python -c "import torch" 2>/dev/null && echo "torch already installed, skipping." || \
    conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

echo ""
echo "--- Pip dependencies (requirements.txt) ---"
pip install -r "$REPO_ROOT/requirements.txt"

mkdir -p "$THIRD_PARTY"
cd "$THIRD_PARTY"

echo ""
echo "--- PyTorch3D ---"
python -c "import pytorch3d" 2>/dev/null && echo "pytorch3d already installed, skipping." || {
    if [ ! -d "pytorch3d" ]; then
        echo "Cloning PyTorch3D..."
        git clone https://github.com/facebookresearch/pytorch3d.git
    fi
    echo "Building PyTorch3D from source (this can take 10-20 minutes)..."
    pip install -e pytorch3d
}

echo ""
echo "--- detectron2 ---"
if [ ! -d "detectron2" ]; then
    echo "Cloning detectron2..."
    git clone https://github.com/facebookresearch/detectron2.git
fi
python -c "import detectron2" 2>/dev/null && echo "detectron2 already installed, skipping." || \
    pip install -e detectron2

echo ""
echo "--- CropFormer (Entity repo) ---"
if [ ! -d "Entity" ]; then
    echo "Cloning Entity (CropFormer source)..."
    git clone https://github.com/qqlu/Entity.git
fi
if [ ! -d "detectron2/projects/CropFormer" ]; then
    echo "Copying CropFormer into detectron2/projects..."
    cp -r Entity/Entityv2/CropFormer detectron2/projects/
fi
# MaskClustering ships a customized mask_predict.py that batches over all sequences.
cp "$MC_DIR/mask_predict.py" detectron2/projects/CropFormer/demo_cropformer/mask_predict.py

echo "Building CropFormer entity_api (cheap no-op if already built)..."
(cd detectron2/projects/CropFormer/entity_api/PythonAPI && make)

if ! python -c "import MultiScaleDeformableAttention" 2>/dev/null; then
    echo "Building CropFormer deformable-attention CUDA ops..."
    (cd detectron2/projects/CropFormer/mask2former/modeling/pixel_decoder/ops && sh make.sh)
fi

echo "Installing mmcv (best-effort, not required for inference)..."
mim install mmcv || echo "WARNING: mmcv install failed — continuing, it is only needed for training/dataset registration, not for running the demo."

echo ""
echo "--- Jupyter kernel ---"
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

echo ""
echo "=== Setup complete ==="
echo "CropFormer checkpoint is NOT downloaded by this script (it requires your own HuggingFace"
echo "access grant) — the pipeline notebook will guide you through that on first run."
echo ""
echo "Next steps:"
echo "  conda activate $ENV_NAME"
echo "  cd $MC_DIR && jupyter notebook --no-browser --ip=0.0.0.0 --port 8888"
echo "Then open 'Project/Milestone 1/maskclustering_pipeline.ipynb' and run all cells."
