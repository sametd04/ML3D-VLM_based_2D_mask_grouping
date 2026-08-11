#!/bin/bash
set -e

# ---- Config ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="myenv"
PYTHON_BIN="python3.11"
REQUIREMENTS_FILE="$SCRIPT_DIR/../../MaskClustering/requirements.txt"
CUDA_TOOLKIT_DIR="$HOME/cuda-11.8"
CUDA_RUNFILE="cuda_11.8.0_520.61.05_linux.run"
CUDA_RUNFILE_URL="https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/$CUDA_RUNFILE"

# ---- 1. Check Python 3.11 is available ----
if ! command -v $PYTHON_BIN &> /dev/null; then
    echo "Error: $PYTHON_BIN not found. Install it first (e.g. 'sudo apt install python3.11 python3.11-venv')."
    exit 1
fi

# ---- 2. Install local CUDA 11.8 toolkit (no sudo needed) ----
if [ -d "$CUDA_TOOLKIT_DIR" ]; then
    echo ">>> CUDA 11.8 toolkit already installed at $CUDA_TOOLKIT_DIR, skipping download"
else
    echo ">>> Downloading CUDA 11.8 toolkit installer (~3GB, this will take a while)"
    wget -q --show-progress "$CUDA_RUNFILE_URL" -O "$CUDA_RUNFILE"
    echo ">>> Installing CUDA 11.8 toolkit to $CUDA_TOOLKIT_DIR"
    sh "$CUDA_RUNFILE" --silent --toolkit --installpath="$CUDA_TOOLKIT_DIR" --override
    echo ">>> Cleaning up installer file"
    rm -f "$CUDA_RUNFILE"
fi

# ---- 3. Point this shell at the local CUDA 11.8 toolkit ----
export CUDA_HOME="$CUDA_TOOLKIT_DIR"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
echo ">>> Using CUDA toolkit: $(nvcc --version | grep release)"

# ---- 4. Create and activate venv ----
echo ">>> Creating virtual environment: $SCRIPT_DIR/$ENV_NAME"
$PYTHON_BIN -m venv "$SCRIPT_DIR/$ENV_NAME"
source "$SCRIPT_DIR/$ENV_NAME/bin/activate"
echo ">>> Using Python: $(python --version)"

# ---- 5. Upgrade pip, pin setuptools (pkg_resources removed in setuptools>=82) ----
pip install --upgrade pip
pip install "setuptools<81" wheel

# ---- 6. Install PyTorch (CUDA 11.8, matches MaskClustering's docs) ----
echo ">>> Installing PyTorch 2.0.0 + torchvision 0.15.0 (CUDA 11.8)"
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118

# ---- 7. Pin NumPy <2 so pytorch3d compiles and runs against the 1.x ABI ----
echo ">>> Pinning numpy<2"
pip install "numpy<2"

# ---- 8. Build PyTorch3D against the pinned numpy ----
if python -c "import pytorch3d" &> /dev/null; then
    echo ">>> PyTorch3D already installed and importable, skipping build"
else
    echo ">>> Installing PyTorch3D from source (this can take 10-20+ minutes)"
    mkdir -p "$SCRIPT_DIR/third_party"
    cd "$SCRIPT_DIR/third_party"

    if [ ! -d "pytorch3d" ]; then
        git clone https://github.com/facebookresearch/pytorch3d.git
    fi

    cd pytorch3d
    rm -rf build *.egg-info  # clear any stale build artifacts from earlier attempts
    pip install -e . --no-build-isolation
    cd "$SCRIPT_DIR"
fi

# ---- 9. Install remaining requirements, pinning opencv and numpy to stay <2 ----
#          opencv-python>=4.10 requires numpy>=2, so cap it at 4.9.x
echo ">>> Installing requirements from $REQUIREMENTS_FILE"
pip install -r "$REQUIREMENTS_FILE" "opencv-python<4.10" "numpy<2"

# ---- 10. Extra tools (numpy pin repeated to block any transitive upgrade) ----
echo ">>> Installing gdown, jupyterlab, ipykernel"
pip install gdown jupyterlab ipykernel "numpy<2"

# ---- 11. Register this venv as a Jupyter kernel ----
echo ">>> Registering '$ENV_NAME' as a Jupyter kernel"
python -m ipykernel install --user --name="$ENV_NAME" --display-name "Python ($ENV_NAME)"

echo ""
echo "✅ Setup complete!"
echo "Activate later with: source $SCRIPT_DIR/$ENV_NAME/bin/activate"
echo "Launch Jupyter with: jupyter lab"
echo "Then select kernel: Python ($ENV_NAME)"
echo ""
echo "⚠️  Note: CUDA_HOME is set to $CUDA_TOOLKIT_DIR only in this script's shell session."
echo "If you open a new terminal later and need to rebuild anything, re-export it first:"
echo "  export CUDA_HOME=$CUDA_TOOLKIT_DIR"
echo "  export PATH=\$CUDA_HOME/bin:\$PATH"
echo "  export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"