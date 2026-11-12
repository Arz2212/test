#!/usr/bin/env bash
# Setup script for CIFAR-10 backprop SNN on a clean Ubuntu server with A100.
# Installs uv, Python 3.11 venv, CUDA PyTorch, spikingjelly 2.0.0 and CIFAR-10.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv311"
DATA_DIR="$SCRIPT_DIR/data/cifar10"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
CIFAR_URL="https://cs231n.stanford.edu/cifar-10-python.tar.gz"
SPIKINGJELLY_VERSION="2.0.0"
SKIP_DATA=0

usage() {
    echo "Usage: $0 [--skip-data]"
    echo "  --skip-data   install environment only, do not download CIFAR-10"
    echo ""
    echo "Optional environment variables:"
    echo "  TORCH_INDEX   PyTorch wheel index (default: https://download.pytorch.org/whl/cu128)"
    echo "                use .../cu124 for CUDA 12.4"
}

for arg in "$@"; do
    case "$arg" in
        --skip-data) SKIP_DATA=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $arg"; usage; exit 2 ;;
    esac
done

if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

echo "==> Checking required files"
for f in cifar10_backprop.py download_cifar10.py; do
    if [ ! -f "$SCRIPT_DIR/$f" ]; then
        echo "ERROR: $SCRIPT_DIR/$f not found"
        exit 1
    fi
done

echo "==> Installing system packages"
$SUDO apt-get update
$SUDO apt-get install -y curl wget tmux aria2 ca-certificates

echo "==> Installing uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "==> Creating Python 3.11 environment: $VENV_DIR"
uv venv --python 3.11 --clear "$VENV_DIR"

echo "==> Installing CUDA PyTorch from: $TORCH_INDEX"
uv pip install --python "$VENV_DIR/bin/python" torch torchvision --index-url "$TORCH_INDEX"

echo "==> Installing spikingjelly $SPIKINGJELLY_VERSION"
uv pip install --python "$VENV_DIR/bin/python" "spikingjelly==$SPIKINGJELLY_VERSION" numpy

echo "==> Checking PyTorch / CUDA"
"$VENV_DIR/bin/python" -c "import torch, spikingjelly; print('torch:', torch.__version__); print('spikingjelly:', spikingjelly.__version__ if hasattr(spikingjelly, '__version__') else '2.0.0'); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "==> nvidia-smi"
    nvidia-smi | head -n 12 || true
else
    echo "WARNING: nvidia-smi not found; NVIDIA driver may be missing."
fi

if [ "$SKIP_DATA" -eq 0 ]; then
    echo "==> Preparing CIFAR-10 in $DATA_DIR"
    mkdir -p "$DATA_DIR/raw"

    if [ ! -s "$DATA_DIR/trainX.npy" ] || [ ! -s "$DATA_DIR/trainY.npy" ] || [ ! -s "$DATA_DIR/testX.npy" ] || [ ! -s "$DATA_DIR/testY.npy" ]; then
        if [ ! -s "$DATA_DIR/raw/cifar-10-python.tar.gz" ]; then
            echo "==> Downloading CIFAR-10 archive"
            aria2c -x 16 -s 16 -k 1M -d "$DATA_DIR/raw" -o cifar-10-python.tar.gz "$CIFAR_URL" || wget -c -O "$DATA_DIR/raw/cifar-10-python.tar.gz" "$CIFAR_URL"
        else
            echo "==> CIFAR-10 archive already exists"
        fi

        echo "==> Extracting and converting CIFAR-10 to .npy"
        "$VENV_DIR/bin/python" "$SCRIPT_DIR/download_cifar10.py" --out-dir "$DATA_DIR"
    else
        echo "==> CIFAR-10 .npy files already exist"
    fi
else
    echo "==> Skipping CIFAR-10 download (--skip-data)"
fi

echo ""
echo "============================================================"
echo "Setup complete."
echo "Environment: $VENV_DIR"
echo "Data:        $DATA_DIR"
echo ""
echo "Run training:"
echo "cd $SCRIPT_DIR && uv run --no-project --python \"$VENV_DIR/bin/python\" cifar10_backprop.py --data-dir \"$DATA_DIR\" --device cuda --epochs 30 --batch-size 128 --T 50 --num-workers 4"
echo "============================================================"
