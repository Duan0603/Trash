#!/bin/bash
# Script to install PyTorch 1.10 (CUDA) on Jetson Nano (JetPack 4.6, Python 3.6 Default)

echo "============================================="
echo "  INSTALLING CUDA + PYTORCH 1.10 (Recommended)"
echo "============================================="

# 1. Install Dependencies
echo "[1/5] Installing Dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip libopenblas-base libopenmpi-dev libjpeg-dev zlib1g-dev

# 2. Upgrade pip (Important for newer wheels)
echo "[2/5] Upgrading pip..."
python3 -m pip install --upgrade pip
python3 -m pip install "numpy<1.20"  # Older numpy for compatibility

# 3. Download PyTorch 1.10.0 (Python 3.6)
echo "[3/5] Downloading PyTorch 1.10 (CUDA)..."
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
echo "Installing PyTorch..."
python3 -m pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
rm torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# 4. Install TorchVision 0.11.1 (Matches PyTorch 1.10)
echo "[4/5] Installing TorchVision..."
# We need to compile or find wheel. Easiest is to install via git+https or find wheel.
# NVIDIA recommends simple pip install if matching version found.
# If fail, try:
python3 -m pip install "torchvision==0.11.1"

# 5. Install Ultralytics (Compatible Version)
echo "[5/5] Installing YOLOv8 (Ultralytics)..."
# Ultralytics dropped 3.6 support recently. Trying older stable.
python3 -m pip install "ultralytics>=8.0.0" "pandas" "requests" "matplotlib" "seaborn" "tqdm" "psutil"

# Install Opencv (already on JetPack, but ensure python bindings)
sudo apt-get install -y python3-opencv

echo ""
echo "============================================="
echo "  CAI DAT THANH CONG!"
echo "============================================="
echo "De chay Code:"
echo "  python3 inference.py"
