#!/bin/bash
# Script to install Python 3.8 + PyTorch CUDA on Jetson Nano (JetPack 4.6)

echo "============================================="
echo "  INSTALLING CUDA + PYTHON 3.8 FOR JETSON"
echo "============================================="

# 1. Install Python 3.8
echo "[1/6] Installing Python 3.8..."
sudo apt-get update
sudo apt-get install -y python3.8 python3.8-dev python3.8-distutils libopenblas-base libopenmpi-dev

# 2. Install Pip for 3.8
echo "[2/6] Installing pip for Python 3.8..."
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3.8 get-pip.py
rm get-pip.py

# 3. Install PyTorch 1.13 (CUDA enabled)
echo "[3/6] Downloading PyTorch 1.13 (CUDA)..."
# Link for JetPack 4.6 / Python 3.8
wget https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.13.0a0+git7c98f3a-cp38-cp38-linux_aarch64.whl -O torch_jetson.whl

echo "Installing PyTorch..."
python3.8 -m pip install torch_jetson.whl
rm torch_jetson.whl

# 4. Install TorchVision
echo "[4/6] Installing TorchVision..."
# TorchVision 0.14 corresponds to PyTorch 1.13
python3.8 -m pip install torchvision==0.14.1

# 5. Install Ultralytics & OpenCV
echo "[5/6] Installing YOLOv8 & libs..."
python3.8 -m pip install ultralytics pandas psutil seaborn tqdm matplotlib
# Use standard opencv-python (it works well on 3.8)
python3.8 -m pip install opencv-python

# 6. Install Jetson GPIO
echo "[6/6] Installing GPIO..."
python3.8 -m pip install Jetson.GPIO
sudo groupadd -f -r gpio
sudo usermod -aG gpio $USER

echo ""
echo "============================================="
echo "  CAI DAT THANH CONG!"
echo "============================================="
echo "De chay Code voi CUDA:"
echo "  python3.8 inference.py"
